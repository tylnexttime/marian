# -*- coding: utf-8 -*-
"""RAM-limited calibration rows (17/11/2026). On a laptop with less RAM available than the walk's first row reads
in (every expert on the CPU, -lm none), that row used to fail and end the walk with no row at all, so the planner
refused. Now a row whose estimated RAM does not fit is skipped before it starts, a row that runs RAM out is recorded
as ram_exhausted, neither ends the walk, the cache arms get a memory-mapped twin of the first row as their base, and
the planner says which rows were skipped and why."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from marian import launch, machine, plan, profile
from marian.engines import Config
from marian.profile import CacheOptions, ThreadOptions

from test_m13 import FakeMarian, fake_machine, toy_moe

MiB = 2**20


class Estimate(unittest.TestCase):
    S = {"block_count": 40, "bytes": 21 * 1024 * MiB, "moe": True}
    GEOM = SimpleNamespace(total_expert_bytes=19 * 1024 * MiB)

    def test_moe_share_of_the_expert_bytes(self):
        need = lambda k: profile.host_ram_need_mib(self.S, Config(n_gpu_layers=999, n_cpu_moe=k), self.GEOM)  # noqa: E731
        self.assertEqual(need(40), 19 * 1024)                    # all experts on the CPU: every expert byte
        self.assertEqual(need(20), 19 * 512)
        self.assertEqual(need(0), 0)

    def test_without_geometry_the_whole_file_is_counted(self):
        c = Config(n_gpu_layers=999, n_cpu_moe=40)
        self.assertEqual(profile.host_ram_need_mib(self.S, c, None), 21 * 1024)
        self.assertEqual(profile.host_ram_need_mib(self.S, c, object()), 21 * 1024)   # a stub with no total

    def test_mmap_rows_are_not_prechecked(self):
        c = Config(n_gpu_layers=999, n_cpu_moe=40, load_mode="mmap")
        self.assertIsNone(profile.host_ram_need_mib(self.S, c, self.GEOM))

    def test_dense_share_of_the_layers_on_the_cpu(self):
        s = {"block_count": 31, "bytes": 32 * 1024 * MiB, "moe": False}
        self.assertEqual(profile.host_ram_need_mib(s, Config(n_gpu_layers=16), None), 16 * 1024)
        self.assertEqual(profile.host_ram_need_mib(s, Config(n_gpu_layers=32), None), 0)


class Precheck(unittest.TestCase):
    def test_skip_records_why_and_starts_nothing(self):
        row = {"status": None}
        self.assertFalse(profile.ram_precheck(row, 19456, 15459))
        self.assertEqual(row["status"], "skipped_ram")
        self.assertIn("19456 MiB", row["note"])
        self.assertIn("15459 MiB available", row["note"])

    def test_room_to_spare_starts(self):
        row = {}
        self.assertTrue(profile.ram_precheck(row, 10000, 15459))
        self.assertNotIn("status", row)
        self.assertEqual(row["ram_need_mib"], 10000)
        self.assertTrue(profile.ram_precheck({}, None, 100))      # nothing to estimate: start

    def test_the_reserve_is_kept(self):
        self.assertFalse(profile.ram_precheck({}, 14000, 15459))   # would leave 1,459 MiB, under the 1,536 reserve
        self.assertTrue(profile.ram_precheck({}, 13900, 15459))

    def test_exhausted_relabels_a_failure_only(self):
        row = {"status": "start_failed", "note": "failed to start (see log)", "ram_avail_low_mib": 241}
        self.assertTrue(profile.ram_exhausted(row))
        self.assertEqual((row["status"], row["failed_as"]), ("ram_exhausted", "start_failed"))
        self.assertIn("241 MiB", row["note"])
        self.assertFalse(profile.ram_exhausted({"status": "start_failed", "ram_avail_low_mib": 9000}))
        self.assertFalse(profile.ram_exhausted({"status": "ok", "ram_avail_low_mib": 100}))


class RamLimitedMarian(FakeMarian):
    """FakeMarian (toy model: 2 expert layers) on a box where reading every expert in (-lm none, n_cpu_moe 2) runs RAM
    out: when `fail_load` is set, that start drops available RAM to 241 MiB while it loads, then fails."""

    def __init__(self, fail_load=False, **kw):
        super().__init__(**kw)
        self.fail_load, self.loading = fail_load, False

    def start(self, model, c, log):
        if self.fail_load and c.n_cpu_moe == 2 and c.load_mode == "none" and not c.moe_cache_mib:
            self.started.append(c)
            self.loading = True
            threading.Event().wait(0.12)                   # long enough for the sampler to see the low
            self.loading = False
            with open(log, "a", encoding="utf-8") as f:
                f.write("E CUDA error: out of memory\n")
            return False, "failed to start (see log)"
        return super().start(model, c, log)


class Walk(unittest.TestCase):
    def run_profile(self, eng, ram, geom):
        with tempfile.TemporaryDirectory() as d, fake_machine(eng), \
                mock.patch.object(machine, "ram_mib", lambda: (32000, ram())), \
                mock.patch.object(profile, "model_geometry", lambda m: geom), \
                mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 500), \
                mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 12000):
            out = profile.run(toy_moe(Path(d)), eng, [4096], ["f16"], 2, None, Path(d) / "profiles", 16,
                              cache=CacheOptions(), sweep=ThreadOptions(enabled=False))
            return json.loads(out.read_text(encoding="utf-8"))

    def test_a_row_that_does_not_fit_ram_is_skipped_and_the_walk_goes_on(self):
        eng = RamLimitedMarian()
        geom = SimpleNamespace(total_expert_bytes=16000 * MiB)        # n_cpu_moe 2 reads 16,000 MiB in
        doc = self.run_profile(eng, lambda: 12000, geom)
        walk = [r for r in doc["rows"] if r["arm"] in ("walk", "walk-mmap")]
        self.assertEqual([(r["arm"], r["n_cpu_moe"], r["status"]) for r in walk],
                         [("walk", 2, "skipped_ram"), ("walk", 0, "ok"), ("walk-mmap", 2, "ok")])
        self.assertFalse(any(c.n_cpu_moe == 2 and c.load_mode == "none" for c in eng.started))   # never started
        self.assertEqual(walk[2]["load_mode"], "mmap")
        self.assertTrue(any(r["arm"] == "cache" and r["status"] == "ok" for r in doc["rows"]))  # cache arms ran
        self.assertNotIn("skipped", doc["cache_search"]["cells"][0])
        self.assertEqual([(s["n_cpu_moe"], s["status"]) for s in doc["skipped"] if s.get("arm") == "walk"],
                         [(2, "skipped_ram")])
        self.assertIsNotNone(plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000))
        said = launch.skipped_summary(doc["rows"])
        self.assertIn("skipped_ram", said)
        self.assertIn("16000 MiB", said)

    def test_a_row_that_runs_ram_out_is_recorded_and_the_walk_goes_on(self):
        eng = RamLimitedMarian(fail_load=True)
        doc = self.run_profile(eng, lambda: 241 if eng.loading else 16000, None)
        walk = [r for r in doc["rows"] if r["arm"] in ("walk", "walk-mmap")]
        self.assertEqual([(r["arm"], r["n_cpu_moe"], r["status"]) for r in walk],
                         [("walk", 2, "ram_exhausted"), ("walk", 0, "ok"), ("walk-mmap", 2, "ok")])
        self.assertEqual(walk[0]["failed_as"], "start_failed")
        self.assertIsNotNone(plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000))

    def test_a_failure_with_ram_to_spare_is_not_a_ram_casualty(self):
        class LoadFails(FakeMarian):
            def start(self, model, c, log):
                if c.n_cpu_moe == 0 and not c.moe_cache_mib:
                    self.started.append(c)
                    return False, "failed to start (see log)"
                return super().start(model, c, log)
        eng = LoadFails()
        doc = self.run_profile(eng, lambda: 16000, None)
        walk = [r for r in doc["rows"] if r["arm"] == "walk"]
        # one ok row gives no VRAM prediction, so the failure is retried once and then recorded load_failed
        # (tests/test_load_retry.py); with 16 GB free it is never ram_exhausted
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk], [(2, "ok"), (0, "load_failed")])
        self.assertEqual(sum(1 for c in eng.started if c.n_cpu_moe == 0 and not c.moe_cache_mib), 2)
        self.assertFalse(any(r["arm"] == "walk-mmap" for r in doc["rows"]))   # the first row ran: no twin needed

    def test_nothing_ran_and_the_planner_says_why(self):
        rows = [dict(Config(ctx=4096, n_gpu_layers=999, n_cpu_moe=2).to_row(), arm="walk", status="skipped_ram",
                     note="needs about 16000 MiB of RAM with -lm none; 12000 MiB available, 1536 MiB kept free")]
        why = launch.explain(rows, 3000, "decode", "toy.gguf", free=None, margin=300, exact_only=False, stock=False,
                             ram_free=None, ram_floor=6144)
        self.assertIn("no row that ran", why)
        self.assertIn("skipped_ram", why)
        self.assertNotIn("larger --ctx", why)


if __name__ == "__main__":
    unittest.main()
