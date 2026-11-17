# -*- coding: utf-8 -*-
"""Transient load failures in the walk (17/11/2026). A walk row failed to load with VRAM at 7.0 of 16 GB and 4.9 GB of
RAM free, between rows that ran, and the walk stopped there. Now such a row is retried once; failing again it is
recorded load_failed and the walk goes on. Only VRAM evidence ends the walk: an overflow, a peak near full, or a peak
predicted near full from the walk's own rows. Two load_failed rows in a row also end it (a limit we cannot name)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marian import launch, machine, plan, profile
from marian.profile import CacheOptions, ThreadOptions

from test_m13 import FakeMarian, fake_machine
from test_offline import write_gguf

LAYERS = 8


def toy8(d: Path) -> Path:
    model = d / "toy8.gguf"
    write_gguf(model, {"general.architecture": "toy", "toy.block_count": LAYERS, "toy.expert_count": 8,
                       "toy.expert_used_count": 2, "toy.context_length": 4096})
    return model


class Walker(FakeMarian):
    """An 8-layer MoE on the synthetic 8,151 MiB GPU (1,000 MiB used by others): each layer of experts moved to the GPU
    adds 900 MiB, so n_cpu_moe 8, 6, 4, 2 use 1,000 to 6,400 MiB and n_cpu_moe 0 would need 8,200 (over the top).
    `flaky` maps n_cpu_moe to how many of its first starts fail as a transient load failure."""

    def __init__(self, flaky=None, **kw):
        super().__init__(**kw)
        self.flaky = dict(flaky or {})

    def share(self) -> int:
        c = self.running
        return 0 if c is None else (LAYERS - c.n_cpu_moe) * 900

    def start(self, model, c, log):
        self.started.append(c)
        if self.flaky.get(c.n_cpu_moe, 0) > 0:
            self.flaky[c.n_cpu_moe] -= 1
            with open(log, "a", encoding="utf-8") as f:
                f.write("E CUDA error: out of memory\n")
            return False, "failed to start (see log)"
        if 1000 + (LAYERS - c.n_cpu_moe) * 900 > 8151:   # a real VRAM failure: nothing ran, the peak stays low
            return False, "failed to start (see log)"
        self.running = c
        return True, "started"

    def complete(self, text, n):
        import threading
        threading.Event().wait(0.12)
        return {"prompt_n": 3000, "prompt_per_second": 800.0,
                "predicted_per_second": 20.0 + (LAYERS - self.running.n_cpu_moe) * 2}


def walk_of(eng):
    with tempfile.TemporaryDirectory() as d, fake_machine(eng), \
            mock.patch.object(profile, "model_geometry", lambda m: None):
        out = profile.run(toy8(Path(d)), eng, [4096], ["f16"], 2, None, Path(d) / "profiles", 16,
                          cache=CacheOptions(enabled=False), sweep=ThreadOptions(enabled=False))
        doc = json.loads(out.read_text(encoding="utf-8"))
    return doc, [r for r in doc["rows"] if r["arm"] == "walk"]


def starts(eng, k):
    return sum(1 for c in eng.started if c.n_cpu_moe == k)


class Prediction(unittest.TestCase):
    CELL = [{"arm": "walk", "status": "ok", "n_cpu_moe": 8, "vram_peak_mib": 2000},
            {"arm": "walk", "status": "ok", "n_cpu_moe": 6, "vram_peak_mib": 3600},
            {"arm": "walk", "status": "skipped_ram", "n_cpu_moe": 5}]

    def test_straight_line_from_the_last_two_ok_rows(self):
        self.assertEqual(profile.predicted_vram_mib({"n_cpu_moe": 2}, self.CELL, "n_cpu_moe"), 6800)
        self.assertIsNone(profile.predicted_vram_mib({"n_cpu_moe": 2}, self.CELL[:1], "n_cpu_moe"))

    def test_evidence(self):
        ev = lambda row: profile.vram_evidence(row, self.CELL, "n_cpu_moe", 8151, 300)  # noqa: E731
        self.assertFalse(ev({"status": "start_failed", "n_cpu_moe": 4, "vram_peak_mib": 3000}))  # 5,200 predicted
        self.assertTrue(ev({"status": "start_failed", "n_cpu_moe": 0, "vram_peak_mib": 3000}))   # 8,400 predicted
        self.assertTrue(ev({"status": "start_failed", "n_cpu_moe": 4, "vram_peak_mib": 7900}))   # measured near full
        self.assertTrue(ev({"status": "overflow", "n_cpu_moe": 4, "vram_peak_mib": 3000}))
        self.assertTrue(profile.vram_evidence({"status": "start_failed"}, [], "n_cpu_moe", 0, 300))  # no total


class Walk(unittest.TestCase):
    def test_a_transient_failure_is_retried_and_the_retry_counts(self):
        eng = Walker(flaky={4: 1})
        doc, walk = walk_of(eng)
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk],
                         [(8, "ok"), (6, "ok"), (4, "ok"), (2, "ok"), (0, "start_failed")])
        self.assertEqual(walk[2]["first_attempt"]["status"], "start_failed")
        self.assertEqual(starts(eng, 4), 2)
        self.assertEqual(starts(eng, 0), 1)       # predicted 8,200 MiB: VRAM evidence, no retry, the walk ends

    def test_failing_three_times_is_load_failed_and_the_walk_goes_on(self):
        eng = Walker(flaky={4: 3})        # twice as recorded, once without pinned memory (E91)
        doc, walk = walk_of(eng)
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk],
                         [(8, "ok"), (6, "ok"), (4, "load_failed"), (2, "ok"), (0, "start_failed")])
        self.assertEqual(walk[2]["failed_as"], "start_failed")
        self.assertIn("failed twice", walk[2]["note"])
        self.assertEqual(starts(eng, 4), 3)
        self.assertIn("load_failed", launch.skipped_summary(doc["rows"]))
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000)
        self.assertEqual(best["n_cpu_moe"], 2)    # the planner reaches past the failed row

    def test_two_load_failures_in_a_row_end_the_walk(self):
        eng = Walker(flaky={6: 3, 4: 3})
        _, walk = walk_of(eng)
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk],
                         [(8, "ok"), (6, "load_failed"), (4, "load_failed")])
        self.assertEqual(starts(eng, 2), 0)

    def test_without_two_ok_rows_a_failure_is_retried(self):
        eng = Walker(flaky={6: 1})
        _, walk = walk_of(eng)
        self.assertEqual(walk[1]["status"], "ok")
        self.assertEqual(starts(eng, 6), 2)


if __name__ == "__main__":
    unittest.main()
