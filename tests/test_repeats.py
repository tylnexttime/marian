"""Offline tests of calibration's probe repeats (1-3, default 1) and `marian recalibrate` (Chris, 30/10/2026; debt
row 23). No GPU and no server: a fake engine and fake memory readings.
python -m pytest -q tests"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marian import cli, machine, plan, profile
from marian.profile import CacheOptions, ThreadOptions, sha256_head

import test_m13 as m13
import test_prefill_ubatch as pu


class Seq:
    """A minimal engine whose decode speed follows a fixed sequence, one value per completion."""

    def __init__(self, tg):
        self.tg, self.calls = list(tg), 0

    def tokenize(self, text):
        return 30

    def complete(self, text, n):
        v = self.tg[self.calls % len(self.tg)]
        self.calls += 1
        return {"prompt_n": 32, "prompt_per_second": 10.0 * v, "predicted_per_second": v}


class Probe(unittest.TestCase):
    def test_one_probe_by_default_and_the_row_is_unchanged(self):
        eng = Seq([12.0, 40.0, 41.0])
        r = profile.probe(eng, 4096, 16)
        self.assertEqual(eng.calls, 1)
        self.assertEqual(r["tg_tps"], 12.0)
        self.assertNotIn("tg_tps_runs", r)
        self.assertNotIn("tg_spread", r)

    def test_three_probes_keep_every_reading_the_median_and_the_spread(self):
        eng = Seq([12.0, 40.0, 41.0])
        r = profile.probe(eng, 4096, 16, repeats=3)
        self.assertEqual(eng.calls, 3)
        self.assertEqual(r["tg_tps_runs"], [12.0, 40.0, 41.0])
        self.assertEqual(r["tg_tps"], 40.0)                         # the median outvotes the cold first probe
        self.assertEqual(r["pp_tps"], 400.0)
        self.assertEqual(r["tg_spread"], round((41.0 - 12.0) / 40.0, 3))

    def test_a_repeat_count_outside_1_to_3_is_refused(self):
        for n in (0, 4, -1):
            with self.assertRaises(ValueError):
                profile.probe(Seq([1.0]), 4096, 16, repeats=n)


def run_profile(eng, d: Path, repeats=1):
    with m13.fake_machine(eng, 16000), \
            mock.patch.object(profile, "model_geometry", lambda m: object()), \
            mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 500), \
            mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 12000):
        model = d / "toy.gguf"
        if not model.exists():
            m13.toy_moe(d)
        out = profile.run(model, eng, [4096], ["f16"], 2, None, d / "profiles", 16, cache=CacheOptions(),
                          sweep=ThreadOptions(enabled=False), repeats=repeats)
        return json.loads(out.read_text(encoding="utf-8")), out


class Calibration(unittest.TestCase):
    def test_every_row_is_probed_three_times_on_one_server(self):
        eng = pu.PrefillFake()
        with tempfile.TemporaryDirectory() as d:
            doc, _ = run_profile(eng, Path(d), repeats=3)
        ok = [r for r in doc["rows"] if r["status"] == "ok"]
        self.assertTrue(ok)
        for r in ok:
            self.assertEqual(len(r["tg_tps_runs"]), 3)
            self.assertIn("tg_spread", r)
        self.assertEqual(len(eng.started), len(doc["rows"]))         # one server per row, not one per probe
        self.assertEqual(doc["probe"]["repeats"], 3)
        self.assertEqual(doc["calibration"]["repeats"], 3)

    def test_the_default_records_its_parameters_and_no_repeat_fields(self):
        with tempfile.TemporaryDirectory() as d:
            doc, _ = run_profile(pu.PrefillFake(), Path(d))
        self.assertEqual(doc["probe"], {"fill": 0.75, "n_predict": 16})
        self.assertFalse(any("tg_tps_runs" in r for r in doc["rows"]))
        k = doc["calibration"]
        self.assertEqual((k["ctx"], k["kv"], k["step"], k["n_predict"], k["repeats"]), ([4096], ["f16"], 2, 16, 1))
        self.assertNotIn("n_ubatch", k["cache"])

    def test_a_bad_repeat_count_is_refused_before_any_server(self):
        eng = pu.PrefillFake()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            run_profile(eng, Path(d), repeats=4)
        self.assertEqual(eng.started, [])


class Params(unittest.TestCase):
    def test_recorded_parameters_are_used_as_recorded(self):
        k = profile.calibration_params({"calibration": {"ctx": [8192], "kv": ["q8_0"], "repeats": 2}})
        self.assertTrue(k["recorded"])
        self.assertEqual((k["ctx"], k["kv"], k["repeats"]), ([8192], ["q8_0"], 2))

    def test_an_older_profile_is_read_back_from_its_rows_and_probe(self):
        old = {"probe": {"prompt_tokens": 32, "n_predict": 128},
               "rows": [{"ctx": 8192, "kv": "f16"}, {"ctx": 8192, "kv": "f16"}, {"ctx": 4096, "kv": "q8_0"}]}
        k = profile.calibration_params(old)
        self.assertFalse(k["recorded"])
        self.assertEqual((k["ctx"], k["kv"], k["probe_prompt"], k["n_predict"], k["repeats"]),
                         ([4096, 8192], ["f16", "q8_0"], 32, 128, 1))

    def test_the_archive_is_never_loaded_by_the_planner(self):
        p = Path("x") / "fp-mid-llama.cpp.json"
        a = profile.archive_name(p, {"measured_utc": "2026-10-29T05:12:01+00:00"})
        self.assertEqual(a.name, "fp-mid-llama.cpp.json.20261029T051201Z")
        self.assertFalse(a.name.endswith(".json"))


class Compare(unittest.TestCase):
    def test_rows_that_moved_beyond_the_band_are_listed(self):
        row = {"ctx": 8192, "kv": "f16", "n_cpu_moe": 30, "threads": None, "load_mode": "mmap", "moe_cache_mib": 3136,
               "arm": "rung", "status": "ok"}
        old = {"rows": [dict(row, tg_tps=39.39), dict(row, moe_cache_mib=3648, tg_tps=18.07)]}
        new = {"rows": [dict(row, tg_tps=11.6), dict(row, moe_cache_mib=3648, tg_tps=19.0),
                        dict(row, moe_cache_mib=4096, tg_tps=30.0)]}
        c = profile.compare(old, new)
        self.assertEqual((c["matched"], c["moved"]), (2, 1))          # 4096 was not measured before: not matched
        moved = [r for r in c["rows"] if r["moved"]]
        self.assertEqual(moved[0]["old_tg_tps"], 39.39)
        self.assertEqual(moved[0]["ratio"], round(11.6 / 39.39, 3))


class Recalibrate(unittest.TestCase):
    def cli(self, d: Path, eng, *args):
        out, err = io.StringIO(), io.StringIO()
        with m13.fake_machine(eng, 16000), mock.patch.object(cli, "PROFILES", d / "profiles"), \
                mock.patch.object(cli, "LlamaCpp", lambda path, port: eng), \
                mock.patch.object(profile, "model_geometry", lambda m: object()), \
                mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 500), \
                mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 12000), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(["recalibrate", str(d / "toy.gguf"), "--engine", "x", *args])
        return rc, out.getvalue(), err.getvalue()

    def test_without_a_profile_it_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            m13.toy_moe(Path(d))
            rc, _, err = self.cli(Path(d), pu.PrefillFake())
        self.assertEqual(rc, 2)
        self.assertIn("no profile to recalibrate", err)

    def test_it_reruns_the_recorded_calibration_and_keeps_the_old_profile(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old, path = run_profile(pu.PrefillFake(), d)
            rc, out, _ = self.cli(d, pu.PrefillFake(), "--repeats", "2")
            self.assertEqual(rc, 0)
            kept = [p.name for p in (d / "profiles").iterdir() if ".json.2" in p.name]
            new = json.loads(path.read_text(encoding="utf-8"))
            with m13.fake_machine(None):                               # the fingerprint the profiles were written under
                docs = plan.load_docs(d / "profiles", d / "toy.gguf")
        self.assertEqual(len(kept), 1)                                 # the old session, beside the new
        self.assertEqual(new["calibration"]["repeats"], 2)             # the flag overrides the recorded count
        self.assertEqual(new["calibration"]["ctx"], old["calibration"]["ctx"])
        self.assertEqual(len(docs), 1)                                 # the planner sees only the new profile
        self.assertIn("configurations measured in both sessions", out)

    def test_a_failed_rerun_leaves_the_old_profile_where_the_planner_reads_it(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, path = run_profile(pu.PrefillFake(), d)
            before = path.read_bytes()
            with mock.patch.object(profile, "run", side_effect=RuntimeError("server died")), \
                    self.assertRaises(RuntimeError):
                self.cli(d, pu.PrefillFake())
            self.assertEqual(path.read_bytes(), before)


class Cli(unittest.TestCase):
    def test_calibrate_passes_repeats_and_refuses_four(self):
        seen = []

        def fake_run(*a, repeats=1, **k):
            seen.append(repeats)
            raise SystemExit(0)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cli.profile, "run", fake_run):
            for extra in ([], ["--repeats", "3"]):
                with self.assertRaises(SystemExit):
                    pu.run_cli(Path(d), "calibrate", "--engine", "x", *extra)
            with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):
                pu.run_cli(Path(d), "calibrate", "--engine", "x", "--repeats", "4")
        self.assertEqual(seen, [1, 3])
        self.assertEqual(e.exception.code, 2)                          # argparse: not a choice


if __name__ == "__main__":
    unittest.main()
