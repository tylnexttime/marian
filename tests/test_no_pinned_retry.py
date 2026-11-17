# -*- coding: utf-8 -*-
"""The pinned band (17/11/2026, E90 and E91). With -lm none the CPU expert weights go into one pinned host buffer; on a
machine where that buffer lands just under the page-locked limit, the pin succeeds and the upload aborts, on every
attempt, while GGML_CUDA_NO_PINNED=1 loads the same row. E89's retry repeats the same attempt, so such a row failed
twice and was lost as load_failed. Now a row that fails twice without RAM or VRAM evidence gets a third attempt with
no_pinned, which the profile row records and the planner carries to launch. Every other row's environment is
inherited unchanged."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marian import engines, launch, plan
from marian.engines import Config, LlamaCpp

from test_load_retry import Walker, starts, walk_of


class Banded(Walker):
    """Walker with rows in the pinned band: they fail every time they start with pinned memory, and load without."""

    def __init__(self, band=(), **kw):
        super().__init__(**kw)
        self.band = set(band)

    def start(self, model, c, log):
        if c.n_cpu_moe in self.band and not c.no_pinned:
            self.started.append(c)
            with open(log, "a", encoding="utf-8") as f:
                f.write("E CUDA error: out of memory\n")
            return False, "failed to start (see log)"
        return super().start(model, c, log)


def attempts(eng, k):
    return [c.no_pinned for c in eng.started if c.n_cpu_moe == k]


class Walk(unittest.TestCase):
    def test_a_row_in_the_pinned_band_is_recovered_without_pinned_memory(self):
        eng = Banded(band={4})
        _, walk = walk_of(eng)
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk],
                         [(8, "ok"), (6, "ok"), (4, "ok"), (2, "ok"), (0, "start_failed")])
        row = walk[2]
        self.assertIs(row["no_pinned"], True)
        self.assertEqual(row["first_attempt"]["status"], "start_failed")
        self.assertEqual(row["second_attempt"]["status"], "start_failed")
        self.assertEqual(attempts(eng, 4), [False, False, True])     # twice as recorded, then once without pinning

    def test_other_rows_keep_their_environment(self):
        eng = Banded(band={4})
        _, walk = walk_of(eng)
        self.assertEqual([r["n_cpu_moe"] for r in walk if r.get("no_pinned")], [4])
        self.assertTrue(all(not c.no_pinned for c in eng.started if c.n_cpu_moe != 4))

    def test_a_row_that_runs_on_its_second_attempt_gets_no_third(self):
        eng = Banded(flaky={4: 1})
        _, walk = walk_of(eng)
        self.assertEqual(walk[2]["status"], "ok")
        self.assertNotIn("no_pinned", walk[2])
        self.assertNotIn("second_attempt", walk[2])
        self.assertEqual(starts(eng, 4), 2)

    def test_failing_without_pinned_memory_too_is_load_failed_and_the_walk_goes_on(self):
        eng = Banded(flaky={4: 3})
        _, walk = walk_of(eng)
        self.assertEqual([(r["n_cpu_moe"], r["status"]) for r in walk],
                         [(8, "ok"), (6, "ok"), (4, "load_failed"), (2, "ok"), (0, "start_failed")])
        self.assertIn("without pinned memory", walk[2]["note"])
        self.assertEqual(walk[2]["second_attempt"]["status"], "start_failed")
        self.assertEqual(attempts(eng, 4), [False, False, True])

    def test_a_vram_failure_gets_no_third_attempt(self):
        eng = Banded(band={4})
        _, walk = walk_of(eng)
        self.assertEqual(attempts(eng, 0), [False])                  # predicted over the top: the walk ends

    def test_the_planner_carries_no_pinned_to_launch(self):
        eng = Banded(band={2})
        doc, walk = walk_of(eng)
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000)
        self.assertEqual(best["n_cpu_moe"], 2)
        cfg = plan.to_config(best)
        self.assertTrue(cfg.no_pinned)
        self.assertEqual(launch.settings(cfg)["server_env"], {"GGML_CUDA_NO_PINNED": "1"})
        self.assertIn("GGML_CUDA_NO_PINNED=1", launch.settings_line(cfg))
        other = plan.to_config(walk[1])
        self.assertEqual(launch.settings(other)["server_env"], {})
        self.assertNotIn("environment", launch.settings_line(other))


class Rows(unittest.TestCase):
    def test_the_row_records_it_only_when_set(self):
        c = Config(ctx=8192, n_cpu_moe=30)
        self.assertNotIn("no_pinned", c.to_row())
        c.no_pinned = True
        r = c.to_row()
        self.assertIs(r["no_pinned"], True)
        self.assertTrue(Config.from_row(r).no_pinned)
        self.assertFalse(Config.from_row({"ctx": 8192, "kv": "f16", "n_cpu_moe": 30}).no_pinned)

    def test_keys_recorded_before_it_still_match(self):
        c = Config(ctx=8192, n_cpu_moe=30)
        self.assertNotIn("no_pinned", json.loads(c.key()))
        self.assertNotEqual(c.key(), Config(ctx=8192, n_cpu_moe=30, no_pinned=True).key())


class Engine(unittest.TestCase):
    def started_env(self, c):
        seen = {}

        class Proc:
            def poll(self):
                return None

        def popen(cmd, **kw):
            seen["env"] = kw.get("env")
            return Proc()

        with tempfile.TemporaryDirectory() as d, mock.patch.object(engines.subprocess, "Popen", popen), \
                mock.patch.object(LlamaCpp, "_get", lambda self, path: {}), \
                mock.patch.object(LlamaCpp, "stop", lambda self: None):
            log = Path(d) / "s.log"
            ok, _ = LlamaCpp("llama-server").start(Path("m.gguf"), c, log)
            text = log.read_text(encoding="utf-8")
        self.assertTrue(ok)
        return seen["env"], text

    def test_no_pinned_adds_the_variable_and_keeps_the_rest(self):
        with mock.patch.dict(os.environ, {"MARIAN_TEST_MARKER": "kept"}):
            env, text = self.started_env(Config(ctx=4096, n_cpu_moe=30, no_pinned=True))
        self.assertEqual(env["GGML_CUDA_NO_PINNED"], "1")
        self.assertEqual(env["MARIAN_TEST_MARKER"], "kept")
        self.assertIn("GGML_CUDA_NO_PINNED=1 ", text)               # the log says how the server was started

    def test_other_rows_inherit_the_environment_unchanged(self):
        env, text = self.started_env(Config(ctx=4096, n_cpu_moe=30))
        self.assertIsNone(env)
        self.assertNotIn("GGML_CUDA_NO_PINNED", text)


if __name__ == "__main__":
    unittest.main()
