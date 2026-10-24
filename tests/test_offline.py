"""Offline tests: no GPU, no server. python -m unittest discover -s tests"""
import struct
import tempfile
import unittest
from pathlib import Path

from marian import gguf
from marian.plan import choose, to_config
from marian.profile import splits


def write_gguf(path: Path, kv: dict) -> None:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = b""
    for k, v in kv.items():
        if isinstance(v, str):
            body += s(k) + struct.pack("<I", 8) + s(v)
        elif isinstance(v, list):
            body += s(k) + struct.pack("<I", 9) + struct.pack("<IQ", 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)
        else:
            body += s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body)


class Gguf(unittest.TestCase):
    def test_moe_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            write_gguf(p, {"general.architecture": "toy", "toy.block_count": 30, "toy.expert_count": 128,
                           "toy.expert_used_count": 8, "toy.context_length": 4096, "toy.scores": [1, 2, 3]})
            s = gguf.summary(p)
            self.assertEqual((s["block_count"], s["moe"], s["expert_used_count"]), (30, True, 8))

    def test_not_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.gguf"
            p.write_bytes(b"NOPE" + b"\0" * 20)
            with self.assertRaises(ValueError):
                gguf.summary(p)


class Splits(unittest.TestCase):
    def test_moe_walks_from_all_experts_on_cpu(self):
        self.assertEqual(splits({"block_count": 30, "moe": True}, 4)[:3], [("n_cpu_moe", 30), ("n_cpu_moe", 26), ("n_cpu_moe", 22)])

    def test_dense_walks_up_from_a_quarter(self):
        self.assertEqual(splits({"block_count": 40, "moe": False}, 10)[0], ("n_gpu_layers", 10))


class Plan(unittest.TestCase):
    ROWS = [{"ctx": 4096, "kv": "f16", "n_cpu_moe": 18, "status": "ok", "tg_tps": 31, "pp_tps": 800},
            {"ctx": 8192, "kv": "f16", "n_cpu_moe": 20, "status": "ok", "tg_tps": 28, "pp_tps": 700},
            {"ctx": 8192, "kv": "q8_0", "n_cpu_moe": 16, "status": "ok", "tg_tps": 30, "pp_tps": 690},
            {"ctx": 8192, "kv": "f16", "n_cpu_moe": 14, "status": "overflow", "tg_tps": 6, "pp_tps": 80}]

    def test_smallest_fitting_context_then_fastest(self):
        self.assertEqual(choose(self.ROWS, 3000)["ctx"], 4096)
        self.assertEqual(choose(self.ROWS, 6000)["kv"], "q8_0")

    def test_objective_prompt_speed(self):
        self.assertEqual(choose(self.ROWS, 6000, "pp")["kv"], "f16")

    def test_never_extrapolates_and_never_picks_an_overflow(self):
        self.assertIsNone(choose(self.ROWS, 9000))
        self.assertNotEqual(choose(self.ROWS, 6000)["status"], "overflow")

    def test_config_from_row(self):
        c = to_config(self.ROWS[2])
        self.assertEqual((c.ctx, c.cache_type, c.n_cpu_moe, c.n_gpu_layers), (8192, "q8_0", 16, 999))

    # rows measured on a desktop that already used 1,500 MiB of the GPU; one legacy row without the baseline
    FIT = [{"ctx": 8192, "kv": "f16", "n_cpu_moe": 18, "status": "ok", "tg_tps": 31, "pp_tps": 800,
            "vram_peak_mib": 7600, "vram_baseline_mib": 1500, "vram_model_mib": 6100},
           {"ctx": 8192, "kv": "q8_0", "n_cpu_moe": 20, "status": "ok", "tg_tps": 25, "pp_tps": 700,
            "vram_peak_mib": 6900, "vram_baseline_mib": 1500, "vram_model_mib": 5400},
           {"ctx": 8192, "kv": "q4_0", "n_cpu_moe": 22, "status": "ok", "tg_tps": 20, "pp_tps": 650, "vram_peak_mib": 6000},
           {"ctx": 8192, "kv": "f16", "n_cpu_moe": 24, "status": "ok", "tg_tps": 18, "pp_tps": 600}]

    def test_fit_now_uses_the_model_share_not_the_whole_peak(self):
        self.assertEqual(choose(self.FIT, 6000)["tg_tps"], 31)                          # no VRAM check: fastest
        self.assertEqual(choose(self.FIT, 6000, vram_free_mib=6400)["tg_tps"], 31)      # 6,100 + 300 fits exactly
        self.assertEqual(choose(self.FIT, 6000, vram_free_mib=6300)["tg_tps"], 25)      # busier now: the q8_0 row
        self.assertIsNone(choose(self.FIT, 6000, vram_free_mib=6300, margin_mib=1000))
        self.assertIsNone(choose([self.FIT[2]], 6000, vram_free_mib=6299))             # legacy row: whole peak + margin
        self.assertEqual(choose([self.FIT[2]], 6000, vram_free_mib=6300)["tg_tps"], 20)

    def test_fit_now_never_trusts_a_row_without_a_vram_reading(self):
        self.assertIsNone(choose([self.FIT[3]], 6000, vram_free_mib=100_000))
        self.assertIsNone(choose(self.FIT, 6000, vram_free_mib=5000))


class PlanCli(unittest.TestCase):
    def _run(self, *args):
        import contextlib
        import io
        import json
        from unittest import mock
        from marian import cli, machine
        from marian.profile import sha256_head

        def smi(query):
            if query == "memory.free":
                return [["6350"]]
            return [["0"]] if query == "memory.used" else [["0", "Test GPU", "8151", "1.0", "5", "4"]]

        with tempfile.TemporaryDirectory() as d, mock.patch.object(machine, "nvidia_smi", smi):
            model = Path(d) / "m.gguf"
            write_gguf(model, {"general.architecture": "toy", "toy.block_count": 2})
            prof = Path(d) / "profiles"
            prof.mkdir()
            name = f"{machine.detect().fingerprint()}-{sha256_head(model)}-llama.cpp.json"
            (prof / name).write_text(json.dumps({"engine": {"name": "llama.cpp"}, "rows": Plan.FIT}), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(cli, "PROFILES", prof), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(["plan", str(model), *args])
        return rc, out.getvalue(), err.getvalue()

    def test_profiles_dir_env_clone_and_installed(self):
        import os
        from unittest import mock
        from marian import cli
        with mock.patch.dict(os.environ, {"MARIAN_PROFILES": "X:/somewhere"}):
            self.assertEqual(cli.profiles_dir(), Path("X:/somewhere"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MARIAN_PROFILES", None)
            self.assertEqual(cli.profiles_dir(), Path(cli.__file__).resolve().parents[1] / "profiles")  # this clone
            with tempfile.TemporaryDirectory() as d:                      # an installed copy: no pyproject beside it
                fake = Path(d) / "site-packages" / "marian" / "cli.py"
                fake.parent.mkdir(parents=True)
                fake.write_text("")
                with mock.patch.object(cli, "__file__", str(fake)):
                    self.assertEqual(cli.profiles_dir(), Path.home() / ".marian" / "profiles")

    def test_json_success_carries_row_config_and_server_args(self):
        import json
        rc, out, _ = self._run("--need", "6000", "--json")
        doc = json.loads(out)
        self.assertEqual((rc, doc["ok"], doc["row"]["tg_tps"], doc["config"]["n_cpu_moe"]), (0, True, 31, 18))
        self.assertIn("-ncmoe", doc["server_args"])
        self.assertTrue(doc["vram_basis"].startswith("model share"))
        self.assertIsNone(doc["vram_free_mib"])

    def test_json_fit_now_measures_free_vram(self):
        import json
        rc, out, _ = self._run("--need", "6000", "--fit-now", "--json")
        doc = json.loads(out)
        self.assertEqual((rc, doc["vram_free_mib"], doc["row"]["tg_tps"]), (0, 6350, 25))   # 6,100+300 > 6,350

    def test_json_failure_is_still_one_object_and_exit_2(self):
        import json
        rc, out, _ = self._run("--need", "6000", "--vram-free", "4000", "--json")
        doc = json.loads(out)
        self.assertEqual((rc, doc["ok"]), (2, False))
        self.assertIn("4000 MiB of VRAM free", doc["reason"])
        rc, out, err = self._run("--need", "99999")
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("larger --ctx", err)


class Sampler(unittest.TestCase):
    def test_peak_sampler_starts_and_stops(self):
        from marian.profile import PeakSampler
        s = PeakSampler(every=0.05)
        s.start()
        s.stop()                       # 21/10/2026: a field named _stop shadowed Thread._stop and crashed here
        self.assertFalse(s.is_alive())


class NoHostname(unittest.TestCase):
    def test_written_profile_carries_nothing_derived_from_the_hostname(self):
        """A short hostname hash can be reversed from a list of likely names, and profiles get published."""
        import hashlib
        import json
        from unittest import mock
        from marian import machine, profile

        host = "marian-test-host-7f3a"
        digest = hashlib.sha256(host.encode()).hexdigest()

        class FakeEngine:                    # never starts a server: one start_failed row, no GPU, no sleep
            name = "fake"
            def version(self): return "0"
            def start(self, model, c, log): return False, "not started (offline test)"
            def stop(self): pass

        def smi(query):
            return [["0"]] if query == "memory.used" else [["0", "Test GPU", "8151", "1.0", "5", "4"]]

        with tempfile.TemporaryDirectory() as d, mock.patch("platform.node", return_value=host), \
                mock.patch.object(machine, "nvidia_smi", smi):
            model = Path(d) / "m.gguf"
            write_gguf(model, {"general.architecture": "toy", "toy.block_count": 2, "toy.expert_count": 4,
                               "toy.expert_used_count": 2, "toy.context_length": 4096})
            out = profile.run(model, FakeEngine(), [4096], ["f16"], 2, None, Path(d) / "profiles")
            text = out.read_text(encoding="utf-8")
        doc = json.loads(text)
        self.assertEqual(doc["rows"][0]["status"], "start_failed")
        self.assertNotIn(host, text)
        self.assertNotIn(digest[:6], text)   # any hash prefix of 6+ hex characters
        self.assertEqual([k for k in doc["machine"] if "host" in k.lower()], [])

    def test_profile_records_the_desktops_vram_and_the_models_share(self):
        """The GPU already holds 1,500 MiB of other programs; with the server up it holds 7,600."""
        import json
        import threading
        import time as real_time
        from unittest import mock
        from marian import machine, profile

        running = threading.Event()

        class FakeEngine:
            name = "fake"
            def version(self): return "0"
            def start(self, model, c, log): running.set(); return True, "started"
            def stop(self): running.clear()
            def tokenize(self, text): return 30
            def complete(self, text, n):
                threading.Event().wait(0.3)           # long enough for the sampler to see the server up
                return {"prompt_n": 3000, "prompt_per_second": 800.0, "predicted_per_second": 30.0}

        def smi(query):
            if query == "memory.used":
                return [["7600" if running.is_set() else "1500"]]
            return [["0", "Test GPU", "8151", "1.0", "5", "4"]]

        orig_sleep = real_time.sleep                  # profile.time IS the time module: keep the real one
        with tempfile.TemporaryDirectory() as d, mock.patch.object(machine, "nvidia_smi", smi), \
                mock.patch.object(profile.time, "sleep", lambda s: orig_sleep(min(s, 0.02))):
            model = Path(d) / "m.gguf"
            write_gguf(model, {"general.architecture": "toy", "toy.block_count": 2, "toy.expert_count": 4,
                               "toy.expert_used_count": 2, "toy.context_length": 4096})
            out = profile.run(model, FakeEngine(), [4096], ["f16"], 2, None, Path(d) / "profiles")
            row = json.loads(out.read_text(encoding="utf-8"))["rows"][0]
        self.assertEqual((row["status"], row["vram_baseline_mib"], row["vram_peak_mib"], row["vram_model_mib"]),
                         ("ok", 1500, 7600, 6100))


if __name__ == "__main__":
    unittest.main()
