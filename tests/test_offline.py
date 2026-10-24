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


if __name__ == "__main__":
    unittest.main()
