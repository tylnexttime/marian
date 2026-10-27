"""Offline tests of M1.3 (calibration v1, launch checks, rollback, diagnostics): no GPU, no server, a fake engine.
python -m unittest discover -s tests"""
import contextlib
import io
import json
import tempfile
import threading
import time as real_time
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from marian import cli, engines, launch, machine, plan, profile
from marian.engines import Config, LlamaCpp, parse_help, parse_version, same_build
from marian.profile import CacheOptions, ThreadOptions, find_cliff, read_server_log, sha256_head, tier_ladder

from test_offline import write_gguf

# excerpts of `llama-server --help`, option lines at column 0 and descriptions indented, as llama.cpp prints them
HELP_OLD_STOCK = """\
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the
                                        CPU
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM
"""
HELP_STOCK = HELP_OLD_STOCK + """\
--moe-cache-mib N                       GPU cache size in MiB for the MoE experts kept in the CPU. with
                                        multiple GPUs, it is split among them like the layers (--tensor-split)
                                        a description may mention --moe-miss-source without the build having it
"""
HELP_MARIAN = HELP_STOCK + """\
--moe-miss-source {mmap,ring,tier}      Marian P3 [EXPERIMENTAL]: where the MoE cache uploads its misses from:
                                        plus a pinned RAM tier of --moe-tier-mib
--moe-tier-mib N                        Marian P3 [EXPERIMENTAL]: pinned RAM tier in MiB for --moe-miss-source
--moe-ring-mib N                        Marian P3 [EXPERIMENTAL]: pinned staging ring in MiB for
"""
VERSION = "version: 0.7.0-dev (build 3203, commit 2c1cf30a)\nbuilt with MSVC 19.44.35228.0 for Windows AMD64\n"
GPU = ["0", "Test GPU", "8151", "1.0", "5", "4"]
FLOOR_LINE = ("0.04.9 E llama_init_from_model: MoE cache: {b} MiB on CUDA0 is below the exact-mode floor of 690 MiB "
              "(slots for one ubatch of 32 tokens in every group of layers); use --moe-cache-mib 700 or more, or a "
              "smaller --ubatch-size\n")


class Capabilities(unittest.TestCase):
    def test_stock_marian_and_old_builds(self):
        old, stock, marian = parse_help(HELP_OLD_STOCK), parse_help(HELP_STOCK), parse_help(HELP_MARIAN)
        self.assertEqual((old["moe_cache"], old["marian"]), (False, False))
        self.assertEqual((stock["moe_cache"], stock["marian"]), (True, False))   # a mention in a description is not a flag
        self.assertEqual((marian["moe_cache"], marian["marian"]), (True, True))
        self.assertEqual(marian["moe_flags"], ["--moe-cache-mib", "--moe-miss-source", "--moe-ring-mib", "--moe-tier-mib"])
        self.assertIn("-ncmoe", marian["flags"])

    def test_help_is_read_once(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[1])
            return mock.Mock(stdout=HELP_MARIAN if cmd[1] == "--help" else VERSION, stderr="")

        eng = LlamaCpp("llama-server")
        with mock.patch.object(engines.subprocess, "run", run):
            self.assertTrue(eng.capabilities()["marian"])
            eng.capabilities()
            self.assertEqual(eng.version(), "0.7.0-dev (build 3203, commit 2c1cf30a)")
            eng.version()
        self.assertEqual(calls, ["--help", "--version"])

    def test_versions(self):
        self.assertEqual(parse_version(VERSION), "0.7.0-dev (build 3203, commit 2c1cf30a)")
        m0 = "0.7.0-dev (build 3203, commit"                       # what M0 recorded: the commit cut off
        self.assertTrue(same_build(m0, "0.7.0-dev (build 3203, commit 2c1cf30a)"))
        self.assertFalse(same_build("0.7.0-dev (build 3203, commit 2c1cf30a)", "0.7.0-dev (build 3203, commit 99999999)"))
        self.assertFalse(same_build(m0, "0.7.0-dev (build 11760, commit"))


class ConfigRoundTrip(unittest.TestCase):
    CONFIGS = [Config(ctx=4096, n_cpu_moe=18, threads=8),
               Config(ctx=8192, cache_type="q8_0", n_gpu_layers=33),
               Config(ctx=4096, n_cpu_moe=30, load_mode="mmap", moe_cache_mib=2500),
               Config(ctx=4096, n_cpu_moe=30, load_mode="mmap", moe_cache_mib=2500, miss_source="ring"),
               Config(ctx=16384, n_cpu_moe=48, load_mode="mmap", moe_cache_mib=9400, miss_source="tier", tier_mib=6144,
                      ring_mib=128)]

    def test_to_config_restores_every_recorded_field(self):
        for c in self.CONFIGS:
            row = json.loads(json.dumps(c.to_row()))                # through JSON, as a profile stores it
            self.assertEqual(asdict(plan.to_config(row)), asdict(c))

    def test_m0_rows_still_load(self):
        c = plan.to_config({"ctx": 4096, "kv": "f16", "n_cpu_moe": 18, "threads": 8, "status": "ok"})
        self.assertEqual((c.load_mode, c.moe_cache_mib, c.miss_source), ("none", None, None))

    def test_command_emits_the_cache_and_the_miss_source(self):
        cmd = LlamaCpp("llama-server").command(Path("m.gguf"), self.CONFIGS[4])
        s = " ".join(cmd)
        self.assertIn("-ncmoe 48 --moe-cache-mib 9400 --moe-miss-source tier --moe-tier-mib 6144 --moe-ring-mib 128", s)
        self.assertIn("-lm mmap", s)
        self.assertEqual(self.CONFIGS[4].marian_flags(), ["--moe-miss-source", "--moe-tier-mib", "--moe-ring-mib"])
        self.assertEqual([c.pinned_mib() for c in self.CONFIGS], [0, 0, 0, 64, 6144 + 128])
        self.assertEqual(Config(n_cpu_moe=30, moe_cache_mib=2500, miss_source="tier", tier_mib=4096).pinned_mib(), 4096 + 64)

    def test_exact_class(self):
        rows = [({"ctx": 1, "kv": "f16", "n_cpu_moe": 0}, True), ({"ctx": 1, "kv": "f16", "n_cpu_moe": 18}, False),
                ({"ctx": 1, "kv": "f16", "n_cpu_moe": 30, "moe_cache_mib": 800}, True),
                ({"ctx": 1, "kv": "f16", "n_gpu_layers": 41, "_block_count": 40}, True),
                ({"ctx": 1, "kv": "f16", "n_gpu_layers": 30, "_block_count": 40}, False),
                ({"ctx": 1, "kv": "f16", "n_cpu_moe": 18, "exact_class": True}, True)]   # a recorded flag wins
        self.assertEqual([plan.exact_class(r) for r, _ in rows], [e for _, e in rows])


def row(ctx=4096, tg=10.0, share=3000, **kw):
    r = {"ctx": ctx, "kv": "f16", "status": "ok", "tg_tps": tg, "pp_tps": 500.0, "vram_model_mib": share,
         "vram_peak_mib": share + 1000, "vram_baseline_mib": 1000}
    r.update(kw)
    return r


ROWS = [row(n_cpu_moe=30, tg=15.0, share=2500, exact_class=False),
        row(n_cpu_moe=18, tg=26.0, share=6500, exact_class=False),
        row(n_cpu_moe=0, tg=60.0, share=12000, exact_class=True),
        row(n_cpu_moe=30, tg=30.0, share=6000, moe_cache_mib=3000, load_mode="mmap", exact_class=True),
        row(n_cpu_moe=30, tg=34.0, share=6000, moe_cache_mib=3000, load_mode="mmap", miss_source="ring", exact_class=True),
        row(n_cpu_moe=30, tg=40.0, share=6000, moe_cache_mib=3000, load_mode="mmap", miss_source="tier", tier_mib=4096,
            exact_class=True)]


class Filters(unittest.TestCase):
    def test_stock_keeps_plain_upstream_placement_only(self):
        self.assertEqual(plan.choose(ROWS, 3000, vram_free_mib=8000)["miss_source"], "tier")
        r = plan.choose(ROWS, 3000, vram_free_mib=8000, stock=True)
        self.assertEqual((r["n_cpu_moe"], r.get("moe_cache_mib")), (18, None))
        self.assertTrue(all(plan.is_stock(x) for x in ROWS[:3]))
        self.assertFalse(any(plan.is_stock(x) for x in ROWS[3:]))     # upstream's cache too: measured on the fork

    def test_exact_only_drops_cpu_experts(self):
        self.assertEqual(plan.choose(ROWS, 3000, vram_free_mib=6400, exact_only=True)["tg_tps"], 40.0)
        self.assertIsNone(plan.choose(ROWS, 3000, vram_free_mib=6000, exact_only=True))
        self.assertEqual(plan.choose(ROWS, 3000, vram_free_mib=6000)["n_cpu_moe"], 30)          # stock 30 fits
        self.assertEqual(plan.choose(ROWS, 3000, exact_only=True, stock=True)["n_cpu_moe"], 0)

    def test_ram_envelope_at_launch(self):
        # the tier pins 4,096 + 64 MiB; with 10,000 MiB available and a 6,144 floor only 3,856 may be pinned
        self.assertEqual(plan.choose(ROWS, 3000, vram_free_mib=8000, ram_free_mib=10000)["miss_source"], "ring")
        self.assertEqual(plan.choose(ROWS, 3000, vram_free_mib=8000, ram_free_mib=10304)["miss_source"], "tier")
        self.assertIsNone(plan.choose(ROWS[4:], 3000, ram_free_mib=6100))


class RamEnvelope(unittest.TestCase):
    LADDER = CacheOptions().tier_ladder_mib

    def test_never_pins_past_the_floor(self):
        self.assertEqual(tier_ladder(16000, 6144, None, self.LADDER, 3), [4096, 6144, 8192])    # cap 9,792
        for free in range(0, 40000, 777):
            for t in tier_ladder(free, 6144, None, self.LADDER, 3):
                self.assertLessEqual(t + 64, free - 6144)
        self.assertEqual(tier_ladder(7000, 6144, None, self.LADDER, 3), [])

    def test_useful_size_and_steps(self):
        self.assertEqual(tier_ladder(30000, 6144, 5000, self.LADDER, 3), [1024, 2048, 4096])   # the uncached experts
        self.assertEqual(tier_ladder(30000, 6144, None, self.LADDER, 2), [12288, 16384])
        self.assertEqual(tier_ladder(30000, 6144, None, self.LADDER, 0), [])


class Cliff(unittest.TestCase):
    """A synthetic GPU: 8,000 MiB, the model's other weights 2,000, the desktop 1,000, margin 300. Decode rises with
    the budget; above `cliff` the driver pages (speed collapses) while the VRAM reading saturates below the margin."""

    def curve(self, cliff=4400, total=8000, margin=300):
        probes = []

        def probe_budget(b):
            probes.append(b)
            if b < 700:
                return {"status": "refused", "floor_min_mib": 700, "tg_tps": 0}
            used = 1000 + 2000 + b
            r = {"status": "ok", "vram_peak_mib": min(used, total - margin - 1), "tg_tps": 10 + b / 500}
            if b > cliff:
                r["tg_tps"] = 2.5                                     # paging: 18.8 -> 2.5 tok/s
            if used >= total + 2000:
                r["vram_peak_mib"] = total                            # far over: the reading shows full
            profile.judge(r, total, margin, 0.0, 0.6)
            return r
        return probe_budget, probes

    def test_bisection_finds_the_cliff_and_never_probes_above_a_known_overflow(self):
        probe_budget, probes = self.curve()
        best = find_cliff(probe_budget, 500, 9000, 256)
        self.assertTrue(4400 - 256 <= best <= 4400, best)
        self.assertEqual(probes[:3], [500, 700, 9000])                # refused at 500: the server's own floor next
        bad = 9000
        for b in probes[3:]:
            self.assertLess(b, bad)
            if b > 4400:
                bad = b
        self.assertLessEqual(len(probes), 3 + 6)                     # log2((9000 - 700) / 256) bisection steps

    def test_top_fits_or_floor_fails(self):
        probe_budget, probes = self.curve(cliff=10**9)
        self.assertEqual((find_cliff(probe_budget, 700, 3000, 256), probes), (3000, [700, 3000]))
        probes = []                                                   # even the floor overflows: no cache arm
        self.assertIsNone(find_cliff(lambda b: probes.append(b) or {"status": "overflow", "tg_tps": 4.0}, 700, 3000))
        self.assertEqual(probes, [700])

    def test_server_log(self):
        said = read_server_log(FLOOR_LINE.format(b=500))
        self.assertEqual(said["floor_min_mib"], 700)
        self.assertIn("690 MiB", said["refusal"])
        said = read_server_log("W llama_moe_cache: --moe-miss-source is not used, misses are uploaded from the host "
                               "tensors: no mapping for C:\\models\\x.gguf\n")
        self.assertNotIn("C:\\", said["source_not_used"])               # profiles get published: no local paths
        said = read_server_log("W llama_moe_source: Marian P3 (experimental) tier: ring 15 x 4.03 MiB, tier 4096 MiB, "
                               "4156.47 MiB pinned (+4096.00 MiB PAGEABLE: pinning failed), sector 4096\n")
        self.assertEqual((said["pinned_mib"], said["pageable_mib"]), (4156, 4096))


class FakeMarian:
    """A Marian build on a synthetic 8,151 MiB GPU: refuses budgets below 700 MiB, pages above a 5,000 MiB cache."""
    name = "llama.cpp"

    def __init__(self, help_text=HELP_MARIAN):
        self.running: Config | None = None
        self.started: list[Config] = []
        self.help = help_text

    def version(self):
        return "0.7.0-dev (build 3203, commit 2c1cf30a)"

    def capabilities(self):
        return parse_help(self.help)

    def share(self) -> int:
        c = self.running
        if c is None:
            return 0
        if c.moe_cache_mib:
            return 1500 + c.moe_cache_mib
        return 1500 + (2 - c.n_cpu_moe) * 600

    def start(self, model, c, log):
        self.started.append(c)
        with open(log, "a", encoding="utf-8") as f:
            if c.moe_cache_mib and c.moe_cache_mib < 700:
                f.write(FLOOR_LINE.format(b=c.moe_cache_mib))
                return False, "failed to start (see log)"
            if c.miss_source:
                pinned = 60.5 + (c.tier_mib or 0)
                f.write(f"W llama_moe_source: Marian P3 (experimental) {c.miss_source}: ring 15 x 4.03 MiB, "
                        f"{pinned:.2f} MiB pinned, sector 4096, unbuffered reads\n")
        self.running = c
        return True, "started"

    def stop(self):
        self.running = None

    def tokenize(self, text):
        return 30

    def complete(self, text, n):
        threading.Event().wait(0.12)              # long enough for the sampler to see the server up (sleep is patched)
        c = self.running
        tg = 20.0 + (2 - c.n_cpu_moe) * 5
        if c.moe_cache_mib:
            tg = 25.0 + c.moe_cache_mib / 1000 if c.moe_cache_mib <= 5000 else 3.0
            tg += {"ring": 2.0, "tier": 4.0 + (c.tier_mib or 0) / 10000}.get(c.miss_source, 0.0)
        return {"prompt_n": 3000, "prompt_per_second": 800.0, "predicted_per_second": tg}


@contextlib.contextmanager
def fake_machine(engine=None, ram_available=16000):
    def smi(query):
        if query == "memory.used":
            return [[str(1000 + (engine.share() if engine else 0))]]
        if query == "memory.free":
            return [[str(8151 - 1000 - (engine.share() if engine else 0))]]
        return [GPU]
    orig_sleep = real_time.sleep
    with mock.patch.object(machine, "nvidia_smi", smi), \
            mock.patch.object(machine, "ram_mib", lambda: (32000, ram_available)), \
            mock.patch.object(machine, "commit_mib", lambda: (64000, 30000)), \
            mock.patch.object(profile.time, "sleep", lambda s: orig_sleep(min(s, 0.01))), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


def toy_moe(d: Path) -> Path:
    model = d / "toy.gguf"
    write_gguf(model, {"general.architecture": "toy", "toy.block_count": 2, "toy.expert_count": 8,
                       "toy.expert_used_count": 2, "toy.context_length": 4096})
    return model


class CacheArms(unittest.TestCase):
    def run_profile(self, eng, ram_available=16000, **opt):
        with tempfile.TemporaryDirectory() as d, fake_machine(eng, ram_available), \
                mock.patch.object(profile, "model_geometry", lambda m: object()), \
                mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 500), \
                mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 12000):
            out = profile.run(toy_moe(Path(d)), eng, [4096], ["f16"], 2, None, Path(d) / "profiles", 16,
                              cache=CacheOptions(**opt), sweep=ThreadOptions(enabled=False))
            return json.loads(out.read_text(encoding="utf-8"))

    def test_marian_build_gets_cliff_and_miss_source_arms(self):
        doc = self.run_profile(FakeMarian())
        rows = doc["rows"]
        walk = [r for r in rows if r["arm"] == "walk"]
        cache = [r for r in rows if r["arm"] == "cache"]
        source = [r for r in rows if r["arm"] == "source"]
        self.assertEqual([(r["n_cpu_moe"], r["exact_class"]) for r in walk], [(2, False), (0, True)])
        # the computed floor (500) was below the server's: refused, recorded, and the server's 700 used next
        self.assertEqual((cache[0]["moe_cache_mib"], cache[0]["status"], cache[0]["floor_min_mib"]), (500, "refused", 700))
        self.assertEqual((cache[1]["moe_cache_mib"], cache[1]["status"]), (700, "ok"))
        cell = doc["cache_search"]["cells"][0]
        self.assertEqual(cell["hi_mib"], 8151 - 1000 - 300 - 1500)           # VRAM room beside the experts-on-CPU share
        self.assertTrue(5000 - 256 <= cell["best_mib"] <= 5000, cell)
        self.assertTrue(any(r["status"] == "overflow" for r in cache))
        self.assertTrue(all(r["exact_class"] and r["load_mode"] == "mmap" and r["n_cpu_moe"] == 2 for r in cache + source))
        self.assertEqual([r.get("miss_source") for r in source], ["ring", "tier", "tier", "tier"])
        tiers = [r["tier_mib"] for r in source[1:]]
        self.assertEqual(tiers, cell["ram_envelope"]["tier_mib"])
        self.assertTrue(all(t + 64 <= 16000 - 6144 and t <= 12000 - cell["best_mib"] for t in tiers))
        self.assertTrue(all(r["moe_cache_mib"] == cell["best_mib"] and r["status"] == "ok" for r in source))
        self.assertEqual(source[1]["pinned_mib"], round(60.5 + tiers[0]))
        self.assertTrue(doc["engine"]["marian"])
        self.assertEqual((doc["marian_profile"], walk[0]["ram_free_before_mib"]), (3, 16000))
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000)
        self.assertEqual((best["miss_source"], best["tier_mib"]), ("tier", max(tiers)))

    def test_ram_envelope_limits_the_ladder(self):
        doc = self.run_profile(FakeMarian(), ram_available=7300)            # 7,300 - 6,144 - 64: only 1,024 fits
        tiers = [r["tier_mib"] for r in doc["rows"] if r.get("miss_source") == "tier"]
        self.assertEqual(tiers, [1024])

    def test_stock_build_gets_the_walk_only(self):
        doc = self.run_profile(FakeMarian(HELP_STOCK))
        self.assertEqual({r["arm"] for r in doc["rows"]}, {"walk"})
        self.assertIn("not a Marian build", doc["cache_search"]["skipped"])

    def test_a_cache_that_holds_every_expert_needs_no_miss_source(self):
        with tempfile.TemporaryDirectory() as d, fake_machine(FakeMarian()), \
                mock.patch.object(profile, "model_geometry", lambda m: object()), \
                mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 700), \
                mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 2000):
            eng = FakeMarian()
            out = profile.run(toy_moe(Path(d)), eng, [4096], ["f16"], 2, None, Path(d) / "p", 16)
            doc = json.loads(out.read_text(encoding="utf-8"))
        cell = doc["cache_search"]["cells"][0]
        self.assertEqual((cell["best_mib"], cell["probes"]), (2000, 2))
        self.assertIn("every expert fits", cell["skipped_sources"])
        self.assertFalse(any(c.miss_source for c in eng.started))


class FakeLlama(LlamaCpp):
    """LlamaCpp that never runs an executable: the help, version and server are canned."""
    help_text, version_text, starts = HELP_MARIAN, "0.7.0-dev (build 3203, commit 2c1cf30a)", []

    def version(self):
        return self.version_text

    def capabilities(self):
        return parse_help(self.help_text)

    def start(self, model, c, log, timeout=900, attempts=3):
        FakeLlama.starts.append(c)
        return False, "not started (offline test)"


class Serve(unittest.TestCase):
    """The gate, the launch checks and the rollback through `marian serve` and `marian plan`."""

    def setUp(self):
        FakeLlama.help_text, FakeLlama.version_text, FakeLlama.starts = HELP_MARIAN, "0.7.0-dev (build 3203, commit 2c1cf30a)", []

    def run_cli(self, d: Path, *args, doc=None, ram_available=16000, model_id=None):
        model = toy_moe(d)
        prof = d / "profiles"
        prof.mkdir(exist_ok=True)
        out, err = io.StringIO(), io.StringIO()
        with fake_machine(None, ram_available), mock.patch.object(cli, "PROFILES", prof), \
                mock.patch.object(cli, "LlamaCpp", FakeLlama):
            if doc is not None:
                fp = machine.detect().fingerprint()
                mid = sha256_head(model)
                d2 = json.loads(json.dumps(doc))
                d2["machine"]["fingerprint"] = fp
                d2["model"]["id"] = model_id or mid
                d2["model"]["bytes"] = model.stat().st_size
                (prof / f"{fp}-{mid}-llama.cpp.json").write_text(json.dumps(d2), encoding="utf-8")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main([args[0], str(model), *args[1:]])
        return rc, out.getvalue(), err.getvalue()

    DOC = {"marian_profile": 2, "machine": {"fingerprint": "?"},
           "engine": {"name": "llama.cpp", "version": "0.7.0-dev (build 3203, commit 2c1cf30a)", "marian": True},
           "model": {"file": "toy.gguf", "id": "?", "block_count": 2, "moe": True},
           "rows": [row(n_cpu_moe=2, tg=15.0, share=2500, exact_class=False),
                    row(n_cpu_moe=0, tg=30.0, share=6500, exact_class=True),
                    row(n_cpu_moe=2, tg=40.0, share=4000, moe_cache_mib=2500, load_mode="mmap", miss_source="tier",
                        tier_mib=4096, exact_class=True)]}

    def test_gate_refuses_without_a_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "llama-server", "--need", "3000", "--json")
        doc = json.loads(out)
        self.assertEqual((rc, doc["ok"], doc["gate"]), (2, False, "no_calibration"))
        self.assertIn("marian calibrate", doc["reason"])
        self.assertEqual(FakeLlama.starts, [])

    def test_serve_calibrate_runs_the_quick_profile_then_serves(self):
        calls = []

        def fake_run(model, eng, ctxs, kvs, step, threads, out_dir, n_predict, cache=None, **kw):
            calls.append((ctxs, kvs, n_predict))
            fp, mid = machine.detect().fingerprint(), sha256_head(model)
            d = json.loads(json.dumps(self.DOC))
            d["machine"]["fingerprint"], d["model"]["id"], d["model"]["bytes"] = fp, mid, model.stat().st_size
            p = out_dir / f"{fp}-{mid}-llama.cpp.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            return p

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cli.profile, "run", fake_run):
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--calibrate", "--json")
        self.assertEqual(calls, [([4096], ["f16"], 32)])
        self.assertEqual(rc, 2)                                  # FakeLlama never starts, but the checks passed
        self.assertTrue(json.loads(out.splitlines()[0])["ok"])
        self.assertEqual(FakeLlama.starts[0].miss_source, "tier")

    def test_dry_run_picks_the_tier_row_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json", doc=self.DOC)
        doc = json.loads(out)
        self.assertEqual((rc, doc["ok"], doc["exact_class"], doc["row"]["tier_mib"]), (0, True, True, 4096))
        self.assertIn("--moe-miss-source", doc["server_args"])
        self.assertEqual(FakeLlama.starts, [])

    def test_launch_check_failures_are_reasons(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json",
                                      doc=self.DOC, model_id="0123456789ab")
        doc = json.loads(out)                     # a profile file whose content names another model
        self.assertEqual((rc, doc["ok"]), (2, False))
        self.assertTrue(any(r.startswith("model identity") for r in doc["reasons"]))
        FakeLlama.version_text = "0.7.0-dev (build 4000, commit abcdef12)"
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json", doc=self.DOC)
        doc = json.loads(out)
        self.assertEqual(rc, 2)
        self.assertTrue(any(r.startswith("engine build") for r in doc["reasons"]))
        FakeLlama.help_text = HELP_STOCK
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json",
                                      "--stock", doc=self.DOC)
        doc = json.loads(out)                     # the rollback: another (stock) build only warns
        self.assertEqual((rc, doc["ok"], doc["stock"], doc["row"]["n_cpu_moe"]), (0, True, True, 0))
        self.assertTrue(any(w.startswith("engine build") for w in doc["warnings"]))
        self.assertNotIn("--moe-cache-mib", doc["server_args"])

    def test_check_vram_ram_and_flags(self):
        eng = FakeLlama("x")
        FakeLlama.help_text = HELP_STOCK
        with tempfile.TemporaryDirectory() as d, fake_machine(None):
            model = toy_moe(Path(d))
            mid = sha256_head(model)
            doc = json.loads(json.dumps(self.DOC))
            doc["machine"]["fingerprint"], doc["model"]["id"] = machine.detect().fingerprint(), mid
            reasons, warns = launch.check(self.DOC["rows"][2], doc, model, mid, eng, vram_free_mib=4200, margin_mib=300,
                                          ram_free_mib=9000, ram_floor_mib=6144)
            _, warns2 = launch.check(self.DOC["rows"][0], doc, model, mid, eng, vram_free_mib=None)
        text = " | ".join(reasons)
        self.assertIn("not a Marian build", text)
        self.assertIn("VRAM: the row needs 4000 MiB plus a 300 MiB margin; 4200 MiB is free now", text)
        self.assertIn("RAM: the row pins 4160 MiB", text)
        self.assertEqual(warns, [])
        self.assertIn("free VRAM could not be read", warns2[0])

    def test_plan_filters(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_cli(Path(d), "plan", "--need", "3000", "--stock", "--json", doc=self.DOC)
            stock = json.loads(out)
            rc2, out, _ = self.run_cli(Path(d), "plan", "--need", "3000", "--exact-only", "--vram-free", "4200", "--json",
                                       doc=self.DOC)
            none = json.loads(out)
        self.assertEqual((rc, stock["row"]["n_cpu_moe"], stock["row"]["tg_tps"]), (0, 0, 30.0))
        self.assertEqual((rc2, none["ok"]), (2, False))
        self.assertIn("4200 MiB of VRAM free", none["reason"])


class Diag(unittest.TestCase):
    def test_bundle_has_no_prompt_and_no_home_path(self):
        secret = "PLEASE SUMMARISE MY BANK STATEMENT"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = toy_moe(d)
            prof = d / "profiles"
            prof.mkdir()
            home = str(Path.home())
            with fake_machine(None), mock.patch.object(cli, "PROFILES", prof), mock.patch.object(cli, "LlamaCpp", FakeLlama):
                fp, mid = machine.detect().fingerprint(), sha256_head(model)
                doc = json.loads(json.dumps(Serve.DOC))
                doc["machine"]["fingerprint"], doc["model"]["id"] = fp, mid
                doc["rows"][0]["note"] = f"failed to start, see {home}\\AppData\\x.log"
                (prof / f"{fp}-{mid}-llama.cpp.json").write_text(json.dumps(doc), encoding="utf-8")
                (prof / "serve.log").write_text(
                    f"# attempt 1: {home}\\bin\\llama-server.exe -m {model} -c 4096\n"
                    f"I srv  load_model: loading model '{model}'\n"
                    f'D srv  log_server_r: request: {{"prompt": "{secret}", "n_predict": 64}}\n'
                    f'D srv  handle_compl: {{"messages": [{{"role": "user", "content": "{secret}"}}]}}\n'
                    "I slot print_timing: id  0 | task 0 | prompt eval time =  2266.70 ms /    22 tokens\n"
                    f"I srv  ... cache at {home.replace(chr(92), '/')}/.cache and /home/someone/models/x.gguf\n",
                    encoding="utf-8")
                out = d / "diag.zip"
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = cli.main(["diag", str(out), str(model), "--engine", str(d / "bin" / "llama-server.exe")])
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
                text = "\n".join(z.read(n).decode("utf-8") for n in names)
                chosen = json.loads(z.read("plan.json"))["chosen"]
            forms = {home, home.replace("\\", "/"), str(d), str(d).replace("\\", "/"), str(d.resolve())}
        self.assertEqual(rc, 0)
        for n in ("machine.json", "versions.json", "plan.json", "logs/serve.log.tail.txt", "README.txt"):
            self.assertIn(n, names)
        self.assertTrue(any(n.startswith("profiles/") for n in names))
        self.assertNotIn(secret, text)
        for form in forms:
            self.assertNotIn(form.lower(), text.lower())
        self.assertNotIn("/home/someone", text)
        self.assertIn("prompt eval time", text)                         # timings stay
        self.assertIn("<model-dir>", text)
        self.assertEqual((chosen["row"]["tier_mib"], chosen["server_args"][:2]), (4096, ["-m", "toy.gguf"]))


if __name__ == "__main__":
    unittest.main()
