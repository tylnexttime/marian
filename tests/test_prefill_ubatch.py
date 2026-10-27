"""Offline tests of the two settings registers E51 and E54 settled (M1.3): the P3 prefill source
(`--moe-prefill-source mmap|p3`, patch 0009) in calibration, plan and launch, and the declared ubatch (`-ub N -b N`)
as a profile parameter. No GPU and no server: a fake engine and fake memory readings.
python -m pytest -q tests"""
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from marian import cli, launch, machine, plan, profile
from marian.engines import Config, LlamaCpp, parse_help
from marian.profile import CacheOptions, ThreadOptions, prefill_pair, sha256_head

import test_m13 as m13

HELP_0009 = m13.HELP_MARIAN + """\
--moe-prefill-source {mmap,p3}          Marian [EXPERIMENTAL]: where large-batch (prefill) misses come from: mmap
                                        (upstream's selective copy) or p3 (the ring and tier); p3 needs
                                        --moe-miss-source ring or tier
"""
VERSION = "0.7.0-dev (build 3203, commit 2c1cf30a)"


class PrefillFake(m13.FakeMarian):
    """FakeMarian with patch 0009: P3 prefill multiplies prefill speed by pp_gain and decode by tg_gain. Like the
    fork, it refuses p3 beside memory-mapped misses."""

    def __init__(self, pp_gain=1.76, tg_gain=1.0, help_text=HELP_0009):
        super().__init__(help_text)
        self.pp_gain, self.tg_gain = pp_gain, tg_gain

    def start(self, model, c, log):
        if c.prefill_source == "p3" and c.miss_source not in ("ring", "tier"):
            self.started.append(c)
            return False, "--moe-prefill-source p3 needs --moe-miss-source ring or tier"
        return super().start(model, c, log)

    def complete(self, text, n):
        t = super().complete(text, n)
        if self.running is not None and self.running.prefill_source == "p3":
            t = dict(t, prompt_per_second=t["prompt_per_second"] * self.pp_gain,
                     predicted_per_second=t["predicted_per_second"] * self.tg_gain)
        return t


def run_profile(eng, ram_available=16000, ubatch=512, **opt):
    """profile.run on the toy MoE with the fake engine; returns (document, profile file name)."""
    with tempfile.TemporaryDirectory() as d, m13.fake_machine(eng, ram_available), \
            mock.patch.object(profile, "model_geometry", lambda m: object()), \
            mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 500), \
            mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 12000):
        out = profile.run(m13.toy_moe(Path(d)), eng, [4096], ["f16"], 2, None, Path(d) / "profiles", 16,
                          cache=CacheOptions(**opt), sweep=ThreadOptions(enabled=False), ubatch=ubatch)
        return json.loads(out.read_text(encoding="utf-8")), out.name


def doc_of(rows, ubatch=None, schema=3):
    d = {"marian_profile": schema, "machine": {"fingerprint": "?"},
         "engine": {"name": "llama.cpp", "version": VERSION, "marian": True},
         "model": {"file": "toy.gguf", "id": "?", "block_count": 2, "moe": True}, "rows": rows}
    if ubatch is not None:
        d["ubatch"] = ubatch
    return d


def tier_row(tg=40.0, pp=500.0, **kw):
    return m13.row(n_cpu_moe=2, tg=tg, share=4000, moe_cache_mib=2500, load_mode="mmap", miss_source="tier",
                   tier_mib=4096, exact_class=True, pp_tps=pp, **kw)


DOC_512 = doc_of([m13.row(n_cpu_moe=0, tg=30.0, share=6500, exact_class=True), tier_row()], ubatch=512)
DOC_2048 = doc_of([dict(m13.row(n_cpu_moe=0, tg=29.0, share=6800, exact_class=True), ubatch=2048),
                   dict(tier_row(tg=38.0, pp=900.0, prefill_source="p3"), ubatch=2048)], ubatch=2048)


def run_cli(d: Path, *args, docs=(), help_text=HELP_0009):
    """cli.main on the toy model with profiles written as given ((file suffix, document) pairs, '' for the default
    name); FakeLlama never starts a server."""
    model = m13.toy_moe(d)
    prof = d / "profiles"
    prof.mkdir(exist_ok=True)
    out, err = io.StringIO(), io.StringIO()
    m13.FakeLlama.help_text, m13.FakeLlama.version_text, m13.FakeLlama.starts = help_text, VERSION, []
    with m13.fake_machine(None), mock.patch.object(cli, "PROFILES", prof), mock.patch.object(cli, "LlamaCpp", m13.FakeLlama):
        fp, mid = machine.detect().fingerprint(), sha256_head(model)
        for suffix, doc in docs:
            d2 = json.loads(json.dumps(doc))
            d2["machine"]["fingerprint"], d2["model"]["id"], d2["model"]["bytes"] = fp, mid, model.stat().st_size
            (prof / f"{fp}-{mid}-llama.cpp{suffix}.json").write_text(json.dumps(d2), encoding="utf-8")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main([args[0], str(model), *args[1:]])
    return rc, out.getvalue(), err.getvalue()


# --- the prefill source ---------------------------------------------------------------------------------------------

class PrefillConfig(unittest.TestCase):
    def test_capability_is_read_from_help(self):
        self.assertTrue(parse_help(HELP_0009)["prefill_source"])
        old = parse_help(m13.HELP_MARIAN)                    # a Marian build before 0009 is still Marian
        self.assertEqual((old["marian"], old["prefill_source"]), (True, False))

    def test_p3_is_never_paired_with_mmap(self):
        for miss in (None, "mmap"):
            c = Config(n_cpu_moe=30, load_mode="mmap", moe_cache_mib=2500, miss_source=miss, prefill_source="p3")
            self.assertIn("ring or tier", c.problem())
            with self.assertRaises(ValueError):
                LlamaCpp("llama-server").command(Path("m.gguf"), c)
        ok = Config(n_cpu_moe=30, load_mode="mmap", moe_cache_mib=2500, miss_source="ring", prefill_source="p3")
        self.assertIsNone(ok.problem())
        self.assertIn("--moe-prefill-source p3", " ".join(LlamaCpp("llama-server").command(Path("m.gguf"), ok)))
        self.assertIn("--moe-prefill-source", ok.marian_flags())          # so the stock rollback never takes it

    def test_mmap_is_the_default_spelt_one_way(self):
        c = Config(n_cpu_moe=30, moe_cache_mib=2500, miss_source="tier", tier_mib=4096, prefill_source="mmap")
        self.assertIsNone(c.prefill_source)
        self.assertNotIn("--moe-prefill-source", LlamaCpp("x").command(Path("m.gguf"), c))
        self.assertEqual(plan.to_config(dict(c.to_row(), prefill_source="mmap")).key(), c.key())
        old_key = json.dumps({k: v for k, v in c.__dict__.items() if k not in ("prefill_source", "ubatch")},
                             sort_keys=True)
        self.assertEqual(c.key(), old_key)                  # a `confirmed` record written before still matches

    def test_planner_and_launch_check_refuse_a_hand_edited_pairing(self):
        bad = m13.row(n_cpu_moe=2, tg=99.0, share=4000, moe_cache_mib=2500, load_mode="mmap", prefill_source="p3",
                      exact_class=True)
        rows = plan.rows_of([(Path("p.json"), doc_of([bad, tier_row()]))])
        self.assertEqual(plan.choose(rows, 3000)["miss_source"], "tier")   # the faster broken row is not a candidate
        with tempfile.TemporaryDirectory() as d, m13.fake_machine(None):
            model = m13.toy_moe(Path(d))
            doc = doc_of([bad])
            doc["machine"]["fingerprint"], doc["model"]["id"] = machine.detect().fingerprint(), sha256_head(model)
            m13.FakeLlama.help_text, m13.FakeLlama.version_text = HELP_0009, VERSION
            reasons, _ = launch.check(bad, doc, model, sha256_head(model), m13.FakeLlama("x"), vram_free_mib=8000)
        self.assertTrue(any(r.startswith("configuration:") and "ring or tier" in r for r in reasons), reasons)

    def test_launch_check_needs_0009_for_a_p3_row(self):
        r = tier_row(prefill_source="p3")
        with tempfile.TemporaryDirectory() as d, m13.fake_machine(None):
            model = m13.toy_moe(Path(d))
            doc = doc_of([r])
            doc["machine"]["fingerprint"], doc["model"]["id"] = machine.detect().fingerprint(), sha256_head(model)
            m13.FakeLlama.help_text, m13.FakeLlama.version_text = m13.HELP_MARIAN, VERSION
            reasons, _ = launch.check(r, doc, model, sha256_head(model), m13.FakeLlama("x"), vram_free_mib=8000,
                                      ram_free_mib=30000)
        self.assertTrue(any("patch 0009" in x for x in reasons), reasons)


class PrefillPairRule(unittest.TestCase):
    def pair(self, mm, p3):
        a = {"status": "ok", "pp_tps": mm[0], "tg_tps": mm[1], "miss_source": "tier"}
        b = {"status": "ok", "pp_tps": p3[0], "tg_tps": p3[1], "miss_source": "tier", "prefill_source": "p3"}
        return prefill_pair(a, b), a, b

    def test_each_case(self):
        v, a, b = self.pair((35, 15.0), (61, 15.1))           # E51's shape: prefill +76%, decode the same
        self.assertEqual((v["kept"], a["status"], b["status"]), ("p3", "dominated", "ok"))
        v, a, b = self.pair((573, 40.0), (260, 40.0))         # a model that fits in the page cache
        self.assertEqual((v["kept"], a["status"], b["status"]), ("mmap", "ok", "dominated"))
        v, a, b = self.pair((100, 20.0), (105, 20.5))         # within the noise band on both: the default stays
        self.assertEqual((v["kept"], b["status"]), ("mmap", "dominated"))
        self.assertIn("tie", v["why"])
        v, a, b = self.pair((100, 20.0), (150, 16.0))         # a split: both stay, the objective decides
        self.assertEqual((v["kept"], a["status"], b["status"]), ("both", "ok", "ok"))
        v, a, b = self.pair((100, 20.0), (100, 25.0))         # decode alone beyond the band
        self.assertEqual((v["kept"], a["status"]), ("p3", "dominated"))


class PrefillCalibration(unittest.TestCase):
    def test_ring_and_tier_rows_get_both_sources_and_the_faster_is_kept(self):
        eng = PrefillFake(pp_gain=1.76)
        doc, _ = run_profile(eng)
        source = [r for r in doc["rows"] if r["arm"] == "source"]
        plain = [r for r in source if r.get("prefill_source") is None]
        p3 = [r for r in source if r.get("prefill_source") == "p3"]
        key = lambda r: (r["miss_source"], r.get("tier_mib"))                          # noqa: E731
        self.assertEqual([key(r) for r in plain], [key(r) for r in p3])               # every ring and tier row, twice
        self.assertEqual([r["miss_source"] for r in plain], ["ring", "tier", "tier", "tier"])
        self.assertTrue(all(r["status"] == "dominated" for r in plain))
        self.assertTrue(all(r["status"] == "ok" for r in p3))
        pairs = doc["cache_search"]["cells"][0]["prefill_pairs"]
        self.assertEqual([p["kept"] for p in pairs], ["p3"] * 4)
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000)
        self.assertEqual((best["miss_source"], best["prefill_source"]), ("tier", "p3"))

    def test_p3_loses_where_the_page_cache_holds_the_model(self):
        doc, _ = run_profile(PrefillFake(pp_gain=0.45))
        p3 = [r for r in doc["rows"] if r.get("prefill_source") == "p3"]
        self.assertTrue(p3 and all(r["status"] == "dominated" for r in p3))
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000, objective="pp")
        self.assertIsNone(best.get("prefill_source"))

    def test_a_split_keeps_both_and_the_objective_decides(self):
        doc, _ = run_profile(PrefillFake(pp_gain=1.5, tg_gain=0.8))
        rows = plan.rows_of([(Path("p.json"), doc)])
        self.assertIsNone(plan.choose(rows, 3000, objective="tg").get("prefill_source"))
        self.assertEqual(plan.choose(rows, 3000, objective="pp")["prefill_source"], "p3")

    def test_p3_is_never_started_beside_mmap_misses(self):
        eng = PrefillFake()
        run_profile(eng)
        p3 = [c for c in eng.started if c.prefill_source == "p3"]
        self.assertTrue(p3)
        self.assertTrue(all(c.miss_source in ("ring", "tier") for c in p3))
        self.assertFalse(any(c.prefill_source for c in eng.started if not c.miss_source))  # cliff and walk rows

    def test_a_build_without_0009_measures_mmap_only(self):
        eng = PrefillFake(help_text=m13.HELP_MARIAN)
        doc, _ = run_profile(eng)
        self.assertFalse(any(c.prefill_source for c in eng.started))
        self.assertIn("patch 0009", doc["cache_search"]["cells"][0]["prefill_pairs"])
        self.assertFalse(doc["engine"]["prefill_source"])


# --- the declared ubatch ------------------------------------------------------------------------------------------

class DeclaredUbatch(unittest.TestCase):
    def test_default_passes_no_flag_and_a_declared_one_passes_both(self):
        self.assertNotIn("-ub", LlamaCpp("x").command(Path("m.gguf"), Config(ubatch=512)))
        cmd = LlamaCpp("x").command(Path("m.gguf"), Config(ubatch=2048))
        self.assertEqual(cmd[cmd.index("-ub"):cmd.index("-ub") + 4], ["-ub", "2048", "-b", "2048"])
        row = json.loads(json.dumps(Config(n_cpu_moe=0, ubatch=2048).to_row()))
        self.assertEqual((row["ubatch"], plan.to_config(row).ubatch), (2048, 2048))

    def test_calibration_measures_every_row_at_the_declared_ubatch(self):
        eng = PrefillFake()
        doc, name = run_profile(eng, ubatch=2048)
        self.assertTrue(eng.started and all(c.ubatch == 2048 for c in eng.started))
        cmds = [LlamaCpp("x").command(Path("m.gguf"), c) for c in eng.started if not c.problem()]
        self.assertTrue(all("-ub" in c and c[c.index("-ub") + 1] == "2048" for c in cmds))
        self.assertEqual((doc["marian_profile"], doc["ubatch"]), (3, 2048))
        self.assertTrue(name.endswith("-ub2048.json"), name)                  # beside the 512 profile, not over it
        self.assertTrue(all(r["ubatch"] == 2048 for r in doc["rows"]))
        cell = doc["ubatch_report"][0]
        for k in ("best_decode", "best_prefill"):
            self.assertEqual(set(cell["exact"][k]) >= {"tg_tps", "pp_tps"}, True)
        _, name512 = run_profile(PrefillFake())
        self.assertFalse(name512.endswith("-ub512.json"))                     # the default keeps M0's name

    def test_declared_ubatch_flows_into_the_server_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--ubatch", "2048", "--dry-run",
                                 "--json", docs=[("", DOC_512), ("-ub2048", DOC_2048)])
        j = json.loads(out)
        self.assertEqual((rc, j["ok"], j["ubatch"], j["ubatch_flag"], j["prefill_source"]), (0, True, 2048, True, "p3"))
        a = j["server_args"]
        self.assertEqual(a[a.index("-ub"):a.index("-ub") + 4], ["-ub", "2048", "-b", "2048"])
        self.assertIn("--moe-prefill-source", a)
        self.assertEqual(m13.FakeLlama.starts, [])

    def test_default_deployment_takes_the_512_profile(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run",
                                   docs=[("", DOC_512), ("-ub2048", DOC_2048)])
        self.assertEqual(rc, 0)
        self.assertNotIn("ubatch", json.loads(out))
        self.assertIn("settings: ubatch 512 (the default, no flag); prefill source mmap", err)

    def test_calibrate_and_serve_calibrate_pass_the_declared_ubatch(self):
        calls = []

        def fake_run(model, eng, ctxs, kvs, step, threads, out_dir, n_predict, cache=None, sweep=None, ubatch=512):
            calls.append(ubatch)
            fp, mid = machine.detect().fingerprint(), sha256_head(model)
            d = json.loads(json.dumps(DOC_2048))
            d["machine"]["fingerprint"], d["model"]["id"], d["model"]["bytes"] = fp, mid, model.stat().st_size
            p = out_dir / profile.profile_name(f"{fp}-{mid}-llama.cpp", ubatch)
            p.write_text(json.dumps(d), encoding="utf-8")
            return p

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, \
                mock.patch.object(cli.profile, "run", fake_run):
            rc1, out1, _ = run_cli(Path(d1), "calibrate", "--engine", "x", "--ubatch", "2048")
            rc2, out2, _ = run_cli(Path(d2), "serve", "--engine", "x", "--need", "3000", "--ubatch", "2048",
                                   "--calibrate", "--json", docs=[("", DOC_512)])   # 512 only: the gate calibrates
        self.assertEqual(calls, [2048, 2048])
        self.assertEqual(rc1, 0)
        self.assertIn("ubatch 2048, ctx 4096 kv f16", out1)                    # the trade report
        self.assertTrue(json.loads(out2.splitlines()[0])["ok"])
        self.assertEqual(m13.FakeLlama.starts[0].ubatch, 2048)

    def test_trade_report_puts_other_ubatches_beside(self):
        d512 = dict(DOC_512, measured_utc="earlier")
        d2048 = dict(DOC_2048, ubatch_report=profile.ubatch_report(
            [dict(r, ctx=r["ctx"]) for r in DOC_2048["rows"]]))
        lines = launch.trade_lines(d2048, [d512])
        self.assertTrue(lines[0].startswith("ubatch 2048, ctx 4096 kv f16: exact-class best decode 38.0 tok/s"))
        self.assertIn("beside it, ubatch 512 (measured earlier, another session)", lines[1])


class UbatchRefusal(unittest.TestCase):
    def test_serving_a_512_profile_at_2048_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--ubatch", "2048", "--json",
                                 docs=[("", DOC_512)])
        j = json.loads(out)
        self.assertEqual((rc, j["ok"], j["gate"]), (2, False, "ubatch"))
        self.assertIn("at ubatch 512 only", j["reason"])
        self.assertIn("never served at another", j["reason"])
        self.assertIn("--ubatch 2048", j["reason"])
        self.assertEqual(m13.FakeLlama.starts, [])

    def test_serving_a_2048_profile_at_the_default_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--json",
                                 docs=[("-ub2048", DOC_2048)])
            rc2, out2, _ = run_cli(Path(d), "plan", "--need", "3000", "--json", docs=[("-ub2048", DOC_2048)])
        self.assertEqual((rc, json.loads(out)["gate"]), (2, "ubatch"))
        j = json.loads(out2)
        self.assertEqual((rc2, j["ok"]), (2, False))
        self.assertTrue(j["reason"].startswith("ubatch: this deployment declares ubatch 512"), j["reason"])

    def test_the_funnel_never_crosses_ubatches(self):
        rows = plan.rows_of([(Path("a.json"), DOC_512), (Path("b-ub2048.json"), DOC_2048)])
        cur = plan.choose(rows, 3000, ubatch=2048)
        self.assertEqual(plan.ubatch_of(cur), 2048)
        self.assertTrue(all(plan.ubatch_of(r) == 2048 for r in plan.candidates(rows, 3000, ubatch=2048)))
        ru = plan.runner_up(rows, cur, 3000, ubatch=2048, allow_non_exact=True)
        self.assertEqual(plan.ubatch_of(ru), 2048)
        nxt = plan.next_smaller(rows, cur, 3000, pressure={"vram"}, vram_free_mib=None, ram_free_mib=None, ubatch=2048)
        self.assertTrue(nxt is None or plan.ubatch_of(nxt) == 2048)
        self.assertIsNone(plan.next_smaller(rows[:2] + [cur], cur, 3000, pressure=set(), vram_free_mib=None,
                                            ram_free_mib=None, ubatch=2048))      # the 512 rows are never offered

    def test_launch_check_refuses_a_row_from_another_ubatch(self):
        with tempfile.TemporaryDirectory() as d, m13.fake_machine(None):
            model = m13.toy_moe(Path(d))
            doc = json.loads(json.dumps(DOC_512))
            doc["machine"]["fingerprint"], doc["model"]["id"] = machine.detect().fingerprint(), sha256_head(model)
            m13.FakeLlama.help_text, m13.FakeLlama.version_text = HELP_0009, VERSION
            reasons, _ = launch.check(doc["rows"][0], doc, model, sha256_head(model), m13.FakeLlama("x"),
                                      vram_free_mib=9000, ubatch=2048)
        self.assertTrue(any(r.startswith("ubatch: the row was measured at ubatch 512") for r in reasons), reasons)

    def test_serve_options_carry_the_ubatch_to_the_governor(self):
        rows = plan.rows_of([(Path("a.json"), DOC_512), (Path("b-ub2048.json"), DOC_2048)])
        seen = {}
        real = plan.next_smaller

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        from marian import serve
        with tempfile.TemporaryDirectory() as d:
            eng = mock.Mock(start=lambda m, c, log: (False, "offline"), stop=lambda: None)
            sup = serve.Supervisor(eng, Path("m.gguf"), rows, serve.ServeOptions(ubatch=2048, drain_s=0), serve.Gate(),
                                   Path(d) / "serve.log", say=lambda m: None)
            sup.row = plan.choose(rows, 3000, ubatch=2048)
            with mock.patch.object(plan, "next_smaller", spy):
                sup.restart({"vram"}, "test", {"vram_free_mib": 100, "ram_avail_mib": 20000, "ram_total_mib": 32000})
        self.assertEqual(seen.get("ubatch"), 2048)


class OldSchema(unittest.TestCase):
    def test_a_schema_2_profile_is_a_512_profile(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json",
                                 docs=[("", m13.Serve.DOC)])
            rc2, out2, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--ubatch", "2048", "--json",
                                   docs=[("", m13.Serve.DOC)])
        j = json.loads(out)
        self.assertEqual((rc, j["ok"], j["ubatch"], j["prefill_source"]), (0, True, 512, "mmap"))
        self.assertNotIn("-ub", j["server_args"])
        self.assertNotIn("--moe-prefill-source", j["server_args"])
        self.assertTrue(any("predate the P3 prefill rows" in w for w in j["warnings"]), j["warnings"])
        self.assertEqual((rc2, json.loads(out2)["gate"]), (2, "ubatch"))
        self.assertIn("at ubatch 512 only", json.loads(out2)["reason"])

    def test_a_schema_1_profile_still_plans(self):
        m0 = doc_of([{"ctx": 4096, "kv": "f16", "n_cpu_moe": 0, "threads": None, "status": "ok", "tg_tps": 30.0,
                      "pp_tps": 500.0, "vram_peak_mib": 7000}], schema=1)
        m0["engine"].pop("marian")
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "plan", "--need", "3000", "--json", docs=[("", m0)])
        j = json.loads(out)
        self.assertEqual((rc, j["ok"], j["ubatch"]), (0, True, 512))

    def test_a_newer_schema_is_refused(self):
        future = dict(DOC_512, marian_profile=99)
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = run_cli(Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json",
                                 docs=[("", future)])
        j = json.loads(out)
        self.assertEqual((rc, j["ok"], j["gate"]), (2, False, "schema"))
        self.assertIn("schema 99 is newer", j["reason"])


class DiagShowsBoth(unittest.TestCase):
    def test_diag_plan_names_the_ubatch_and_the_prefill_source(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = m13.toy_moe(d)
            prof = d / "profiles"
            prof.mkdir()
            m13.FakeLlama.help_text, m13.FakeLlama.version_text = HELP_0009, VERSION
            with m13.fake_machine(None), mock.patch.object(cli, "PROFILES", prof), \
                    mock.patch.object(cli, "LlamaCpp", m13.FakeLlama):
                fp, mid = machine.detect().fingerprint(), sha256_head(model)
                for suffix, doc in (("", DOC_512), ("-ub2048", DOC_2048)):
                    d2 = json.loads(json.dumps(doc))
                    d2["machine"]["fingerprint"], d2["model"]["id"] = fp, mid
                    (prof / f"{fp}-{mid}-llama.cpp{suffix}.json").write_text(json.dumps(d2), encoding="utf-8")
                out = d / "diag.zip"
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = cli.main(["diag", str(out), str(model), "--engine", "llama-server", "--need", "3000",
                                   "--ubatch", "2048"])
            with zipfile.ZipFile(out) as z:
                pj = json.loads(z.read("plan.json"))
                ver = json.loads(z.read("versions.json"))
        self.assertEqual(rc, 0)
        self.assertEqual((pj["ubatch"], pj["ubatches_measured"], pj["profile_schemas"]), (2048, [512, 2048], [3]))
        ch = pj["chosen"]
        self.assertEqual((ch["ubatch"], ch["prefill_source"]), (2048, "p3"))
        self.assertIn("-ub", ch["server_args"])
        self.assertTrue(ver["engine"]["prefill_source"])


if __name__ == "__main__":
    unittest.main()
