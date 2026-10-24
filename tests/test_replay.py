"""Offline tests of the routing-id trace format and `marian replay`: no GPU, no server, SYNTHETIC traces only.
python -m unittest discover -s tests"""
import json
import random
import struct
import tempfile
import unittest
from array import array
from pathlib import Path

from marian import geometry as G
from marian import idstrace, replay, synth
from marian.cli import main as cli_main
from marian.idstrace import Step, Trace

DATA = Path(__file__).resolve().parent / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs" / "m1.2"


def tiny_geometry(n_layers, n_expert, k, nb2=(1000, 1000, 1000)):
    layers = {il: G.Layer(il, n_expert, [G.ExpertTensor(r, "x", (0, 0), b) for r, b in zip(("gate", "up", "down"), nb2)])
              for il in range(n_layers)}
    return G.Geometry(k, layers, "SYNTHETIC", {"synthetic": True})


def session(seed=1, **kw):
    sp = synth.SynthSpec(seed=seed, **kw)
    tr, reqs, geom = synth.synth_trace(sp)
    idstrace.assign_requests(tr, reqs)
    return tr, reqs, geom


class IdsFormat(unittest.TestCase):
    def test_reads_the_file_the_cpp_writer_wrote(self):
        # written by llama.cpp tests/test-moe-trace-ids.cpp (patch 0004) on Linux; the Windows build writes the same bytes
        tr = idstrace.read_ids(DATA / "ids-sample-cpp.bin")
        self.assertEqual((tr.t_open_us, tr.truncated_bytes, len(tr.steps)), (777, 0, 4))
        s = tr.steps
        self.assertEqual((s[0].kind, s[0].layer, s[0].n_tokens, s[0].k, s[0].t_us, list(s[0].ids)),
                         ("S", 0, 2, 4, 1000, [7, 3, 1, 0, 3, 5, 6, 7]))
        self.assertEqual(list(s[2].ids), [511, 0, 0xffff, 0xffff])          # ids that do not fit in u16
        self.assertEqual((s[3].kind, s[3].layer, s[3].n_tokens, s[3].k, list(s[3].ids)), ("L", 1, 3, 2, [10, 20, 11, 21, 12, 22]))

    def test_roundtrip_and_a_record_cut_by_a_kill(self):
        steps = [Step("S", 10 + i, i % 3, 1, 4, array("H", [i, i + 1, i + 2, 500])) for i in range(9)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.ids.bin"
            idstrace.write_ids(p, steps, 5)
            back = idstrace.read_ids(p)
            self.assertEqual([(x.t_us, x.layer, list(x.ids)) for x in back.steps], [(x.t_us, x.layer, list(x.ids)) for x in steps])
            p.write_bytes(p.read_bytes()[:-3])
            cut = idstrace.read_ids(p)
            self.assertEqual((len(cut.steps), cut.truncated_bytes), (8, 24 - 3))

    def test_size_estimate_in_the_notes(self):
        # 16-byte record head + 10 u16 ids per single-token step; 2,400 tokens x 48 layers
        self.assertEqual(struct.calcsize("<BBHIq"), 16)
        self.assertEqual(2400 * 48 * (16 + 2 * 10), 4_147_200)

    def test_swapped_head_of_the_first_writer_is_refused(self):
        # the first 0004 wrote a 2-token S step as n_tokens=1, k=20 (the ids reach the cache as a 1-D reshape);
        # with the model's k the reader must refuse it, and check_heads must flag it against the S line
        bad = [Step("S", 100, 0, 1, 20, array("H", range(20)))]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.ids.bin"
            idstrace.write_ids(p, bad)
            with self.assertRaises(ValueError):
                idstrace.read_ids(p, expected_k=10)
            tr = idstrace.read_ids(p)                       # readable without the model's k ...
            ok, joined, msgs = idstrace.check_heads(tr, [{"kind": "S", "t_us": 100, "layer": 0, "n_ids": 20}])
            self.assertEqual((ok, joined), (1, 1))          # ... and n_tokens*k still equals n_ids, so the k check is the one that bites
            good = [Step("S", 100, 0, 2, 10, array("H", range(20)))]
            idstrace.write_ids(p, good)
            self.assertEqual(len(idstrace.read_ids(p, expected_k=10).steps), 1)
            ok, joined, msgs = idstrace.check_heads(idstrace.read_ids(p), [{"kind": "S", "t_us": 100, "layer": 0, "n_ids": 30}])
            self.assertEqual((ok, joined, len(msgs)), (0, 1, 1))

    def test_a_kind_that_is_not_s_or_l_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.ids.bin"
            idstrace.write_ids(p, [Step("S", 1, 0, 1, 2, array("H", [1, 2]))])
            b = bytearray(p.read_bytes()); b[32] = ord("X"); p.write_bytes(bytes(b))
            with self.assertRaises(ValueError):
                idstrace.read_ids(p)

    def test_not_an_ids_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(b"NOPE" * 10)
            with self.assertRaises(ValueError):
                idstrace.read_ids(p)


class Geometry(unittest.TestCase):
    def test_swift_slots_match_the_m11_table(self):
        g = G.load(DOCS / "swift-1.5-iq2_xs.geometry.json")
        self.assertEqual((len(g.layers), g.n_expert_used, round(g.total_expert_bytes / 2**30, 2)), (48, 10, 33.02))
        self.assertEqual([(x.type, len(x.layers), x.expert_bytes) for x in g.groups()],
                         [("iq2_s", 34, 1510400), ("iq2_xxs", 11, 1305600), ("iq1_m", 3, 1177600)])
        table = {2400: (1105, 356, 320), 4000: (1973, 637, 320), 6500: (3329, 1076, 320), 9400: (4838, 1564, 426)}
        for mib, want in table.items():
            self.assertEqual(tuple(s.n_slots for s in G.cache_slots(g, mib)), want, mib)
        self.assertTrue(G.cache_slots(g, 4000)[2].at_floor)
        self.assertFalse(any(s.at_floor for s in G.cache_slots(g, 9400)))

    def test_budget_below_the_floor_is_refused(self):
        g = G.load(DOCS / "swift-1.5-iq2_xs.geometry.json")
        with self.assertRaises(G.BudgetBelowFloor) as e:
            G.cache_slots(g, 1222)
        self.assertEqual(e.exception.min_mib, 1223)
        G.cache_slots(g, 1223)

    def test_alloc_size_pads_each_bank_and_keeps_a_spare_slot(self):
        g = tiny_geometry(2, 8, 2, (100, 100, 60)).groups()[0]
        self.assertEqual(g.alloc_size(3), 2 * 512 + 256)         # pad(100*4, 128) twice + pad(60*4, 128)
        self.assertEqual(g.slots_for(2 * 512 + 256), 3)
        self.assertEqual(g.slots_for(10**9), 16)                 # never more than every expert of the group

    def test_from_a_gguf_header(self):
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b
        kv = s("general.architecture") + struct.pack("<I", 8) + s("toy") + s("toy.expert_used_count") + struct.pack("<II", 4, 2)
        tensors = [("blk.0.ffn_gate_exps.weight", (512, 4, 8), 22), ("blk.0.ffn_down_exps.weight", (4, 512, 8), 0),
                   ("blk.1.ffn_gate_exps.weight", (512, 4, 8), 29)]
        sizes = [4 * 2 * 82 * 8, 512 * 16 * 8, 4 * 2 * 56 * 8]
        body, off = b"", 0
        for (name, ne, t), n in zip(tensors, sizes):
            body += s(name) + struct.pack("<I", 3) + struct.pack("<3Q", *ne) + struct.pack("<IQ", t, off)
            off += (n + 31) // 32 * 32
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "toy.gguf"
            p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2) + kv + body)
            g = G.from_gguf([p])
        self.assertEqual(g.n_expert_used, 2)
        self.assertEqual([t.nb2 for t in g.layers[0].tensors], [4 * 2 * 82, 512 * 16])   # gate before down
        self.assertEqual(g.layers[1].tensors[0].type, "iq1_m")
        self.assertEqual(len(g.groups()), 2)


class SlruHand(unittest.TestCase):
    """1 layer, 8 experts, 4 slots (protected holds at most 2), 2 experts per token, worked through by hand."""

    def test_seven_steps(self):
        s = replay.SLRU([0], 8, 4)

        def step(ids):
            rh = sum(s.slot_map[0][e] >= 0 for e in ids)
            uniq = list(dict.fromkeys(ids))
            fills, n_hit = s.plan(0, uniq)
            return rh, n_hit, fills

        def segs():
            out = []
            for seg in (0, 1):
                x, lst = s.head[seg], []
                while x >= 0:
                    lst.append(x)
                    x = s.next[x]
                out.append(lst)
            return out
        self.assertEqual(step([5, 2]), (0, 0, [(2, 0), (5, 1)]))          # sorted misses take empty slots in order
        self.assertEqual(segs(), [[2, 3, 0, 1], []])
        self.assertEqual(step([2, 7]), (1, 1, [(7, 2)]))                  # the hit moves to protected
        self.assertEqual(segs(), [[3, 1, 2], [0]])
        self.assertEqual(step([1, 6]), (0, 0, [(1, 3), (6, 1)]))          # an empty slot, then the probation LRU (5)
        self.assertEqual(step([7, 1]), (2, 2, []))                        # protected 3 > 2: slot 0 is demoted
        self.assertEqual(segs(), [[1, 0], [2, 3]])
        self.assertEqual(step([5, 2]), (1, 1, [(5, 1)]))                  # 2 back to protected, 7 (slot 2) demoted
        self.assertEqual(segs(), [[2, 1], [3, 0]])
        self.assertEqual(step([6, 7, 6, 1]), (2, 2, [(6, 1)]))            # 2 tokens: row hits count per row
        self.assertEqual(segs(), [[0, 1], [2, 3]])
        # 4 distinct misses: slots holding experts this step already selected are skipped, protected is used last
        self.assertEqual(step([0, 3, 4, 5]), (0, 0, [(0, 0), (3, 1), (4, 2), (5, 3)]))
        self.assertEqual(segs(), [[0, 1, 2, 3], []])
        self.assertIsNone(s.plan(0, [0, 1, 2, 3, 4]))                     # more distinct experts than slots


class SlruAgainstTheFork(unittest.TestCase):
    """Fixtures from docs/m1.2/tools/slru_xcheck.sh: the fork's own moe_cache_slru, cut verbatim from
    src/llama-moe-cache.cpp and compiled, on the same steps."""

    def check(self, name):
        lines = [x for x in (DATA / f"slru-xcheck-{name}.in").read_text().splitlines() if x.strip()]
        n_layer, n_expert, n_slots = map(int, lines[0].split())
        s = replay.SLRU(range(n_layer), n_expert, n_slots)
        want = (DATA / f"slru-xcheck-{name}.out").read_text().splitlines()
        self.assertEqual(len(want), len(lines) - 1)
        for line, w in zip(lines[1:], want):
            v = list(map(int, line.split()))
            il, ids = v[0], v[2:]
            rh = sum(s.slot_map[il][e] >= 0 for e in ids)
            r = s.plan(il, list(dict.fromkeys(ids)))
            got = "REFUSED" if r is None else f"{rh} {r[1]} {len(r[0])}" + "".join(f" {e}:{sl}" for e, sl in r[0])
            self.assertEqual(got, w, line[:80])

    def test_small_with_a_refused_step(self):
        self.check("small")

    def test_swift_like_512_experts_10_per_token(self):
        self.check("swiftlike")


class TierAgainstTheFork(unittest.TestCase):
    """Fixtures from docs/m1.2/tools/tier_xcheck.sh: the fork's own llama_moe_source (patch 0003) in tier mode on the CPU
    backend, fed the miss streams of SYNTHETIC traces (make_xcheck_inputs.py)."""

    def check(self, name):
        lines = [x for x in (DATA / f"tier-xcheck-{name}.in").read_text().splitlines() if x.strip()]
        n_layer, n_expert, ring, tier = map(int, lines[0].split())
        layers = {}
        for il in range(n_layer):
            v = list(map(int, lines[1 + il].split()))
            layers[il] = G.Layer(il, n_expert, [G.ExpertTensor(r, "x", (0, 0), b) for r, b in zip(("gate", "up", "down"), v[2:])])
        g = G.Geometry(4, layers)
        t = replay.CurrentTier(replay.TierGeometry(g, g.groups(), 4096, ring / G.MiB), tier / G.MiB)
        want = (DATA / f"tier-xcheck-{name}.out").read_text().split()
        self.assertEqual(len(want), len(lines) - 1 - n_layer)
        for line, w in zip(lines[1 + n_layer:], want):
            v = list(map(int, line.split()))
            il, dec, nu = v[0], v[1], v[2]
            t.begin(il, bool(dec), v[3:3 + nu])
            fl = v[4 + nu::2]
            self.assertEqual("".join(t.fill(e) for e in fl) or "-", w, line[:80])
        return t

    def test_tiny_tier_small_ring(self):
        self.assertEqual(self.check("tiny").n_wrap, 17)

    def test_mid(self):
        self.assertEqual(self.check("mid").n_wrap, 64)

    def test_big(self):
        self.check("big")


class KnownAnswers(unittest.TestCase):
    def test_a_cache_that_holds_everything_uploads_each_expert_once(self):
        tr, _, geom = session(n_layers=4, n_expert=32, k=2)
        run = replay.run_vram(tr, geom, 10_000)
        self.assertTrue(all(s.n_slots == 4 // 2 * 32 for s in run.slots))
        fills = [(s.layer, e) for s, o in zip(tr.steps, run.out) if o for e, _ in o.fills]
        selected = {(s.layer, e) for s in tr.steps if s.kind == "S" for e in s.ids}
        self.assertEqual(len(fills), len(set(fills)))
        self.assertEqual(set(fills), selected)

    def test_large_batches_never_change_the_cache(self):
        tr, _, geom = session(n_layers=2, n_expert=64, k=4, prompt_tokens=100)
        self.assertTrue(any(s.kind == "L" for s in tr.steps))
        run = replay.run_vram(tr, geom, 300)
        self.assertTrue(all(o is None for s, o in zip(tr.steps, run.out) if s.kind == "L"))

    def iid_misses(self, n, n_expert, s=1.0, seed=3):
        rng = random.Random(seed)
        w = [1 / (i + 1) ** s for i in range(n_expert)]
        z = sum(w)
        draws = rng.choices(range(n_expert), w, k=n)
        steps = [Step("S", i, 0, 1, 1, array("H", [e]), req=0) for i, e in enumerate(draws)]
        misses = [replay.MissStep(i, 0, 0, True, [e], [e], 1000) for i, e in enumerate(draws)]
        return Trace(steps, requests=[{"phase": "decode", "idx": 0}]), misses, [x / z for x in w]

    def test_a_static_tier_of_the_top_m_serves_their_probability(self):
        geom = tiny_geometry(1, 64, 1)
        tg = replay.TierGeometry(geom, geom.groups())
        tr, misses, p = self.iid_misses(40_000, 64)
        m = 10
        tier_mib = m * tg.slot_bytes[0] / G.MiB
        prof = {e: 1000 - e for e in range(64)}                           # expert e has rank e
        contents = replay.static_contents(tg, prof, tier_mib, {0: geom.layers[0].expert_bytes})
        self.assertEqual(contents, set(range(m)))
        share = replay.tier_share(tr, misses, replay.static_replay(misses, contents, 64), None, True)["share"]
        self.assertAlmostEqual(share, sum(p[:m]), delta=0.01)             # 0.636 for Zipf(1) over 64

    def test_bounds_order_on_a_stationary_stream(self):
        geom = tiny_geometry(1, 64, 1)
        tg = replay.TierGeometry(geom, geom.groups())
        tr, misses, _ = self.iid_misses(20_000, 64)
        tier_mib = 12 * tg.slot_bytes[0] / G.MiB
        test_prof = replay.miss_profile(tr, misses, {0}, 64)
        sh = lambda src: replay.tier_share(tr, misses, src, None, True)["share"]          # noqa: E731
        static = sh(replay.static_replay(misses, replay.static_contents(tg, test_prof, tier_mib, {0: 3000}), 64))
        belady = sh(replay.belady_replay(misses, tg, tier_mib))
        current = sh(replay.CurrentTier(tg, tier_mib).replay(misses))
        self.assertGreaterEqual(belady, static)
        self.assertGreaterEqual(static, current - 0.01)
        self.assertGreater(current, 0.3)

    def test_belady_by_hand(self):
        geom = tiny_geometry(1, 8, 1)
        tg = replay.TierGeometry(geom, geom.groups())
        seq = [0, 1, 2, 0, 1, 2, 0]
        misses = [replay.MissStep(i, 0, 0, True, [e], [e], 3000) for i, e in enumerate(seq)]
        src = replay.belady_replay(misses, tg, 2 * tg.slot_bytes[0] / G.MiB)
        # holds 0 and 1; 2 is bypassed twice: first both cached experts are needed sooner, then it is never used again
        self.assertEqual("".join(x[0] for x in src), "rrrttrt")

    def test_a_tier_that_holds_everything_admits_each_decode_miss_once(self):
        tr, _, geom = session(n_layers=4, n_expert=64, k=4)
        run = replay.run_vram(tr, geom, 300)
        misses = replay.miss_stream(tr, run, 4)
        tg = replay.TierGeometry(geom, geom.groups())
        src = replay.CurrentTier(tg, 10_000).replay(misses)
        admitted = [(m.layer, e) for m, ss in zip(misses, src) for e, x in zip(m.fills, ss) if x == "a"]
        decode_missed = {(m.layer, e) for m in misses if m.decode for e in m.fills}
        self.assertEqual(len(admitted), len(set(admitted)))
        self.assertEqual(set(admitted), decode_missed)
        for m, ss in zip(misses, src):
            if not m.decode:
                self.assertNotIn("a", ss)                                 # prompt chunks only read the tier

    def test_ranked_core_hits_from_the_first_token(self):
        tr, _, geom = session(n_layers=4, n_expert=64, k=4)            # 2 layers per group: 110 slots, floor 64
        sp = replay.split_prompts(tr)
        prof = replay.selection_profile(tr, replay.tokens_of(tr, sp.train, 4, 4, ("decode",)), 64)
        run = replay.run_vram(tr, geom, 300, "core50", prof)
        for gr, gs in zip(run.groups, run.slots):
            self.assertEqual(gr.n_core, min(int(0.5 * gs.n_slots), gs.n_slots - gs.n_floor))
            self.assertGreaterEqual(gr.slru.n_slots, gs.n_floor)          # the capacity invariant
            self.assertGreater(gr.n_core, 0)
        first = lambda r: sum(o.row_hits for o in [x for x in r.out if x is not None][:2])   # noqa: E731
        self.assertEqual(first(replay.run_vram(tr, geom, 300)), 0)        # the first pass: SLRU starts empty,
        self.assertGreater(first(run), 0)                                 # the core does not

    def test_fetch_model_points(self):
        self.assertEqual([round(replay.fetch_ms(x), 6) for x in (0, 0.75, 0.9)], [46, 15, 8])


class SplitAndTokens(unittest.TestCase):
    def test_split_by_prompt_never_by_token(self):
        tr, reqs, _ = session()
        sp = replay.split_prompts(tr)
        self.assertFalse(sp.train & sp.test)
        self.assertFalse(set(sp.train_keys) & set(sp.test_keys))
        warm = next(i for i, r in enumerate(reqs) if r["phase"] == "warm")
        d0 = next(i for i, r in enumerate(reqs) if r["phase"] == "decode" and r["idx"] == 0)
        self.assertEqual(warm in sp.train, d0 in sp.train)                # warm is decode prompt 0 again
        self.assertEqual(sp.test_keys, ["decode:1", "decode:3", "decode:5"])
        self.assertEqual(replay.split_prompts(tr, ["decode:2"]).test_keys, ["decode:2"])

    def test_tokens_and_requests_without_a_request_log(self):
        tr, reqs, _ = session(n_layers=4)
        tr2 = Trace(tr.steps)
        idstrace.infer_requests(tr2)
        self.assertEqual(len(tr2.requests), len(reqs))                    # the server's warm-up joins the first request
        toks = replay.tokens_of(tr, None, 4, 4, ("decode",))
        self.assertEqual(len(toks), 6 * 48)
        self.assertTrue(all(len(t) == 4 for t in toks))

    def test_deterministic(self):
        a = replay.run_all(*session(seed=5)[::2], 300, tiers=(256,))
        b = replay.run_all(*session(seed=5)[::2], 300, tiers=(256,))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))


class CheckAgainstAServerTrace(unittest.TestCase):
    def test_exact_at_the_traced_budget_and_a_mismatch_is_found(self):
        tr, _, geom = session()
        run = replay.run_vram(tr, geom, 300)
        s_rows, f_rows = replay.as_server_rows(tr, run)
        c = replay.check(tr, run, s_rows, f_rows)
        self.assertEqual((c["s_lines"]["match"], c["s_lines"]["n"]), (len(s_rows), len(s_rows)))
        self.assertEqual(c["fill_steps"]["match"], c["fill_steps"]["n"])
        s_rows[100]["row_hits"] += 1
        f_rows[5]["slot"] += 1
        c = replay.check(tr, run, s_rows, f_rows)
        self.assertEqual(c["s_lines"]["match"], len(s_rows) - 1)
        self.assertEqual(c["fill_steps"]["match"], c["fill_steps"]["n"] - 1)
        other = replay.run_vram(tr, geom, 400)                            # another budget: the replay sees it
        self.assertLess(replay.check(tr, other, replay.as_server_rows(tr, run)[0], None)["s_lines"]["match"], len(s_rows))

    def test_measured_hit_rate_equals_the_replay_on_its_own_trace(self):
        tr, _, geom = session()
        rows = replay.as_server_rows(tr, replay.run_vram(tr, geom, 300))[0]
        rep = replay.run_all(tr, geom, 300, csv_rows=rows, tiers=(256,), p2=())
        self.assertEqual(rep["check"]["median_diff_points"], 0.0)
        self.assertEqual(rep["check"]["server_decode_all"]["tokens"], rep["slru"]["decode_all"]["tokens"])

    def test_csv_roundtrip(self):
        tr, _, geom = session(n_layers=2)
        rows = replay.as_server_rows(tr, replay.run_vram(tr, geom, 300))[0]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.trace.csv"
            idstrace.write_trace_csv(p, rows)
            with p.open("a") as f:
                f.write("S,123,4")                                        # a line cut by the kill
            back = idstrace.read_csv(p)
        self.assertEqual(len(back), len(rows))
        self.assertEqual(back[7]["row_hits"], rows[7]["row_hits"])


class Cli(unittest.TestCase):
    def test_synth_then_replay(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cli_main(["synth", d, "--n-prompts", "4"]), 0)
            md = Path(d) / "r.md"
            rc = cli_main(["replay", f"{d}/synthetic.ids.bin", "--geometry", f"{d}/synthetic.geometry.json",
                           "--requests", f"{d}/synthetic.requests.jsonl", "--budget-mib", "300,400", "--tier-mib", "128,512",
                           "--md", str(md), "--json", str(Path(d) / "r.json")])
            self.assertEqual(rc, 0)
            text = md.read_text()
            self.assertIn("SYNTHETIC", text)
            self.assertIn("--moe-cache-mib 400", text)
            self.assertEqual(len(json.loads((Path(d) / "r.json").read_text())), 2)
            self.assertEqual(cli_main(["replay", f"{d}/synthetic.ids.bin", "--geometry", f"{d}/synthetic.geometry.json",
                                       "--budget-mib", "50"]), 2)          # below the floor: refused like the server


if __name__ == "__main__":
    unittest.main()
