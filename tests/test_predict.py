"""Offline tests of `marian predict` (register E47): SYNTHETIC traces with a known answer, written before the tool read
any real trace. python -m unittest discover -s tests"""
import json
import random
import tempfile
import unittest
from array import array
from pathlib import Path

from marian import geometry as G
from marian import idstrace, predict, replay, synth
from marian.cli import main as cli_main
from marian.idstrace import Step, Trace

DOCS = Path(__file__).resolve().parents[1] / "docs" / "m1.2"
NE = 32


def tiny_geometry(n_layers, n_expert=NE, k=2):
    layers = {il: G.Layer(il, n_expert, [G.ExpertTensor(r, "x", (0, 0), 1000) for r in ("gate", "up", "down")])
              for il in range(n_layers)}
    return G.Geometry(k, layers, "SYNTHETIC", {"synthetic": True})


def default_prompt_route(key, row, il, k=2):
    return [(row * 7 + il + 11 * x) % NE for x in range(k)]


def build(route, *, n_layers=6, k=2, n_dec=60, requests=None, n_prompt=5, prompt_route=default_prompt_route):
    """A session shaped like the real one: the server's warm-up outside every request, then per request a prompt pass on
    layers 0..n-2 closed by the 1-row step at the last layer (llama.cpp computes it for the output row only), then n_dec
    single-token passes. route(key, j, il, below, prev) gives token j's ids at layer il; below = its ids at il-1, prev =
    the previous token's ids at il (None at the first)."""
    if requests is None:
        requests = [("decode", i, n_dec, n_prompt) for i in range(10)]
    steps, reqs, t = [], [], [1000]

    def step(layer, rows):
        steps.append(Step("S" if len(rows) <= 32 else "L", t[0], layer, len(rows), k, array("H", [e for r in rows for e in r])))
        t[0] += 10
    for il in range(n_layers):
        step(il, [list(range(k))] * (2 if il < n_layers - 1 else 1))
    for phase, idx, nd, npr in requests:
        t[0] += 1000
        key = (phase, idx)
        r = {"phase": phase, "idx": idx, "t0_us": t[0]}
        for il in range(n_layers - 1):
            step(il, [prompt_route(key, row, il) for row in range(npr)])
        step(n_layers - 1, [prompt_route(key, npr - 1, n_layers - 1)])
        prev = {}
        for j in range(nd):
            below = None
            for il in range(n_layers):
                ids = route(key, j, il, below, prev.get(il))
                step(il, [ids])
                prev[il], below = ids, ids
        r["t1_us"] = t[0]
        reqs.append(r)
        t[0] += 1000
    tr = Trace(steps)
    idstrace.assign_requests(tr, reqs)
    return tr, reqs


def run(tr, n_layers=6, k=2, perms=20, **kw):
    return predict.run_all(tr, tiny_geometry(n_layers, NE, k), 1.0, permutations=perms, seed=5, **kw)


def zipf_route(seed, k=2, s=1.1):
    """iid draws, without replacement within a row, from a fixed Zipf distribution per layer: no time, no layer coupling."""
    rng = random.Random(seed)
    w = {}

    def route(key, j, il, below, prev):
        if il not in w:
            perm = list(range(NE))
            random.Random(100 + il).shuffle(perm)
            w[il] = [0.0] * NE
            for r, e in enumerate(perm):
                w[il][e] = 1 / (r + 1) ** s
        out = []
        while len(out) < k:
            e = rng.choices(range(NE), w[il])[0]
            if e not in out:
                out.append(e)
        return out
    return route


class Tokens(unittest.TestCase):
    def test_decode_tokens_and_the_last_layer_prompt_row(self):
        tr, reqs = build(lambda key, j, il, b, p: [j % NE, (j + 1) % NE], n_dec=7)
        toks = predict.build_tokens(tr, 6, 2, NE)
        self.assertEqual(sorted(toks), list(range(10)))                 # the warm-up outside every request is not a request
        self.assertTrue(all(len(v) == 7 for v in toks.values()))         # the 1-row step at layer 5 closes the prompt, no token
        self.assertTrue(all(len(t.steps) == 6 and t.n_sel == 12 for v in toks.values() for t in v))
        in_tok = {i for v in toks.values() for t in v for i in t.steps}
        steps0 = [i for i, s in enumerate(tr.steps) if s.req == 0]
        counts, rows = predict.prefill_profile(tr, steps0, in_tok, 6, NE)
        self.assertEqual(rows, [5, 5, 5, 5, 5, 1])
        self.assertEqual(sum(counts[5]), 2)

    def test_top_k_splits_ties_at_the_last_place(self):
        full, tie, w = predict.top_k([5, 5, 3, 3, 3, 0], 3)
        self.assertEqual((full, tie), (0b11, 0b11100))
        self.assertAlmostEqual(w, 1 / 3)
        self.assertAlmostEqual(predict.overlap((full, tie, w), 0b100101), 1 + 1 / 3)
        self.assertEqual(predict.top_k([4, 0, 0, 0], 2, include_zero=False), (0b1, 0, 0.0))   # unused experts never count
        full, tie, w = predict.top_k([4, 0, 0, 0], 2)
        self.assertEqual((full, tie), (0b1, 0b1110))
        self.assertAlmostEqual(w, 1 / 3)

    def test_slots_per_layer_for_swift_at_9400(self):
        g = G.load(DOCS / "swift-1.5-iq2_xs.geometry.json")
        n = predict.slots_per_layer(g, 9400)
        self.assertEqual(set(n), {142})                                 # 4838/34, 1564/11, 426/3, floored
        self.assertEqual(len(n), 48)


class Persistence(unittest.TestCase):
    def test_perfectly_persistent_routing_is_100_percent(self):
        tr, _ = build(lambda key, j, il, b, p: [(3 * key[1] + il) % NE, (3 * key[1] + il + 16) % NE])
        rep = run(tr)
        for w in predict.WINDOWS:
            self.assertEqual(rep["persistence"]["test"][w]["real"], 1.0)
            self.assertEqual(rep["persistence"]["test"][w]["tokens"], 5 * (60 - w))       # a full window only
        self.assertEqual(rep["itinerary"]["test"][1]["persist"], 1.0)
        # identical tokens: the shuffle cannot change anything, and the instrument check must say so
        self.assertEqual(rep["persistence"]["test"][1]["null_mean"], 1.0)
        self.assertFalse(rep["checks"]["null_differs"]["pass"])

    def test_iid_routing_equals_its_null(self):
        tr, _ = build(zipf_route(7), n_dec=120)
        rep = run(tr)
        for w in (1, 4):
            p = rep["persistence"]["test"][w]
            self.assertLess(abs(p["real"] - p["null_mean"]), 0.02, (w, p))
            self.assertGreater(p["real"], 0.05)
        it = rep["itinerary"]["test"][1]
        self.assertLess(abs(it["cooc"] - it["null_mean"]), 0.03, it)       # no layer coupling either
        self.assertGreater(rep["checks"]["prior_above_chance"]["prior"], 2 * 2 / NE)

    def test_sticky_routing_beats_its_null(self):
        base = zipf_route(9)
        rng = random.Random(3)

        def route(key, j, il, below, prev):
            if prev is not None and rng.random() < 0.8:
                return prev
            return base(key, j, il, below, prev)
        rep = run(build(route)[0])
        p = rep["persistence"]["test"][1]
        self.assertGreater(p["real"] - p["null_mean"], 0.3)
        self.assertTrue(rep["checks"]["null_differs"]["pass"])
        self.assertEqual(p["p"], 1 / 21)
        self.assertEqual(rep["rules"]["persistence"]["verdict"], predict.VERDICTS["persistence"][1])


class Itinerary(unittest.TestCase):
    def test_a_deterministic_layer_map_is_predicted_exactly(self):
        f = lambda e: (5 * e + 3) % NE                                  # noqa: E731, a bijection of the experts
        rng = random.Random(11)

        def route(key, j, il, below, prev):
            return rng.sample(range(NE), 2) if il == 0 else [f(e) for e in below]
        rep = run(build(route, n_dec=100)[0])
        for h in predict.HORIZONS:
            it = rep["itinerary"]["test"][h]
            self.assertEqual(it["cooc"], 1.0, h)
            self.assertLess(it["null_mean"], 0.5)
            self.assertEqual(it["p"], 1 / 21)
            self.assertEqual(it["tokens"], 5 * 99)                        # tokens with a previous token
            self.assertEqual(it["n_pred"], 5 * 99 * (6 - h))
        self.assertLess(rep["itinerary"]["test"][1]["prior"], 0.3)
        self.assertEqual(rep["rules"]["itinerary"]["verdict"], predict.VERDICTS["itinerary"][0])
        for c in rep["classes"].values():
            self.assertEqual(c["itinerary"][1]["cooc"], 1.0)

    def test_null_pairs_tokens_not_layers(self):
        # one test token pattern per request: the null still pairs the right layer with another token's same layer
        rep = run(build(lambda key, j, il, b, p: [il, il + 8], n_dec=20)[0])
        it = rep["itinerary"]["test"][1]
        self.assertEqual((it["cooc"], it["null_mean"], it["prior"], it["persist"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(it["p"], 1.0)


class Prefill(unittest.TestCase):
    def test_prefill_set_against_train_frequency(self):
        def route(key, j, il, below, prev):
            if key[0] == "pp4k":
                return [0, 1]
            return [20 + (j + il) % 10, 20 + (j + il + 5) % 10]

        def prompt_route(key, row, il):
            if key[0] == "pp4k":
                return [5, 6] if il == 5 else [row % 10, (row + 5) % 10]
            return default_prompt_route(key, row, il)
        reqs = [("decode", i, 30, 5) for i in range(10)] + [("pp4k", 0, 12, 40)]
        tr, _ = build(route, requests=reqs, prompt_route=prompt_route)
        toks = predict.build_tokens(tr, 6, 2, NE)
        sp = predict.Split.of(tr, predict.DEFAULT_TRAIN, predict.DEFAULT_TEST)
        train = [t for r in sp.train for t in toks.get(r, [])]
        m = predict.prefill_measure(tr, toks, 10, train, [10] * 6, 6, NE)
        self.assertEqual(m["decode_tokens"], 12)
        self.assertEqual(m["prefill_rows"], 5 * 40 + 1)
        self.assertAlmostEqual(m["recall"], 5 / 6)                     # the last layer saw only the prompt's output row
        self.assertEqual(m["baseline"], 0.0)
        self.assertEqual(m["thin_layers"], [5])
        self.assertEqual(m["recall_without_thin"], 1.0)


class Oracle(unittest.TestCase):
    """4 layers, bandwidths 1 and 2 GB/s (1e6 and 2e6 bytes per ms), worked through by hand.
    token 0: T 10 ms, host 2 ms (C 8), bytes 3e6, 1e6, 0, 0
    token 1: T 16 ms, host 4 ms (C 12), bytes 0, 0, 0, 8e6
    token 2: T 6 ms, host 0 (C 6), no bytes; token 3 is the request's last (no interval)."""

    def trace(self):
        starts, hosts = [10_000, 20_000, 36_000, 42_000], [[1000, 1000, 0, 0], [2000, 0, 0, 2000], [0] * 4, [0] * 4]
        nbytes = [[3_000_000, 1_000_000, 0, 0], [0, 0, 0, 8_000_000], [0] * 4, [0] * 4]
        steps, rows = [], []
        for j, t0 in enumerate(starts):
            for il in range(4):
                steps.append(Step("S", t0 + 100 * il, il, 1, 1, array("H", [j]), req=0))
                rows.append({"kind": "S", "t_us": t0 + 100 * il, "layer": il, "n_ids": 1, "us_wait": 0,
                             "us_host": hosts[j][il], "bytes": nbytes[j][il]})
        tr = Trace(steps, requests=[{"phase": "decode", "idx": 1, "t0_us": 0, "t1_us": 10**6}])
        return tr, rows

    def test_by_hand(self):
        tr, rows = self.trace()
        toks = predict.build_tokens(tr, 4, 1, 8)
        o = predict.oracle(tr, rows, [toks[0]], 4, {"slow": 1.0, "pinned": 2.0}, "pinned")
        self.assertEqual((o["tokens"], o["skipped_last"], o["skipped_no_line"]), (3, 1, 0))
        self.assertAlmostEqual(o["measured"]["tok_s_all"], 3000 / 32)
        self.assertAlmostEqual(o["measured"]["tok_s_median"], 100.0)
        want = {("slow", "none"): (38, 12), ("slow", "layer1"): (32, 9), ("slow", "token"): (26, 8),
                ("pinned", "none"): (32, 10), ("pinned", "layer1"): (27, 8), ("pinned", "token"): (26, 8)}
        for (src, hz), (tot, med) in want.items():
            c = o["sources"][src][hz]
            self.assertAlmostEqual(c["tok_s_all"], 3000 / tot, msg=(src, hz))
            self.assertAlmostEqual(c["tok_s_median"], 1000 / med, msg=(src, hz))
        self.assertAlmostEqual(o["sources"]["slow"]["layer1"]["hidden"], 0.5)          # 6 of 12 copy ms hidden
        self.assertAlmostEqual(o["request"]["tok_s_all"], 3000 / 32)
        self.assertAlmostEqual(o["C_ms_median"], 8.0)

    def test_a_token_without_its_s_lines_is_skipped(self):
        tr, rows = self.trace()
        rows = [r for r in rows if not (r["t_us"] == 20_000 + 200)]
        o = predict.oracle(tr, rows, [predict.build_tokens(tr, 4, 1, 8)[0]], 4, {"pinned": 2.0}, "pinned")
        self.assertEqual((o["tokens"], o["skipped_no_line"]), (2, 1))


class Determinism(unittest.TestCase):
    def test_same_trace_passes_and_a_changed_id_is_found(self):
        route = zipf_route(4)
        a, _ = build(route, n_dec=20)
        b, _ = build(zipf_route(4), n_dec=20)
        c = predict.compare_decode_ids(a, b, 6, 2)
        self.assertTrue(c["pass"])
        self.assertEqual((c["compared"], c["identical"]), (200, 200))
        i = next(i for i, s in enumerate(b.steps) if s.req == 3 and s.n_tokens == 1 and s.layer == 2)
        b.steps[i + 6 * 4].ids[0] ^= 1                                   # token 4 of request 3 (the tail row is step i - 3)
        c = predict.compare_decode_ids(a, b, 6, 2)
        self.assertFalse(c["pass"])
        self.assertEqual(c["identical"], 199)
        self.assertEqual(c["first_mismatch"]["request"], "decode:3")
        self.assertEqual(c["first_mismatch"]["layer"], 2)

    def test_a_missing_token_fails_and_nothing_compared_fails(self):
        a, _ = build(zipf_route(4), n_dec=20)
        b, _ = build(zipf_route(4), requests=[("decode", i, 20 if i != 5 else 19, 5) for i in range(10)])
        self.assertFalse(predict.compare_decode_ids(a, b, 6, 2)["pass"])
        e, _ = build(zipf_route(4), requests=[("pp4k", 0, 5, 5)])
        self.assertFalse(predict.compare_decode_ids(a, e, 6, 2)["pass"])   # no shared request: a check that compared nothing


class Cli(unittest.TestCase):
    def test_synth_then_predict_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cli_main(["synth", d, "--n-prompts", "10", "--n-layers", "4", "--n-expert", "32", "--k", "2",
                                       "--decode-tokens", "24", "--stick", "0.5"]), 0)
            tr = idstrace.read_ids(f"{d}/synthetic.ids.bin")
            idstrace.assign_requests(tr, idstrace.read_requests(f"{d}/synthetic.requests.jsonl"))
            geom = G.load(f"{d}/synthetic.geometry.json")
            rows = replay.as_server_rows(tr, replay.run_vram(tr, geom, 300))[0]
            for r in rows:
                r.update(us_wait=1, us_host=5 * r["fills"])                # synth steps are 50 us apart
            idstrace.write_trace_csv(f"{d}/synthetic.trace.csv", rows)
            args = ["predict", f"{d}/synthetic.ids.bin", "--geometry", f"{d}/synthetic.geometry.json", "--budget-mib", "300",
                    "--requests", f"{d}/synthetic.requests.jsonl", "--trace", f"{d}/synthetic.trace.csv",
                    "--check-against", f"{d}/synthetic.ids.bin", "--permutations", "10", "--seed", "3"]
            self.assertEqual(cli_main(args + ["--md", f"{d}/a.md", "--json", f"{d}/a.json"]), 0)
            self.assertEqual(cli_main(args + ["--json", f"{d}/b.json"]), 0)
            a, b = Path(f"{d}/a.json").read_text(), Path(f"{d}/b.json").read_text()
            self.assertEqual(a, b)
            rep = json.loads(a)
            self.assertTrue(rep["checks"]["determinism"]["pass"])
            self.assertGreater(rep["oracle"]["test"]["tokens"], 0)
            text = Path(f"{d}/a.md").read_text(encoding="utf-8")
            for s in ("SYNTHETIC", "Instrument checks", "rule:", "Oracle"):
                self.assertIn(s, text)
            self.assertNotIn(d, text)                                    # file names only, never a local path


if __name__ == "__main__":
    unittest.main()
