"""marian predict: how predictable is a MoE model's expert routing, at which horizon, and what would perfect prediction
be worth? Register E47 (pre-registered 27/10/2026), offline, on a routing-id trace (patch 0004), its request log and its
P0 trace.

Tokens. A decode token is a run of single-token S steps of one request with increasing layers, kept when it covers at
least 80% of the MoE layers (`idstrace.decode_tokens`, the selection replay and analyse_m10.py use); only tokens with
every layer are read. Prefill is every other step of the request: L records, multi-token S records, and the 1-row step
that closes each prompt pass (llama.cpp computes the last layer for the rows that need logits only, so that step has
n_tokens == 1, but it covers one layer and is no token). Steps outside every request (the server's warm-up, anything
between requests) are never read. Train and test are decode-phase requests chosen by prompt (default: decode prompts
0, 2, 4, 6, 8 train, 1, 3, 5, 7, 9 test); `warm` (prompt 0's first tokens again) is never read; `post` requests are a
class of their own.

Measures (decode tokens of the test prompts unless stated; experts compared per layer, sets kept as bit masks):
  1 persistence  share of token t's experts at layer l that layer l selected in the previous w decode tokens of the
                 same request (a token counts at w only with w predecessors). Null: the same after shuffling token order
                 within each request.
  2 itinerary    layer l+h's top-k predicted from layer l's set of the same token by summed co-occurrence counts of the
                 train decode tokens; precision of the top-k (= recall: both sets hold k). Baselines: the train per-layer
                 frequency prior, and the previous token's layer l+h set. All three on tokens with a previous token.
                 Null: each token's predictions scored against another random test token's sets (a random cyclic
                 permutation, so never the token itself; one partner per token for all its layers).
  3 prefill      per layer, the N most-used experts of a pp4k/pp16k request's prefill (N = the layer's group's cache
                 slots / the group's layers, at the budget); recall = share of that request's decode activations in the
                 set. Baseline: the N most frequent experts of the train decode tokens. An unused expert is never in a set.
  4 classes      1 and 2 per test prompt, and for the post requests; their nulls shuffle and pair within the class.
  5 oracle       a model, not a measurement: per decode token from the P0 trace, T = token interval (its layer-0 line to
                 the next token's), H = the sum of us_host, B_l = bytes of layer l, C = T - H; T' = C + sum over layers of
                 max(0, B_l / bandwidth - window) with window C / n_layers (one layer ahead), C (one token ahead), or 0
                 at the pinned bandwidth (staged before the request); tok/s = 1000 / T'. A request's last token has no
                 interval and is left out.
A top-k with ties at its last place counts each tied expert as (places left) / (experts tied): the expected score under
a uniformly random tie-break, so no order among equals is invented. The nulls are seeded (random.Random(f"{seed}/...")):
a report is reproducible byte for byte.
"""
from __future__ import annotations

import random
import statistics as st
from dataclasses import dataclass
from pathlib import Path

from . import geometry as G
from .idstrace import Trace, decode_tokens, prompt_key

WINDOWS = (1, 2, 4, 8, 16)
HORIZONS = (1, 2, 3, 4)
DEFAULT_TRAIN = ("decode:0", "decode:2", "decode:4", "decode:6", "decode:8")
DEFAULT_TEST = ("decode:1", "decode:3", "decode:5", "decode:7", "decode:9")
# docs/m1.0/prompts.json, key "decode", in order
DEFAULT_LABELS = ("story", "python code", "arithmetic word problem", "nurse explanation", "Polish letter",
                  "German instructions", "C++ code", "TCP/UDP explainer", "SQL", "French Revolution summary")
# GB/s = 1e9 bytes/s, as docs/m1.0/copy-probe.txt prints them (1,510,400 bytes in 3,650.7 us = 0.41 GB/s)
BANDWIDTHS = {"mmap": 0.41, "pageable": 6.6, "pinned": 10.3}
REFERENCE_TOK_S = 14.5          # the 6 GiB tier's warm decode, all-tokens rate (register E44)

# E47's pre-registered rules (the test register, 27/10/2026 07:05); in VERDICTS, index 0 = the rule's condition met
PERSIST_POINTS = 5.0
ITIN_POINTS, ITIN_P = 10.0, 0.05
PREFILL_POINTS = 10.0
ORACLE_GAIN = 0.10
VERDICTS = {
    "persistence": ("drop token-horizon id prefetch from M2", "token-horizon id prefetch stays a candidate"),
    "itinerary": ("build the hidden-state logging patch",
                  "layer-ahead prediction from ids is dead; the hidden-state route needs its own case before any patch"),
    "prefill": ("'warm the cache from the prompt's own prefill' becomes an M1.3/M2 candidate (promoted, not adopted)",
                "not promoted"),
    "oracle": ("prefetch is not the decode lever on 32 GB: the RAM tier's coverage is",
               "the rule's condition is not met; the register names no decision for this branch"),
}
RULES = {
    "persistence": "if w = 1 persistence on test is within 5 points of its null mean, drop token-horizon id prefetch "
                   "from M2; otherwise it stays a candidate",
    "itinerary": "build the hidden-state logging patch only if the co-occurrence predictor at h = 1 beats the better of "
                 "the two baselines by at least 10 points AND the permutation null at p < 0.05",
    "prefill": "if recall beats the frequency baseline by at least 10 points in both requests, warming the cache from "
               "the prompt's own prefill becomes an M1.3/M2 candidate (n = 2 requests: can promote, never adopt)",
    "oracle": "if perfect layer-horizon prediction from the pinned source gains under 10% over the measured warm tier "
              "decode (14.5 tok/s, E44), prefetch is not the decode lever on 32 GB: the RAM tier's coverage is",
}
EPS = 1e-12


# ----------------------------------------------------------------------------------------------------------- tokens

@dataclass
class Tok:
    req: int
    pos: int                # index among the request's decode tokens
    steps: list[int]        # trace step of each layer, -1 where the layer is missing
    sets: list[int]         # per layer, the selected experts as a bit mask (0 where missing)
    exps: list[tuple]       # per layer, the selected experts, sorted
    big: int                # every layer's mask at bit offset layer * stride

    @property
    def n_sel(self) -> int:
        return self.big.bit_count()

    @property
    def complete(self) -> bool:
        return all(self.sets)


def build_tokens(trace: Trace, n_layers: int, k: int, stride: int) -> dict[int, list[Tok]]:
    """Decode tokens per request index, in decode order (every request that has a step, possibly with none)."""
    by_req: dict[int, list[int]] = {}
    for i, s in enumerate(trace.steps):
        if s.req >= 0:
            by_req.setdefault(s.req, []).append(i)
    out: dict[int, list[Tok]] = {}
    for r in sorted(by_req):
        toks = []
        for pos, t in enumerate(decode_tokens(trace.steps, by_req[r], k, n_layers)):
            steps, sets, exps = [-1] * n_layers, [0] * n_layers, [()] * n_layers
            for i in t:
                s = trace.steps[i]
                if not 0 <= s.layer < n_layers:
                    continue
                e_ok = sorted({e for e in s.ids if e < stride})     # 0xffff: an id that did not fit
                steps[s.layer], exps[s.layer] = i, tuple(e_ok)
                sets[s.layer] = sum(1 << e for e in e_ok)
            big = 0
            for il in range(n_layers - 1, -1, -1):
                big = (big << stride) | sets[il]
            toks.append(Tok(r, pos, steps, sets, exps, big))
        out[r] = toks
    return out


def layer_counts(tokens: list[Tok], n_layers: int, n_expert: int) -> list[list[int]]:
    c = [[0] * n_expert for _ in range(n_layers)]
    for t in tokens:
        for il in range(n_layers):
            row = c[il]
            for e in t.exps[il]:
                row[e] += 1
    return c


def top_k(score: list[int], n: int, include_zero: bool = True) -> tuple[int, int, float]:
    """The top n of score as (mask of the experts strictly inside, mask of the experts tied at the n-th place, weight of
    each tied one = places left / experts tied). With include_zero False an expert scoring 0 is never in."""
    if n <= 0:
        return 0, 0, 0.0
    if n >= len(score):
        return sum(1 << e for e, s in enumerate(score) if include_zero or s > 0), 0, 0.0
    kth = sorted(score, reverse=True)[n - 1]
    full = tie = n_full = n_tie = 0
    for e, s in enumerate(score):
        if s > kth:
            full |= 1 << e
            n_full += 1
        elif s == kth:
            tie |= 1 << e
            n_tie += 1
    if kth <= 0 and not include_zero:
        return full, 0, 0.0
    return full, tie, (n - n_full) / n_tie


def overlap(pred: tuple[int, int, float], mask: int) -> float:
    full, tie, w = pred
    o = (full & mask).bit_count()
    return o + (tie & mask).bit_count() * w if tie else o


def _p(null: list[float], real: float) -> float:
    """One-sided permutation p-value: (1 + nulls at or above real) / (1 + nulls)."""
    return (1 + sum(x >= real - EPS for x in null)) / (1 + len(null))


def _summ(null: list[float], real: float) -> dict:
    null = [x for x in null if x == x]                      # a permutation with nothing to score gives nan
    if real != real:
        null = []
    return {"real": real, "null_mean": st.mean(null) if null else None, "null_sd": st.pstdev(null) if null else None,
            "null_min": min(null) if null else None, "null_max": max(null) if null else None,
            "p": _p(null, real) if null else None}


# ---------------------------------------------------------------------------------------------- 1 token persistence

def persistence(groups: list[list[Tok]], ws=WINDOWS) -> dict[int, list[int]]:
    """{w: [hits, selections, tokens]} over groups (each one request's tokens, in decode order)."""
    out = {w: [0, 0, 0] for w in ws}
    ws = sorted(ws)
    for toks in groups:
        for i, t in enumerate(toks):
            u = j = 0
            for w in ws:
                if i < w:
                    break
                while j < w:
                    j += 1
                    u |= toks[i - j].big
                o = out[w]
                o[0] += (t.big & u).bit_count()
                o[1] += t.n_sel
                o[2] += 1
    return out


def persistence_shuffles(groups: list[list[Tok]], ws, n_perm: int, rng: random.Random) -> list[list[dict]]:
    """[shuffle][group] -> persistence of that group with its token order shuffled."""
    res = []
    for _ in range(n_perm):
        per = []
        for toks in groups:
            order = list(toks)
            rng.shuffle(order)
            per.append(persistence([order], ws))
        res.append(per)
    return res


def _share(acc: dict, w: int) -> float:
    return acc[w][0] / acc[w][1] if acc[w][1] else float("nan")


def _sum_groups(per: list[dict], ws) -> dict:
    return {w: [sum(p[w][i] for p in per) for i in range(3)] for w in ws}


def persistence_report(groups: list[list[Tok]], shuffles: list[list[dict]], ws=WINDOWS) -> dict:
    real = persistence(groups, ws)
    out = {}
    for w in ws:
        null = [_share(_sum_groups(per, ws), w) for per in shuffles]
        out[w] = {"tokens": real[w][2], **_summ(null, _share(real, w))}
    return out


# ------------------------------------------------------------------------------------------------- 2 layer itinerary

def cooc_predictions(train: list[Tok], evals: list[Tok], n_layers: int, n_expert: int, k: int, hs=HORIZONS,
                     progress=None) -> list[dict]:
    """[eval token] -> {h: [masks of the predicted top-k of every layer l+h packed like Tok.big, [(layer, tie mask,
    weight)] of the predictions with ties at the k-th place, number of predictions]}."""
    out = [{h: [0, [], 0] for h in hs} for _ in evals]
    for h in hs:
        if progress:
            progress(f"itinerary h={h}")
        for il in range(n_layers - h):
            m = [[0] * n_expert for _ in range(n_expert)]
            for t in train:
                tgt = t.exps[il + h]
                for a in t.exps[il]:
                    row = m[a]
                    for b in tgt:
                        row[b] += 1
            shift = (il + h) * n_expert
            for j, t in enumerate(evals):
                src = t.exps[il]
                if not src:
                    continue
                score = list(map(sum, zip(*[m[a] for a in src])))
                full, tie, w = top_k(score, k)
                o = out[j][h]
                o[0] |= full << shift
                if tie:
                    o[1].append((il + h, tie, w))
                o[2] += 1
    return out


def _itin_hits(pred: list, target: Tok) -> float:
    big, ties, _ = pred
    o = (big & target.big).bit_count()
    for il, tie, w in ties:
        o += (tie & target.sets[il]).bit_count() * w
    return o


def itinerary_null(evals: list[int], preds: list[dict], toks: list[Tok], hs, k: int, n_perm: int,
                   rng: random.Random) -> dict[int, list[float]]:
    """Precision of each eval token's predictions scored against another eval token (a random cyclic permutation)."""
    n = len(evals)
    res = {h: [] for h in hs}
    if n < 2:
        return res
    den = {h: k * sum(preds[j][h][2] for j in evals) for h in hs}
    for _ in range(n_perm):
        order = list(evals)
        for i in range(n - 1, 0, -1):                       # Sattolo: one cycle through every token, no fixed point
            j = rng.randrange(i)
            order[i], order[j] = order[j], order[i]
        for h in hs:
            hit = sum(_itin_hits(preds[a][h], toks[b]) for a, b in zip(evals, order))
            res[h].append(hit / den[h] if den[h] else float("nan"))
    return res


def itinerary(evals: list[int], prev: dict[int, int], preds: list[dict], prior: list[tuple], toks: list[Tok],
              n_layers: int, n_expert: int, k: int, hs=HORIZONS) -> dict:
    """Precision of co-occurrence, prior and persistence on the eval tokens (indices into toks; prev[j] = the token
    before j in its request)."""
    out = {}
    for h in hs:
        keep = ((1 << ((n_layers - h) * n_expert)) - 1) << (h * n_expert)       # layers h .. n_layers-1
        prior_pred = [0, [], 0]
        for il in range(h, n_layers):
            full, tie, w = prior[il]
            prior_pred[0] |= full << (il * n_expert)
            if tie:
                prior_pred[1].append((il, tie, w))
        hit_c = hit_f = hit_p = 0.0
        n_pred = 0
        for j in evals:
            t, pr = toks[j], preds[j][h]
            hit_c += _itin_hits(pr, t)
            hit_f += _itin_hits(prior_pred, t)
            hit_p += (toks[prev[j]].big & t.big & keep).bit_count()
            n_pred += pr[2]
        den = k * n_pred
        out[h] = {"tokens": len(evals), "n_pred": n_pred, "cooc": hit_c / den if den else float("nan"),
                  "prior": hit_f / den if den else float("nan"), "persist": hit_p / den if den else float("nan")}
    return out


# ---------------------------------------------------------------------------------------- 3 prefill-to-decode recall

def slots_per_layer(geom: G.Geometry, budget_mib: float, n_ubatch: int = 512, alignment: int = G.CUDA_ALIGNMENT) -> list[int]:
    """A layer's share of its group's cache slots at the budget (slots are shared by a group's layers): n_slots // layers."""
    n = {}
    for gs in G.cache_slots(geom, budget_mib, n_ubatch, alignment):
        for il in gs.group.layers:
            n[il] = gs.n_slots // len(gs.group.layers)
    return [n[il] for il in sorted(geom.layers)]


def prefill_profile(trace: Trace, step_idx: list[int], in_tokens: set[int], n_layers: int,
                    n_expert: int) -> tuple[list[list[int]], list[int]]:
    """Expert use per layer of a request's prefill (every row of every step that is not in a decode token), and the
    rows routed per layer."""
    counts = [[0] * n_expert for _ in range(n_layers)]
    rows = [0] * n_layers
    for i in step_idx:
        if i in in_tokens:
            continue
        s = trace.steps[i]
        if not 0 <= s.layer < n_layers:
            continue
        rows[s.layer] += s.n_tokens
        c = counts[s.layer]
        for e in s.ids:
            if e < n_expert:
                c[e] += 1
    return counts, rows


def prefill_measure(trace: Trace, toks_by_req: dict[int, list[Tok]], req: int, train: list[Tok], n_per_layer: list[int],
                    n_layers: int, n_expert: int, train_counts: list[list[int]] | None = None) -> dict:
    idx = [i for i, s in enumerate(trace.steps) if s.req == req]
    toks = [t for t in toks_by_req.get(req, []) if t.complete]
    in_tok = {i for t in toks_by_req.get(req, []) for i in t.steps if i >= 0}
    counts, rows = prefill_profile(trace, idx, in_tok, n_layers, n_expert)
    tc = train_counts if train_counts is not None else layer_counts(train, n_layers, n_expert)
    pre = [top_k(counts[il], n_per_layer[il], include_zero=False) for il in range(n_layers)]
    base = [top_k(tc[il], n_per_layer[il], include_zero=False) for il in range(n_layers)]
    med = st.median(rows) if rows else 0
    thin = [il for il in range(n_layers) if rows[il] < 0.5 * med]

    def recall(sets, layers):
        hit = sum(overlap(sets[il], t.sets[il]) for t in toks for il in layers)
        tot = sum(t.sets[il].bit_count() for t in toks for il in layers)
        return hit / tot if tot else float("nan")
    every = range(n_layers)
    wide = [il for il in every if il not in thin]
    return {"decode_tokens": len(toks), "prefill_rows": sum(rows), "prefill_steps": sum(1 for i in idx if i not in in_tok),
            "rows_per_layer": rows, "recall": recall(pre, every), "baseline": recall(base, every), "thin_layers": thin,
            "recall_without_thin": recall(pre, wide), "baseline_without_thin": recall(base, wide),
            "set_sizes": [overlap(p, (1 << n_expert) - 1) for p in pre]}


# --------------------------------------------------------------------------------------------------------- 5 oracle

def oracle(trace: Trace, csv_rows: list[dict], groups: list[list[Tok]], n_layers: int, bandwidths: dict[str, float],
           pinned: str) -> dict:
    """The oracle model on the given requests' decode tokens (each group one request's complete tokens, in order)."""
    lines = {(r["t_us"], r["layer"]): r for r in csv_rows if r.get("kind") == "S"}
    per = []                                    # (T ms, H ms, [bytes per layer])
    skipped_last = skipped_no_line = skipped_gap = skipped_c = 0
    for toks in groups:
        skipped_last += bool(toks)
        for a, b in zip(toks, toks[1:]):
            if min(b.steps) != max(a.steps) + 1:        # a dropped token in between: the interval would span two
                skipped_gap += 1
                continue
            ls = [lines.get((trace.steps[i].t_us, trace.steps[i].layer)) for i in a.steps]
            if any(x is None or x.get("us_host") is None or x.get("bytes") is None for x in ls):
                skipped_no_line += 1
                continue
            t_ms = (trace.steps[b.steps[0]].t_us - trace.steps[a.steps[0]].t_us) / 1e3
            h_ms = sum(x["us_host"] for x in ls) / 1e3
            if t_ms - h_ms <= 0:
                skipped_c += 1
                continue
            per.append((t_ms, h_ms, [x["bytes"] for x in ls]))
    out = {"tokens": len(per), "skipped_last": skipped_last, "skipped_no_line": skipped_no_line,
           "skipped_gap": skipped_gap, "skipped_nonpositive_c": skipped_c}
    if not per:
        return out

    def agg(ts):
        return {"tok_s_all": 1e3 * len(ts) / sum(ts), "tok_s_median": 1e3 / st.median(ts)}
    cs = [t - h for t, h, _ in per]
    out["measured"] = agg([t for t, _, _ in per])
    out.update(T_ms_median=st.median(t for t, _, _ in per), H_ms_median=st.median(h for _, h, _ in per),
               C_ms_median=st.median(cs), C_ms_mean=st.mean(cs), MiB_median=st.median(sum(b) for _, _, b in per) / G.MiB,
               host_GB_s=sum(sum(b) for _, _, b in per) / (sum(h for _, h, _ in per) * 1e6))

    def model(rate_gb_s: float, horizon: str) -> dict:
        rate = rate_gb_s * 1e6                  # bytes per ms
        ts, copy, hidden = [], 0.0, 0.0
        for (_, _, bs), c in zip(per, cs):
            win = c / n_layers if horizon == "layer1" else c if horizon == "token" else 0.0
            ex = 0.0
            for b in bs:
                cp = b / rate
                copy += cp
                hidden += min(cp, win)
                ex += max(0.0, cp - win)
            ts.append(c + ex)
        return {**agg(ts), "hidden": hidden / copy if copy else float("nan"), "copy_ms_mean": copy / len(per)}
    out["sources"] = {src: {hz: model(bw, hz) for hz in ("none", "layer1", "token")} for src, bw in bandwidths.items()}
    out["request"] = model(bandwidths[pinned], "none")      # staged to pinned before the request, nothing hidden
    return out


# ---------------------------------------------------------------------------------------------- determinism check

def _req_key(r: dict) -> str:
    return f"{r.get('phase', 'all')}:{r.get('idx', 0)}"


def compare_decode_ids(a: Trace, b: Trace, n_layers: int, k: int) -> dict:
    """Decode ids of two runs of the same prompts, token for token, layer for layer, in the router's order, per shared
    request (phase:idx). Passes only if something was compared and every token and token count agrees."""
    def per_req(tr: Trace) -> dict:
        by: dict[int, list[int]] = {}
        for i, s in enumerate(tr.steps):
            if s.req >= 0:
                by.setdefault(s.req, []).append(i)
        out = {}
        for r, idx in by.items():
            toks = decode_tokens(tr.steps, idx, k, n_layers)
            in_tok = {i for t in toks for i in t}
            pre = [(tr.steps[i].kind, tr.steps[i].layer, tr.steps[i].n_tokens, tuple(tr.steps[i].ids)) for i in idx if i not in in_tok]
            out[_req_key(tr.requests[r])] = ([[(tr.steps[i].layer, tuple(tr.steps[i].ids)) for i in t] for t in toks], pre)
        return out
    ka, kb = per_req(a), per_req(b)
    shared = [key for key in ka if key in kb]
    compared = identical = pre_same = 0
    counts, first = [], None
    for key in shared:
        (ta, pa), (tb, pb) = ka[key], kb[key]
        pre_same += pa == pb
        if len(ta) != len(tb):
            counts.append({"request": key, "tokens_a": len(ta), "tokens_b": len(tb)})
        for j, (x, y) in enumerate(zip(ta, tb)):
            compared += 1
            if x == y:
                identical += 1
            elif first is None:
                lay = next((p[0] for p, q in zip(x, y) if p != q), None)
                first = {"request": key, "token": j, "layer": lay}
    return {"requests": shared, "compared": compared, "identical": identical, "count_mismatch": counts,
            "first_mismatch": first, "prefill_identical": pre_same,
            "pass": compared > 0 and identical == compared and not counts}


# ------------------------------------------------------------------------------------------------------------- run

def norm_keys(keys) -> list[str]:
    """'3' or 'decode:3' -> 'decode:3'."""
    return [k if ":" in k else f"decode:{int(k)}" for k in keys]


@dataclass
class Split:
    train: list[int]
    test: list[int]
    post: list[int]
    pp: list[int]
    train_keys: list[str]
    test_keys: list[str]

    @staticmethod
    def of(trace: Trace, train=DEFAULT_TRAIN, test=DEFAULT_TEST) -> "Split":
        trk, tek = norm_keys(train), norm_keys(test)
        if set(trk) & set(tek):
            raise ValueError(f"a prompt on both sides of the split: {sorted(set(trk) & set(tek))}")
        ph = lambda r: r.get("phase", "all")                                            # noqa: E731
        rq = list(enumerate(trace.requests))
        return Split([i for i, r in rq if ph(r) == "decode" and prompt_key(r) in trk],
                     [i for i, r in rq if ph(r) == "decode" and prompt_key(r) in tek],
                     [i for i, r in rq if ph(r) == "post"],
                     [i for i, r in rq if ph(r).startswith("pp")], trk, tek)


def _label(key: str, labels) -> str:
    ph, _, idx = key.partition(":")
    return labels[int(idx)] if ph == "decode" and idx.isdigit() and int(idx) < len(labels) else ""


def run_all(trace: Trace, geom: G.Geometry, budget_mib: float, *, train=DEFAULT_TRAIN, test=DEFAULT_TEST,
            labels=DEFAULT_LABELS, csv_rows: list[dict] | None = None, other: Trace | None = None, other_name: str = "",
            permutations: int = 100, seed: int = 47, bandwidths: dict | None = None, pinned: str = "pinned",
            reference_tok_s: float = REFERENCE_TOK_S, n_ubatch: int = 512, alignment: int = G.CUDA_ALIGNMENT,
            names: dict | None = None, progress=None) -> dict:
    say = progress or (lambda *_: None)
    bandwidths = dict(BANDWIDTHS if bandwidths is None else bandwidths)
    n_layers, k = len(geom.layers), geom.n_expert_used
    if sorted(geom.layers) != list(range(n_layers)):
        raise ValueError("marian predict reads models whose every layer is a MoE layer (layers 0..n-1)")
    ne = max(lay.n_expert for lay in geom.layers.values())
    hs = tuple(h for h in HORIZONS if h < n_layers)
    sp = Split.of(trace, train, test)
    by_req = build_tokens(trace, n_layers, k, ne)
    full = {r: [t for t in ts if t.complete] for r, ts in by_req.items()}
    rep: dict = {"synthetic": bool(geom.source.get("synthetic")), "model": geom.model, "inputs": names or {},
                 "budget_mib": budget_mib, "n_layers": n_layers, "n_expert": ne, "k": k, "permutations": permutations,
                 "seed": seed, "split": {"train": sp.train_keys, "test": sp.test_keys}, "horizons": list(hs),
                 "steps": len(trace.steps), "steps_outside_requests": sum(s.req < 0 for s in trace.steps),
                 "incomplete_tokens": sum(len(ts) - len(full[r]) for r, ts in by_req.items())}
    side = {**{r: "train" for r in sp.train}, **{r: "test" for r in sp.test}, **{r: "post" for r in sp.post},
            **{r: "pp" for r in sp.pp}}
    dec = {prompt_key(trace.requests[r]): r for r in sp.train + sp.test}
    reqs = []
    for r, q in enumerate(trace.requests):
        n_steps = sum(1 for s in trace.steps if s.req == r)
        tok_steps = sum(len(t.steps) for t in by_req.get(r, []))
        row = {"req": r, "request": _req_key(q), "prompt": prompt_key(q), "label": _label(prompt_key(q), labels),
               "side": side.get(r, "-"), "decode_tokens": len(full.get(r, [])), "prefill_steps": n_steps - tok_steps}
        if q.get("phase") == "post" and prompt_key(q) in dec:            # the same prompt's routing, token for token?
            mine, theirs = full.get(r, []), full.get(dec[prompt_key(q)], [])
            row["same_ids_as"] = (prompt_key(q) if mine and len(mine) <= len(theirs)
                                  and all(x.big == y.big for x, y in zip(mine, theirs)) else None)
        reqs.append(row)
    rep["requests"] = reqs

    train_toks = [t for r in sp.train for t in full.get(r, [])]
    test_groups = [full.get(r, []) for r in sp.test]
    post_groups = [full.get(r, []) for r in sp.post]
    classes = [(f"{prompt_key(trace.requests[r])} {_label(prompt_key(trace.requests[r]), labels)}".strip(), [r])
               for r in sp.test]
    if any(post_groups):
        classes.append(("post-pp16K", list(sp.post)))

    # 1 persistence: one set of shuffles per request, shared by the test aggregate and the class rows
    say("persistence")
    rng = random.Random(f"{seed}/persistence")
    shuf = {}
    for r in sp.test + sp.post:
        shuf[r] = persistence_shuffles([full.get(r, [])], WINDOWS, permutations, rng)
    def pers(rs):                                                                     # noqa: E306
        return persistence_report([full.get(r, []) for r in rs], [[shuf[r][i][0] for r in rs] for i in range(permutations)])
    rep["persistence"] = {"test": pers(sp.test)}

    # 2 itinerary
    toks: list[Tok] = []
    prev: dict[int, int] = {}
    ev_of: dict[int, list[int]] = {}
    for r in sp.test + sp.post:
        ev_of[r] = []
        for i, t in enumerate(full.get(r, [])):
            toks.append(t)
            if i:
                prev[len(toks) - 1] = len(toks) - 2
                ev_of[r].append(len(toks) - 1)
    preds = cooc_predictions(train_toks, toks, n_layers, ne, k, hs, say)
    tcounts = layer_counts(train_toks, n_layers, ne)
    prior = [top_k(tcounts[il], k) for il in range(n_layers)]
    ev_test = [j for r in sp.test for j in ev_of[r]]
    say("itinerary nulls")
    it = itinerary(ev_test, prev, preds, prior, toks, n_layers, ne, k, hs)
    null = itinerary_null(ev_test, preds, toks, hs, k, permutations, random.Random(f"{seed}/itinerary/test"))
    for h in hs:
        s = _summ(null[h], it[h]["cooc"])
        it[h].update({x: s[x] for x in ("null_mean", "null_sd", "null_min", "null_max", "p")})
    rep["itinerary"] = {"test": it}

    # 4 classes
    rep["classes"] = {}
    for name, rs in classes:
        ev = [j for r in rs for j in ev_of[r]]
        if not ev:
            continue
        c_it = itinerary(ev, prev, preds, prior, toks, n_layers, ne, k, hs)
        c_null = itinerary_null(ev, preds, toks, hs, k, permutations, random.Random(f"{seed}/itinerary/{name}"))
        for h in hs:
            s = _summ(c_null[h], c_it[h]["cooc"])
            c_it[h].update(null_mean=s["null_mean"], p=s["p"])
        rep["classes"][name] = {"requests": [_req_key(trace.requests[r]) for r in rs],
                                "persistence": pers(rs), "itinerary": c_it}

    # 3 prefill to decode
    rep["prefill"] = {"n_per_layer": None, "requests": {}}
    if sp.pp:
        npl = slots_per_layer(geom, budget_mib, n_ubatch, alignment)
        rep["prefill"]["n_per_layer"] = sorted(set(npl)) if len(set(npl)) > 1 else npl[0]
        for r in sp.pp:
            say(f"prefill {_req_key(trace.requests[r])}")
            rep["prefill"]["requests"][_req_key(trace.requests[r])] = prefill_measure(
                trace, by_req, r, train_toks, npl, n_layers, ne, tcounts)

    # 5 oracle
    if csv_rows is not None:
        say("oracle")
        rep["oracle"] = {"test": oracle(trace, csv_rows, test_groups, n_layers, bandwidths, pinned),
                         "all_decode": oracle(trace, csv_rows, [full.get(r, []) for r in sorted(sp.train + sp.test)],
                                              n_layers, bandwidths, pinned),
                         "bandwidths_gb_s": bandwidths, "pinned": pinned, "reference_tok_s": reference_tok_s}

    # instrument checks
    p1 = rep["persistence"]["test"][1]
    chance = k / ne
    rep["checks"] = {
        "determinism": ({"other": other_name, **compare_decode_ids(other, trace, n_layers, k)} if other is not None
                        else {"pass": None, "note": "not run (no --check-against)"}),
        "null_differs": {"real": p1["real"], "null_mean": p1["null_mean"], "null_min": p1["null_min"],
                         "null_max": p1["null_max"],
                         "pass": (p1["real"] > p1["null_max"] + EPS or p1["real"] < p1["null_min"] - EPS)
                         if p1["null_max"] is not None else None},
        "prior_above_chance": {"prior": it[1]["prior"], "chance": chance, "pass": it[1]["prior"] >= 2 * chance}}
    rep["rules"] = rules(rep)
    return rep


def rules(rep: dict) -> dict:
    out = {}
    p1 = rep["persistence"]["test"][1]
    if p1["null_mean"] is not None:
        d = 100 * (p1["real"] - p1["null_mean"])
        met = abs(d) <= PERSIST_POINTS
        out["persistence"] = {"number": f"w = 1: {100 * p1['real']:.2f}% against a null mean of {100 * p1['null_mean']:.2f}%, "
                                        f"{d:+.2f} points", "met": met, "verdict": VERDICTS["persistence"][0 if met else 1]}
    i1 = rep["itinerary"]["test"][1]
    if i1["p"] is not None:
        best = max(i1["prior"], i1["persist"])
        d = 100 * (i1["cooc"] - best)
        met = d >= ITIN_POINTS and i1["p"] < ITIN_P
        out["itinerary"] = {"number": f"h = 1: co-occurrence {100 * i1['cooc']:.2f}%, better baseline "
                                      f"{'prior' if i1['prior'] >= i1['persist'] else 'persistence'} {100 * best:.2f}%, "
                                      f"{d:+.2f} points; null mean {100 * i1['null_mean']:.2f}%, p = {i1['p']:.4f}",
                            "met": met, "verdict": VERDICTS["itinerary"][0 if met else 1]}
    pre = rep["prefill"]["requests"]
    if pre:
        ds = {key: 100 * (m["recall"] - m["baseline"]) for key, m in pre.items()}
        met = len(ds) == 2 and all(d >= PREFILL_POINTS for d in ds.values())
        out["prefill"] = {"number": "; ".join(f"{key}: recall {100 * pre[key]['recall']:.2f}% vs baseline "
                                              f"{100 * pre[key]['baseline']:.2f}%, {d:+.2f} points" for key, d in ds.items()),
                          "met": met, "verdict": VERDICTS["prefill"][0 if met else 1]
                          + ("" if len(ds) == 2 else f" (the rule needs both requests; {len(ds)} found)")}
    o = (rep.get("oracle") or {}).get("test") or {}
    if o.get("sources") and rep["oracle"]["pinned"] in o["sources"]:
        x = o["sources"][rep["oracle"]["pinned"]]["layer1"]["tok_s_all"]
        ref = rep["oracle"]["reference_tok_s"]
        gain = x / ref - 1
        met = gain < ORACLE_GAIN
        out["oracle"] = {"number": f"pinned, one layer ahead: {x:.2f} tok/s (all tokens) against {ref:g}: {100 * gain:+.1f}%",
                         "met": met, "verdict": VERDICTS["oracle"][0 if met else 1]}
    for key in out:
        out[key]["rule"] = RULES[key]
    return out


# ---------------------------------------------------------------------------------------------------------- report

def _pct(x, d=2):
    return "-" if x is None or x != x else f"{100 * x:.{d}f}%"


def _pts(x, d=2):
    return "-" if x is None or x != x else f"{100 * x:+.{d}f}"


def _ok(x):
    return "not run" if x is None else ("PASS" if x else "FAIL")


def to_markdown(rep: dict) -> str:
    L = []
    if rep.get("synthetic"):
        L += ["**SYNTHETIC trace and geometry: every number below is a property of the generator, not of any model.**", ""]
    nm = rep.get("inputs") or {}
    L += [f"`marian predict` (register E47) on {', '.join(f'`{v}`' for v in nm.values() if v) or 'a trace'}: "
          f"{rep['model'] or 'model'}, {rep['n_layers']} MoE layers, {rep['k']} of {rep['n_expert']} experts per token, "
          f"slots at `--moe-cache-mib {rep['budget_mib']:g}`; {rep['permutations']} permutations, seed {rep['seed']}.",
          f"Split by prompt: train {', '.join(rep['split']['train'])}; test {', '.join(rep['split']['test'])}.", "",
          "**Tokens.** A decode token = the single-token S steps of one forward pass of a request, every layer present "
          "(the replay's and analyse_m10.py's selection). Prefill = L records, multi-token S records, and the 1-row step "
          "at the last layer that closes each prompt pass (llama.cpp computes the last layer for the logits row only). "
          f"Steps outside every request (server warm-up): {rep['steps_outside_requests']} of {rep['steps']}, never read. "
          f"Tokens dropped for a missing layer: {rep['incomplete_tokens']}.", "",
          "| request | prompt | label | side | decode tokens | prefill steps | note |", "|---|---|---|---|---|---|---|"]
    for r in rep["requests"]:
        note = ""
        if "same_ids_as" in r:
            note = (f"routing identical to {r['same_ids_as']}'s first {r['decode_tokens']} tokens" if r["same_ids_as"]
                    else "routing differs from its prompt's decode request")
        L.append(f"| {r['request']} | {r['prompt']} | {r['label']} | {r['side']} | {r['decode_tokens']} | {r['prefill_steps']} | {note} |")

    c = rep["checks"]
    det = c["determinism"]
    det_txt = det.get("note") or (f"`{det.get('other')}`: {det['identical']} of {det['compared']} decode tokens identical "
                                  f"(ids in router order, every layer) over {len(det['requests'])} shared requests; "
                                  f"token-count mismatches {len(det['count_mismatch'])}; first mismatch "
                                  f"{det['first_mismatch'] or 'none'}; prefill steps identical in {det['prefill_identical']} "
                                  f"of {len(det['requests'])} requests")
    nd = c["null_differs"]
    L += ["", "## Instrument checks", "", "| check | result | verdict |", "|---|---|---|",
          f"| determinism: the other run's decode ids equal this one's, token for token | {det_txt} | {_ok(det['pass'])} |",
          f"| the null differs from real (persistence, w = 1, test) | real {_pct(nd['real'])}, null mean {_pct(nd['null_mean'])}, "
          f"range {_pct(nd['null_min'])} to {_pct(nd['null_max'])} (declared: real outside the null's range) | {_ok(nd['pass'])} |",
          f"| the frequency prior is well above chance (h = 1, test) | prior {_pct(c['prior_above_chance']['prior'])}, chance "
          f"{_pct(c['prior_above_chance']['chance'])} (declared: at least twice chance) | {_ok(c['prior_above_chance']['pass'])} |"]

    L += ["", "## 1. Token persistence (test prompts)", "",
          "Share of a token's experts at a layer that the same layer selected in the previous w decode tokens of its request "
          "(tokens with w predecessors). Null: token order shuffled within each request.", "",
          "| w | tokens | real | null mean (sd) | null range | real - null (points) | p |", "|---|---|---|---|---|---|---|"]
    for w, p in rep["persistence"]["test"].items():
        L.append(f"| {w} | {p['tokens']} | {_pct(p['real'])} | {_pct(p['null_mean'])} ({_pct(p['null_sd'])}) | "
                 f"{_pct(p['null_min'])} to {_pct(p['null_max'])} | {_pts(p['real'] - p['null_mean'] if p['null_mean'] is not None else None)} | "
                 f"{'-' if p['p'] is None else format(p['p'], '.4f')} |")

    L += ["", "## 2. Layer itinerary (test prompts)", "",
          "Layer l+h's top-k predicted from layer l's set of the same token by co-occurrence counts of the train decode "
          "tokens; precision of the top-k (= recall). Baselines: the train per-layer frequency prior; the previous token's "
          "layer l+h set. Tokens with a previous token. Null: predictions scored against another random test token.", "",
          "| h | tokens | predictions | co-occurrence | prior | persistence | margin over the better baseline (points) | "
          "null mean (sd) | null max | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h, x in rep["itinerary"]["test"].items():
        L.append(f"| {h} | {x['tokens']} | {x['n_pred']} | {_pct(x['cooc'])} | {_pct(x['prior'])} | {_pct(x['persist'])} | "
                 f"{_pts(x['cooc'] - max(x['prior'], x['persist']))} | {_pct(x['null_mean'])} ({_pct(x['null_sd'])}) | "
                 f"{_pct(x['null_max'])} | {'-' if x['p'] is None else format(x['p'], '.4f')} |")

    pre = rep["prefill"]
    L += ["", "## 3. Prefill-to-decode working set", ""]
    if not pre["requests"]:
        L.append("No pp4k/pp16k request in this trace.")
    else:
        L += [f"N per layer = the layer's group's slots / its layers at the budget: {pre['n_per_layer']}. Recall = share of the "
              "request's decode activations whose expert is in the prefill's N most-used at that layer; baseline = the N most "
              "frequent of the train decode tokens. Thin layers = layers whose prefill routed under half the median rows.", "",
              "| request | prefill rows | decode tokens | recall | baseline | difference (points) | thin layers | "
              "recall / baseline without thin layers |", "|---|---|---|---|---|---|---|---|"]
        for key, m in pre["requests"].items():
            L.append(f"| {key} | {m['prefill_rows']} | {m['decode_tokens']} | {_pct(m['recall'])} | {_pct(m['baseline'])} | "
                     f"{_pts(m['recall'] - m['baseline'])} | {m['thin_layers'] or 'none'} | "
                     f"{_pct(m['recall_without_thin'])} / {_pct(m['baseline_without_thin'])} |")

    L += ["", "## 4. Per class", "", "Persistence (real; null mean at w = 1, shuffled within the class's requests):", "",
          "| class | tokens (w = 1) | " + " | ".join(f"w = {w}" for w in WINDOWS) + " | null w = 1 | real - null w = 1 (points) |",
          "|---|---|" + "---|" * len(WINDOWS) + "---|---|"]
    for name, cl in rep["classes"].items():
        p = cl["persistence"]
        L.append(f"| {name} | {p[1]['tokens']} | " + " | ".join(_pct(p[w]['real']) for w in WINDOWS) +
                 f" | {_pct(p[1]['null_mean'])} | {_pts(p[1]['real'] - p[1]['null_mean'] if p[1]['null_mean'] is not None else None)} |")
    L += ["", "Itinerary (precision; null = predictions paired with another token of the same class):", "",
          "| class | tokens | h = 1 co-occurrence | prior | persistence | null | p | " + " | ".join(f"h = {h}" for h in rep["horizons"][1:]) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(rep["horizons"][1:])]
    for name, cl in rep["classes"].items():
        x = cl["itinerary"]
        L.append(f"| {name} | {x[1]['tokens']} | {_pct(x[1]['cooc'])} | {_pct(x[1]['prior'])} | {_pct(x[1]['persist'])} | "
                 f"{_pct(x[1]['null_mean'])} | {'-' if x[1]['p'] is None else format(x[1]['p'], '.4f')} | "
                 + " | ".join(_pct(x[h]['cooc']) for h in rep["horizons"][1:]) + " |")

    L += ["", "## 5. Oracle bound (a model, not a measurement)", ""]
    orc = rep.get("oracle")
    if not orc:
        L.append("No P0 trace given (`--trace`).")
    else:
        for which, title in (("test", "test prompts"), ("all_decode", "all decode prompts")):
            o = orc[which]
            if not o.get("sources"):
                L += [f"{title}: no token with an interval and all its S lines.", ""]
                continue
            m = o["measured"]
            L += [f"**{title}**: {o['tokens']} tokens (left out: {o['skipped_last']} last tokens of a request, "
                  f"{o['skipped_no_line']} without all their S lines, {o['skipped_gap']} not adjacent, {o['skipped_nonpositive_c']} "
                  f"with C <= 0). Measured {m['tok_s_all']:.2f} tok/s all tokens ({m['tok_s_median']:.2f} median); per token "
                  f"median T {o['T_ms_median']:.1f} ms, H {o['H_ms_median']:.1f} ms, C {o['C_ms_median']:.1f} ms "
                  f"(mean {o['C_ms_mean']:.1f}), {o['MiB_median']:.1f} MiB of misses; host rate {o['host_GB_s']:.2f} GB/s.", "",
                  "| source (GB/s) | no prediction (calibration) | one layer ahead | one token ahead | staged before the request (pinned) |",
                  "|---|---|---|---|---|"]
            for src, cells in o["sources"].items():
                f = lambda x: f"{x['tok_s_all']:.2f} ({x['tok_s_median']:.2f}), hidden {_pct(x['hidden'], 1)}"    # noqa: E731
                L.append(f"| {src} {orc['bandwidths_gb_s'][src]:g} | {cells['none']['tok_s_all']:.2f} ({cells['none']['tok_s_median']:.2f}) | "
                         f"{f(cells['layer1'])} | {f(cells['token'])} | {o['request']['tok_s_all']:.2f} ({o['request']['tok_s_median']:.2f}) |")
            L += ["", "tok/s all tokens (median); hidden = share of the copy time that fits in the window. Windows: C / "
                  f"{rep['n_layers']} per layer one layer ahead, C per layer one token ahead (as written: each layer's copy may "
                  "use the whole previous token), 0 when staged (the copy runs at the pinned rate). Reference: the measured "
                  f"warm tier decode, {orc['reference_tok_s']:g} tok/s (E44).", ""]

    L += ["## Pre-registered rules", ""]
    for key in ("persistence", "itinerary", "prefill", "oracle"):
        r = rep["rules"].get(key)
        if r:
            L.append(f"- **{key}** rule: {r['rule']} | number: {r['number']} | verdict per the rule: **{r['verdict']}**")
        else:
            L.append(f"- **{key}** rule: {RULES[key]} | number: not computed (no data for it) | verdict: none")
    return "\n".join(L) + "\n"


def file_names(**paths) -> dict:
    """File names only for the report, never a local path."""
    return {k: Path(v).name for k, v in paths.items() if v}
