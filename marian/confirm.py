"""Live confirmation (M1.3, design section 7 step 5): does the plan run as fast as its profile row says, and does it
beat the runner-up head to head?

A profile row is one measurement taken minutes or days before serving, on a desktop that may have changed since. Two
checks keep the plan honest:

1. **The live check** (every `marian serve`, passive): the front proxy reads the decode timings llama-server returns
   with every answer. After the first N answers of at least 16 generated tokens it compares their all-tokens decode
   rate with the row's profiled rate; more than X % below (default 15 %) and it says "below profile" in one line.
   The profile measures decode at 75 % of the context, and most live requests decode at a shallower depth, which is
   not slower, so the gap is not an artefact of depth. But serve's request loop ran 10-16 % under calibrate's figure
   on a clean machine (register E53), so a gap of that size is expected and is not flagged as slow. The outcome is
   recorded in the row as `live_check`.
2. **Keep-if-faster** (`marian serve --confirm`): before serving, the plan (A) and the runner-up (B: the next best
   row under the same filters) run K rounds of interleaved windows, A B then B A (ABBA, so neither arm always goes
   first; design 9.3), each window a fresh server start, a short warm-up and the same short probe. The runner-up
   replaces the plan only if it is faster in at least 3 of 4 rounds AND its median speed ratio is above 1.02: Strata
   v0.2.5 keeps a faster path only when a live A/B over 64 windows confirms it (theirs), and two numbers from a profile
   are not that. Server restarts make our windows costly, so K is small (4 by default); the 2 % bar is Strata's.
   The outcome is recorded in both rows as `confirmed` (date, rounds, rates, ratio, outcome), and the planner then
   prefers the winner (plan.py).

**Exact-class preference:** an exact-class plan is only ever A/B-tested against an exact-class runner-up, unless the
user allowed non-exact rows (`--stock` or `--allow-non-exact`): a faster different answer is not a confirmation.
"""
from __future__ import annotations

import datetime as dt
import http.client
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from . import plan
from .profile import FILLER

MIN_TOKENS = 16           # answers shorter than this say little about decode speed (the README's 8-token rule, doubled)


@dataclass
class ConfirmOptions:
    rounds: int = 4             # K: rounds of A/B windows, alternating AB and BA
    slow: float = 0.15          # X: a live rate more than this fraction below the profiled rate is reported
    min_gain: float = 0.02      # the runner-up must be faster by more than this median ratio (Strata's 2 %)...
    min_share: float = 0.75     # ...in at least this share of the rounds
    fill_tokens: int = 2048     # each window's prompt, capped at 75 % of the row's context
    n_predict: int = 64
    warmup: int = 16


def today() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


SERVE_GAP = "serve's request loop measured 10-16% under calibrate's decode figure on a clean machine (register E53)"


def live_verdict(live_tps: float, profiled_tps: float | None, slow: float, served: bool = False) -> dict:
    """Is a live decode rate more than `slow` below the profiled one? `served`: the rate comes from serve's own
    request loop (the passive check), which runs below calibrate's figure even on a clean machine (SERVE_GAP), so the
    flag is `below_profile`, not `slow`: a gap of that size is expected, not a fault. The `--confirm` windows use the
    calibrate-style probe and keep the `slow` flag."""
    flag = "below_profile" if served else "slow"
    if not profiled_tps:
        return {"live_tps": round(live_tps, 2), "profiled_tps": profiled_tps, flag: False,
                "message": "the row has no profiled decode rate to compare with"}
    ratio = live_tps / profiled_tps
    below = ratio < 1 - slow
    out = {"live_tps": round(live_tps, 2), "profiled_tps": profiled_tps, "ratio": round(ratio, 3), flag: below}
    pct = abs(100 * (1 - ratio))
    if below and served:
        out["message"] = (f"live decode {live_tps:.1f} tok/s is {pct:.0f}% below profile ({profiled_tps:.1f} tok/s; "
                          f"threshold {100 * slow:.0f}%). {SERVE_GAP}, so a gap near that size is expected; a much "
                          "larger one means the desktop changed: `marian serve --confirm` A/B-tests the plan against "
                          "the runner-up, `marian calibrate` measures again")
    elif below:
        out["message"] = (f"live decode {live_tps:.1f} tok/s is {pct:.0f}% below the profiled "
                          f"{profiled_tps:.1f} tok/s for this configuration (threshold {100 * slow:.0f}%): the desktop may "
                          "have changed since profiling; `marian serve --confirm` A/B-tests it against the runner-up, "
                          "`marian calibrate` measures again")
    elif ratio >= 1:
        out["message"] = (f"live decode {live_tps:.1f} tok/s against {profiled_tps:.1f} profiled "
                          f"({pct:.0f}% above): at or above the profile")
    else:
        out["message"] = (f"live decode {live_tps:.1f} tok/s against {profiled_tps:.1f} profiled "
                          f"({pct:.0f}% below): within the {100 * slow:.0f}% threshold")
    return out


class LiveCheck:
    """The passive check: fed the `timings` of each answer by the front proxy; reports once, after n useful answers."""

    def __init__(self, profiled_tps: float | None, n: int = 8, slow: float = 0.15, on_done=None):
        self.profiled, self.n, self.slow, self.on_done = profiled_tps, n, slow, on_done
        self.samples: list[tuple[int, float]] = []
        self.result: dict | None = None

    def reset(self, profiled_tps: float | None) -> None:
        """A supervised restart changed the configuration: compare the next answers with the new row instead."""
        self.profiled, self.samples, self.result = profiled_tps, [], None

    def add(self, timings: dict | None) -> dict | None:
        if self.result is not None or not timings:
            return None
        n, ms = timings.get("predicted_n"), timings.get("predicted_ms")
        if not n or not ms or n < MIN_TOKENS:
            return None
        self.samples.append((int(n), float(ms)))
        if len(self.samples) < self.n:
            return None
        rate = sum(x for x, _ in self.samples) / (sum(y for _, y in self.samples) / 1000)
        self.result = dict(live_verdict(rate, self.profiled, self.slow, served=True), answers=len(self.samples),
                           date=today())
        if self.on_done:
            self.on_done(self.result)
        return self.result


def rounds_order(k: int) -> list[str]:
    """AB, BA, AB, BA, ...: k rounds, so neither arm always runs first (page-cache state carries between windows)."""
    return [("AB" if i % 2 == 0 else "BA") for i in range(k)]


def decide(a: list[float | None], b: list[float | None], opt: ConfirmOptions) -> dict:
    """Keep the plan unless the runner-up wins at least min_share of the rounds AND the median ratio clears min_gain.
    A round where a window failed counts for the arm that ran; a plan window that failed counts against the plan."""
    ratios, wins = [], 0
    for x, y in zip(a, b):
        if x and y:
            r = y / x
            ratios.append(r)
            wins += r > 1
        elif y and not x:
            wins += 1
            ratios.append(math.inf)
        else:
            ratios.append(0.0)                    # the runner-up failed (or both did): the plan keeps that round
    k = len(ratios)
    med = statistics.median(ratios) if ratios else 0.0
    need = math.ceil(opt.min_share * k)
    replace = k > 0 and wins >= need and med > 1 + opt.min_gain
    return {"rounds": k, "runner_up_wins": wins, "wins_needed": need, "median_ratio": round(med, 4) if math.isfinite(med)
            else "inf", "min_gain": opt.min_gain, "outcome": "replaced" if replace else "kept"}


def probe_window(engine, ctx: int, opt: ConfirmOptions) -> dict:
    """The same short probe for both arms: filler up to min(fill_tokens, 75 % of the context), n_predict tokens."""
    target = max(64, min(opt.fill_tokens, int(ctx * 0.75) - opt.n_predict))
    unit = engine.tokenize(FILLER)
    t = engine.complete(FILLER * max(1, target // max(unit, 1)), opt.n_predict)
    return {"tg_tps": round(t.get("predicted_per_second", 0), 2), "pp_tps": round(t.get("prompt_per_second", 0), 1),
            "prompt_n": t.get("prompt_n")}


def window(engine, model: Path, row: dict, log: Path, opt: ConfirmOptions) -> dict:
    """One window: a fresh server start with the row's configuration, a warm-up (the first decode after a load pays
    the expert cache's cold misses), the probe, stop."""
    c = plan.to_config(row)
    ok, why = engine.start(model, c, log)
    try:
        if not ok:
            return {"ok": False, "why": why}
        engine.complete("Hello.", opt.warmup)
        return dict(probe_window(engine, c.ctx, opt), ok=True)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return {"ok": False, "why": f"{type(e).__name__}"}
    finally:
        engine.stop()


def describe(row: dict) -> str:
    c = plan.to_config(row)
    parts = [f"ctx {c.ctx}", f"kv {c.cache_type}"]
    parts.append(f"-ncmoe {c.n_cpu_moe}" if c.n_cpu_moe is not None else f"-ngl {c.n_gpu_layers}")
    if c.moe_cache_mib:
        parts.append(f"cache {c.moe_cache_mib} MiB")
    if c.miss_source:
        parts.append(f"misses from {c.miss_source}" + (f" {c.tier_mib} MiB" if c.tier_mib else ""))
    if c.prefill_source:
        parts.append(f"prefill from {c.prefill_source}")
    if c.ubatch:
        parts.append(f"ubatch {c.ubatch}")
    if c.threads:
        parts.append(f"{c.threads} threads")
    return ", ".join(parts)


def ab(engine, model: Path, a_row: dict, b_row: dict, log: Path, opt: ConfirmOptions, say=print) -> dict:
    """K rounds of interleaved windows; returns the decision with every rate."""
    rates = {"A": [], "B": []}
    for i, order in enumerate(rounds_order(opt.rounds), 1):
        got = {}
        for arm in order:
            w = window(engine, model, a_row if arm == "A" else b_row, log, opt)
            got[arm] = w.get("tg_tps") if w.get("ok") else None
            say(f"confirm round {i}/{opt.rounds}, {'plan' if arm == 'A' else 'runner-up'}: "
                + (f"{w['tg_tps']} tok/s" if w.get("ok") else f"failed ({w.get('why')})"))
            time.sleep(2)
        rates["A"].append(got["A"])
        rates["B"].append(got["B"])
    d = decide(rates["A"], rates["B"], opt)
    d.update(plan_tps=rates["A"], runner_up_tps=rates["B"], order=rounds_order(opt.rounds))
    good = [x for x in rates["A"] if x]
    if good:
        d["live_check"] = live_verdict(statistics.median(good), a_row.get("tg_tps"), opt.slow)
    return d


def records(a_row: dict, b_row: dict | None, d: dict | None, opt: ConfirmOptions) -> list[tuple[dict, dict]]:
    """(row, `confirmed` record) for each row the A/B touched."""
    base = {"date": today(), "method": f"{opt.rounds} rounds of interleaved windows (ABBA), fresh server per window, "
                                       f"{opt.n_predict}-token probe after a {opt.warmup}-token warm-up"}
    if b_row is None or d is None:
        return [(a_row, dict(base, outcome="kept", note="no runner-up under the same filters"))]
    shared = {k: d[k] for k in ("rounds", "runner_up_wins", "wins_needed", "median_ratio", "min_gain", "plan_tps",
                                "runner_up_tps", "order")}
    a_rec = dict(base, **shared, role="plan", against=plan.key(b_row), outcome=d["outcome"])
    if d["outcome"] == "replaced":
        a_rec["by"] = plan.key(b_row)
    b_rec = dict(base, **shared, role="runner-up", against=plan.key(a_row),
                 outcome="won" if d["outcome"] == "replaced" else "lost")
    return [(a_row, a_rec), (b_row, b_rec)]


def write_back(profile_dir: Path, updates: list[tuple[dict, str, dict]]) -> list[str]:
    """Write fields into the rows of their profile files (row `_file` and `_index`, from plan.rows_of), atomically:
    a crash mid-write must not leave a profile half-written. Returns the files changed."""
    by_file: dict[str, list[tuple[int, str, dict]]] = {}
    for row, field, value in updates:
        if row.get("_file") is None or row.get("_index") is None:
            continue
        by_file.setdefault(row["_file"], []).append((row["_index"], field, value))
    changed = []
    for name, items in by_file.items():
        p = profile_dir / name
        doc = json.loads(p.read_text(encoding="utf-8"))
        for i, field, value in items:
            if 0 <= i < len(doc.get("rows", [])):
                doc["rows"][i][field] = value
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        changed.append(name)
    return changed
