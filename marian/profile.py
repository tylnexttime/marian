"""Profile a model on this machine: speed and memory per (context size, KV type, GPU/CPU split).

For each context size and KV type, the GPU/CPU split is walked from the most conservative (most work on the CPU) to
the most aggressive. A point is an OVERFLOW when VRAM is within a margin of full, or when speed collapses against the
best point so far (on Windows the driver pages GPU memory into system RAM instead of failing, so the symptom is a
cliff, measured 21/10/2026: 42.7 -> 5.5 tok/s). The walk stops at the first overflow.

The probe fills about 75 % of the context with neutral text, then generates a fixed number of tokens, so decode is
measured at depth, not on an empty cache.

Each row also records the GPU memory already in use just before its server started (`vram_baseline_mib`: the desktop,
other programs) and the model's own share (`vram_model_mib` = peak minus that baseline), so a planner can re-check
the fit against the VRAM free at launch. The baseline is read once per row; other programs' use can still drift
while the row runs.

**Cache arms (M1.3, calibration v1).** When the engine is a Marian build (its --help lists the P3 flags) and the
model is MoE, each (context, KV) cell also gets, after the walk:

1. **A VRAM cliff finder** over `--moe-cache-mib`, every expert layer kept in host memory (`-ncmoe` = all layers), as
   every measured cache arm ran. The bottom of the bracket is the exactness floor (patch 0001: below it the server
   refuses to start; that refusal is correct and is recorded as `refused`, never treated as a crash, and the minimum
   it names replaces a floor computed from the GGUF if they differ). The top is the budget that caches every expert,
   capped by the VRAM left beside the model's other weights, KV and buffers (the share of the walk's all-experts-on-CPU
   point) and the margin. The floor is probed first (the speed reference), then the top; if the top overflows, the
   bracket is bisected down to `resolution_mib`, never probing above a known overflow. An overflow is the walk's rule.
2. **The miss sources at the best budget**: the staging ring, then a small ladder of pinned RAM tier sizes, ascending,
   inside a RAM safety envelope: never pin more than (RAM available now - floor), floor 6 GiB by default. A tier is
   pinned, so the desktop cannot page it: with about 25 GB of 32 GB in desktop use, plain `-ncmoe` beat the cache
   (15.0 against 13.4 tok/s, 24/10/2026), and the tier costs prefill 4-5x against stock. So it is measured here, per
   machine, never assumed. When every expert fits in the cache there are no misses, and the sources are skipped.
3. **Drift rungs** (debt row 20, after register E63): the best budget fits with nothing to spare beyond the plan's
   margin, so in E50 a 263 MiB drop in free VRAM before launch left the planner only the floor row (6.7 against 31.4
   tok/s). Below the best budget, every `drift_step_mib` (512) for `drift_rungs` (2) steps, never at or below the
   floor, the cache row and every miss-source configuration that survived at the best budget are measured again
   (`arm` rung), under the same RAM envelope. A drift smaller than one step then costs one rung, not the ladder.

Cache arms load memory-mapped (`-lm mmap`): the ring and tier read a miss by its offset in the model's mapped files
(without a mapping the fork logs "--moe-miss-source is not used"), and every measured cache arm ran mapped. The walk
keeps M0's `-lm none`.

Every row records `exact_class`: every expert computed on the GPU (full offload, or any cache arm) or not (stock
`-ncmoe` > 0, which changed answers in A22). `plan --exact-only` filters on it. Rows also record the RAM available
before the server started and its low point, the commit headroom low point, and the server's peak working set and
commit where the OS gives them.

**CPU thread sweep (M1.3).** A row that computes on the CPU (stock `-ncmoe` > 0, or a dense model with layers left
on the CPU) is bound by memory bandwidth there, and llama.cpp's default thread count is not always its best: Maya #433
saw 15 -> 41 tok/s with fewer threads (theirs, a user report). So, unless `--threads` fixes the count, each (context,
KV) cell's fastest CPU-computing walk point is measured again at a small ladder of thread counts (physical cores,
cores - 2, cores / 2, 8, 4; never above the logical processors) and every result is kept as a row (`arm` threads), so
the planner simply picks the fastest. Rows that compute every expert on the GPU (full offload, cache arms) skip it:
their threads only feed the GPU.

**P3 prefill rows (M1.3, after register E51).** On a build with patch 0009 (`--moe-prefill-source`), every ring and
tier row is measured twice: with the default prefill source (`mmap`, upstream's large-batch path) and with `p3` (the
large-batch misses from the same ring and tier). Both are exact; which is faster depends on the machine: E51 measured
p3 at +76% for a 16K prompt on a machine under memory pressure, and at less than half the memory map's speed on a
model that fits in the page cache. So the pair is judged here, on prefill and decode together (`prefill_pair`): a
source that is faster on one by more than the noise band (10%, the README's rule for few-round results) and not
slower on the other beyond it keeps its row, and its twin is marked `dominated`, which the planner skips. A tie on
both keeps the default (`mmap`). A split (one faster on prefill, the other on decode) keeps both rows, and the
planner's objective decides (`--objective pp` for read-heavy work). `p3` is never measured, or launched, beside
memory-mapped misses: the fork refuses that pairing.

**The declared ubatch (M1.3, after register E54).** A ubatch is part of the exactness reference, so it is declared by
the deployment (`marian calibrate --ubatch 2048`), never chosen here. Every row of a profile is measured at the
declared ubatch (`-ub N -b N`; the default, 512, passes no flag), the profile records it (schema 3), and a profile at
a ubatch other than 512 gets its own file (`...-ub2048.json`), so it sits beside the default one rather than replacing
it. `ubatch_report` puts the best decode and the best prefill of each cell side by side, so the trade a declared
ubatch makes is visible; this code encodes no expectation of what that trade is.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import geometry, gguf, machine
from .engines import DEFAULT_UBATCH, P3_MISS_SOURCES, RING_DEFAULT_MIB, Config, LlamaCpp

PROFILE_SCHEMA = 3        # 3: the declared ubatch and P3 prefill rows; 2: cache arms (M1.3); 1: M0
PAIR_NOISE = 0.10         # prefill_pair: a speed difference within 10% is a tie (one probe per arm)

FILLER = ("The workshop opens at eight. Apprentices sign in, check the job board and collect their tools. "
          "Safety briefings come first, then practical tasks, then a short review before lunch. ")

# what the fork's log says, matched on the message text (the prefix and timestamp vary by log format)
FLOOR_RE = re.compile(r"below the exact-mode floor of (\d+) MiB[^\r\n]*?use --moe-cache-mib (\d+) or more")
NOT_USED_RE = re.compile(r"--moe-miss-source is not used, misses are uploaded from the host tensors: ([^\r\n]*)")
PINNED_RE = re.compile(r"Marian P3 \(experimental\) \w+:[^\r\n]*?([\d.]+) MiB pinned(?: \(\+([\d.]+) MiB PAGEABLE)?")
PATHLIKE = re.compile(r"[A-Za-z]:[\\/][^\s'\"]*|/(?:home|Users|mnt|media)/[^\s'\"]*")


@dataclass
class CacheOptions:
    """The cache arms of a profile (Marian builds and MoE models only)."""
    enabled: bool = True
    resolution_mib: int = 256            # the cliff finder stops when its bracket is this narrow (design 6.1)
    max_cache_mib: int | None = None     # never offer the cache more than this (VRAM kept for other programs)
    ram_floor_mib: int = 6144            # the RAM envelope: never pin more than (RAM available - this)
    tier_ladder_mib: tuple[int, ...] = (1024, 2048, 4096, 6144, 8192, 12288, 16384)
    tier_steps: int = 3                  # measure at most this many tier sizes: the largest the envelope allows
    ring: bool = True
    n_ubatch: int = 512                  # the server's default; the exactness floor depends on it (run() sets it to
                                         # the declared ubatch)
    prefill_p3: bool = True              # measure each ring and tier row with --moe-prefill-source p3 too (0009 builds)
    drift_step_mib: int = 512            # drift rungs (debt row 20, register E63): budgets this far apart below the best
    drift_rungs: int = 2                 # how many rungs below the best budget; 0 measures the best budget only


@dataclass
class ThreadOptions:
    """The CPU thread sweep of a profile (rows that compute on the CPU only)."""
    enabled: bool = True
    ladder: tuple[int, ...] | None = None    # None: thread_ladder() of this machine


def thread_ladder(physical: int | None, logical: int | None) -> list[int]:
    """Thread counts to try, largest first: physical cores, cores - 2, cores / 2, 8 and 4, each between 1 and the
    logical processors, without repeats. Without a physical count, the logical count stands in."""
    cores = physical or logical or 0
    if cores <= 0:
        return []
    top = logical or cores
    out: list[int] = []
    for t in (cores, cores - 2, cores // 2, 8, 4):
        if 1 <= t <= top and t not in out:
            out.append(t)
    return sorted(out, reverse=True)


def thread_sweep(model: Path, engine, s: dict, ctx: int, kv: str, cell: list[dict], log: Path, total_vram: int,
                 n_predict: int, margin_mib: int, ladder: list[int], probe_prompt: int | None = None
                 ) -> tuple[list[dict], dict]:
    """The ladder at the fastest CPU-computing point of one walk cell; returns its rows and a summary."""
    summary: dict = {"ctx": ctx, "kv": kv}
    cpu = [r for r in cell if r["status"] == "ok" and not r.get("exact_class")]
    if not cpu:
        summary["skipped"] = "no measured point computes on the CPU (every expert or layer is on the GPU)"
        return [], summary
    best = max(cpu, key=lambda r: r["tg_tps"])
    c0 = Config.from_row(best)
    knob = ("n_cpu_moe", c0.n_cpu_moe) if c0.n_cpu_moe is not None else ("n_gpu_layers", c0.n_gpu_layers)
    summary.update(split={knob[0]: knob[1]}, default_tg_tps=best["tg_tps"], default_threads=best.get("threads"))
    rows = []
    for t in ladder:
        c = Config.from_row(best)
        c.threads = t
        row = dict(c.to_row(), arm="threads", exact_class=c.exact_class(s["block_count"]))
        measure(engine, model, c, row, log, n_predict, probe_prompt)
        judge(row, total_vram, margin_mib, 0.0, 0.6)       # VRAM only: fewer threads may rightly be slower
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(5)
    summary["results"] = [{"threads": r["threads"], "status": r["status"], "tg_tps": r.get("tg_tps"),
                           "pp_tps": r.get("pp_tps")} for r in rows]
    ok = [r for r in rows if r["status"] == "ok"]
    top = max(ok, key=lambda r: r["tg_tps"], default=None)
    if top is not None and top["tg_tps"] > best["tg_tps"]:
        summary.update(best_threads=top["threads"], best_tg_tps=top["tg_tps"])
    else:
        summary.update(best_threads=best.get("threads"), best_tg_tps=best["tg_tps"],
                       note="the engine's default thread count was fastest")
    return rows, summary


def sha256_head(path: Path, n: int = 64 * 2**20) -> str:
    """Hash of the first 64 MiB plus the size: a cheap, stable model identity (full hashes are in the registry)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


class PeakSampler(threading.Thread):
    def __init__(self, every: float = 0.5):
        super().__init__(daemon=True)
        self.every, self.vram, self.ram_low, self._halt = every, 0, 10**9, threading.Event()  # not _stop: Thread has one
        self.commit_low: int | None = None

    def run(self):
        while not self._halt.is_set():
            used = machine.gpu_memory_used_mib()
            if used:
                self.vram = max(self.vram, used[0])
            self.ram_low = min(self.ram_low, machine.ram_mib()[1])
            commit = machine.commit_mib()
            if commit:
                self.commit_low = commit[1] if self.commit_low is None else min(self.commit_low, commit[1])
            time.sleep(self.every)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def probe_target(ctx: int, n_predict: int, probe_prompt: int | None = None) -> int:
    """The probe's prompt length in tokens. By default 75% of the context minus the answer (M0's choice); with
    probe_prompt, the deployment's declared prompt length (dev-debt row 18: a 4K calibration otherwise judged P3 prefill
    on about 3K tokens, though its gain was measured at 16K). The prompt and the answer must fit the context."""
    if probe_prompt is None:
        return int(ctx * 0.75) - n_predict
    if probe_prompt < 1 or probe_prompt + n_predict > ctx:
        raise ValueError(f"a {probe_prompt}-token probe prompt plus {n_predict} answer tokens does not fit a "
                         f"{ctx}-token context")
    return probe_prompt


def probe(engine: LlamaCpp, ctx: int, n_predict: int, probe_prompt: int | None = None) -> dict:
    target = probe_target(ctx, n_predict, probe_prompt)
    unit = engine.tokenize(FILLER)
    text = FILLER * max(1, target // max(unit, 1))
    t = engine.complete(text, n_predict)
    return {"prompt_tokens": t.get("prompt_n"), "pp_tps": round(t.get("prompt_per_second", 0), 1),
            "tg_tps": round(t.get("predicted_per_second", 0), 2)}


def splits(summary: dict, step: int) -> list[tuple[str, int]]:
    """The walk, conservative to aggressive. It always ends on the all-on-GPU point (the exact one for a MoE model),
    even when the step does not divide the layer count (M0 skipped it: 30 layers in steps of 4 ended at 2)."""
    n = summary["block_count"] or 0
    if summary["moe"]:
        ks = list(range(n, -1, -step))                  # all experts on CPU -> none
        return [("n_cpu_moe", k) for k in ks + ([0] if ks[-1] != 0 else [])]
    ks = list(range(max(0, n // 4), n + 2, step))       # a quarter on GPU -> all (n + 1 includes the output layer)
    return [("n_gpu_layers", k) for k in ks + ([n + 1] if ks[-1] != n + 1 else [])]


def read_server_log(text: str) -> dict:
    """What a server's log says about a cache arm: a refusal below the exactness floor (and the minimum it names), a
    miss source the fork did not use (and why), and the memory the source pinned (or failed to pin)."""
    out: dict = {}
    m = FLOOR_RE.search(text)
    if m:
        out["refusal"] = f"below the exact-mode floor of {m.group(1)} MiB; the server asks for {m.group(2)} MiB or more"
        out["floor_min_mib"] = int(m.group(2))
    m = NOT_USED_RE.search(text)
    if m:
        out["source_not_used"] = PATHLIKE.sub("<path>", m.group(1).strip())[:200]
    m = PINNED_RE.search(text)
    if m:
        out["pinned_mib"] = round(float(m.group(1)))
        if m.group(2):
            out["pageable_mib"] = round(float(m.group(2)))      # pinning failed for this much; it runs pageable
    return out


def _log_since(log: Path, offset: int) -> str:
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def measure(engine, model: Path, c: Config, row: dict, log: Path, n_predict: int, probe_prompt: int | None = None) -> dict:
    """Start one server for configuration c, probe it, stop it. Records speed, VRAM (peak, the desktop's baseline, the
    model's share), RAM (available before, lowest), commit headroom, the server's peak working set and commit, and what
    its log says; a refusal below the exactness floor becomes status `refused`, a miss source the fork did not use
    becomes `not_applied` (the row would not be what it claims)."""
    before = machine.gpu_memory_used_mib()
    baseline = before[0] if before else None
    row["ram_free_before_mib"] = machine.ram_mib()[1]
    offset = log.stat().st_size if log.exists() else 0
    sampler = PeakSampler()
    sampler.start()
    ok, why = engine.start(model, c, log)
    try:
        if not ok:
            row.update(status="start_failed", note=why)
        else:
            try:
                row.update(probe(engine, c.ctx, n_predict, probe_prompt))
                row["status"] = "ok"
            except (OSError, ValueError, http.client.HTTPException) as e:   # a timeout or a dropped server is a row
                row.update(status="probe_failed", note=PATHLIKE.sub("<path>", f"{type(e).__name__}: {e}")[:200])
            proc = getattr(engine, "proc", None)
            if row["status"] == "ok" and proc is not None and proc.poll() is None:
                row.update(machine.process_memory_mib(proc.pid))
    finally:
        engine.stop()
        sampler.stop()
    row.update(vram_peak_mib=sampler.vram, vram_baseline_mib=baseline,
               vram_model_mib=(max(sampler.vram - baseline, 0) if baseline is not None and sampler.vram else None),
               ram_avail_low_mib=sampler.ram_low)
    if sampler.commit_low is not None:
        row["commit_avail_low_mib"] = sampler.commit_low
    said = read_server_log(_log_since(log, offset))
    if "refusal" in said:
        row.update(status="refused", note=said["refusal"], floor_min_mib=said["floor_min_mib"])
    if "source_not_used" in said and c.miss_source in ("ring", "tier"):
        row.update(status="not_applied", note=f"the fork did not use --moe-miss-source {c.miss_source}: {said['source_not_used']}")
    for k in ("pinned_mib", "pageable_mib"):
        if k in said:
            row[k] = said[k]
    return row


def judge(row: dict, total_vram: int, margin_mib: int, best: float, collapse: float) -> None:
    """M0's overflow rule, applied to an ok row in place: VRAM within the margin of full, or decode below collapse x
    the best point so far (Windows pages GPU memory into system RAM instead of failing)."""
    if row["status"] != "ok":
        return
    near_full = total_vram and row["vram_peak_mib"] >= total_vram - margin_mib
    cliff = best and row["tg_tps"] < best * collapse
    if near_full or cliff:
        row["status"] = "overflow"
        row["note"] = "VRAM near full" if near_full else f"speed collapse vs best {best}"


def find_cliff(probe_budget, lo: int, hi: int, resolution: int = 256, collapse: float = 0.6) -> int | None:
    """The largest --moe-cache-mib in [lo, hi] that runs without overflow, or None if even lo does not.

    probe_budget(mib) starts a server at that budget and returns its row (status ok | overflow | refused | ...; the
    caller has already applied the VRAM-near-full rule). Here a row that is ok but decodes below collapse x the best
    good cache point is marked an overflow: the paging cliff. lo first (it is the speed reference), then hi; if hi is
    bad, bisect between the highest good and the lowest bad budget, in 64 MiB steps, until they are `resolution`
    apart. Every probe lies between a known good and a known bad budget, so none is ever above a known overflow. Speed
    is assumed to rise with the budget until the cliff (a larger cache only adds hits)."""
    best = 0.0

    def good(row: dict) -> bool:
        nonlocal best
        if row["status"] == "ok" and best and row["tg_tps"] < best * collapse:
            row["status"], row["note"] = "overflow", f"speed collapse vs best cache point {best}"
        if row["status"] != "ok":
            return False
        best = max(best, row["tg_tps"])
        return True

    first = probe_budget(lo)
    if first["status"] == "refused" and (first.get("floor_min_mib") or 0) > lo:
        lo = first["floor_min_mib"]                     # the server's own floor beats the one computed from the GGUF
        if lo > hi:
            return None
        first = probe_budget(lo)
    if not good(first):
        return None
    if hi <= lo:
        return lo
    if good(probe_budget(hi)):
        return hi
    ok_b, bad_b = lo, hi
    while bad_b - ok_b > resolution:
        mid = (ok_b + bad_b) // 2 // 64 * 64
        if mid <= ok_b:
            break
        if good(probe_budget(mid)):
            ok_b = mid
        else:
            bad_b = mid
    return ok_b


def tier_ladder(ram_available_mib: int, floor_mib: int, useful_mib: int | None, ladder: tuple[int, ...], steps: int,
                ring_mib: int = RING_DEFAULT_MIB) -> list[int]:
    """Tier sizes to measure, ascending: the largest `steps` ladder sizes inside the RAM envelope, where a tier plus
    its staging ring never pins more than (RAM available - floor). A tier larger than the experts the VRAM cache does
    not hold (useful_mib) buys nothing, so that caps it too."""
    cap = ram_available_mib - floor_mib - ring_mib
    if useful_mib is not None:
        cap = min(cap, useful_mib)
    fits = [t for t in sorted(ladder) if 0 < t <= cap]
    return fits[-steps:] if steps > 0 else []


def model_geometry(model: Path) -> geometry.Geometry | None:
    """Expert geometry from the GGUF headers (every shard of a split model), or None if it cannot be read: then the
    floor comes from the server's own refusal and the top of the bracket from VRAM alone."""
    m = re.match(r"(.*-)(\d{5})-of-(\d{5})\.gguf$", model.name)
    shards = [model]
    if m:
        shards = [model.with_name(f"{m.group(1)}{i:05d}-of-{m.group(3)}.gguf") for i in range(1, int(m.group(3)) + 1)]
    try:
        return geometry.from_gguf(shards)
    except (OSError, ValueError):
        return None


def _speed_verdict(a: float | None, b: float | None, noise: float) -> int:
    """+1 if a is faster than b beyond the noise band, -1 if slower beyond it, 0 for a tie or a missing figure."""
    if not a or not b:
        return 0
    r = a / b
    return 1 if r > 1 + noise else -1 if r < 1 / (1 + noise) else 0


def prefill_pair(mmap_row: dict, p3_row: dict, noise: float = PAIR_NOISE) -> dict:
    """Judge one ring or tier configuration measured with both prefill sources, on prefill AND decode; mark the
    loser `dominated` in place (the planner only uses ok rows) and return the verdict for the profile.

    p3 keeps its row when it is faster on one figure beyond the noise band and not slower on the other beyond it;
    the default (mmap) keeps its row in the mirror case and on a tie, so p3 must earn its place. A split keeps both,
    and the planner's objective chooses between them."""
    out = {"miss_source": mmap_row.get("miss_source"), "tier_mib": mmap_row.get("tier_mib"), "noise": noise}
    if mmap_row.get("status") != "ok" or p3_row.get("status") != "ok":
        out["kept"] = "mmap" if mmap_row.get("status") == "ok" else ("p3" if p3_row.get("status") == "ok" else None)
        out["why"] = f"not both measured (mmap {mmap_row.get('status')}, p3 {p3_row.get('status')})"
        return out
    pp = _speed_verdict(p3_row.get("pp_tps"), mmap_row.get("pp_tps"), noise)
    tg = _speed_verdict(p3_row.get("tg_tps"), mmap_row.get("tg_tps"), noise)
    out.update(prefill={"mmap": mmap_row.get("pp_tps"), "p3": p3_row.get("pp_tps")},
               decode={"mmap": mmap_row.get("tg_tps"), "p3": p3_row.get("tg_tps")})
    if 1 in (pp, tg) and -1 not in (pp, tg):
        loser, out["kept"] = mmap_row, "p3"
        out["why"] = "p3 is faster beyond the noise band on " + " and ".join(
            n for n, v in (("prefill", pp), ("decode", tg)) if v == 1) + ", and not slower on the other"
    elif 1 in (pp, tg):
        out["kept"], out["why"] = "both", ("split: p3 is faster on " + ("prefill" if pp == 1 else "decode") +
                                           " and slower on " + ("decode" if pp == 1 else "prefill") +
                                           "; the planner's objective chooses")
        return out
    else:
        loser, out["kept"] = p3_row, "mmap"
        out["why"] = ("a tie on prefill and decode: the default stays" if (pp, tg) == (0, 0) else
                      "p3 is slower beyond the noise band on " + " and ".join(
                          n for n, v in (("prefill", pp), ("decode", tg)) if v == -1) + ", and not faster on the other")
    loser["status"] = "dominated"
    loser["note"] = f"prefill source {'p3' if loser is p3_row else 'mmap'} lost to its twin: {out['why']}"
    return out


def ubatch_report(rows: list[dict]) -> list[dict]:
    """Per (context, KV) cell, the fastest decode row and the fastest prefill row, each with both figures, overall
    and among exact-class rows: what a declared ubatch buys and costs, side by side."""
    out = []
    for ctx, kv in sorted({(r["ctx"], r["kv"]) for r in rows}):
        cell = {"ctx": ctx, "kv": kv}
        for label, pool in (("all", [r for r in rows if r["ctx"] == ctx and r["kv"] == kv and r["status"] == "ok"]),
                            ("exact", [r for r in rows if r["ctx"] == ctx and r["kv"] == kv and r["status"] == "ok"
                                       and r.get("exact_class")])):
            best = {}
            for k, name in (("tg_tps", "best_decode"), ("pp_tps", "best_prefill")):
                top = max(pool, key=lambda r: r.get(k) or 0, default=None)
                if top is not None:
                    best[name] = {"tg_tps": top.get("tg_tps"), "pp_tps": top.get("pp_tps"), "arm": top.get("arm"),
                                  "config": Config.from_row(top).key()}
            cell[label] = best
        out.append(cell)
    return out


def cache_arms(model: Path, engine, s: dict, geom, ctx: int, kv: str, threads: int | None, walk_rows: list[dict],
               log: Path, total_vram: int, n_predict: int, margin_mib: int, collapse: float,
               opt: CacheOptions, *, ubatch: int = DEFAULT_UBATCH, p3: bool = False,
               probe_prompt: int | None = None) -> tuple[list[dict], dict]:
    """The cliff finder and the miss sources for one (context, KV) cell; returns its rows and a summary of the search.
    With p3 (the engine has --moe-prefill-source), every ring and tier row gets a P3-prefill twin, judged by
    prefill_pair."""
    n = s["block_count"] or 0
    summary: dict = {"ctx": ctx, "kv": kv}
    rows: list[dict] = []
    base = next((r for r in walk_rows if r.get("n_cpu_moe") == n and r["status"] == "ok"), None)
    if base is None:
        summary["skipped"] = "the all-experts-on-CPU point did not run, so the VRAM left for a cache is unknown"
        return rows, summary
    lo = geometry.floor_mib(geom, opt.n_ubatch) if geom else 256        # without geometry, a refusal names the floor
    full = geometry.full_mib(geom, opt.n_ubatch) if geom else None
    share = base.get("vram_model_mib") or base.get("vram_peak_mib")
    used = machine.gpu_memory_used_mib()
    room = total_vram - used[0] - margin_mib - share if (total_vram and used and share) else None
    tops = [x for x in (full, room, opt.max_cache_mib) if x is not None]
    summary.update(floor_mib=lo, full_mib=full, vram_room_mib=room, max_cache_mib=opt.max_cache_mib)
    if not tops:
        summary["skipped"] = "neither the model's geometry nor the free VRAM bounds the cache budget"
        return rows, summary
    hi = min(tops)
    summary["hi_mib"] = hi
    if hi < lo:
        summary["skipped"] = f"the exactness floor ({lo} MiB) is above the largest budget that fits ({hi} MiB)"
        return rows, summary

    def run_one(budget: int, arm: str, **src) -> dict:
        c = Config(ctx=ctx, cache_type=kv, threads=threads, n_gpu_layers=999, n_cpu_moe=n, load_mode="mmap",
                   moe_cache_mib=budget, ubatch=ubatch, **src)
        row = dict(c.to_row(), arm=arm, exact_class=c.exact_class(n))
        measure(engine, model, c, row, log, n_predict, probe_prompt)
        judge(row, total_vram, margin_mib, 0.0, collapse)            # VRAM near full; speed is judged by the caller
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(5)
        return row

    best = find_cliff(lambda b: run_one(b, "cache"), lo, hi, opt.resolution_mib, collapse)
    summary["best_mib"] = best
    summary["probes"] = len(rows)
    if best is None:
        summary["skipped_sources"] = "no cache budget ran without overflow"
        return rows, summary

    def rungs() -> None:
        """Drift rungs (debt row 20, register E63): the best budget fits with nothing to spare beyond the margin, so
        a small drop in free VRAM before launch would leave the planner only the floor row. Below the best budget,
        every drift_step_mib down to the floor, measure the cache row and each miss-source configuration that
        survived at the best budget, under the same RAM envelope; a drift of less than one step then costs one rung."""
        floor = min(r["moe_cache_mib"] for r in rows if r["arm"] == "cache" and r["status"] == "ok")
        kept = [{k: r[k] for k in ("miss_source", "tier_mib", "ring_mib", "prefill_source") if r.get(k) is not None}
                for r in rows if r["arm"] == "source" and r["status"] == "ok" and r["moe_cache_mib"] == best]
        done: list[dict] = []
        summary["rungs"] = {"step_mib": opt.drift_step_mib, "sources": kept, "measured": done}
        for k in range(1, max(opt.drift_rungs, 0) + 1):
            b = -(-(best - k * opt.drift_step_mib) // 64) * 64       # 64 MiB aligned, rounded up: within one step
            if b <= floor:
                break
            r = run_one(b, "rung")
            rung = {"mib": b, "cache": r["status"], "sources": []}
            done.append(rung)
            if r["status"] != "ok":
                break
            for src in kept:
                pins = Config(ctx=ctx, cache_type=kv, moe_cache_mib=b, **src).pinned_mib()
                if pins and pins > machine.ram_mib()[1] - opt.ram_floor_mib:
                    rung["sources"].append(dict(src, status="skipped: the RAM envelope"))
                    continue
                s_row = run_one(b, "rung", **src)
                if s_row["status"] == "ok" and src.get("miss_source") == "tier" \
                        and s_row["ram_avail_low_mib"] < opt.ram_floor_mib:
                    s_row["status"] = "ram_envelope"
                    s_row["note"] = (f"RAM available fell to {s_row['ram_avail_low_mib']} MiB, below the "
                                     f"{opt.ram_floor_mib} MiB floor")
                rung["sources"].append(dict(src, status=s_row["status"]))

    if full is not None and best >= full:
        summary["skipped_sources"] = "every expert fits in the VRAM cache: no misses to serve"
        rungs()
        return rows, summary
    pairs: list[dict] = []
    if opt.prefill_p3 and p3:
        summary["prefill_pairs"] = pairs
    elif opt.prefill_p3:
        summary["prefill_pairs"] = "skipped: the engine has no --moe-prefill-source (patch 0009)"

    def twin(r: dict, **src) -> dict | None:
        """The same ring or tier row with P3 prefill, judged against it; never for memory-mapped misses."""
        if not (opt.prefill_p3 and p3) or r["status"] != "ok" or src.get("miss_source") not in P3_MISS_SOURCES:
            return None
        t = run_one(best, "source", prefill_source="p3", **src)
        if t["status"] == "ok" and src["miss_source"] == "tier" and t["ram_avail_low_mib"] < opt.ram_floor_mib:
            t["status"] = "ram_envelope"
            t["note"] = f"RAM available fell to {t['ram_avail_low_mib']} MiB, below the {opt.ram_floor_mib} MiB floor"
        pairs.append(prefill_pair(r, t))
        return t

    if opt.ring:
        free = machine.ram_mib()[1]
        if RING_DEFAULT_MIB <= free - opt.ram_floor_mib:
            twin(run_one(best, "source", miss_source="ring"), miss_source="ring")
        else:
            summary["ring"] = f"skipped: {free} MiB available leaves less than the {opt.ram_floor_mib} MiB floor"
    free = machine.ram_mib()[1]
    sizes = tier_ladder(free, opt.ram_floor_mib, (full - best) if full is not None else None, opt.tier_ladder_mib,
                        opt.tier_steps)
    summary["ram_envelope"] = {"ram_available_mib": free, "floor_mib": opt.ram_floor_mib,
                               "pin_max_mib": free - opt.ram_floor_mib, "tier_mib": sizes}
    for t in sizes:
        now = machine.ram_mib()[1]                           # the desktop may have grown since the ladder was cut
        if t + RING_DEFAULT_MIB > now - opt.ram_floor_mib:
            summary["ram_envelope"]["stopped"] = f"tier {t} MiB: {now} MiB available now leaves less than the floor"
            break
        r = run_one(best, "source", miss_source="tier", tier_mib=t)
        if r["status"] == "ok" and r["ram_avail_low_mib"] < opt.ram_floor_mib:
            r["status"] = "ram_envelope"
            r["note"] = f"RAM available fell to {r['ram_avail_low_mib']} MiB, below the {opt.ram_floor_mib} MiB floor"
        if r["status"] != "ok":
            summary["ram_envelope"]["stopped"] = f"tier {t} MiB: {r['status']}"
            break
        tw = twin(r, miss_source="tier", tier_mib=t)
        if tw is not None and tw["status"] in ("ram_envelope", "overflow"):
            summary["ram_envelope"]["stopped"] = f"tier {t} MiB with P3 prefill: {tw['status']}"
            break
    rungs()
    return rows, summary


def _capabilities(engine) -> dict:
    caps = getattr(engine, "capabilities", None)            # an engine without it (or a broken --help) is not Marian
    try:
        return caps() if caps else {}
    except Exception:                                      # noqa: BLE001
        return {}


BATCH_ENV = ("LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH")


def logical_batch(ubatch: int, caps: dict) -> int | None:
    """The logical batch (-b) the server runs at: N for a declared ubatch N (-ub N -b N); for the default ubatch,
    llama-server's own default as its --help states it (None if unread). Register E59 (28/10/2026): at a fixed ubatch
    the logical batch changes answers too, so it is part of the exactness reference and every profile records it."""
    return ubatch if ubatch != DEFAULT_UBATCH else caps.get("default_batch")


def launch_env_problem() -> str | None:
    """llama-server reads -b and -ub from these variables too; set, they would move the reference without a flag."""
    set_ = [k for k in BATCH_ENV if os.environ.get(k)]
    return (f"environment: {', '.join(set_)} set; llama-server would take its batch or ubatch from it, and both are part "
            "of the exactness reference (registers E54, E59). Unset it and declare the ubatch with --ubatch"
            if set_ else None)


def profile_name(stem: str, ubatch: int) -> str:
    """The profile's file name: the default ubatch keeps the name M0 used; a declared one gets its own file, so a
    deployment at 2048 does not replace the profile at 512 (each keeps one previous copy)."""
    return f"{stem}.json" if ubatch == DEFAULT_UBATCH else f"{stem}-ub{ubatch}.json"


def run(model: Path, engine: LlamaCpp, ctxs: list[int], kvs: list[str], step: int, threads: int | None,
        out_dir: Path, n_predict: int = 64, vram_margin_mib: int = 300, collapse: float = 0.6,
        cache: CacheOptions | None = None, sweep: ThreadOptions | None = None, ubatch: int = DEFAULT_UBATCH,
        probe_prompt: int | None = None) -> Path:
    if ubatch < 1:
        raise ValueError(f"ubatch {ubatch} is not a positive token count")
    for ctx in ctxs:                                      # refuse before any server starts, not halfway through
        probe_target(ctx, n_predict, probe_prompt)
    cache = cache or CacheOptions()
    cache.n_ubatch = ubatch                               # the exactness floor is computed at the declared ubatch
    sweep = sweep or ThreadOptions()
    s = gguf.summary(model)
    m = machine.detect()
    total_vram = m.gpus[0].vram_mib if m.gpus else 0
    caps = _capabilities(engine)
    why = launch_env_problem()
    if why:
        raise ValueError(why)
    with_cache = bool(cache.enabled and s["moe"] and caps.get("marian"))
    geom = model_geometry(model) if with_cache else None
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{m.fingerprint()}-{sha256_head(model)}-{engine.name}"
    log = out_dir / (profile_name(stem, ubatch)[:-len(".json")] + ".server.log")
    rows, searches, sweeps = [], [], []
    ladder: list[int] = []
    if sweep.enabled and threads is None:
        ladder = list(sweep.ladder) if sweep.ladder else thread_ladder(machine.physical_cores(), m.logical_cpus)
    for kv in kvs:
        for ctx in ctxs:
            best, cell = 0.0, []
            for knob, k in splits(s, step):
                c = Config(ctx=ctx, cache_type=kv, threads=threads, ubatch=ubatch)
                if knob == "n_cpu_moe":
                    c.n_gpu_layers, c.n_cpu_moe = 999, k
                else:
                    c.n_gpu_layers = k
                row = dict(c.to_row(), arm="walk", exact_class=c.exact_class(s["block_count"]))
                measure(engine, model, c, row, log, n_predict, probe_prompt)
                judge(row, total_vram, vram_margin_mib, best, collapse)
                best = max(best, row["tg_tps"]) if row["status"] == "ok" else best
                cell.append(row)
                print(json.dumps(row), flush=True)
                if row["status"] != "ok":
                    break
                time.sleep(5)
            rows += cell
            if ladder:
                trows, tsum = thread_sweep(model, engine, s, ctx, kv, cell, log, total_vram, n_predict,
                                           vram_margin_mib, ladder, probe_prompt)
                rows += trows
                sweeps.append(tsum)
            if with_cache:
                crows, summary = cache_arms(model, engine, s, geom, ctx, kv, threads, cell, log, total_vram, n_predict,
                                            vram_margin_mib, collapse, cache, ubatch=ubatch,
                                            p3=bool(caps.get("prefill_source")), probe_prompt=probe_prompt)
                rows += crows
                searches.append(summary)
    doc = {"marian_profile": PROFILE_SCHEMA,
           "measured_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "machine": m.to_dict(),
           "engine": {"name": engine.name, "version": engine.version(), "marian": bool(caps.get("marian")),
                      "prefill_source": bool(caps.get("prefill_source")), "moe_flags": caps.get("moe_flags", [])},
           "ubatch": ubatch,
           "batch": logical_batch(ubatch, caps),
           "model": {"file": model.name, "id": sha256_head(model), **s}, "probe": ({"fill": 0.75, "n_predict": n_predict} if probe_prompt is None else
                     {"prompt_tokens": probe_prompt, "n_predict": n_predict}),
           "rows": rows}
    if with_cache:
        doc["cache_search"] = {"ram_floor_mib": cache.ram_floor_mib, "resolution_mib": cache.resolution_mib,
                               "cells": searches}
    elif cache.enabled and s["moe"]:
        doc["cache_search"] = {"skipped": "the engine is not a Marian build (its --help lacks the P3 flags)"}
    if ladder:
        doc["thread_search"] = {"ladder": ladder, "physical_cores": machine.physical_cores(),
                                "logical_cpus": m.logical_cpus, "cells": sweeps}
    elif sweep.enabled:
        doc["thread_search"] = {"skipped": "--threads fixed the thread count" if threads is not None
                                else "the processor count could not be read"}
    doc["ubatch_report"] = ubatch_report(rows)
    out = out_dir / profile_name(stem, ubatch)
    if out.exists():                                     # a new profile replaces the old one; keep one previous copy
        out.replace(out.with_name(out.name + ".prev"))
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out
