"""Profile a model on this machine: speed and memory per (context size, KV type, GPU/CPU split).

For each context size and KV type, the GPU/CPU split is walked from the most conservative (most work on the CPU) to
the most aggressive. A point is an OVERFLOW when VRAM is within a margin of full, or when speed collapses against the
best point so far (on Windows the driver pages GPU memory into system RAM instead of failing, so the symptom is a
cliff, measured 21/10/2026: 42.7 -> 5.5 tok/s). The walk stops at the first overflow.

The probe fills about 75 % of the context with neutral text, then generates a fixed number of tokens, so decode is
measured at depth, not on an empty cache.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import time
from pathlib import Path

from . import gguf, machine
from .engines import Config, LlamaCpp

FILLER = ("The workshop opens at eight. Apprentices sign in, check the job board and collect their tools. "
          "Safety briefings come first, then practical tasks, then a short review before lunch. ")


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
        self.every, self.vram, self.ram_low, self._stop = every, 0, 10**9, threading.Event()

    def run(self):
        while not self._stop.is_set():
            used = machine.gpu_memory_used_mib()
            if used:
                self.vram = max(self.vram, used[0])
            self.ram_low = min(self.ram_low, machine.ram_mib()[1])
            time.sleep(self.every)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)


def probe(engine: LlamaCpp, ctx: int, n_predict: int) -> dict:
    target = int(ctx * 0.75) - n_predict
    unit = engine.tokenize(FILLER)
    text = FILLER * max(1, target // max(unit, 1))
    t = engine.complete(text, n_predict)
    return {"prompt_tokens": t.get("prompt_n"), "pp_tps": round(t.get("prompt_per_second", 0), 1),
            "tg_tps": round(t.get("predicted_per_second", 0), 2)}


def splits(summary: dict, step: int) -> list[tuple[str, int]]:
    n = summary["block_count"] or 0
    if summary["moe"]:
        return [("n_cpu_moe", k) for k in range(n, -1, -step)]          # all experts on CPU -> none
    return [("n_gpu_layers", k) for k in range(max(0, n // 4), n + 2, step)]  # a quarter on GPU -> all


def run(model: Path, engine: LlamaCpp, ctxs: list[int], kvs: list[str], step: int, threads: int | None,
        out_dir: Path, n_predict: int = 64, vram_margin_mib: int = 300, collapse: float = 0.6) -> Path:
    s = gguf.summary(model)
    m = machine.detect()
    total_vram = m.gpus[0].vram_mib if m.gpus else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{m.fingerprint()}-{sha256_head(model)}-{engine.name}"
    log = out_dir / f"{stem}.server.log"
    rows = []
    for kv in kvs:
        for ctx in ctxs:
            best = 0.0
            for knob, k in splits(s, step):
                c = Config(ctx=ctx, cache_type=kv, threads=threads)
                if knob == "n_cpu_moe":
                    c.n_gpu_layers, c.n_cpu_moe = 999, k
                else:
                    c.n_gpu_layers = k
                row = {"ctx": ctx, "kv": kv, knob: k, "threads": threads}
                sampler = PeakSampler()
                sampler.start()
                ok, why = engine.start(model, c, log)
                try:
                    if not ok:
                        row.update(status="start_failed", note=why)
                    else:
                        row.update(probe(engine, ctx, n_predict))
                        row["status"] = "ok"
                finally:
                    engine.stop()
                    sampler.stop()
                row.update(vram_peak_mib=sampler.vram, ram_avail_low_mib=sampler.ram_low)
                if row["status"] == "ok":
                    near_full = total_vram and sampler.vram >= total_vram - vram_margin_mib
                    cliff = best and row["tg_tps"] < best * collapse
                    if near_full or cliff:
                        row["status"] = "overflow"
                        row["note"] = "VRAM near full" if near_full else f"speed collapse vs best {best}"
                    best = max(best, row["tg_tps"]) if row["status"] == "ok" else best
                rows.append(row)
                print(json.dumps(row), flush=True)
                if row["status"] in ("overflow", "start_failed"):
                    break
                time.sleep(5)
    doc = {"marian_profile": 1, "measured_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "machine": m.to_dict(), "engine": {"name": engine.name, "version": engine.version()},
           "model": {"file": model.name, "id": sha256_head(model), **s}, "probe": {"fill": 0.75, "n_predict": n_predict},
           "rows": rows}
    out = out_dir / f"{stem}.json"
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out
