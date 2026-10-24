"""Choose a configuration from measured profiles: the fastest point that fits the input, never a guess.

`need` is the context the request needs (input tokens + the answer budget). Only rows measured on THIS machine
(same fingerprint) for THIS model (same id) count. A row qualifies when its status is ok and its context is at least
`need`; among those, the fastest for the chosen objective wins (generation speed by default; prompt speed for
read-heavy work such as screening or classifying long documents, which reads far more than it writes).

With `vram_free_mib` (the VRAM free at launch), a row must also fit now: its own VRAM share plus a margin must not
exceed what is free. A row's share is `vram_model_mib` (its peak minus the GPU memory other programs used when it was
measured); a row from an older profile without that figure counts its whole peak, which is the safe side. This guards
the dangerous direction, a busier desktop at launch than at profiling time, which on Windows pages GPU memory and
collapses speed. It cannot add rows that were skipped as overflows on a busy desktop: only a new profile finds those.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import machine
from .engines import Config
from .profile import sha256_head


def load(profile_dir: Path, model: Path) -> list[dict]:
    fp, mid = machine.detect().fingerprint(), sha256_head(model)
    rows = []
    for p in profile_dir.glob(f"{fp}-{mid}-*.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        rows += [dict(r, _engine=doc["engine"]["name"], _file=p.name) for r in doc["rows"]]
    return rows


def vram_share(row: dict) -> tuple[int | None, str]:
    """(MiB this row needs on the GPU, basis): its own share if recorded, else its whole peak, else unknown."""
    if row.get("vram_model_mib") is not None:
        return row["vram_model_mib"], "model share (peak minus other programs' use at profile time)"
    if row.get("vram_peak_mib"):
        return row["vram_peak_mib"], "whole peak (the profile predates baseline recording)"
    return None, "unknown (no VRAM reading in the profile)"


def choose(rows: list[dict], need: int, objective: str = "tg", vram_free_mib: int | None = None,
           margin_mib: int = 300) -> dict | None:
    key = "tg_tps" if objective == "tg" else "pp_tps"
    fits = [r for r in rows if r.get("status") == "ok" and r["ctx"] >= need]
    if vram_free_mib is not None:
        fits = [r for r in fits if (s := vram_share(r)[0]) is not None and s + margin_mib <= vram_free_mib]
    if not fits:
        return None
    smallest = min(r["ctx"] for r in fits)                 # the smallest measured context that fits: less VRAM
    return max((r for r in fits if r["ctx"] == smallest), key=lambda r: (r.get(key) or 0))


def to_config(row: dict) -> Config:
    c = Config(ctx=row["ctx"], cache_type=row["kv"], threads=row.get("threads"))
    if "n_cpu_moe" in row:
        c.n_gpu_layers, c.n_cpu_moe = 999, row["n_cpu_moe"]
    else:
        c.n_gpu_layers = row["n_gpu_layers"]
    return c
