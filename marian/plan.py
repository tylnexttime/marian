"""Choose a configuration from measured profiles: the fastest point that fits the input, never a guess.

`need` is the context the request needs (input tokens + the answer budget). Only rows measured on THIS machine
(same fingerprint) for THIS model (same id) count. A row qualifies when its status is ok and its context is at least
`need`; among those, the fastest for the chosen objective wins (generation speed by default; prompt speed for
read-heavy work such as screening or classifying long documents, which reads far more than it writes).
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


def choose(rows: list[dict], need: int, objective: str = "tg") -> dict | None:
    key = "tg_tps" if objective == "tg" else "pp_tps"
    fits = [r for r in rows if r.get("status") == "ok" and r["ctx"] >= need]
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
