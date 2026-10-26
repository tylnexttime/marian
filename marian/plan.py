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

With `ram_free_mib` (the RAM available at launch), a row that pins host memory (the fork's ring or tier) must leave
`ram_floor_mib` available beside what it pins: pinned memory cannot be paged, so the desktop would be squeezed instead.

Two filters (M1.3): `exact_only` keeps exact-class rows (every expert computed on the GPU: full offload or a cache
arm; stock `-ncmoe` > 0 changes answers), and `stock` keeps the rows a stock llama.cpp runs with plain upstream
placement flags (`-ncmoe` / `-ngl`, no expert cache, no Marian flag): the rollback. The expert cache is upstream's,
but it is EXPERIMENTAL there (three changes in two weeks) and its rows were measured on the fork, whose exactness
floor a stock build does not have, so the rollback leaves it out.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import machine
from .engines import Config
from .profile import sha256_head


def load_docs(profile_dir: Path, model: Path, model_id: str | None = None) -> list[tuple[Path, dict]]:
    """Every profile of this model on this machine, as (file, document)."""
    fp, mid = machine.detect().fingerprint(), model_id or sha256_head(model)
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(profile_dir.glob(f"{fp}-{mid}-*.json"))]


def rows_of(docs: list[tuple[Path, dict]], engine_name: str | None = None) -> list[dict]:
    """The rows of these profiles, each tagged with its file, engine and the model's layer count (exact_class needs
    it for a dense model). With engine_name, only that engine's profiles count."""
    rows = []
    for p, doc in docs:
        eng = (doc.get("engine") or {}).get("name")
        if engine_name is not None and eng != engine_name:
            continue
        blocks = (doc.get("model") or {}).get("block_count")
        rows += [dict(r, _engine=eng, _file=p.name, _block_count=blocks) for r in doc.get("rows", [])]
    return rows


def load(profile_dir: Path, model: Path) -> list[dict]:
    return rows_of(load_docs(profile_dir, model))


def vram_share(row: dict) -> tuple[int | None, str]:
    """(MiB this row needs on the GPU, basis): its own share if recorded, else its whole peak, else unknown."""
    if row.get("vram_model_mib") is not None:
        return row["vram_model_mib"], "model share (peak minus other programs' use at profile time)"
    if row.get("vram_peak_mib"):
        return row["vram_peak_mib"], "whole peak (the profile predates baseline recording)"
    return None, "unknown (no VRAM reading in the profile)"


def exact_class(row: dict) -> bool:
    """Recorded since M1.3; derived for older rows from their placement (M0 rows: exact only at -ncmoe 0)."""
    if "exact_class" in row:
        return bool(row["exact_class"])
    return to_config(row).exact_class(row.get("_block_count"))


def is_stock(row: dict) -> bool:
    """Plain upstream placement: no expert cache and no Marian-only flag."""
    c = to_config(row)
    return not c.moe_cache_mib and not c.marian_flags()


def pinned_mib(row: dict) -> int:
    return to_config(row).pinned_mib()


def choose(rows: list[dict], need: int, objective: str = "tg", vram_free_mib: int | None = None,
           margin_mib: int = 300, *, exact_only: bool = False, stock: bool = False, ram_free_mib: int | None = None,
           ram_floor_mib: int = 6144) -> dict | None:
    key = "tg_tps" if objective == "tg" else "pp_tps"
    fits = [r for r in rows if r.get("status") == "ok" and r["ctx"] >= need]
    if exact_only:
        fits = [r for r in fits if exact_class(r)]
    if stock:
        fits = [r for r in fits if is_stock(r)]
    if vram_free_mib is not None:
        fits = [r for r in fits if (s := vram_share(r)[0]) is not None and s + margin_mib <= vram_free_mib]
    if ram_free_mib is not None:
        fits = [r for r in fits if pinned_mib(r) <= max(ram_free_mib - ram_floor_mib, 0)]
    if not fits:
        return None
    smallest = min(r["ctx"] for r in fits)                 # the smallest measured context that fits: less VRAM
    return max((r for r in fits if r["ctx"] == smallest), key=lambda r: (r.get(key) or 0))


def to_config(row: dict) -> Config:
    """Every recorded field back into a launch configuration: placement, KV, threads, load mode, cache budget, miss
    source, tier and ring (M0 dropped the cache budget here)."""
    return Config.from_row(row)
