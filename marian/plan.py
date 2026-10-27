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

**Live confirmation (M1.3).** A row may carry a `confirmed` record from `marian serve --confirm`: a live A/B against
the runner-up over interleaved windows on this machine. A row that LOST such an A/B (`outcome` replaced) gives way to
the row that beat it whenever that row also qualifies: a measurement taken live, head to head, outranks two profile
numbers taken minutes apart. `runner_up` names the row such an A/B is run against, and `next_smaller` the row the
memory governor restarts on; both keep an exact-class plan exact unless non-exact rows were allowed.

**The declared ubatch (M1.3, after register E54).** A ubatch changes the compute shape, so a row is exact only against
an all-VRAM reference at its own ubatch, and its speeds were measured at it. Every query here therefore takes the
deployment's declared `ubatch` (default 512) and keeps only rows measured at it: a plan measured at one ubatch is
never served at another, as an exact-only plan never mixes in a non-exact row. The filter sits in `candidates`, the
funnel that `choose`, `runner_up` and `next_smaller` all go through, so neither `--confirm` nor the governor can cross
it. A profile records its ubatch (schema 3); a schema 1 or 2 profile was measured without `-ub`, so at 512.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import machine
from .engines import DEFAULT_UBATCH, Config
from .profile import PROFILE_SCHEMA, sha256_head


def load_docs(profile_dir: Path, model: Path, model_id: str | None = None) -> list[tuple[Path, dict]]:
    """Every profile of this model on this machine, as (file, document)."""
    fp, mid = machine.detect().fingerprint(), model_id or sha256_head(model)
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(profile_dir.glob(f"{fp}-{mid}-*.json"))]


def doc_ubatch(doc: dict) -> int:
    """The ubatch a profile was measured at. Schema 3 records it; schema 1 and 2 profiles passed no `-ub`, so their
    servers ran at llama-server's default, 512."""
    return int(doc.get("ubatch") or DEFAULT_UBATCH)


def schema_problem(doc: dict) -> str | None:
    """Why this profile cannot be used, or None. Older schemas load (their missing fields have safe defaults); a newer
    one may carry settings this version would silently drop, so it is refused."""
    v = doc.get("marian_profile")
    if isinstance(v, int) and v > PROFILE_SCHEMA:
        return (f"profile schema {v} is newer than this Marian reads (schema {PROFILE_SCHEMA}): it may record settings "
                "this version would drop; update Marian, or calibrate again with this version")
    return None


def rows_of(docs: list[tuple[Path, dict]], engine_name: str | None = None) -> list[dict]:
    """The rows of these profiles, each tagged with its file, engine, the model's layer count (exact_class needs it
    for a dense model) and the profile's ubatch. With engine_name, only that engine's profiles count."""
    rows = []
    for p, doc in docs:
        eng = (doc.get("engine") or {}).get("name")
        if engine_name is not None and eng != engine_name:
            continue
        blocks = (doc.get("model") or {}).get("block_count")
        ub = doc_ubatch(doc)
        rows += [dict(r, _engine=eng, _file=p.name, _block_count=blocks, _index=i, _ubatch=ub)
                 for i, r in enumerate(doc.get("rows", []))]
    return rows


def ubatch_of(row: dict) -> int:
    """The ubatch a row was measured at: its own record, else its profile's, else the default."""
    return int(row.get("ubatch") or row.get("_ubatch") or DEFAULT_UBATCH)


def ubatches(rows: list[dict]) -> list[int]:
    """The ubatches these rows were measured at, for messages."""
    return sorted({ubatch_of(r) for r in rows})


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


def key(row: dict) -> str:
    """The launch configuration of a row, as a comparable string (the arm and the measurements are not part of it)."""
    return to_config(row).key()


def ram_need_mib(row: dict) -> int:
    """Host RAM the row took when it was measured: the fall in available RAM from before its start to its low point,
    else its peak working set, else what it pins. A comparison figure for the governor, not a promise."""
    before, low = row.get("ram_free_before_mib"), row.get("ram_avail_low_mib")
    if before is not None and low is not None and before >= low:
        return max(before - low, pinned_mib(row))
    if row.get("working_set_peak_mib") is not None:
        return max(row["working_set_peak_mib"], pinned_mib(row))
    return pinned_mib(row)


def candidates(rows: list[dict], need: int, vram_free_mib: int | None = None, margin_mib: int = 300, *,
               exact_only: bool = False, stock: bool = False, ram_free_mib: int | None = None,
               ram_floor_mib: int = 6144, ubatch: int = DEFAULT_UBATCH) -> list[dict]:
    """Every row that fits the request, the declared ubatch, the filters and the memory free now; unordered. A row
    whose configuration could not be launched (P3 prefill beside memory-mapped misses, in a hand-edited profile) is
    never a candidate."""
    fits = [r for r in rows if r.get("status") == "ok" and r["ctx"] >= need and ubatch_of(r) == ubatch
            and to_config(r).problem() is None]
    if exact_only:
        fits = [r for r in fits if exact_class(r)]
    if stock:
        fits = [r for r in fits if is_stock(r)]
    if vram_free_mib is not None:
        fits = [r for r in fits if (s := vram_share(r)[0]) is not None and s + margin_mib <= vram_free_mib]
    if ram_free_mib is not None:
        fits = [r for r in fits if pinned_mib(r) <= max(ram_free_mib - ram_floor_mib, 0)]
    return fits


def _best(fits: list[dict], objective: str) -> dict | None:
    """The fastest row at the smallest measured context (less VRAM), then the live-confirmation rule: a row that lost
    an A/B gives way to the row that beat it, if that row is among the candidates."""
    if not fits:
        return None
    k = "tg_tps" if objective == "tg" else "pp_tps"
    smallest = min(r["ctx"] for r in fits)
    pool = [r for r in fits if r["ctx"] == smallest]
    best = max(pool, key=lambda r: (r.get(k) or 0))
    seen = set()
    while (c := best.get("confirmed") or {}).get("outcome") == "replaced" and c.get("by") and id(best) not in seen:
        seen.add(id(best))
        winner = next((r for r in pool if key(r) == c["by"]), None)
        if winner is None:
            break
        best = winner
    return best


def choose(rows: list[dict], need: int, objective: str = "tg", vram_free_mib: int | None = None,
           margin_mib: int = 300, *, exact_only: bool = False, stock: bool = False, ram_free_mib: int | None = None,
           ram_floor_mib: int = 6144, ubatch: int = DEFAULT_UBATCH) -> dict | None:
    return _best(candidates(rows, need, vram_free_mib, margin_mib, exact_only=exact_only, stock=stock,
                            ram_free_mib=ram_free_mib, ram_floor_mib=ram_floor_mib, ubatch=ubatch), objective)


def runner_up(rows: list[dict], chosen: dict, need: int, objective: str = "tg", vram_free_mib: int | None = None,
              margin_mib: int = 300, *, exact_only: bool = False, stock: bool = False, ram_free_mib: int | None = None,
              ram_floor_mib: int = 6144, allow_non_exact: bool = False,
              ubatch: int = DEFAULT_UBATCH) -> dict | None:
    """The best row with another launch configuration under the same filters: what a live A/B confirms the plan
    against. An exact-class plan only ever meets exact-class runners-up, unless non-exact rows were allowed (--stock or
    --allow-non-exact): a faster answer that is a different answer is not a confirmation."""
    fits = candidates(rows, need, vram_free_mib, margin_mib, exact_only=exact_only, stock=stock,
                      ram_free_mib=ram_free_mib, ram_floor_mib=ram_floor_mib, ubatch=ubatch)
    fits = [r for r in fits if key(r) != key(chosen)]
    if exact_class(chosen) and not allow_non_exact:
        fits = [r for r in fits if exact_class(r)]
    k = "tg_tps" if objective == "tg" else "pp_tps"
    return max(fits, key=lambda r: (r.get(k) or 0), default=None)


def next_smaller(rows: list[dict], current: dict, need: int, objective: str = "tg", *, pressure: set[str],
                 vram_free_mib: int | None, ram_free_mib: int | None, margin_mib: int = 300,
                 ram_floor_mib: int = 6144, exact_only: bool = False, stock: bool = False,
                 allow_non_exact: bool = False, ubatch: int = DEFAULT_UBATCH) -> dict | None:
    """The row the governor restarts on: the fastest that fits the request and the filters, takes less of the memory
    under pressure than the current row, and fits what stopping the current server leaves free (the readings are
    taken while it runs, so its own share comes back). An exact-class row is only replaced by an exact-class row unless
    non-exact rows were allowed. None: nothing smaller fits, and the governor gives up rather than guess."""
    cur_v = vram_share(current)[0] or 0
    cur_r = ram_need_mib(current)
    fits = candidates(rows, need, exact_only=exact_only, stock=stock, ubatch=ubatch)
    fits = [r for r in fits if key(r) != key(current)]
    if exact_class(current) and not allow_non_exact:
        fits = [r for r in fits if exact_class(r)]
    if "vram" in pressure:
        fits = [r for r in fits if (s := vram_share(r)[0]) is not None and s < cur_v]
    if "ram" in pressure:
        fits = [r for r in fits if ram_need_mib(r) < cur_r]
    fits = [r for r in fits if fits_after_stop(r, current, vram_free_mib=vram_free_mib, ram_free_mib=ram_free_mib,
                                               margin_mib=margin_mib, ram_floor_mib=ram_floor_mib)]
    return _best(fits, objective)


def fits_after_stop(row: dict, current: dict, *, vram_free_mib: int | None, ram_free_mib: int | None,
                    margin_mib: int = 300, ram_floor_mib: int = 6144) -> bool:
    """Would `row` fit what stopping the `current` server leaves free, with the floors kept? The readings are taken
    while the current server runs, so its own share comes back. Used for the restart's next row and the step-up."""
    cur_v = vram_share(current)[0] or 0
    cur_r = ram_need_mib(current)
    if vram_free_mib is not None:
        s = vram_share(row)[0]
        if s is None or s + margin_mib > vram_free_mib + cur_v:
            return False
    if ram_free_mib is not None:
        if ram_need_mib(row) - cur_r + ram_floor_mib > ram_free_mib:
            return False
        if pinned_mib(row) > max(ram_free_mib + cur_r - ram_floor_mib, 0):
            return False
    return True


def to_config(row: dict) -> Config:
    """Every recorded field back into a launch configuration: placement, KV, threads, load mode, cache budget, miss
    source, tier, ring, prefill source and the ubatch it was measured at (M0 dropped the cache budget here). The
    ubatch comes from the row, else from its profile, so a row is always launched at the ubatch it was measured at."""
    c = Config.from_row(row)
    if c.ubatch is None:
        c.ubatch = ubatch_of(row)
    return c.normalise()
