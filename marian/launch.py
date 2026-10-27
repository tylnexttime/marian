"""Before and around `marian serve` (M1.3 deployment deliverables): the calibration gate, launch checks, diagnostics.

**The calibration gate** (Chris's rule: every machine runs its pre-tuning tests before it serves). `serve` refuses
when there is no profile for this machine's fingerprint, this model's id and this engine's name, unless it is asked
to calibrate first (`serve --calibrate`: one quick profile, then serve). Nothing is ever launched on a guess.

**Launch checks**, on the row the planner chose, each failure a reason (exit 2): the model file is the one profiled
(id and size), the profile is this machine's, the engine is the one profiled (name, then build: timings, the
exactness floor and the miss sources belong to the build that was measured; the stock rollback only warns, because
it runs plain upstream flags on whatever build is at hand), the engine has every flag the row passes, VRAM free now
covers the row's share plus the margin, and RAM available now covers what the row pins plus the RAM floor. Two more
since the prefill rows (after registers E51 and E54): the row was measured at the ubatch the deployment declares (a
plan measured at one ubatch is never served at another, because the ubatch is part of the exactness reference), and
a row with P3 prefill runs only on a build that lists `--moe-prefill-source` and only beside ring or tier misses.
A profile of a newer schema than this version reads is refused; an older one loads, at ubatch 512.

**The diagnostics bundle** (`marian diag OUT.zip`): the machine fingerprint dict and live memory readings, the
model's profiles, the plan it would choose, the tails of the serve and profiling logs, and the Marian and engine
versions. No prompt text (the profiles hold only the probe's filler; log lines that carry request or response bodies
are dropped) and no absolute paths: the home directory becomes ~, the model, engine and profile folders become
placeholders, and any other absolute folder is reduced to <dir>.
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

from . import machine, plan
from .engines import DEFAULT_UBATCH, LlamaCpp, same_build
from .profile import launch_env_problem, logical_batch, sha256_head, ubatch_report

QUICK_CTXS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
PROMPTY = re.compile(r'"(?:prompt|messages|content|input|text|system)"\s*:'
                     r'|\b(?:request|response|prompt|body|text|content)\s*[:=]\s*[\[{"\']', re.I)
USERS = re.compile(r"(?i)\b[a-z]:[\\/]+(?:users|documents and settings)[\\/]+[^\\/\r\n\"']+")
POSIX_HOME = re.compile(r"(?<![\w.:/])/(?:home|Users)/[^/\s\"']+")
WIN_DIR = re.compile(r"(?i)\b[a-z]:[\\/]+(?:[^\\/\s\"'<>|]+[\\/]+)*")
POSIX_DIR = re.compile(r"(?<![\w.:/~<>-])/(?:[^/\s\"'<>]+/)+")


def quick_ctx(need: int) -> int:
    """The one context a quick calibration measures: the smallest usual size that holds the request."""
    return next((c for c in QUICK_CTXS if c >= need), need)


def _ub_arg(ubatch: int) -> str:
    return "" if ubatch == DEFAULT_UBATCH else f" --ubatch {ubatch}"


def gate_reason(model: Path, engine_name: str, ubatch: int = DEFAULT_UBATCH, measured: list[int] | None = None) -> str:
    if measured:
        return ubatch_reason(model.name, ubatch, measured, engine_name)
    return (f"no calibration for {model.name} with {engine_name} on this machine (fingerprint "
            f"{machine.detect().fingerprint()}): every machine runs its pre-tuning tests before serving. Run "
            f"`marian calibrate {model.name} --engine PATH{_ub_arg(ubatch)}` first, or `marian serve ... --calibrate` "
            "to calibrate and then serve")


def ubatch_reason(model_name: str, ubatch: int, measured: list[int], engine_name: str | None = None) -> str:
    """Why a deployment at one ubatch is refused rows measured at another."""
    at = ", ".join(str(u) for u in measured)
    who = f" with {engine_name}" if engine_name else ""
    return (f"ubatch: this deployment declares ubatch {ubatch}, and {model_name}{who} was calibrated on this machine "
            f"at ubatch {at} only. A ubatch is part of the exactness reference (register E54: exact against all-VRAM "
            f"at the same ubatch, not at another), so a plan measured at one ubatch is never served at another. Run "
            f"`marian calibrate {model_name} --engine PATH{_ub_arg(ubatch)}`, or serve with "
            + " or ".join(f"--ubatch {u}" for u in measured))


def explain(rows: list[dict], need: int, objective: str, model_name: str, *, free: int | None, margin: int,
            exact_only: bool, stock: bool, ram_free: int | None, ram_floor: int, ubatch: int = DEFAULT_UBATCH) -> str:
    """Why choose() found nothing, from the widest question to the narrowest."""
    if not rows:
        return f"no profile for {model_name} on this machine; run `marian profile` (or `marian calibrate`) first"
    if not any(plan.ubatch_of(r) == ubatch for r in rows):
        return ubatch_reason(model_name, ubatch, plan.ubatches(rows))
    rows = [r for r in rows if plan.ubatch_of(r) == ubatch]
    if plan.choose(rows, need, objective, ubatch=ubatch) is None:
        return (f"no measured configuration fits {need} tokens on this machine for {model_name}; "
                "run `marian profile` with a larger --ctx first")
    kinds = [k for k, on in (("exact-class", exact_only), ("stock", stock)) if on]
    if kinds and plan.choose(rows, need, objective, exact_only=exact_only, stock=stock, ubatch=ubatch) is None:
        return (f"no {' and '.join(kinds)} configuration fits {need} tokens for {model_name}: the profile has rows "
                "that fit, but none of that class")
    if free is not None and plan.choose(rows, need, objective, free, margin, exact_only=exact_only, stock=stock,
                                        ubatch=ubatch) is None:
        return (f"measured configurations fit {need} tokens, but none fits the {free} MiB of VRAM free now "
                f"with a {margin} MiB margin")
    return (f"the configurations that fit {need} tokens pin more host memory than the {ram_free} MiB available now "
            f"leaves above the {ram_floor} MiB RAM floor")


def check(row: dict, doc: dict, model: Path, model_id: str, engine, *, stock: bool = False,
          vram_free_mib: int | None = None, margin_mib: int = 300, ram_free_mib: int | None = None,
          ram_floor_mib: int = 6144, ubatch: int = DEFAULT_UBATCH) -> tuple[list[str], list[str]]:
    """(reasons not to launch, warnings) for this row of this profile document."""
    reasons, warnings = [], []
    why = plan.schema_problem(doc)
    if why:
        reasons.append(why)
    measured = plan.ubatch_of(dict(row, _ubatch=plan.doc_ubatch(doc)))
    if measured != ubatch:
        reasons.append(f"ubatch: the row was measured at ubatch {measured} and this deployment declares {ubatch}; a "
                       "plan measured at one ubatch is never served at another (the ubatch is part of the exactness "
                       f"reference, register E54); calibrate at {ubatch} or declare {measured}")
    pm = doc.get("model") or {}
    if pm.get("id") != model_id:
        reasons.append(f"model identity: the profile is for model id {pm.get('id')} ({pm.get('file')}), "
                       f"this file is {model_id}")
    elif pm.get("bytes") is not None and pm["bytes"] != model.stat().st_size:
        reasons.append(f"model identity: the profile's model has {pm['bytes']} bytes, this file {model.stat().st_size}")
    fp = machine.detect().fingerprint()
    if (doc.get("machine") or {}).get("fingerprint") != fp:
        reasons.append(f"machine: the profile was measured on {(doc.get('machine') or {}).get('fingerprint')}, "
                       f"this machine is {fp}")
    pe = doc.get("engine") or {}
    if pe.get("name") != engine.name:
        reasons.append(f"engine: the profile is for {pe.get('name')}, this engine is {engine.name}")
    else:
        v = engine.version()
        if not same_build(pe.get("version"), v):
            msg = f"engine build: profiled with {pe.get('version')!r}, this engine is {v!r}"
            if stock:
                warnings.append(msg + "; the stock rows use plain upstream flags, their speeds may differ on this build")
            else:
                reasons.append(msg + "; calibrate again (`marian calibrate`) or serve the stock rows (--stock)")
    c = plan.to_config(dict(row, _ubatch=plan.doc_ubatch(doc)))
    caps = engine.capabilities()
    why = launch_env_problem()
    if why:
        reasons.append(why)
    rec, now = doc.get("batch"), logical_batch(c.n_ubatch, caps)
    if rec is None:
        if not stock and now is not None:
            warnings.append("the profile does not record its logical batch (-b; measured before register E59), so a "
                            "change in llama-server's default -b would go unnoticed; `marian calibrate` records it")
    elif now is None:
        reasons.append(f"batch: the profile was measured at -b {rec} and this engine's --help does not state its "
                       "default -b, so the logical batch cannot be checked")
    elif now != rec:
        reasons.append(f"batch: the profile was measured at -b {rec} and this engine would run at -b {now}; the "
                       "logical batch is part of the exactness reference (register E59); calibrate again")
    bad = c.problem()
    if bad:
        reasons.append(f"configuration: {bad}")
    if c.prefill_source is not None and caps.get("marian") and not caps.get("prefill_source"):
        reasons.append("engine: the row passes --moe-prefill-source, which this Marian build's --help does not list "
                       "(patch 0009); calibrate again with this build, or use a build with 0009")
    if (isinstance(doc.get("marian_profile"), int) and doc["marian_profile"] < 3 and caps.get("prefill_source")
            and any((r.get("miss_source") in ("ring", "tier")) for r in doc.get("rows", []))):
        warnings.append(f"the profile is schema {doc['marian_profile']}: its ring and tier rows predate the P3 prefill "
                        "rows, so --moe-prefill-source p3 was never measured here; `marian calibrate` measures both")
    if c.moe_cache_mib and not caps.get("moe_cache"):
        reasons.append("engine: the row uses --moe-cache-mib, which this engine's --help does not list")
    if c.marian_flags() and not caps.get("marian"):
        reasons.append(f"engine: the row passes {', '.join(c.marian_flags())}, and this engine is not a Marian build; "
                       "serve the stock rows (--stock) or use a Marian build")
    share, _ = plan.vram_share(row)
    if vram_free_mib is None:
        warnings.append("free VRAM could not be read, so the row's VRAM was not checked")
    elif share is None:
        reasons.append("VRAM: the row has no VRAM reading, so its fit cannot be checked")
    elif share + margin_mib > vram_free_mib:
        reasons.append(f"VRAM: the row needs {share} MiB plus a {margin_mib} MiB margin; {vram_free_mib} MiB is free now")
    pinned = c.pinned_mib()
    if pinned:
        if ram_free_mib is None:
            reasons.append(f"RAM: the row pins {pinned} MiB and the RAM available could not be read")
        elif pinned > ram_free_mib - ram_floor_mib:
            reasons.append(f"RAM: the row pins {pinned} MiB; {ram_free_mib} MiB available now would leave less than "
                           f"the {ram_floor_mib} MiB floor")
    return reasons, warnings


def settings(cfg) -> dict:
    """The two non-placement settings of a configuration, spelt out (for --dry-run, --json and diag)."""
    return {"ubatch": cfg.n_ubatch, "ubatch_flag": cfg.ubatch is not None,
            "prefill_source": cfg.prefill_source or "mmap", "prefill_flag": cfg.prefill_source is not None}


def settings_line(cfg) -> str:
    st = settings(cfg)
    ub = (f"ubatch {st['ubatch']} (declared: -ub {st['ubatch']} -b {st['ubatch']})" if st["ubatch_flag"]
          else f"ubatch {st['ubatch']} (the default, no flag)")
    pf = ("prefill source p3 (--moe-prefill-source p3: large-batch misses from the ring and tier)"
          if st["prefill_flag"] else "prefill source mmap (the default, upstream's path, no flag)")
    return f"settings: {ub}; {pf}"


def trade_lines(doc: dict, others: list[dict]) -> list[str]:
    """What a calibration found at its ubatch, decode and prefill side by side per cell, with the same figures from
    this machine's profiles at other ubatches beside them. Those were measured in another session, so the lines say
    so: the comparison shows the trade, it does not settle it."""
    ub = plan.doc_ubatch(doc)
    out = []

    def fmt(best: dict) -> str:
        d, pr = best.get("best_decode"), best.get("best_prefill")
        if not d:
            return "no ok row"
        return (f"best decode {d['tg_tps']} tok/s (prefill {d['pp_tps']}), "
                f"best prefill {pr['pp_tps']} tok/s (decode {pr['tg_tps']})")

    for cell in doc.get("ubatch_report") or ubatch_report(doc.get("rows", [])):   # schema 1 and 2 have none
        out.append(f"ubatch {ub}, ctx {cell['ctx']} kv {cell['kv']}: exact-class {fmt(cell.get('exact') or {})}; "
                   f"all rows {fmt(cell.get('all') or {})}")
        for o in others:
            ou = plan.doc_ubatch(o)
            if ou == ub:
                continue
            twin = next((c for c in (o.get("ubatch_report") or []) if (c["ctx"], c["kv"]) == (cell["ctx"], cell["kv"])),
                        None)
            if twin is None:
                twin = next((c for c in ubatch_report(o.get("rows", []))
                             if (c["ctx"], c["kv"]) == (cell["ctx"], cell["kv"])), None)
            if twin is not None:
                out.append(f"  beside it, ubatch {ou} (measured {o.get('measured_utc', '?')}, another session): "
                           f"exact-class {fmt(twin.get('exact') or {})}; all rows {fmt(twin.get('all') or {})}")
    return out


# --- the diagnostics bundle ---------------------------------------------------------------------------------------

def marian_version() -> str:
    try:
        from importlib.metadata import version
        return version("marian")
    except Exception:                                    # noqa: BLE001 - not installed: read the clone's pyproject
        pass
    py = Path(__file__).resolve().parents[1] / "pyproject.toml"
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', py.read_text(encoding="utf-8")) if py.exists() else None
    return m.group(1) if m else "?"


def _forms(path: str | Path) -> set[str]:
    """The spellings a folder can appear in: as given, resolved, and on Windows the 8.3 short form (C:\\Users\\NAME~1,
    which temporary paths use) and the long form; each with either slash."""
    p = Path(path)
    forms = {str(p)}
    try:
        forms.add(str(p.resolve()))
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            for fn in (ctypes.windll.kernel32.GetShortPathNameW, ctypes.windll.kernel32.GetLongPathNameW):
                for f in list(forms):
                    buf = ctypes.create_unicode_buffer(1024)
                    if fn(f, buf, 1024):
                        forms.add(buf.value)
        except (OSError, AttributeError):
            pass
    out = set()
    for f in forms:
        out |= {f.rstrip("\\/"), f.replace("\\", "/").rstrip("/"), f.replace("/", "\\").rstrip("\\")}
    return {f for f in out if len(f) > 3}


class Scrubber:
    """Replaces the user's folders in text: named folders by placeholders, the home folder by ~, any other absolute
    folder by <dir> (the file name is kept)."""

    def __init__(self, named: dict[str, str]):
        pairs = [(f, ph) for d, ph in named.items() if d for f in _forms(d)]
        pairs += [(f, "~") for f in _forms(Path.home())]
        self.pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    def text(self, s: str) -> str:
        for old, new in self.pairs:
            s = re.sub(re.escape(old), lambda _m, n=new: n, s, flags=re.I)
        s = USERS.sub("~", s)
        s = POSIX_HOME.sub("~", s)
        s = WIN_DIR.sub("<dir>/", s)
        return POSIX_DIR.sub("<dir>/", s)

    def obj(self, o):
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, list):
            return [self.obj(x) for x in o]
        if isinstance(o, dict):
            return {k: self.obj(v) for k, v in o.items()}
        return o


def log_tail(path: Path, scrub: Scrubber, n: int = 200) -> str:
    """The last n lines of a server log, with every line that could carry a prompt or an answer dropped."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return ""
    if not lines:
        return ""
    kept = [scrub.text(x) for x in lines if not PROMPTY.search(x)]
    return "\n".join(kept) + f"\n# {len(lines) - len(kept)} of the last {len(lines)} lines dropped as possible prompt text\n"


def write_diag(out: Path, profile_dir: Path, model: Path | None, engine: LlamaCpp | None, need: int,
               port: int = 9951, ubatch: int = DEFAULT_UBATCH) -> list[str]:
    named = {str(profile_dir): "<profiles>"}
    if model is not None:
        named[str(model.resolve().parent)] = "<model-dir>"
    if engine is not None:
        named[str(engine.exe.resolve().parent)] = "<engine-dir>"
    scrub = Scrubber(named)
    m = machine.detect()
    free, used = machine.gpu_memory_free_mib(), machine.gpu_memory_used_mib()
    total, avail = machine.ram_mib()
    files: dict[str, str] = {}
    files["machine.json"] = json.dumps(scrub.obj({**m.to_dict(), "now": {
        "vram_free_mib": free, "vram_used_mib": used, "ram_total_mib": total, "ram_available_mib": avail,
        "commit_mib": machine.commit_mib()}}), indent=1)
    ver = {"marian": marian_version(), "python": platform.python_version(), "os": platform.platform(terse=True)}
    if engine is not None:
        try:
            caps = engine.capabilities()
            ver["engine"] = {"name": engine.name, "version": engine.version(), "marian": caps.get("marian"),
                             "prefill_source": caps.get("prefill_source"), "moe_flags": caps.get("moe_flags")}
        except (OSError, ValueError) as e:
            ver["engine"] = {"name": engine.name, "error": type(e).__name__}
    files["versions.json"] = json.dumps(scrub.obj(ver), indent=1)
    if model is not None:
        mid = sha256_head(model)
        docs = plan.load_docs(profile_dir, model, mid)
        for p, doc in docs:
            files[f"profiles/{p.name}"] = json.dumps(scrub.obj(doc), indent=1)
            tail = log_tail(p.with_name(p.stem + ".server.log"), scrub)
            if tail:
                files[f"logs/{p.stem}.server.log.tail.txt"] = tail
        rows = plan.rows_of(docs)
        vram_free = free[0] if free else None
        row = plan.choose(rows, need, vram_free_mib=vram_free, ram_free_mib=avail, ubatch=ubatch)
        chosen = None
        if row is not None:
            cfg = plan.to_config(row)
            chosen = {"row": {k: v for k, v in row.items() if not k.startswith("_")}, "config": asdict(cfg),
                      "server_args": LlamaCpp("llama-server", port).command(Path(model.name), cfg)[1:],
                      "exact_class": plan.exact_class(row), "stock": plan.is_stock(row), "profile_file": row["_file"],
                      **settings(cfg)}
        files["plan.json"] = json.dumps(scrub.obj({"model": model.name, "model_id": mid, "need": need,
                                                    "ubatch": ubatch, "ubatches_measured": plan.ubatches(rows),
                                                    "profile_schemas": sorted({d.get("marian_profile") or 1
                                                                               for _, d in docs}),
                                                    "vram_free_mib": vram_free, "ram_available_mib": avail,
                                                    "rows_measured": len(rows), "chosen": chosen}), indent=1)
    files["logs/serve.log.tail.txt"] = log_tail(profile_dir / "serve.log", scrub)
    files["README.txt"] = ("Marian diagnostics bundle. No prompt text: log lines that could carry a request or an "
                           "answer are dropped. No absolute paths: the home folder is ~, the model, engine and profile "
                           "folders are <model-dir>, <engine-dir> and <profiles>, any other folder is <dir>.\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    out.write_bytes(buf.getvalue())
    return sorted(files)

