"""marian: measure, plan and launch local LLM servers by what this machine can actually do.

    python -m marian machine
    python -m marian inspect MODEL.gguf
    python -m marian profile MODEL.gguf --engine PATH/llama-server --ctx 4096,8192,16384 --kv f16,q8_0 [--step 2]
    python -m marian plan MODEL.gguf --need 6000 [--objective pp] [--fit-now | --vram-free MIB] [--json]
    python -m marian serve MODEL.gguf --engine PATH/llama-server --need 6000 [--port 9951] [--fit-now]

`--fit-now` re-checks each measured row against the VRAM free at this moment (see plan.py); `--vram-free` gives that
figure instead of measuring it. `--json` prints one JSON object on stdout for other programs: {"ok": true, "row",
"config", "server_args", "vram_basis", ...} or {"ok": false, "reason", ...}; the exit code is 0 or 2 as without it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import gguf, machine, plan, profile
from .engines import LlamaCpp

def profiles_dir() -> Path:
    """Where profiles live: $MARIAN_PROFILES if set; the repository's profiles/ when running from a clone; else
    ~/.marian/profiles (an installed package must not write next to its own code in site-packages)."""
    if os.environ.get("MARIAN_PROFILES"):
        return Path(os.environ["MARIAN_PROFILES"])
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists() and (root / "marian" / "cli.py").exists():
        return root / "profiles"
    return Path.home() / ".marian" / "profiles"


PROFILES = profiles_dir()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marian")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("machine")
    p = sub.add_parser("inspect"); p.add_argument("model", type=Path)
    p = sub.add_parser("profile"); p.add_argument("model", type=Path); p.add_argument("--engine", required=True)
    p.add_argument("--ctx", default="4096,8192,16384,32768"); p.add_argument("--kv", default="f16,q8_0")
    p.add_argument("--step", type=int, default=2); p.add_argument("--threads", type=int)
    p.add_argument("--n-predict", type=int, default=64); p.add_argument("--port", type=int, default=9951)
    for name in ("plan", "serve"):
        p = sub.add_parser(name); p.add_argument("model", type=Path); p.add_argument("--need", type=int, required=True)
        p.add_argument("--objective", choices=["tg", "pp"], default="tg")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--fit-now", action="store_true", help="re-check each row against the VRAM free now")
        g.add_argument("--vram-free", type=int, metavar="MIB", help="re-check each row against this much free VRAM")
        p.add_argument("--margin", type=int, default=300, metavar="MIB", help="VRAM kept free on top of a row's share")
        p.add_argument("--port", type=int, default=9951)
        if name == "plan":
            p.add_argument("--json", action="store_true", help="one JSON object on stdout, for other programs")
        else:
            p.add_argument("--engine", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "machine":
        print(json.dumps(machine.detect().to_dict(), indent=1))
    elif a.cmd == "inspect":
        print(json.dumps(gguf.summary(a.model), indent=1))
    elif a.cmd == "profile":
        eng = LlamaCpp(a.engine, a.port)
        out = profile.run(a.model, eng, [int(x) for x in a.ctx.split(",")], a.kv.split(","), a.step, a.threads,
                          PROFILES, a.n_predict)
        print(f"profile written: {out}")
    elif a.cmd in ("plan", "serve"):
        rows = plan.load(PROFILES, a.model)
        free = a.vram_free
        if a.fit_now:
            now = machine.gpu_memory_free_mib()
            free = now[0] if now else None
            if free is None:
                return _plan_fail(a, "--fit-now: could not read free VRAM (no nvidia-smi?)", rows, None)
        row = plan.choose(rows, a.need, a.objective, vram_free_mib=free, margin_mib=a.margin)
        if row is None:
            if not rows:
                why = f"no profile for {a.model.name} on this machine; run `marian profile` first"
            elif free is not None and plan.choose(rows, a.need, a.objective) is not None:
                why = (f"measured configurations fit {a.need} tokens, but none fits the {free} MiB of VRAM free now "
                       f"with a {a.margin} MiB margin")
            else:
                why = (f"no measured configuration fits {a.need} tokens on this machine for {a.model.name}; "
                       "run `marian profile` with a larger --ctx first")
            return _plan_fail(a, why, rows, free)
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        if getattr(a, "json", False):
            cfg = plan.to_config(row)
            print(json.dumps({"ok": True, "model": a.model.name, "need": a.need, "objective": a.objective,
                              "vram_free_mib": free, "margin_mib": a.margin if free is not None else None,
                              "vram_basis": plan.vram_share(row)[1], "row": clean, "config": asdict(cfg),
                              "server_args": LlamaCpp("llama-server", a.port).command(a.model, cfg)[1:],
                              "profile_file": row.get("_file")}))
            return 0
        print(json.dumps(clean))
        if a.cmd == "serve":
            eng = LlamaCpp(a.engine, a.port)
            ok, why = eng.start(a.model, plan.to_config(row), PROFILES / "serve.log")
            print(why)
            if not ok:
                return 2
            print(f"serving {a.model.name} on {eng.base} (Ctrl+C stops)")
            try:
                while eng.proc and eng.proc.poll() is None:
                    time.sleep(2)
            except KeyboardInterrupt:
                pass
            finally:
                eng.stop()
    return 0


def _plan_fail(a, why: str, rows: list[dict], free) -> int:
    if getattr(a, "json", False):
        print(json.dumps({"ok": False, "model": a.model.name, "need": a.need, "objective": a.objective,
                          "vram_free_mib": free, "reason": why, "rows_measured": len(rows)}))
    else:
        print(why, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
