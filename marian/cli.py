"""marian: measure, plan and launch local LLM servers by what this machine can actually do.

    python -m marian machine
    python -m marian inspect MODEL.gguf
    python -m marian profile MODEL.gguf --engine PATH/llama-server --ctx 4096,8192,16384 --kv f16,q8_0 [--step 2]
    python -m marian plan MODEL.gguf --need 6000 [--objective pp]
    python -m marian serve MODEL.gguf --engine PATH/llama-server --need 6000 [--port 9951]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import gguf, machine, plan, profile
from .engines import LlamaCpp

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marian")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("machine")
    p = sub.add_parser("inspect"); p.add_argument("model", type=Path)
    p = sub.add_parser("profile"); p.add_argument("model", type=Path); p.add_argument("--engine", required=True)
    p.add_argument("--ctx", default="4096,8192,16384,32768"); p.add_argument("--kv", default="f16,q8_0")
    p.add_argument("--step", type=int, default=2); p.add_argument("--threads", type=int)
    p.add_argument("--n-predict", type=int, default=64); p.add_argument("--port", type=int, default=9951)
    p = sub.add_parser("plan"); p.add_argument("model", type=Path); p.add_argument("--need", type=int, required=True)
    p.add_argument("--objective", choices=["tg", "pp"], default="tg")
    p = sub.add_parser("serve"); p.add_argument("model", type=Path); p.add_argument("--engine", required=True)
    p.add_argument("--need", type=int, required=True); p.add_argument("--objective", choices=["tg", "pp"], default="tg")
    p.add_argument("--port", type=int, default=9951)
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
        row = plan.choose(rows, a.need, a.objective)
        if row is None:
            print(f"no measured configuration fits {a.need} tokens on this machine for {a.model.name}; "
                  "run `marian profile` with a larger --ctx first", file=sys.stderr)
            return 2
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("_")}))
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


if __name__ == "__main__":
    sys.exit(main())
