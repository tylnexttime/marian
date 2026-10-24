"""marian: measure, plan and launch local LLM servers by what this machine can actually do.

    python -m marian machine
    python -m marian inspect MODEL.gguf
    python -m marian profile MODEL.gguf --engine PATH/llama-server --ctx 4096,8192,16384 --kv f16,q8_0 [--step 2]
    python -m marian plan MODEL.gguf --need 6000 [--objective pp] [--fit-now | --vram-free MIB] [--json]
    python -m marian serve MODEL.gguf --engine PATH/llama-server --need 6000 [--port 9951] [--fit-now]
    python -m marian geometry SHARD.gguf [SHARD.gguf ...] -o MODEL.geometry.json
    python -m marian replay TAG.ids.bin --geometry MODEL.geometry.json --budget-mib 9400 [--requests TAG.requests.jsonl]
                            [--trace TAG.trace.csv] [--fills TAG.fills.csv] [--md OUT.md] [--json OUT.json]
    python -m marian synth OUTDIR [--seed 1]          (a SYNTHETIC trace for trying replay without a GPU)

`--fit-now` re-checks each measured row against the VRAM free at this moment (see plan.py); `--vram-free` gives that
figure instead of measuring it. `plan --json` prints one JSON object on stdout for other programs: {"ok": true, "row",
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

from . import geometry, gguf, idstrace, machine, plan, profile, replay, synth
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
    p = sub.add_parser("geometry", help="expert geometry of a MoE GGUF (headers only) as JSON, for replay")
    p.add_argument("shards", type=Path, nargs="+"); p.add_argument("-o", "--out", type=Path)
    p = sub.add_parser("replay", help="replay a routing-id trace through expert-cache and RAM-tier policies")
    p.add_argument("ids", type=Path, help="MARIAN_MOE_TRACE_IDS file (patch 0004)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--geometry", type=Path, help="JSON from `marian geometry`")
    g.add_argument("--gguf", type=Path, nargs="+", help="the model's shards (only their headers are read)")
    p.add_argument("--budget-mib", required=True, help="--moe-cache-mib of the run to replay; a comma list replays several")
    p.add_argument("--requests", type=Path, help="run_arm's <tag>.requests.jsonl (phases, prompts); else prompts are inferred")
    p.add_argument("--trace", type=Path, help="the same run's MARIAN_MOE_TRACE CSV: compare every S line with the replay")
    p.add_argument("--fills", type=Path, help="the same run's MARIAN_MOE_TRACE_FILLS CSV: compare every upload's slot")
    p.add_argument("--ubatch", type=int, default=512); p.add_argument("--alignment", type=int, default=geometry.CUDA_ALIGNMENT)
    p.add_argument("--test-prompts", help="comma list of prompt keys held out, e.g. decode:1,decode:3 (default: every other)")
    p.add_argument("--p2", default=",".join(replay.DEFAULT_P2), help="VRAM policies compared with SLRU ('' for none)")
    p.add_argument("--tier-mib", default=",".join(map(str, replay.DEFAULT_TIERS)), help="RAM tier sizes ('' for none)")
    p.add_argument("--tier-policies", default=",".join(replay.TIER_POLICIES))
    p.add_argument("--ring-mib", type=float, default=64); p.add_argument("--sector", type=int, default=4096)
    p.add_argument("--label", default=""); p.add_argument("--md", type=Path); p.add_argument("--json", type=Path)
    p = sub.add_parser("synth", help="write a SYNTHETIC routing trace (ids, requests, geometry) to try replay on")
    p.add_argument("out", type=Path)
    for k, v in synth.SynthSpec().__dict__.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)

    if a.cmd == "machine":
        print(json.dumps(machine.detect().to_dict(), indent=1))
    elif a.cmd == "geometry":
        geom = geometry.from_gguf(a.shards)
        if a.out:
            geom.save(a.out)
        groups = geom.groups()
        print(json.dumps({"model": geom.model, "n_expert_used": geom.n_expert_used, "layers": len(geom.layers),
                          "expert_gib": round(geom.total_expert_bytes / 2**30, 3),
                          "groups": [{"type": g.type, "layers": len(g.layers), "expert_bytes": g.expert_bytes} for g in groups]}, indent=1))
    elif a.cmd == "replay":
        return _replay(a)
    elif a.cmd == "synth":
        spec = synth.SynthSpec(**{k: getattr(a, k) for k in synth.SynthSpec().__dict__})
        print(json.dumps(synth.write(a.out, spec)))
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


def _num(x: str):
    v = float(x)
    return int(v) if v.is_integer() else v


def _replay(a) -> int:
    geom = geometry.load(a.geometry) if a.geometry else geometry.from_gguf(a.gguf)
    trace = idstrace.read_ids(a.ids, expected_k=geom.n_expert_used)
    if a.requests:
        idstrace.assign_requests(trace, idstrace.read_requests(a.requests))
    else:
        idstrace.infer_requests(trace)
    csv_rows = idstrace.read_csv(a.trace) if a.trace else None
    if csv_rows is not None:
        ok, joined, bad = idstrace.check_heads(trace, csv_rows)
        print(f"record heads: {ok} of {joined} S records agree with their S line on n_tokens x k", file=sys.stderr)
        if ok != joined:
            for b in bad:
                print(f"  {b}", file=sys.stderr)
            print("refusing to replay a trace whose record heads disagree with the server's own trace", file=sys.stderr)
            return 2
    fill_rows = idstrace.read_csv(a.fills) if a.fills else None
    split = lambda x: [y for y in x.split(",") if y]                                    # noqa: E731
    reps, md = [], []
    for budget in split(a.budget_mib):
        try:
            rep = replay.run_all(trace, geom, _num(budget), n_ubatch=a.ubatch, alignment=a.alignment,
                                 test_keys=split(a.test_prompts) if a.test_prompts else None, p2=tuple(split(a.p2)),
                                 tiers=tuple(_num(x) for x in split(a.tier_mib)), tier_policies=tuple(split(a.tier_policies)),
                                 ring_mib=a.ring_mib, sector=a.sector, csv_rows=csv_rows, fill_rows=fill_rows,
                                 progress=lambda m: print(f"  {m}", file=sys.stderr))
        except geometry.BudgetBelowFloor as e:
            print(str(e), file=sys.stderr)
            return 2
        reps.append(rep)
        md.append(replay.to_markdown(rep, a.label))
    text = "\n".join(md)
    print(text)
    if a.md:
        a.md.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(reps if len(reps) > 1 else reps[0], indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
