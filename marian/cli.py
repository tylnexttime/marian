"""marian: measure, plan and launch local LLM servers by what this machine can actually do.

    python -m marian machine
    python -m marian inspect MODEL.gguf
    python -m marian profile MODEL.gguf --engine PATH/llama-server --ctx 4096,8192,16384 --kv f16,q8_0 [--step 2]
    python -m marian calibrate MODEL.gguf --engine PATH/llama-server [--ctx 4096] [--kv f16] [--ubatch 512] [--repeats 1]
                               (the quick profile)
    python -m marian recalibrate MODEL.gguf --engine PATH/llama-server [--ubatch 512] [--repeats N]
                                 (the same calibration again, unchanged; the old profile is kept beside the new)
    python -m marian plan MODEL.gguf --need 6000 [--objective pp] [--fit-now | --vram-free MIB] [--exact-only | --stock]
                          [--ubatch 512] [--json]
    python -m marian serve MODEL.gguf --engine PATH/llama-server --need 6000 [--port 9951] [--calibrate] [--exact-only |
                           --stock] [--ubatch 512] [--confirm] [--governor watch|restart|off]
                           [--cache-prompt on --gate-prefix FILE] [--dry-run] [--json]
    python -m marian diag OUT.zip [MODEL.gguf] [--engine PATH/llama-server] [--need 6000] [--ubatch 512]
    python -m marian geometry SHARD.gguf [SHARD.gguf ...] -o MODEL.geometry.json
    python -m marian replay TAG.ids.bin --geometry MODEL.geometry.json --budget-mib 9400 [--requests TAG.requests.jsonl]
                            [--trace TAG.trace.csv] [--fills TAG.fills.csv] [--md OUT.md] [--json OUT.json]
    python -m marian synth OUTDIR [--seed 1]          (a SYNTHETIC trace for trying replay without a GPU)
    python -m marian predict TAG.ids.bin --geometry MODEL.geometry.json --budget-mib 9400 --requests TAG.requests.jsonl
                             [--trace TAG.trace.csv] [--check-against OTHER.ids.bin] [--train decode:0,...] [--test ...]
                             [--permutations 100] [--seed 47] [--md OUT.md] [--json OUT.json]      (register E47)

`--fit-now` re-checks each measured row against the VRAM free at this moment (see plan.py); `--vram-free` gives that
figure instead of measuring it. `plan --json` prints one JSON object on stdout for other programs: {"ok": true, "row",
"config", "server_args", "vram_basis", ...} or {"ok": false, "reason", ...}; the exit code is 0 or 2 as without it.

`serve` (M1.3) refuses without a profile for this machine, model and engine unless `--calibrate` runs the quick one
first, always checks the chosen row against the VRAM and RAM free now, and runs the launch checks of launch.py; a
failure is a reason and exit 2 (`--json`: one object). `--stock` is the rollback: plain upstream placement only.
Then it serves through a front proxy on --port (llama-server on --internal-port, default --port + 1): one sequence at a
time, a memory governor (`--governor watch` by default: it warns, and pauses only on low commit headroom;
`restart` also restarts on a measured slowdown), the passive live check, and prefix reuse off until E52 passes
(serve.py); `--confirm` first A/B-tests the plan against the runner-up (confirm.py).

`--ubatch N` declares the deployment's ubatch (default 512, no flag; `-ub N -b N` otherwise). It is part of the
exactness reference (register E54), so it is declared, never chosen: `calibrate --ubatch N` measures every row at N
into its own profile, and `plan` and `serve` use only rows measured at the ubatch they declare, refusing otherwise.

`--repeats N` (1 to 3, default 1) probes each row N times on the same server and keeps every reading, the median
(the row's figure) and the spread. `recalibrate` re-runs a profile's own calibration (its contexts, KV types, probe,
repeats and cache options, read from the profile) as a new session: the old profile is kept beside it, named by when
it was measured, and the configurations whose decode moved more than 15% between the two are listed. Chris
(30/10/2026): one probe is the default, repetition an option, and re-calibration the remedy for a stale or lucky
profile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import geometry, gguf, idstrace, machine, plan, predict, profile, replay, synth
from . import confirm, launch, serve
from .engines import DEFAULT_UBATCH, LlamaCpp
from .profile import CacheOptions, ThreadOptions, sha256_head


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
    _cache_args(p, tier_steps=3)
    p = sub.add_parser("calibrate", help="the quick first-launch profile: one context, one KV type, cache arms on a Marian build")
    p.add_argument("model", type=Path); p.add_argument("--engine", required=True)
    p.add_argument("--ctx", type=int, default=4096); p.add_argument("--kv", default="f16")
    p.add_argument("--step", type=int, default=2); p.add_argument("--threads", type=int)
    p.add_argument("--n-predict", type=int, default=32); p.add_argument("--port", type=int, default=9951)
    _cache_args(p, tier_steps=2)
    p = sub.add_parser("recalibrate", help="run this machine's calibration of the model again, unchanged, as a new session")
    p.add_argument("model", type=Path); p.add_argument("--engine", required=True)
    p.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, metavar="N", help="which profile: the one at this ubatch")
    p.add_argument("--repeats", type=int, choices=range(1, profile.REPEATS_MAX + 1), metavar="N",
                   help="probes per row (1-3); default: what the profile was calibrated with")
    p.add_argument("--port", type=int, default=9951)
    p.add_argument("--json", action="store_true", help="print the old-against-new comparison as JSON")
    for name in ("plan", "serve"):
        p = sub.add_parser(name); p.add_argument("model", type=Path); p.add_argument("--need", type=int, required=True)
        p.add_argument("--objective", choices=["tg", "pp"], default="tg")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--fit-now", action="store_true", help="re-check each row against the VRAM free now")
        g.add_argument("--vram-free", type=int, metavar="MIB", help="re-check each row against this much free VRAM")
        p.add_argument("--margin", type=int, default=300, metavar="MIB", help="VRAM kept free on top of a row's share")
        p.add_argument("--ram-floor", type=int, default=6144, metavar="MIB", help="RAM left available beside pinned memory")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--exact-only", action="store_true", help="only rows that compute every expert on the GPU")
        g.add_argument("--stock", action="store_true", help="only plain upstream placement (-ncmoe/-ngl): the rollback")
        p.add_argument("--port", type=int, default=9951)
        p.add_argument("--json", action="store_true", help="one JSON object on stdout, for other programs")
        p.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, metavar="N",
                       help="the deployment's declared ubatch; only rows measured at it are used (default 512)")
        if name == "serve":
            p.add_argument("--engine", required=True)
            p.add_argument("--calibrate", action="store_true", help="no profile yet: run the quick calibration, then serve")
            p.add_argument("--dry-run", action="store_true", help="gate, plan and launch checks only; start nothing")
            p.add_argument("--internal-port", type=int, help="llama-server's own port (default: --port + 1)")
            p.add_argument("--allow-non-exact", action="store_true",
                           help="let --confirm and the governor move an exact-class plan to a non-exact row")
            p.add_argument("--confirm", action="store_true", help="A/B the plan against the runner-up before serving")
            p.add_argument("--confirm-rounds", type=int, default=4, metavar="K", help="interleaved A/B rounds")
            p.add_argument("--slow", type=float, default=0.15, metavar="X",
                           help="report a live decode rate more than this fraction below the profile")
            p.add_argument("--live-n", type=int, default=8, metavar="N", help="answers the live check reads (0: off)")
            p.add_argument("--cache-prompt", choices=["off", "on"], default="on" if serve.CACHE_PROMPT_DEFAULT else "off",
                           help="llama-server's prompt-prefix reuse (off until register test E52 passes)")
            p.add_argument("--gate-prefix", type=Path, metavar="FILE",
                           help="prime the slot with this fixed prompt prefix after every start (needs --cache-prompt on)")
            p.add_argument("--queue", type=int, default=1, help="requests that may wait behind the running one")
            p.add_argument("--queue-timeout", type=float, default=300, metavar="S")
            p.add_argument("--governor", choices=list(serve.GOVERNOR_LINE), default="watch",
                           help="watch: warn on low memory, pause only on low commit headroom (default); restart: "
                                "also restart smaller on a sustained decode slowdown, and step back up; off: none")
            p.add_argument("--interval", type=float, default=2.0, metavar="S", help="governor reading interval")
            p.add_argument("--persist", type=float, default=10.0, metavar="S",
                           help="--governor restart: a decode slowdown lasting this long restarts on a smaller plan")
            p.add_argument("--step-up", type=float, default=60.0, metavar="S",
                           help="--governor restart: memory fitting the planned row this long steps back up to it")
            p.add_argument("--drain", type=float, default=30.0, metavar="S", help="wait this long for the running request")
            p.add_argument("--max-restarts", type=int, default=3)
    p = sub.add_parser("diag", help="a diagnostics bundle (zip): machine, profiles, plan, log tails; no prompts or paths")
    p.add_argument("out", type=Path); p.add_argument("model", type=Path, nargs="?")
    p.add_argument("--engine"); p.add_argument("--need", type=int, default=4096); p.add_argument("--port", type=int, default=9951)
    p.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, metavar="N", help="the declared ubatch the plan is for")
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
    p = sub.add_parser("predict", help="how predictable is a trace's expert routing, and what is perfect prediction worth (E47)")
    p.add_argument("ids", type=Path, help="MARIAN_MOE_TRACE_IDS file (patch 0004)")
    p.add_argument("--geometry", type=Path, required=True, help="JSON from `marian geometry`")
    p.add_argument("--budget-mib", type=float, required=True, help="--moe-cache-mib of the run (the slots of measure 3)")
    p.add_argument("--requests", type=Path, required=True, help="run_arm's <tag>.requests.jsonl (phases and prompts)")
    p.add_argument("--trace", type=Path, help="the same run's MARIAN_MOE_TRACE CSV: the timings of the oracle (measure 5)")
    p.add_argument("--train", default=",".join(predict.DEFAULT_TRAIN), help="training prompts (decode:N or N)")
    p.add_argument("--test", default=",".join(predict.DEFAULT_TEST), help="test prompts (decode:N or N)")
    p.add_argument("--labels", default=",".join(predict.DEFAULT_LABELS), help="names of decode prompts 0, 1, ...")
    p.add_argument("--check-against", type=Path, help="another run's ids trace of the same prompts: its decode ids must equal these")
    p.add_argument("--check-requests", type=Path, help="that run's requests.jsonl (default: <tag>.requests.jsonl beside it)")
    p.add_argument("--permutations", type=int, default=100); p.add_argument("--seed", type=int, default=47)
    p.add_argument("--bandwidths", default=",".join(f"{k}={v:g}" for k, v in predict.BANDWIDTHS.items()),
                   help="copy sources for the oracle, GB/s (1e9 bytes/s) as the copy probe prints them")
    p.add_argument("--pinned", default="pinned", help="the source a request-horizon prediction stages into")
    p.add_argument("--reference-tok-s", type=float, default=predict.REFERENCE_TOK_S, help="measured decode the oracle rule compares with")
    p.add_argument("--ubatch", type=int, default=512); p.add_argument("--alignment", type=int, default=geometry.CUDA_ALIGNMENT)
    p.add_argument("--md", type=Path); p.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if getattr(a, "ubatch", 1) < 1 and a.cmd not in ("replay", "predict"):
        print(f"--ubatch {a.ubatch}: a ubatch is a positive token count", file=sys.stderr)
        return 2

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
    elif a.cmd == "predict":
        return _predict(a)
    elif a.cmd == "synth":
        spec = synth.SynthSpec(**{k: getattr(a, k) for k in synth.SynthSpec().__dict__})
        print(json.dumps(synth.write(a.out, spec)))
    elif a.cmd == "inspect":
        print(json.dumps(gguf.summary(a.model), indent=1))
    elif a.cmd == "profile":
        eng = LlamaCpp(a.engine, a.port)
        out = profile.run(a.model, eng, [int(x) for x in a.ctx.split(",")], a.kv.split(","), a.step, a.threads,
                          PROFILES, a.n_predict, cache=_cache_opts(a), sweep=ThreadOptions(enabled=not a.no_thread_sweep),
                          ubatch=a.ubatch, probe_prompt=a.probe_prompt, repeats=a.repeats)
        print(f"profile written: {out}")
        _report_trade(out)
    elif a.cmd == "calibrate":
        eng = LlamaCpp(a.engine, a.port)
        out = profile.run(a.model, eng, [a.ctx], [a.kv], a.step, a.threads, PROFILES, a.n_predict, cache=_cache_opts(a),
                          sweep=ThreadOptions(enabled=not a.no_thread_sweep), ubatch=a.ubatch,
                          probe_prompt=a.probe_prompt, repeats=a.repeats)
        print(f"calibration written: {out}")
        _report_trade(out)
    elif a.cmd == "recalibrate":
        return _recalibrate(a)
    elif a.cmd == "diag":
        eng = LlamaCpp(a.engine, a.port) if a.engine else None
        names = launch.write_diag(a.out, PROFILES, a.model, eng, a.need, a.port, ubatch=a.ubatch)
        print(f"diagnostics written: {a.out} ({len(names)} files; no prompt text, home shown as ~)")
    elif a.cmd in ("plan", "serve"):
        return _plan_or_serve(a)
    return 0


def _cache_args(p, tier_steps: int) -> None:
    p.add_argument("--no-cache", action="store_true", help="no cache arms, even on a Marian build")
    p.add_argument("--max-cache-mib", type=int, help="never offer the expert cache more VRAM than this")
    p.add_argument("--ram-floor", type=int, default=6144, metavar="MIB", help="never pin more than (RAM available - this)")
    p.add_argument("--tier-steps", type=int, default=tier_steps, help="pinned RAM tier sizes to measure")
    p.add_argument("--drift-rungs", type=int, default=2, metavar="N",
                   help="cache budgets measured below the best, 512 MiB apart, so a small VRAM drift costs one rung")
    p.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, metavar="N",
                   help="the deployment's declared ubatch: every row is measured at -ub N -b N (512: no flag)")
    p.add_argument("--no-thread-sweep", action="store_true", help="no CPU thread ladder for rows that compute on the CPU")
    p.add_argument("--probe-prompt", type=int, metavar="TOKENS",
                   help="probe prefill at the deployment's prompt length (default: 75%% of the context)")
    p.add_argument("--repeats", type=int, default=1, choices=range(1, profile.REPEATS_MAX + 1), metavar="N",
                   help="probe each row N times (1-3) on the same server; the row keeps the median and the spread")


def _recalibrate(a) -> int:
    """Run the existing profile's calibration again with its own parameters; keep the old profile beside the new."""
    eng = LlamaCpp(a.engine, a.port)
    stem = f"{machine.detect().fingerprint()}-{sha256_head(a.model)}-{eng.name}"
    path = PROFILES / profile.profile_name(stem, a.ubatch)
    if not path.exists():
        print(f"no profile to recalibrate: {path.name} is not in {PROFILES}; run `marian calibrate` first",
              file=sys.stderr)
        return 2
    old = json.loads(path.read_text(encoding="utf-8"))
    k = profile.calibration_params(old)
    if not k["recorded"]:
        print("this profile predates recorded calibration parameters: contexts, KV types and the probe are read from "
              "its rows; the walk step, threads and cache options take calibrate's defaults")
    keep = profile.archive_name(path, old)
    path.replace(keep)
    copt = CacheOptions(**k["cache"]) if k["cache"] else CacheOptions(tier_steps=2)
    try:
        out = profile.run(a.model, eng, k["ctx"], k["kv"], k["step"], k["threads"], PROFILES, k["n_predict"],
                          vram_margin_mib=k["vram_margin_mib"], collapse=k["collapse"], cache=copt,
                          sweep=ThreadOptions(enabled=k["thread_sweep"]), ubatch=a.ubatch,
                          probe_prompt=k["probe_prompt"], repeats=a.repeats or k["repeats"])
    except BaseException:
        if not path.exists():
            keep.replace(path)                           # a failed run leaves the old profile where the planner reads it
        raise
    new = json.loads(out.read_text(encoding="utf-8"))
    cmp = profile.compare(old, new)
    if a.json:
        print(json.dumps(dict(cmp, profile=out.name, previous=keep.name)))
    else:
        print(f"recalibration written: {out}; the previous session is kept as {keep.name}")
        print(f"{cmp['matched']} configurations measured in both sessions; {cmp['moved']} moved more than "
              f"{cmp['band']:.0%} on decode")
        for r in cmp["rows"]:
            if r["moved"]:
                cfg = {x: v for x, v in r["config"].items() if v is not None}
                print(f"  {r['old_tg_tps']} -> {r['new_tg_tps']} tok/s ({r['ratio']:.2f}x)  {cfg}")
        _report_trade(out)
    return 0


def _report_trade(out: Path) -> None:
    """Decode and prefill at the calibrated ubatch, beside this machine's profiles of the model at other ubatches."""
    try:
        doc = json.loads(out.read_text(encoding="utf-8"))
        others = [d for p, d in plan.load_docs(out.parent, Path((doc.get("model") or {}).get("file", "")),
                                               (doc.get("model") or {}).get("id"))
                  if p != out and (d.get("engine") or {}).get("name") == (doc.get("engine") or {}).get("name")]
    except (OSError, ValueError):
        return
    for line in launch.trade_lines(doc, others):
        print(line)


def _cache_opts(a) -> CacheOptions:
    return CacheOptions(enabled=not a.no_cache, max_cache_mib=a.max_cache_mib, ram_floor_mib=a.ram_floor,
                        tier_steps=a.tier_steps, drift_rungs=a.drift_rungs)


def _plan_or_serve(a) -> int:
    serve_cmd = a.cmd == "serve"
    mid = sha256_head(a.model)
    docs = plan.load_docs(PROFILES, a.model, mid)
    internal = (a.internal_port or a.port + 1) if serve_cmd else a.port
    eng = LlamaCpp(a.engine, internal) if serve_cmd else None
    if serve_cmd and a.gate_prefix is not None and a.cache_prompt != "on":
        return _plan_fail(a, "--gate-prefix primes llama-server's prompt reuse, which is off: prefix reuse stays opt-in "
                             "until register test E52 shows it exact; add --cache-prompt on to opt in", [], None)
    for p, d in docs:
        why = plan.schema_problem(d)
        if why:
            return _plan_fail(a, f"{p.name}: {why}", [], None, gate="schema")
    eng_rows = plan.rows_of(docs, eng.name) if serve_cmd else []
    if serve_cmd and not any(plan.ubatch_of(r) == a.ubatch for r in eng_rows):     # the calibration gate
        if not a.calibrate:
            measured = plan.ubatches(eng_rows) if eng_rows else None
            return _plan_fail(a, launch.gate_reason(a.model, eng.name, a.ubatch, measured), [], None,
                              gate="ubatch" if measured else "no_calibration")
        ctx = launch.quick_ctx(a.need)
        print(f"no calibration for this machine, model and engine at ubatch {a.ubatch} yet: calibrating at context "
              f"{ctx} first", file=sys.stderr)
        profile.run(a.model, eng, [ctx], ["f16"], 2, None, PROFILES, 32,
                    cache=CacheOptions(ram_floor_mib=a.ram_floor, tier_steps=2), ubatch=a.ubatch)
        docs = plan.load_docs(PROFILES, a.model, mid)
    rows = plan.rows_of(docs, eng.name if eng else None)
    free, ram_free = a.vram_free, None
    if a.fit_now or serve_cmd:                   # serve always checks the row against the memory free now
        if free is None:
            now = machine.gpu_memory_free_mib()
            free = now[0] if now else None
            if free is None and a.fit_now:
                return _plan_fail(a, "--fit-now: could not read free VRAM (no nvidia-smi?)", rows, None)
        ram_free = machine.ram_mib()[1]
    filters = dict(exact_only=a.exact_only, stock=a.stock)
    row = plan.choose(rows, a.need, a.objective, vram_free_mib=free, margin_mib=a.margin, ram_free_mib=ram_free,
                      ram_floor_mib=a.ram_floor, ubatch=a.ubatch, **filters)
    if row is None:
        why = launch.explain(rows, a.need, a.objective, a.model.name, free=free, margin=a.margin, ram_free=ram_free,
                             ram_floor=a.ram_floor, ubatch=a.ubatch, **filters)
        return _plan_fail(a, why, rows, free)
    warnings: list[str] = []
    if serve_cmd:
        if a.cache_prompt == "off" and "--no-cache-prompt" in eng.capabilities().get("flags", []):
            eng.server_extra = ["--no-cache-prompt"]
        doc = next(d for p, d in docs if p.name == row["_file"])
        reasons, warnings = launch.check(row, doc, a.model, mid, eng, stock=a.stock, vram_free_mib=free,
                                         margin_mib=a.margin, ram_free_mib=ram_free, ram_floor_mib=a.ram_floor,
                                         ubatch=a.ubatch)
        if reasons:
            return _plan_fail(a, "launch checks failed: " + "; ".join(reasons), rows, free,
                              reasons=reasons, warnings=warnings, row=_clean(row))
    cfg = plan.to_config(row)
    if a.json:
        print(json.dumps({"ok": True, "model": a.model.name, "need": a.need, "objective": a.objective,
                          "vram_free_mib": free, "margin_mib": a.margin if free is not None else None,
                          "ram_available_mib": ram_free, "vram_basis": plan.vram_share(row)[1],
                          "exact_class": plan.exact_class(row), "stock": plan.is_stock(row), "warnings": warnings,
                          **launch.settings(cfg),
                          "row": _clean(row), "config": asdict(cfg),
                          "server_args": _server_args(a, eng, cfg),
                          "port": a.port, "internal_port": internal if serve_cmd else None,
                          "cache_prompt": (a.cache_prompt == "on") if serve_cmd else None,
                          "governor": a.governor if serve_cmd else None,
                          "profile_file": row.get("_file")}))
    else:
        print(json.dumps(_clean(row)))
        print(launch.settings_line(cfg), file=sys.stderr)
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        if not plan.exact_class(row):
            print("note: this configuration computes some experts on the CPU, so its answers can differ from "
                  "all-on-GPU (--exact-only excludes it)", file=sys.stderr)
    if not serve_cmd or a.dry_run:
        return 0
    return _serve(a, eng, rows, docs, row, mid, free, ram_free, filters)


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _server_args(a, eng, cfg) -> list[str]:
    """llama-server's arguments without the executable, as a program would launch them (the user-facing port for
    `plan`, the internal port for `serve`, which puts its proxy on --port)."""
    e = LlamaCpp("llama-server", eng.port if eng is not None else a.port)
    if eng is not None:
        e.server_extra = list(eng.server_extra)
    return e.command(a.model, cfg)[1:]


def _serve(a, eng, rows, docs, row, mid, free, ram_free, filters) -> int:
    say = lambda m: print(m, file=sys.stderr if a.json else sys.stdout, flush=True)       # noqa: E731
    log = PROFILES / "serve.log"
    allow = a.allow_non_exact or a.stock
    if a.confirm:
        ru = plan.runner_up(rows, row, a.need, a.objective, free, a.margin, ram_free_mib=ram_free,
                            ram_floor_mib=a.ram_floor, allow_non_exact=allow, ubatch=a.ubatch, **filters)
        copt = confirm.ConfirmOptions(rounds=a.confirm_rounds, slow=a.slow)
        if ru is None:
            say("confirm: no runner-up under the same filters (and exact class); the plan stands unconfirmed")
            d = None
        else:
            say(f"confirm: {confirm.describe(row)} (plan) against {confirm.describe(ru)} (runner-up), "
                f"{a.confirm_rounds} rounds")
            d = confirm.ab(eng, a.model, row, ru, log, copt, say=say)
            if "live_check" in d:
                say(d["live_check"]["message"])
        recs = confirm.records(row, ru, d, copt)
        for r, rec in recs:
            r["confirmed"] = rec
        confirm.write_back(PROFILES, [(r, "confirmed", rec) for r, rec in recs])
        if d is not None:
            say(f"confirm: the runner-up won {d['runner_up_wins']} of {d['rounds']} rounds (needs {d['wins_needed']}), "
                f"median ratio {d['median_ratio']}: "
                + ("serving the runner-up" if d["outcome"] == "replaced" else "keeping the plan"))
            if d["outcome"] == "replaced":
                doc = next(dd for p, dd in docs if p.name == ru["_file"])
                reasons, _ = launch.check(ru, doc, a.model, mid, eng, stock=a.stock, vram_free_mib=free,
                                          margin_mib=a.margin, ram_free_mib=ram_free, ram_floor_mib=a.ram_floor,
                                          ubatch=a.ubatch)
                if reasons:
                    return _plan_fail(a, "launch checks failed for the confirmed runner-up: " + "; ".join(reasons),
                                      rows, free, reasons=reasons)
                row = ru
    gate_text = a.gate_prefix.read_text(encoding="utf-8") if a.gate_prefix is not None else None
    opt = serve.ServeOptions(port=a.port, internal_port=eng.port, need=a.need, objective=a.objective,
                             margin_mib=a.margin, ram_floor_mib=a.ram_floor, queue=a.queue,
                             queue_timeout_s=a.queue_timeout, governor=a.governor, interval_s=a.interval,
                             persist_s=a.persist, step_up_s=a.step_up,
                             drain_s=a.drain, max_restarts=a.max_restarts, cache_prompt=a.cache_prompt == "on",
                             gate_prefix=gate_text, live_n=a.live_n, slow=a.slow, allow_non_exact=allow,
                             ubatch=a.ubatch, **filters)
    try:
        sess, why = serve.open_session(eng, a.model, rows, row, opt, log, profile_dir=PROFILES, say=say)
    except OSError as e:
        eng.stop()
        return _plan_fail(a, f"could not open port {a.port} for the front proxy: {e}", rows, free)
    if sess is None:
        say(why)
        return 2
    say(f"serving {a.model.name} on http://127.0.0.1:{a.port} (llama-server on {eng.port}; one request at a time; "
        f"prompt reuse {'on' if opt.cache_prompt else 'off'}; Ctrl+C stops)")
    try:
        while not sess.sup.done.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()
    if sess.sup.state == "gave_up":
        say(f"stopped: {sess.sup.reason}")
        return 2
    return 0


def _plan_fail(a, why: str, rows: list[dict], free, **extra) -> int:
    if getattr(a, "json", False):
        print(json.dumps({"ok": False, "model": a.model.name, "need": a.need, "objective": a.objective,
                          "vram_free_mib": free, "reason": why, "rows_measured": len(rows), **extra}))
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


def _predict(a) -> int:
    geom = geometry.load(a.geometry)
    trace = idstrace.read_ids(a.ids, expected_k=geom.n_expert_used)
    idstrace.assign_requests(trace, idstrace.read_requests(a.requests))
    csv_rows = idstrace.read_csv(a.trace) if a.trace else None
    if csv_rows is not None:
        ok, joined, bad = idstrace.check_heads(trace, csv_rows)
        print(f"record heads: {ok} of {joined} S records agree with their S line on n_tokens x k", file=sys.stderr)
        if ok != joined:
            for b in bad:
                print(f"  {b}", file=sys.stderr)
            print("refusing to read a trace whose record heads disagree with the server's own trace", file=sys.stderr)
            return 2
    other = None
    if a.check_against:
        other = idstrace.read_ids(a.check_against, expected_k=geom.n_expert_used)
        req = a.check_requests
        if req is None and a.check_against.name.endswith(".ids.bin"):
            req = a.check_against.with_name(a.check_against.name[:-len(".ids.bin")] + ".requests.jsonl")
        if req is not None and req.exists():
            idstrace.assign_requests(other, idstrace.read_requests(req))
        else:
            print("--check-against: no request log beside it; its requests are inferred and will not match by phase",
                  file=sys.stderr)
            idstrace.infer_requests(other)
    split = lambda x: [y.strip() for y in x.split(",") if y.strip()]                     # noqa: E731
    bw = {k.strip(): float(v) for k, v in (x.split("=", 1) for x in split(a.bandwidths))}
    if a.pinned not in bw:
        print(f"--pinned {a.pinned!r} is not one of the --bandwidths ({', '.join(bw)})", file=sys.stderr)
        return 2
    try:
        rep = predict.run_all(trace, geom, a.budget_mib, train=split(a.train), test=split(a.test),
                              labels=tuple(split(a.labels)), csv_rows=csv_rows, other=other,
                              other_name=a.check_against.name if a.check_against else "", permutations=a.permutations,
                              seed=a.seed, bandwidths=bw, pinned=a.pinned, reference_tok_s=a.reference_tok_s,
                              n_ubatch=a.ubatch, alignment=a.alignment,
                              names=predict.file_names(ids=a.ids, requests=a.requests, trace=a.trace, geometry=a.geometry),
                              progress=lambda m: print(f"  {m}", file=sys.stderr))
    except geometry.BudgetBelowFloor as e:
        print(str(e), file=sys.stderr)
        return 2
    text = predict.to_markdown(rep)
    print(text)
    if a.md:
        a.md.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
