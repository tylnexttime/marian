# -*- coding: utf-8 -*-
"""Marian M1.0 analysis: per-arm decode/prefill speed, per-token row-level VRAM hit rate from the P0 trace, host time
per token in the cache callback, bytes uploaded, resources, and a cross-arm exactness check of greedy text and top-3
probabilities. Usage: python -I analyse_m10.py <runs dir> [out.md]"""
import csv, json, statistics as st, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


def load_requests(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if line:
            rows.append(json.loads(line))
    return rows


def load_trace(path):
    s, l = [], []
    if not path.exists():
        return s, l
    with path.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                rec = {k: (int(v) if v not in ("", None) and k != "kind" else v) for k, v in r.items()}
            except ValueError:
                continue          # a line cut by the kill
            (s if r["kind"] == "S" else l).append(rec)
    return s, l


def steps(slines):
    """Group slot-map lines into decode steps: a new step starts when the layer does not increase."""
    out, cur, prev = [], [], None
    for r in slines:
        if prev is not None and r["layer"] <= prev:
            out.append(cur); cur = []
        cur.append(r); prev = r["layer"]
    if cur:
        out.append(cur)
    return out


def arm_summary(tag, runs):
    req = load_requests(runs / f"{tag}.requests.jsonl")
    s, l = load_trace(runs / f"{tag}.trace.csv")
    proc = json.loads((runs / f"{tag}.proc.json").read_text(encoding="utf-8-sig")) if (runs / f"{tag}.proc.json").exists() else {}
    out = {"tag": tag, "proc": proc, "phases": {}}
    for ph in ("warm", "decode", "pp4k", "pp16k", "post"):
        rs = [r for r in req if r["phase"] == ph and r.get("ok")]
        if not rs:
            continue
        d = {"n": len(rs), "gen_tps_median": st.median(r["gen_tps"] for r in rs),
             "prompt_tps_median": st.median(r["prompt_tps"] for r in rs), "prompt_n": sum(r["prompt_n"] for r in rs),
             "gen_n": sum(r["gen_n"] for r in rs)}
        sl = [x for x in s if any(r["t0_us"] <= x["t_us"] <= r["t1_us"] for r in rs)]
        k_used = min((x["n_ids"] for x in s), default=10)                   # one token's selections in one layer
        tok = [st_ for st_ in steps([x for x in sl if x["n_ids"] <= k_used])]   # single-token steps only
        n_cached_layers = len({x["layer"] for x in s})
        tok = [t for t in tok if len(t) >= 0.8 * n_cached_layers]          # whole tokens only
        if tok:
            hit = [sum(x["row_hits"] for x in t) / sum(x["n_ids"] for x in t) for t in tok]
            d.update(tokens_traced=len(tok), hit_median=q(hit, .5), hit_p10=q(hit, .1), hit_mean=st.mean(hit),
                     mib_up_per_token=st.mean(sum(x["bytes"] for x in t) for t in tok) / 2**20,
                     fills_per_token=st.mean(sum(x["fills"] for x in t) for t in tok),
                     ms_wait_per_token=st.median(sum(x["us_wait"] for x in t) for t in tok) / 1e3,
                     ms_host_per_token=st.median(sum(x["us_host"] for x in t) for t in tok) / 1e3)
        ll = [x for x in l if any(r["t0_us"] <= x["t_us"] <= r["t1_us"] for r in rs)]
        if ll:
            d.update(large_copies=len(ll), large_gib_up=sum(x["bytes"] for x in ll) / 2**30,
                     large_cached_share=sum(x["uniq_hits"] for x in ll) / max(1, sum(x["uniq"] for x in ll)),
                     large_ms_wait=sum(x["us_wait"] for x in ll) / 1e3, large_ms_host=sum(x["us_host"] for x in ll) / 1e3)
        out["phases"][ph] = d
    res = runs / f"{tag}.res.csv"
    if res.exists():
        rows = list(csv.DictReader(res.open(encoding="utf-8-sig")))
        f = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "", "-1", "-1.00")]
        out["res"] = {"vram_peak_mib": max(f("vram_used_mib"), default=None), "free_ram_min_gib": min(f("free_ram_gib"), default=None),
                      "commit_peak_gib": max(f("commit_gib"), default=None),
                      "pages_input_median": st.median(f("pages_input_per_s")) if f("pages_input_per_s") else None,
                      "disk_read_median_mb_s": st.median(f("disk_read_mb_s")) if f("disk_read_mb_s") else None}
    out["decode_text"] = {r["idx"]: r.get("content") for r in req if r["phase"] == "decode" and r.get("ok")}
    out["decode_probs"] = {r["idx"]: r.get("probs") for r in req if r["phase"] == "decode" and r.get("ok")}
    return out


def main():
    runs = Path(sys.argv[1]); md = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    tags = sorted({p.name[:-len(".requests.jsonl")] for p in runs.glob("*.requests.jsonl")})
    arms = [arm_summary(t, runs) for t in tags]
    lines = ["| arm | decode tok/s | post-pp16K tok/s | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        P = a["phases"]; dc = P.get("decode", {}); po = P.get("post", {}); r = a.get("res", {}); pr = a["proc"]
        g = lambda d, k, fmt="{:.1f}": (fmt.format(d[k]) if d.get(k) is not None else "-")
        lines.append(f"| {a['tag']} | {g(dc,'gen_tps_median')} | {g(po,'gen_tps_median')} | {g(P.get('pp4k',{}),'prompt_tps_median')} | "
                     f"{g(P.get('pp16k',{}),'prompt_tps_median')} | {g(dc,'hit_median','{:.1%}')} / {g(dc,'hit_p10','{:.1%}')} | "
                     f"{g(dc,'mib_up_per_token')} | {g(dc,'ms_wait_per_token','{:.2f}')} / {g(dc,'ms_host_per_token','{:.2f}')} | "
                     f"{g(r,'vram_peak_mib','{:.0f}')} | {g(r,'free_ram_min_gib','{:.2f}')} | {pr.get('peak_working_set_gib','-')} / {pr.get('peak_private_commit_gib','-')} |")
    # exactness: compare decode text and top-3 probabilities between every pair of arms
    ex = ["", "| arm A | arm B | decode texts identical | top-3 probs identical |", "|---|---|---|---|"]
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            A, B = arms[i], arms[j]
            ks = sorted(set(A["decode_text"]) & set(B["decode_text"]))
            same_t = sum(A["decode_text"][k] == B["decode_text"][k] for k in ks)
            same_p = sum(A["decode_probs"].get(k) == B["decode_probs"].get(k) for k in ks)
            ex.append(f"| {A['tag']} | {B['tag']} | {same_t}/{len(ks)} | {same_p}/{len(ks)} |")
    text = "\n".join(lines + ex) + "\n"
    print(text)
    (runs / "summary.json").write_text(json.dumps([{k: v for k, v in a.items() if not k.startswith("decode_")} for a in arms], indent=1), encoding="utf-8")
    if md:
        md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
