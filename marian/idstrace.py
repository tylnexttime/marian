"""Routing-id traces (Marian patch 0004, MARIAN_MOE_TRACE_IDS) and the CSV traces and request logs beside them.

Binary ids trace, little-endian, packed (the writer is src/llama-moe-trace-ids.h of the fork):
  header (32 bytes): magic b"MRNIDS1\\n", u32 header_bytes, u32 version (1), u32 id_bytes (2), u32 reserved, i64 t_open_us
  records to the end of the file: u8 kind ('S' slot-map lookup, 'L' large-batch copy), u8 k, u16 layer, u32 n_tokens,
  i64 t_us, then n_tokens*k u16 ids, row by row in the router's order (0xffff: an id that did not fit)
A last record cut short (the server is usually killed) is dropped and counted.

The P0 trace (MARIAN_MOE_TRACE, CSV): kind,t_us,layer,n_ids,row_hits,uniq,uniq_hits,fills,bytes,us_wait,us_host.
The fill trace (MARIAN_MOE_TRACE_FILLS, CSV): kind,t_us,layer,n_ids,expert,slot,src,bytes.
Both join an ids record on (t_us, layer). run_arm*.ps1 writes <tag>.requests.jsonl with each request's phase, idx and
t0_us/t1_us on the same clock (QPC on Windows, as std::chrono::steady_clock there).
"""
from __future__ import annotations

import bisect
import csv
import json
import struct
from array import array
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"MRNIDS1\n"
_HEAD = struct.Struct("<8sIIIIq")
_REC = struct.Struct("<BBHIq")


@dataclass
class Step:
    kind: str           # 'S' or 'L'
    t_us: int
    layer: int
    n_tokens: int
    k: int
    ids: array          # n_tokens*k ids, row by row
    req: int = -1       # index of the request it belongs to, -1 if none (server warm-up, between requests)


@dataclass
class Trace:
    steps: list[Step]
    t_open_us: int = 0
    truncated_bytes: int = 0
    requests: list[dict] = field(default_factory=list)


def read_ids(path: str | Path) -> Trace:
    data = Path(path).read_bytes()
    if len(data) < _HEAD.size:
        raise ValueError(f"{path}: shorter than the ids-trace header")
    magic, header_bytes, version, id_bytes, _, t_open = _HEAD.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a Marian ids trace (magic {magic!r})")
    if version != 1 or id_bytes != 2:
        raise ValueError(f"{path}: version {version} with {id_bytes}-byte ids is not supported")
    steps, pos, n = [], header_bytes, len(data)
    while pos + _REC.size <= n:
        kind, k, layer, n_tokens, t_us = _REC.unpack_from(data, pos)
        end = pos + _REC.size + 2 * k * n_tokens
        if end > n:
            break
        ids = array("H")
        ids.frombytes(data[pos + _REC.size:end])
        if ids.itemsize != 2:
            raise RuntimeError("array('H') is not 16-bit on this platform")
        if struct.pack("=H", 1) != struct.pack("<H", 1):
            ids.byteswap()
        steps.append(Step(chr(kind), t_us, layer, n_tokens, k, ids))
        pos = end
    return Trace(steps, t_open, n - pos)


def write_ids(path: str | Path, steps: list[Step], t_open_us: int = 0) -> None:
    out = bytearray(_HEAD.pack(MAGIC, _HEAD.size, 1, 2, 0, t_open_us))
    for s in steps:
        out += _REC.pack(ord(s.kind), s.k, s.layer, s.n_tokens, s.t_us)
        out += struct.pack(f"<{len(s.ids)}H", *s.ids)
    Path(path).write_bytes(bytes(out))


def read_csv(path: str | Path) -> list[dict]:
    """P0 trace or fill trace lines as dicts of ints (kind and src stay strings); lines cut by a kill are skipped."""
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if None in r.values() or None in r:
                continue                    # fewer or more fields than the header: a line cut by the kill
            try:
                rows.append({k: (v if k in ("kind", "src") else int(v) if v != "" else None) for k, v in r.items()})
            except ValueError:
                continue
    return rows


def write_trace_csv(path: str | Path, rows: list[dict]) -> None:
    cols = ["kind", "t_us", "layer", "n_ids", "row_hits", "uniq", "uniq_hits", "fills", "bytes", "us_wait", "us_host"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


def read_requests(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if line:
            rows.append(json.loads(line))
    return rows


def assign_requests(trace: Trace, requests: list[dict]) -> None:
    """Each step gets the index of the request whose [t0_us, t1_us] holds its t_us (requests never overlap)."""
    trace.requests = requests
    order = sorted(range(len(requests)), key=lambda i: requests[i]["t0_us"])
    starts = [requests[i]["t0_us"] for i in order]
    for s in trace.steps:
        j = bisect.bisect_right(starts, s.t_us) - 1
        s.req = order[j] if j >= 0 and s.t_us <= requests[order[j]]["t1_us"] else -1


def infer_requests(trace: Trace) -> None:
    """Without a request log: a request starts at every forward pass with more than one token (a prompt) that follows
    a decode token or opens the trace. Phases are unknown ('all'); each request is its own prompt. The server's warm-up
    run (2 tokens, no decode) cannot be told from the prompt after it, so it joins the first request."""
    reqs: list[dict] = []
    prev_layer, pass_multi, cur, last_was_decode = None, False, -1, True
    for s in trace.steps:
        new_pass = prev_layer is None or s.layer <= prev_layer
        if new_pass:
            pass_multi = s.n_tokens > 1
            if pass_multi and last_was_decode:
                cur = len(reqs)
                reqs.append({"phase": "all", "idx": cur, "t0_us": s.t_us, "t1_us": s.t_us})
            last_was_decode = not pass_multi
        if cur >= 0:
            reqs[cur]["t1_us"] = s.t_us
        s.req = cur
        prev_layer = s.layer
    trace.requests = reqs


def prompt_key(req: dict) -> str:
    """The prompt a request uses, so a split never puts one prompt on both sides: run_arm's 'warm' request is decode
    prompt 0 and its 'post' requests are decode prompts 0-2 again."""
    ph, idx = req.get("phase", "all"), req.get("idx", 0)
    if ph in ("warm", "post", "decode"):
        return f"decode:{idx if ph != 'warm' else 0}"
    return f"{ph}:{idx}"


def decode_tokens(steps: list[Step], idx: list[int], k: int, n_layers: int) -> list[list[int]]:
    """Single-token decode steps grouped into tokens, the way analyse_m10.py does it: S steps of at most k ids, a new
    token when the layer does not increase, tokens with at least 80% of the cached layers kept. Returns step indices."""
    out, cur, prev = [], [], None
    for i in idx:
        s = steps[i]
        if s.kind != "S" or s.n_tokens * s.k > k:
            continue
        if prev is not None and s.layer <= prev:
            out.append(cur)
            cur = []
        cur.append(i)
        prev = s.layer
    if cur:
        out.append(cur)
    return [t for t in out if len(t) >= 0.8 * n_layers]
