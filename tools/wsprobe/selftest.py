# Copyright 2026 tylnexttime (Chris Tyl and Claude Tyl) and contributors
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE at the repository root).
"""wsprobe self-test: can the probe fail, and does it read back exactly what was touched?

1. Write a synthetic GGUF of about 1 GiB (seven tensors, sizes deliberately not page multiples, kinds ple / experts /
   other, two expert layers) and its ranges JSON with gguf_ranges.py, so the whole path is the one used on a model.
2. wsprobe_target maps it like llama.cpp does. A continuous wsprobe (500 ms) watches the target throughout, and at each
   step a one-shot wsprobe --once --dump-runs reads the resident pages back:
     P0 mapped, nothing touched          -> the view is found, 0 pages resident
     P1 touch set T1                     -> resident runs == T1's pages exactly, per-kind and per-layer pages exactly
     P2 trim the working set             -> view still found, 0 pages resident
     P3 touch a different set T2         -> exactly T2 (the probe follows changes, it is not stuck on P1)
     P4 unmap, touch nothing             -> the view is gone and residency falls to 0
   The continuous series must show the same rise and fall.
Each check prints PASS or FAIL; the exit status is the number of failures.

usage: python selftest.py [--dir SCRATCH_DIR] [--keep]
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MiB = 1 << 20
PAGE = 4096

TENSORS = [  # name, bytes (none a page multiple)
    ("token_embd.weight", 48 * MiB + 1000),
    ("per_layer_token_embd.weight", 512 * MiB + 12345),
    ("blk.0.attn_q.weight", 8 * MiB + 5),
    ("blk.0.ffn_up_exps.weight", 160 * MiB + 777),
    ("blk.1.per_layer_inp_gate.weight", 1 * MiB + 3),
    ("blk.1.ffn_down_exps.weight", 256 * MiB + 99),
    ("output.weight", 40 * MiB + 4097),
]


def write_gguf(path: Path) -> None:
    align = 32
    kv = [("general.architecture", 8, "wsprobe-selftest"), ("general.alignment", 4, align)]
    hdr = bytearray(b"GGUF") + struct.pack("<I", 3) + struct.pack("<QQ", len(TENSORS), len(kv))
    for k, t, v in kv:
        hdr += struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", t)
        hdr += (struct.pack("<Q", len(v)) + v.encode()) if t == 8 else struct.pack("<I", v)
    off, offs = 0, []
    for name, nb in TENSORS:  # one dimension of ggml type 24 (I8): one byte per element
        offs.append(off)
        hdr += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", 1) + struct.pack("<Q", nb)
        hdr += struct.pack("<I", 24) + struct.pack("<Q", off)
        off = (off + nb + align - 1) // align * align
    data_offset = (len(hdr) + align - 1) // align * align
    total = data_offset + offs[-1] + TENSORS[-1][1]
    block = bytes((i * 7 + 13) & 0xFF or 1 for i in range(MiB))  # non-zero content
    with open(path, "wb") as f:
        f.write(bytes(hdr) + b"\0" * (data_offset - len(hdr)))
        left = total - data_offset
        while left > 0:
            n = min(left, len(block))
            f.write(block[:n])
            left -= n
    assert path.stat().st_size == total


def pages_of(ranges: list[tuple[int, int]]) -> set[int]:
    s: set[int] = set()
    for a, b in ranges:
        s.update(range(a // PAGE, (b + PAGE - 1) // PAGE))
    return s


def runs_of(pages: set[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for p in sorted(pages):
        if out and out[-1][1] == p * PAGE:
            out[-1][1] = (p + 1) * PAGE
        else:
            out.append([p * PAGE, (p + 1) * PAGE])
    return out


def attribute(pages: set[int], tensors: list[dict]) -> tuple[dict, dict]:
    """Expected per-kind and per-expert-layer page counts: a page belongs to the range holding its first byte."""
    kinds: dict[str, int] = {}
    layers: dict[int, int] = {}
    for p in pages:
        b = p * PAGE
        t = next(t for t in tensors if t["start"] <= b < t["end"])
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
        if t["kind"] == "experts":
            layers[t["layer"]] = layers.get(t["layer"], 0) + 1
    return kinds, layers


class Target:
    def __init__(self, exe: Path, gguf: Path):
        self.p = subprocess.Popen([str(exe), str(gguf)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        line = self.p.stdout.readline().split()
        if not line or line[0] != "ready":
            raise RuntimeError(f"target did not start: {line}")
        self.pid = int(line[1])

    def cmd(self, s: str) -> str:
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()
        r = self.p.stdout.readline().strip()
        if not r.startswith("ok"):
            raise RuntimeError(f"target answered {r!r} to {s!r}")
        return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="scratch directory for the 1 GiB file (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch files")
    a = ap.parse_args(argv)
    work = Path(a.dir) if a.dir else Path(tempfile.mkdtemp(prefix="wsprobe-selftest-"))
    work.mkdir(parents=True, exist_ok=True)
    probe, target_exe = HERE / "wsprobe.exe", HERE / "wsprobe_target.exe"
    for exe in (probe, target_exe):
        if not exe.exists():
            print(f"FAIL {exe.name} not built (run build.bat)")
            return 1
    gguf, rj = work / "selftest.gguf", work / "selftest.ranges.json"
    write_gguf(gguf)
    subprocess.run([sys.executable, "-I", str(HERE / "gguf_ranges.py"), str(gguf), "-o", str(rj)], check=True)
    shard = json.loads(rj.read_text(encoding="utf-8"))["shards"][0]
    T = {t["name"]: t for t in shard["tensors"]}
    size = shard["size"]
    print(f"synthetic GGUF {size / MiB:.1f} MiB, {len(TENSORS)} tensors, data offset {shard['data_offset']}")

    ple, e0, gate1, e1 = (T["per_layer_token_embd.weight"], T["blk.0.ffn_up_exps.weight"],
                          T["blk.1.per_layer_inp_gate.weight"], T["blk.1.ffn_down_exps.weight"])
    out_w, tok = T["output.weight"], T["token_embd.weight"]
    T1 = [(0, PAGE),                                                   # the header page
          (ple["start"] + 100 * MiB, ple["start"] + 164 * MiB),         # 64 MiB inside the PLE
          (e0["start"], e0["start"] + 10 * MiB),                        # start of expert layer 0 (unaligned start)
          (e0["end"] - 2 * MiB, e1["start"] + 3 * MiB),                 # across exps0 -> ple gate of blk.1 -> exps1
          ((out_w["start"] + out_w["end"]) // 2, (out_w["start"] + out_w["end"]) // 2 + 1),  # one page
          (size - 1, size)]                                             # the file's last, partial page
    T2 = [(ple["start"] + 300 * MiB, ple["start"] + 332 * MiB),
          (e1["start"] + 100 * MiB, e1["start"] + 120 * MiB),
          (tok["start"], tok["start"] + MiB)]
    assert gate1["end"] - gate1["start"] < 2 * MiB

    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")

    tgt = Target(target_exe, gguf)
    series = work / "selftest.series.jsonl"
    series.unlink(missing_ok=True)
    cont = subprocess.Popen([str(probe), "--pid", str(tgt.pid), "--ranges", str(rj), "--out", str(series),
                             "--interval-ms", "500"])

    def once(tag: str) -> tuple[dict, dict]:
        runs_f, line_f = work / f"selftest.{tag}.runs.json", work / f"selftest.{tag}.sample.jsonl"
        line_f.unlink(missing_ok=True)
        subprocess.run([str(probe), "--pid", str(tgt.pid), "--ranges", str(rj), "--once", "--dump-runs", str(runs_f),
                        "--out", str(line_f)], check=True)
        sample = next(json.loads(x) for x in line_f.read_text(encoding="utf-8").splitlines()
                      if json.loads(x)["type"] == "sample")
        return sample, json.loads(runs_f.read_text(encoding="utf-8"))

    def exact(tag: str, touched: list[tuple[int, int]]) -> int:
        want_pages = pages_of(touched)
        want_kinds, want_layers = attribute(want_pages, shard["tensors"])
        s, d = once(tag)
        got_runs = d["views"][0]["runs"] if d["views"] else []
        check(f"{tag}: one full view of the shard found", s["views"] == 1 and d["views"] and d["views"][0]["full"] == 1)
        check(f"{tag}: resident runs == touched pages exactly ({len(want_pages)} pages)", got_runs == runs_of(want_pages),
              "" if got_runs == runs_of(want_pages) else f"got {got_runs[:6]}... want {runs_of(want_pages)[:6]}...")
        gk = {k: v for k, v in s["res_pages"].items() if v}
        check(f"{tag}: pages per kind {want_kinds}", gk == want_kinds, f"got {gk}")
        gl = {i: v for i, v in enumerate(s["exp_layer_pages"]) if v}
        check(f"{tag}: expert pages per layer {want_layers}", gl == want_layers, f"got {gl}")
        return len(want_pages)

    time.sleep(1.2)
    s0, _ = once("P0")
    check("P0 mapped, untouched: view found, nothing resident",
          s0["views"] == 1 and sum(s0["res_pages"].values()) == 0, f"views {s0['views']}, pages {s0['res_pages']}")
    for x, y in T1:
        tgt.cmd(f"touch 0 {x} {y}")
    n1 = exact("P1", T1)
    time.sleep(1.2)
    tgt.cmd("trim")
    s2, _ = once("P2")
    check("P2 trimmed, still mapped: view found, nothing resident",
          s2["views"] == 1 and sum(s2["res_pages"].values()) == 0, f"views {s2['views']}, pages {s2['res_pages']}")
    time.sleep(1.2)
    for x, y in T2:
        tgt.cmd(f"touch 0 {x} {y}")
    n2 = exact("P3", T2)
    time.sleep(1.2)
    tgt.cmd("unmap 0")
    s4, _ = once("P4")
    check("P4 unmapped, nothing touched: view gone, residency 0",
          s4["views"] == 0 and sum(s4["res_pages"].values()) == 0, f"views {s4['views']}, pages {s4['res_pages']}")
    time.sleep(1.2)
    tgt.p.stdin.write("quit\n")
    tgt.p.stdin.flush()
    tgt.p.wait(10)
    cont.wait(10)

    rows = [json.loads(x) for x in series.read_text(encoding="utf-8").splitlines()]
    samples = [x for x in rows if x["type"] == "sample"]
    totals = [sum(x["res_pages"].values()) for x in samples]
    # the continuous series must rise to T1, fall to 0 while still mapped, rise to T2, then fall to 0 with no view
    seq = []
    for x, tot in zip(samples, totals):
        key = (x["views"], tot)
        if not seq or seq[-1] != key:
            seq.append(key)
    want_seq = [(1, 0), (1, n1), (1, 0), (1, n2), (0, 0)]
    it = iter(seq)
    in_order = all(any(k == w for k in it) for w in want_seq)
    check(f"continuous series shows {want_seq} in order", in_order, f"distinct states {seq}")
    qms = [x["q_ms"] for x in samples]
    end = rows[-1]
    check("continuous probe ended on process exit", end["type"] == "end" and end["reason"] == "process exited", str(end))
    print(f"continuous: {len(samples)} samples, q_ms max {max(qms):.1f} for {samples[0]['queried']} pages queried per sample")
    summary = {"pass": fails == 0, "fails": fails, "file_mib": round(size / MiB, 1), "t1_pages": n1, "t2_pages": n2,
               "distinct_states": seq, "q_ms_max": max(qms), "samples": len(samples)}
    (work / "selftest.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not a.keep:
        gguf.unlink(missing_ok=True)
    print(f"SELFTEST {'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s)); files in {work}")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
