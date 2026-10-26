# Copyright 2026 tylnexttime (Chris Tyl and Claude Tyl) and contributors
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE at the repository root).
"""Write the tensor byte ranges of one or more GGUF shards as JSON, for wsprobe.

Every tensor gets a kind: "ple" if its name contains "per_layer", "experts" if it contains "_exps", else "other".
Its absolute file range is [data_offset + offset, next tensor's absolute offset) with the file size closing the
last one, so alignment padding is counted with the tensor before it. The bytes before the data section (the GGUF
header and tensor directory) are one extra range of kind "header". "layer" is N from a "blk.N." name, else -1.
The ranges of a shard are sorted, contiguous and cover [0, file size) exactly; this script asserts that.

usage: python gguf_ranges.py SHARD [SHARD ...] -o ranges.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from marian.gguf import tensor_infos
except ImportError:                                   # run from a checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from marian.gguf import tensor_infos

_BLK = re.compile(r"^blk\.(\d+)\.")


def kind_of(name: str) -> str:
    if "per_layer" in name:
        return "ple"
    if "_exps" in name:
        return "experts"
    return "other"


def shard_ranges(path: str | Path) -> dict:
    path = Path(path).resolve()
    size = path.stat().st_size
    _, tensors, data_offset = tensor_infos(path)
    rows = sorted(({"name": t["name"], "start": data_offset + int(t["offset"])} for t in tensors),
                  key=lambda r: r["start"])
    out = [{"name": "(gguf header)", "kind": "header", "layer": -1, "start": 0, "end": data_offset}]
    for i, r in enumerate(rows):
        end = rows[i + 1]["start"] if i + 1 < len(rows) else size
        m = _BLK.match(r["name"])
        out.append({"name": r["name"], "kind": kind_of(r["name"]), "layer": int(m.group(1)) if m else -1,
                    "start": r["start"], "end": end})
    # the ranges must tile the file: a gap or overlap would make every per-kind total wrong
    if out[0]["start"] != 0 or out[-1]["end"] != size:
        raise ValueError(f"{path}: ranges do not cover [0, {size})")
    for a, b in zip(out, out[1:]):
        if a["end"] != b["start"] or a["end"] < a["start"]:
            raise ValueError(f"{path}: ranges not contiguous at {a['name']} -> {b['name']}")
    totals: dict[str, int] = {}
    for r in out:
        totals[r["kind"]] = totals.get(r["kind"], 0) + r["end"] - r["start"]
    return {"path": str(path), "size": size, "data_offset": data_offset, "bytes_by_kind": totals, "tensors": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    doc = {"format": "wsprobe-ranges/1", "shards": [shard_ranges(p) for p in a.shards]}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for s in doc["shards"]:
        gib = {k: round(v / 2**30, 2) for k, v in sorted(s["bytes_by_kind"].items())}
        print(f"{Path(s['path']).name}: {len(s['tensors']) - 1} tensors, GiB by kind {gib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
