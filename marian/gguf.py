"""Read a GGUF file's metadata (header key/values only, never tensors), standard library only.

Used to learn what a model is (architecture, layers, experts, context length) so the profiler knows which knobs
apply: --n-cpu-moe for a mixture-of-experts model, -ngl for a dense one.
"""
from __future__ import annotations

import struct
from pathlib import Path

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


def _read(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f) -> str:
    n = _read(f, "<Q")
    if n > 1 << 24:
        raise ValueError("implausible GGUF string length")
    return f.read(n).decode("utf-8", errors="replace")


def _value(f, vtype, keep_arrays: bool):
    if vtype in _SCALAR:
        return _read(f, _SCALAR[vtype])
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        itype, n = _read(f, "<I"), _read(f, "<Q")
        if itype in _SCALAR and not keep_arrays:
            f.seek(struct.calcsize(_SCALAR[itype]) * n, 1)      # skip big numeric arrays (token scores, types)
            return f"<array {n}>"
        items = [_value(f, itype, keep_arrays) for _ in range(n)]
        return items if keep_arrays else f"<array {n}>"
    raise ValueError(f"unknown GGUF value type {vtype}")


def metadata(path: str | Path, keep_arrays: bool = False) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")
        meta = {"gguf.version": version, "gguf.tensor_count": n_tensors}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"), keep_arrays)
    return meta


def tensor_infos(path: str | Path) -> tuple[dict, list[dict], int]:
    """The header and tensor directory of one GGUF file (or shard): metadata, tensors, and the file offset of the data.

    Each tensor: name, ne (dims, innermost first), type (ggml type id), offset (from the start of the data section).
    Only the header is read, so a partial download of a file's first bytes is enough."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")
        meta = {"gguf.version": version, "gguf.tensor_count": n_tensors}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"), False)
        tensors = []
        for _ in range(n_tensors):
            name = _string(f)
            n_dims = _read(f, "<I")
            if n_dims > 8:
                raise ValueError(f"implausible tensor rank {n_dims} in {path}")
            ne = [_read(f, "<Q") for _ in range(n_dims)]
            tensors.append({"name": name, "ne": ne, "type": _read(f, "<I"), "offset": _read(f, "<Q")})
        align = int(meta.get("general.alignment", 32) or 32)
        data_offset = (f.tell() + align - 1) // align * align
    return meta, tensors, data_offset


def summary(path: str | Path) -> dict:
    """What Marian needs: architecture, block count, expert counts, trained context, MoE or dense."""
    m = metadata(path)
    arch = m.get("general.architecture", "?")
    get = lambda k, d=None: m.get(f"{arch}.{k}", d)        # noqa: E731
    experts = get("expert_count") or 0
    return {
        "architecture": arch,
        "name": m.get("general.name"),
        "file_type": m.get("general.file_type"),
        "block_count": get("block_count"),
        "context_length": get("context_length"),
        "expert_count": experts,
        "expert_used_count": get("expert_used_count") or 0,
        "moe": bool(experts and experts > 1),
        "embedding_length": get("embedding_length"),
        "head_count_kv": get("attention.head_count_kv"),
        "bytes": Path(path).stat().st_size,
    }
