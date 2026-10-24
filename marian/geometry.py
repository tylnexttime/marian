"""Expert geometry of a MoE model, and the MoE cache's slot counts per group as the Marian fork computes them.

Geometry = for each MoE layer: the routed-expert tensors in the order the cache keeps them (gate_up, gate, up, down),
each with its ggml type, shape and bytes per expert (nb[2]), and n_expert_used. It comes from a GGUF's tensor directory
(header only, so a partial download of each shard's first bytes is enough) or from a small JSON file:

    {"model": "...", "n_expert_used": 10,
     "layers": [{"layer": 0, "n_expert": 512,
                 "tensors": [{"role": "gate", "type": "iq2_s", "ne": [2560, 640], "nb2": 524800}, ...]}, ...]}

Slot counts follow `llama_moe_cache::impl` at llama.cpp a37cc461 + Marian 0000-0003 (`src/llama-moe-cache.cpp`): one
device; layers grouped by identical expert layout in layer order; `alloc_size(g, n) = sum_t pad(nb2_t*(n+1), alignment)`
(one spare slot for MMQ, CUDA alignment 128); every group first gets the M1.1 exactness floor
`min(min(n_ubatch, 32)*n_expert_used, n_expert)`; the budget is split by host bytes, a group whose share falls below its
floor is held at the floor and the rest is split again, until stable; a budget below the sum of the floors is refused.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import gguf

MiB = 1024 * 1024
MAX_BATCH = 32          # get_slot_map: ubatches above 32 tokens take the large-batch selective copy
CUDA_ALIGNMENT = 128    # ggml_backend_cuda_buffer_type_get_alignment

# ggml type id -> (name, block size, bytes per block), printed from ggml_type_name/ggml_blck_size/ggml_type_size of the
# pinned ggml (a37cc461); removed and deprecated ids are left out
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20), 6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36), 10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110),
    12: ("q4_K", 256, 144), 13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110), 22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17), 40: ("nvfp4", 64, 36), 41: ("q1_0", 128, 18), 42: ("q2_0", 64, 18),
}
ROLES = ("gate_up", "gate", "up", "down")   # llama_moe_cache_layer_experts order


@dataclass
class ExpertTensor:
    role: str
    type: str
    ne: tuple[int, int]     # ne[0], ne[1] of one expert
    nb2: int                # bytes per expert


@dataclass
class Layer:
    layer: int
    n_expert: int
    tensors: list[ExpertTensor]

    @property
    def expert_bytes(self) -> int:
        return sum(t.nb2 for t in self.tensors)

    def layout(self) -> tuple:
        return (self.n_expert,) + tuple((t.type, t.ne, t.nb2) for t in self.tensors)


@dataclass
class Group:
    index: int
    layers: list[int]
    n_expert: int
    nb2: list[int]
    type: str               # type of the first expert tensor, as the fork's log names the group

    @property
    def expert_bytes(self) -> int:
        return sum(self.nb2)

    @property
    def host_bytes(self) -> int:
        return len(self.layers) * self.n_expert * self.expert_bytes

    def alloc_size(self, n_slots: int, alignment: int = CUDA_ALIGNMENT) -> int:
        return sum((b * (n_slots + 1) + alignment - 1) // alignment * alignment for b in self.nb2)

    def slots_for(self, budget: int, alignment: int = CUDA_ALIGNMENT) -> int:
        """The most slots (up to every expert of the group) whose alloc_size fits in budget bytes."""
        lo, hi = 0, len(self.layers) * self.n_expert
        while lo < hi:                      # alloc_size is monotonic, so this equals the fork's linear walk
            mid = (lo + hi + 1) // 2
            if self.alloc_size(mid, alignment) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo


@dataclass
class Geometry:
    n_expert_used: int
    layers: dict[int, Layer]
    model: str = ""
    source: dict = field(default_factory=dict)

    def groups(self) -> list[Group]:
        """Layers with the same expert layout share a group (banks and SLRU), in order of their first layer."""
        out: list[Group] = []
        keys: list[tuple] = []
        for il in sorted(self.layers):
            lay = self.layers[il]
            key = lay.layout()
            if key in keys:
                out[keys.index(key)].layers.append(il)
            else:
                keys.append(key)
                out.append(Group(len(out), [il], lay.n_expert, [t.nb2 for t in lay.tensors], lay.tensors[0].type))
        return out

    @property
    def total_expert_bytes(self) -> int:
        return sum(lay.n_expert * lay.expert_bytes for lay in self.layers.values())

    def to_dict(self) -> dict:
        return {"model": self.model, "n_expert_used": self.n_expert_used, "source": self.source,
                "layers": [{"layer": lay.layer, "n_expert": lay.n_expert,
                            "tensors": [{"role": t.role, "type": t.type, "ne": list(t.ne), "nb2": t.nb2} for t in lay.tensors]}
                           for lay in (self.layers[i] for i in sorted(self.layers))]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")


def from_dict(d: dict) -> Geometry:
    layers = {}
    for x in d["layers"]:
        ts = [ExpertTensor(t.get("role", "?"), t.get("type", "?"), tuple(t.get("ne", (0, 0))), int(t["nb2"])) for t in x["tensors"]]
        layers[int(x["layer"])] = Layer(int(x["layer"]), int(x["n_expert"]), ts)
    if not layers:
        raise ValueError("geometry has no MoE layer")
    return Geometry(int(d["n_expert_used"]), layers, d.get("model", ""), d.get("source", {}))


def load(path: str | Path) -> Geometry:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def from_gguf(paths: list[str | Path]) -> Geometry:
    """Geometry from the tensor directories of all shards of a GGUF (headers only). The bytes per expert come from the
    type table; where the next tensor of the same file is known, the gap between the two offsets is checked as well."""
    layers_t: dict[int, dict[str, ExpertTensor]] = {}
    n_expert: dict[int, int] = {}
    meta0 = None
    for path in paths:
        meta, tensors, _ = gguf.tensor_infos(path)
        if meta0 is None or "general.architecture" in meta:
            meta0 = meta0 or meta
        align = int(meta.get("general.alignment", 32) or 32)
        by_off = sorted(tensors, key=lambda t: t["offset"])
        nxt = {by_off[i]["name"]: by_off[i + 1]["offset"] for i in range(len(by_off) - 1)}
        for t in tensors:
            parts = t["name"].split(".")
            if len(parts) != 4 or parts[0] != "blk" or not parts[2].startswith("ffn_") or not parts[2].endswith("_exps"):
                continue
            role = parts[2][len("ffn_"):-len("_exps")]
            if role not in ROLES or len(t["ne"]) != 3:
                continue
            if t["type"] not in GGML_TYPES:
                raise ValueError(f"{t['name']}: ggml type {t['type']} is not in the table of the pinned ggml")
            name, blck, tsize = GGML_TYPES[t["type"]]
            ne0, ne1, ne2 = t["ne"]
            if ne0 % blck:
                raise ValueError(f"{t['name']}: ne[0] = {ne0} is not a multiple of the {name} block size {blck}")
            nb2 = ne1 * (ne0 // blck * tsize)
            if t["name"] in nxt:
                gap = nxt[t["name"]] - t["offset"]
                if not (0 <= gap - nb2 * ne2 < align):
                    raise ValueError(f"{t['name']}: {nb2} bytes per expert does not match the gap of {gap} bytes to the next tensor")
            il = int(parts[1])
            layers_t.setdefault(il, {})[role] = ExpertTensor(role, name, (ne0, ne1), nb2)
            if n_expert.setdefault(il, ne2) != ne2:
                raise ValueError(f"layer {il}: expert tensors disagree on n_expert")
    if not layers_t:
        raise ValueError("no routed-expert tensors (blk.N.ffn_*_exps) found")
    meta0 = meta0 or {}
    arch = meta0.get("general.architecture", "")
    k = meta0.get(f"{arch}.expert_used_count")
    if not k:
        raise ValueError("expert_used_count not found in the first shard's metadata")
    layers = {il: Layer(il, n_expert[il], [ts[r] for r in ROLES if r in ts]) for il, ts in layers_t.items()}
    return Geometry(int(k), layers, meta0.get("general.name") or arch,
                    {"gguf": [Path(p).name for p in paths], "architecture": arch})


class BudgetBelowFloor(ValueError):
    def __init__(self, budget: int, floor: int):
        self.min_mib = math.ceil(floor / MiB)
        super().__init__(f"MoE cache: {budget / MiB:.0f} MiB is below the exact-mode floor of {floor / MiB:.0f} MiB; "
                         f"use --moe-cache-mib {self.min_mib} or more")


@dataclass
class GroupSlots:
    group: Group
    n_slots: int
    n_floor: int
    at_floor: bool

    @property
    def mib(self) -> float:
        return self.group.alloc_size(self.n_slots) / MiB


def cache_slots(geom: Geometry, budget_mib: float, n_ubatch: int = 512, alignment: int = CUDA_ALIGNMENT) -> list[GroupSlots]:
    """Slots per group for --moe-cache-mib budget_mib on one GPU, as the fork's constructor computes them."""
    groups = geom.groups()
    budget = int(budget_mib * MiB)          # one device: size*split/split_sum = size
    n_tokens_max = min(n_ubatch, MAX_BATCH)
    floors = [min(n_tokens_max * geom.n_expert_used, g.n_expert) for g in groups]
    dev_floor = sum(g.alloc_size(f, alignment) for g, f in zip(groups, floors))
    if dev_floor > budget:
        raise BudgetBelowFloor(budget, dev_floor)
    at_floor = [False] * len(groups)
    share = [0] * len(groups)
    changed = True
    while changed:
        changed = False
        rest_budget = budget - sum(g.alloc_size(f, alignment) for g, f, a in zip(groups, floors, at_floor) if a)
        rest_bytes = sum(g.host_bytes for g, a in zip(groups, at_floor) if not a)
        for i, g in enumerate(groups):
            if at_floor[i]:
                continue
            share[i] = int(float(rest_budget) * g.host_bytes / rest_bytes)   # (double) rest*host/rest_bytes, truncated
            if g.slots_for(share[i], alignment) < floors[i]:
                at_floor[i] = True
                changed = True
    return [GroupSlots(g, floors[i] if at_floor[i] else g.slots_for(share[i], alignment), floors[i], at_floor[i])
            for i, g in enumerate(groups)]
