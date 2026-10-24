"""marian replay: offline replay of a routing-id trace through expert-cache (VRAM) and RAM-tier policies.

VRAM policies (per cache group, at the slot counts of `geometry.cache_slots`):
  slru        upstream's SLRU exactly as the fork runs it (`moe_cache_slru` in src/llama-moe-cache.cpp): every slot
              starts empty in probation in slot order; hits are promoted (first-appearance order) to protected, which
              holds at most n_slots/2 (its LRU end is demoted to probation's MRU end); misses, sorted by expert id, take
              the first slot from probation's LRU end, then protected's, that is empty, of another layer, or of an
              expert this step did not select, and go to probation's MRU end. Large-batch steps (above 32 tokens)
              read the cache without changing it, so only S steps replay.
  core<P>     DESIGN-M1 P2 ranked core: P% of the group's slots (capped so the evictable slots still hold one 32-token
              ubatch) hold the most-selected experts of the training prompts' decode tokens from the first token and
              are never evicted; the rest is the SLRU above.
  door        P2 TinyLFU doorkeeper: a 4-bit count-min sketch (depth 4) over layer*n_expert+expert, incremented for
              each distinct expert a step selects and halved every 10 x n_slots increments; a miss whose victim slot is
              occupied is declined when the victim's estimate is higher than its own (so a one-off does not evict a
              repeat-use expert; ties admit, as SLRU would), and a declined miss is uploaded into one of n_expert_used
              transient slots (taken from the budget) and not kept; with no transient slot left in the step it is admitted.
  guard       P2 transient slots for prompt chunks: steps above 8 tokens admit nothing while transient slots (up to one
              32-token ubatch, within the capacity invariant) remain.
  Combinations join with '+', e.g. core50+door.

RAM-tier policies (for P3), replayed on the miss stream of a VRAM policy (slru by default), serving VRAM misses:
  current       the M1.1 tier exactly as `llama_moe_source` runs it: per-group pinned slots split by host bytes, sector-
                rounded slot sizes, decayed counts (decode steps of <= 8 tokens, halved every 256 decode steps of a
                layer), CLOCK victims with reference bits set by decode hits, admission only if the newcomer's count beats
                the victim's, slots used in the current step never victims (generation stamps, including ring wraps).
  warm          `current` started from the state it reaches on the training prompts' misses (learned counts and
                contents, as P4's warm start would hand over).
  static-sel    a static tier filled before the first token with the experts the training prompts' decode tokens
                selected most (what P4's decayed counts record), by count per tier byte across groups.
  static-miss   the same, ranked by the training prompts' VRAM misses at the same budget instead of selections.
  oracle-static the best static tier in hindsight: ranked by the test prompts' own misses (an upper bound for static).
  belady        Belady's MIN with bypass over the whole miss stream (future knowledge): optimal for equal-size experts,
                an approximate bound across groups of different sizes.

Metrics are taken on the test prompts only (train/test split by prompt; the session itself is replayed in its real
order, so the cache warms across prompts as it does in a server). Hit rate is the design's section 2 definition:
per single-token decode step, selections resident before the step's fills over n_layer x n_expert_used, median and 10th
percentile over tokens, plus the byte-weighted rate and MiB uploaded per token.
"""
from __future__ import annotations

import heapq
import math
import statistics as st
from dataclasses import dataclass, field

from . import geometry as G
from .idstrace import Step, Trace, decode_tokens, prompt_key

INF = float("inf")


def q(xs, p):
    """Nearest-rank quantile, as analyse_m10.py computes it."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


# ----------------------------------------------------------------------------------------------- VRAM: the fork's SLRU

class SLRU:
    """`moe_cache_slru` of the fork, line for line: one group, slots shared by its layers, key = il*n_expert + e."""

    def __init__(self, layers, n_expert: int, n_slots: int):
        self.n_expert, self.n_slots = n_expert, n_slots
        self.slot_map = {il: [-1] * n_expert for il in layers}
        self.key_of = [-1] * n_slots
        self.prev = [s - 1 for s in range(n_slots)]
        self.next = [s + 1 if s + 1 < n_slots else -1 for s in range(n_slots)]
        self.head = [0 if n_slots else -1, -1]
        self.tail = [n_slots - 1, -1]
        self.prot = [False] * n_slots
        self.n_prot = 0
        self.seen = [0] * n_expert
        self.seen_gen = 0

    def move(self, s: int, protect: bool) -> None:
        frm = self.prot[s]
        if frm == protect and s == self.tail[frm]:
            return
        prev, nxt, head, tail = self.prev, self.next, self.head, self.tail
        if prev[s] >= 0:
            nxt[prev[s]] = nxt[s]
        else:
            head[frm] = nxt[s]
        if nxt[s] >= 0:
            prev[nxt[s]] = prev[s]
        else:
            tail[frm] = prev[s]
        prev[s] = tail[protect]
        nxt[s] = -1
        if tail[protect] >= 0:
            nxt[tail[protect]] = s
        else:
            head[protect] = s
        tail[protect] = s
        self.prot[s] = protect
        self.n_prot += int(protect) - int(frm)

    def touch(self, s: int) -> None:
        self.move(s, True)
        if self.n_prot > self.n_slots // 2:
            self.move(self.head[1], False)

    def victim(self, il: int) -> int:
        ne, key_of, seen, gen = self.n_expert, self.key_of, self.seen, self.seen_gen
        for seg in (0, 1):
            s = self.head[seg]
            while s >= 0:
                key = key_of[s]
                if key < 0 or key // ne != il or seen[key % ne] != gen:
                    return s
                s = self.next[s]
        raise RuntimeError("no SLRU victim")

    def plan(self, il: int, uniq: list[int], admit=None, n_transient: int = 0):
        """uniq: the step's distinct experts in first-appearance order. Returns (fills [(expert, slot or -1 for a
        transient slot)], n_hit), or None if the step selects more distinct experts than there are slots."""
        if len(uniq) > self.n_slots:
            return None
        self.seen_gen += 1
        gen, seen = self.seen_gen, self.seen
        for e in uniq:
            seen[e] = gen
        sm = self.slot_map[il]
        n_hit = 0
        for e in uniq:
            if sm[e] >= 0:
                self.touch(sm[e])
                n_hit += 1
        fills = []
        ne = self.n_expert
        for e in sorted(uniq):
            if sm[e] >= 0:
                continue
            s = self.victim(il)
            old = self.key_of[s]
            if admit is not None and old >= 0 and n_transient > 0 and not admit(il * ne + e, old):
                fills.append((e, -1))
                n_transient -= 1
                continue
            if old >= 0:
                self.slot_map[old // ne][old % ne] = -1
            self.key_of[s] = il * ne + e
            sm[e] = s
            self.move(s, False)
            fills.append((e, s))
        return fills, n_hit


class CountMin4:
    """4-bit count-min sketch with periodic halving (the TinyLFU frequency estimate), deterministic hashing."""
    _MUL = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93)

    def __init__(self, n_keys: int, sample: int, depth: int = 4):
        w = 16
        while w < n_keys:
            w *= 2
        self.mask, self.depth = w - 1, depth
        self.rows = [bytearray(w) for _ in range(depth)]
        self.sample, self.n = max(1, sample), 0

    def _idx(self, key: int, i: int) -> int:
        x = ((key + 1) * self._MUL[i]) & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & self.mask

    def add(self, key: int) -> None:
        for i in range(self.depth):
            r, j = self.rows[i], self._idx(key, i)
            if r[j] < 15:
                r[j] += 1
        self.n += 1
        if self.n >= self.sample:
            for r in self.rows:
                for j in range(len(r)):
                    r[j] >>= 1
            self.n //= 2

    def est(self, key: int) -> int:
        return min(self.rows[i][self._idx(key, i)] for i in range(self.depth))


@dataclass
class VramPolicy:
    name: str = "slru"
    core: float = 0.0       # share of each group's slots in the ranked core
    door: bool = False
    guard: bool = False

    @staticmethod
    def parse(spec: str) -> "VramPolicy":
        p = VramPolicy(spec)
        for part in spec.split("+"):
            if part == "slru":
                continue
            if part.startswith("core"):
                p.core = float(part[4:] or 50) / 100
            elif part == "door":
                p.door = True
            elif part == "guard":
                p.guard = True
            else:
                raise ValueError(f"unknown VRAM policy part {part!r} (slru, core<P>, door, guard)")
        return p


@dataclass
class StepOut:
    group: int
    row_hits: int
    n_ids: int
    uniq: list[int]         # sorted, as llama_moe_source::begin receives them
    n_hit: int
    fills: list[tuple[int, int]]
    expert_bytes: int

    @property
    def bytes(self) -> int:
        return len(self.fills) * self.expert_bytes


class _Group:
    def __init__(self, gs: G.GroupSlots, k: int, policy: VramPolicy, profile: dict | None, stride: int):
        g = gs.group
        self.index, self.n_expert, self.expert_bytes = g.index, g.n_expert, g.expert_bytes
        self.stride = stride                            # profile and core keys: layer*stride + expert
        self.n_floor = gs.n_floor
        spare = gs.n_slots - gs.n_floor                 # slots beyond what one 32-token ubatch needs
        n_core = 0
        if policy.core > 0 and profile:
            n_core = max(0, min(int(policy.core * gs.n_slots), spare))
        self.n_transient = 0
        if policy.door or policy.guard:
            want = gs.n_floor if policy.guard else k
            self.n_transient = max(0, min(want, spare - n_core))
        layers = set(g.layers)
        ranked = sorted(((c, key) for key, c in (profile or {}).items() if key // stride in layers and c > 0),
                        key=lambda x: (-x[0], x[1]))
        self.core = {key for _, key in ranked[:n_core]}
        self.n_core = len(self.core)
        self.slru = SLRU(g.layers, g.n_expert, gs.n_slots - self.n_core - self.n_transient)
        self.sketch = CountMin4(len(g.layers) * g.n_expert, 10 * gs.n_slots) if policy.door else None
        self.policy = policy

    def step(self, s: Step) -> StepOut | None:
        il, core = s.layer, self.core
        sm = self.slru.slot_map[il]
        base = il * self.stride
        row_hits = 0
        for e in s.ids:
            row_hits += (sm[e] >= 0) or (base + e in core)
        uniq, seen = [], set()
        for e in s.ids:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        rest = [e for e in uniq if base + e not in core] if core else uniq
        admit, n_tr = None, 0
        if self.sketch is not None:
            for e in rest:
                self.sketch.add(base + e)
        if self.n_transient:
            prompt = s.n_tokens > 8
            sk, guard = self.sketch, self.policy.guard
            if guard and prompt:
                def admit(new, old):
                    return False
            elif sk is not None:
                def admit(new, old):
                    return sk.est(new) >= sk.est(old)       # decline only for a victim used more often
            n_tr = self.n_transient if admit is not None else 0
        r = self.slru.plan(il, rest, admit, n_tr)
        if r is None:
            return None
        fills, n_hit = r
        return StepOut(self.index, row_hits, len(s.ids), sorted(uniq), n_hit + (len(uniq) - len(rest)), fills, self.expert_bytes)


@dataclass
class VramRun:
    policy: str
    budget_mib: float
    slots: list[G.GroupSlots]
    out: list[StepOut | None]           # aligned with trace.steps; None for steps that do not touch the cache
    groups: list[_Group] = field(default_factory=list)


def run_vram(trace: Trace, geom: G.Geometry, budget_mib: float, policy: str | VramPolicy = "slru",
             profile: dict | None = None, n_ubatch: int = 512, alignment: int = G.CUDA_ALIGNMENT) -> VramRun:
    pol = VramPolicy.parse(policy) if isinstance(policy, str) else policy
    slots = G.cache_slots(geom, budget_mib, n_ubatch, alignment)
    stride = max(gs.group.n_expert for gs in slots)
    groups = [_Group(gs, geom.n_expert_used, pol, profile, stride) for gs in slots]
    layer_group = {il: gr for gs, gr in zip(slots, groups) for il in gs.group.layers}
    out: list[StepOut | None] = []
    for s in trace.steps:
        gr = layer_group.get(s.layer)
        if gr is None or s.kind != "S" or s.n_tokens > G.MAX_BATCH:
            out.append(None)
            continue
        out.append(gr.step(s))
    return VramRun(pol.name, budget_mib, slots, out, groups)


# --------------------------------------------------------------------------------------------- prompts, tokens, stats

@dataclass
class Split:
    train: set[int]         # request indices
    test: set[int]
    train_keys: list[str]
    test_keys: list[str]


def split_prompts(trace: Trace, test_keys: list[str] | None = None, phases=("warm", "decode", "post", "all")) -> Split:
    """Train/test split by prompt, never by token: every request of one prompt lands on one side. Default: the prompts
    in sorted order alternate train, test, train, ... (decode:0 train, decode:1 test, ...). Deterministic."""
    keys = sorted({prompt_key(r) for r in trace.requests if r.get("phase", "all") in phases},
                  key=lambda x: (x.split(":")[0], int(x.split(":")[1]) if x.split(":")[1].isdigit() else x))
    if test_keys is None:
        test_keys = keys[1::2] if len(keys) > 1 else keys
    test_keys = [k for k in keys if k in set(test_keys)]
    train_keys = [k for k in keys if k not in set(test_keys)]
    train = {i for i, r in enumerate(trace.requests) if prompt_key(r) in set(train_keys)}
    test = {i for i, r in enumerate(trace.requests) if prompt_key(r) in set(test_keys)}
    return Split(train, test, train_keys, test_keys)


def tokens_of(trace: Trace, reqs: set[int] | None, n_layers: int, k: int, phases=None) -> list[list[int]]:
    idx = [i for i, s in enumerate(trace.steps)
           if s.req >= 0 and (reqs is None or s.req in reqs)
           and (phases is None or trace.requests[s.req].get("phase", "all") in phases)]
    toks = []
    # tokens never span two requests
    by_req: dict[int, list[int]] = {}
    for i in idx:
        by_req.setdefault(trace.steps[i].req, []).append(i)
    for r in sorted(by_req, key=lambda r: by_req[r][0]):
        toks += decode_tokens(trace.steps, by_req[r], k, n_layers)
    return toks


def hit_stats(run: VramRun, tokens: list[list[int]]) -> dict:
    hits, bw, mib = [], [], []
    for t in tokens:
        outs = [run.out[i] for i in t if run.out[i] is not None]
        if not outs:
            continue
        n = sum(o.n_ids for o in outs)
        hits.append(sum(o.row_hits for o in outs) / n)
        tot = sum(o.n_ids * o.expert_bytes for o in outs)
        bw.append(sum(o.row_hits * o.expert_bytes for o in outs) / tot)
        mib.append(sum(o.bytes for o in outs) / G.MiB)
    if not hits:
        return {"tokens": 0}
    return {"tokens": len(hits), "hit_median": q(hits, .5), "hit_p10": q(hits, .1), "hit_mean": st.mean(hits),
            "byte_hit_mean": st.mean(bw), "mib_up_per_token": st.mean(mib), "mib_up_median": q(mib, .5)}


def selection_profile(trace: Trace, tokens: list[list[int]], n_expert: int) -> dict[int, int]:
    """Decode selections per key layer*n_expert + expert over the given tokens (what P4's counts record)."""
    prof: dict[int, int] = {}
    for t in tokens:
        for i in t:
            s = trace.steps[i]
            for e in s.ids:
                key = s.layer * n_expert + e
                prof[key] = prof.get(key, 0) + 1
    return prof


# ------------------------------------------------------------------------------------------- the miss stream and tiers

@dataclass
class MissStep:
    step: int
    layer: int
    group: int
    decode: bool            # ids <= 8 x n_expert_used: the tier counts, admits and sets reference bits
    uniq: list[int]
    fills: list[int]        # experts, in upload order
    expert_bytes: int


def miss_stream(trace: Trace, run: VramRun, k: int) -> list[MissStep]:
    out = []
    for i, (s, o) in enumerate(zip(trace.steps, run.out)):
        if o is None:
            continue
        out.append(MissStep(i, s.layer, o.group, len(s.ids) <= 8 * k, o.uniq, [e for e, _ in o.fills], o.expert_bytes))
    return out


def miss_profile(trace: Trace, misses: list[MissStep], reqs: set[int], n_expert: int, decode_only: bool = True) -> dict[int, int]:
    prof: dict[int, int] = {}
    for m in misses:
        if trace.steps[m.step].req in reqs and (m.decode or not decode_only):
            for e in m.fills:
                key = m.layer * n_expert + e
                prof[key] = prof.get(key, 0) + 1
    return prof


class TierGeometry:
    """Slot sizes of the P3 source: each tensor's region is round_up(nb2, sector) + sector, a slot holds all of an
    expert's regions; the ring has ring_mib / (largest slot) slots; the tier is split among groups by host bytes."""

    def __init__(self, geom: G.Geometry, groups: list[G.Group], sector: int = 4096, ring_mib: float = 64):
        self.groups = groups
        self.n_expert = max(g.n_expert for g in groups)
        region = lambda nb2: (nb2 + sector - 1) // sector * sector + sector       # noqa: E731
        self.slot_bytes = [max(sum(region(t.nb2) for t in geom.layers[il].tensors) for il in g.layers) for g in groups]
        self.ring_slot = max(self.slot_bytes)
        self.ring_n = max(1, int(ring_mib * G.MiB) // self.ring_slot)
        self.group_of = {il: g.index for g in groups for il in g.layers}

    def tier_slots(self, tier_mib: float) -> list[int]:
        size = int(tier_mib * G.MiB)
        total = sum(g.host_bytes for g in self.groups)
        return [min(int(float(size) * g.host_bytes / total) // sb, len(g.layers) * g.n_expert)
                for g, sb in zip(self.groups, self.slot_bytes)]


class CurrentTier:
    """llama_moe_source in tier mode (patch 0003), the policy state only (no bytes)."""
    DECAY = 256

    def __init__(self, tg: TierGeometry, tier_mib: float):
        self.tg = tg
        self.n_slots = tg.tier_slots(tier_mib) if tier_mib > 0 else []
        self.enabled = tier_mib > 0
        self.key = [[-1] * n for n in self.n_slots]
        self.ref = [[0] * n for n in self.n_slots]
        self.busy = [[0] * n for n in self.n_slots]
        self.hand = [0] * len(self.n_slots)
        ne = tg.n_expert
        self.tier_slot = {il: [-1] * ne for il in tg.group_of}
        self.count = {il: [0] * ne for il in tg.group_of}
        self.n_decode = {il: 0 for il in tg.group_of}
        self.gen = 1
        self.il, self.decode, self.ring_next = -1, False, 0
        self.n_wrap = 0

    def begin(self, il: int, decode: bool, uniq: list[int]) -> None:
        self.il, self.decode, self.ring_next = il, decode, 0
        self.gen += 1
        if decode and self.enabled and il in self.tg.group_of:
            c = self.count[il]
            for e in uniq:
                if c[e] < 65535:
                    c[e] += 1
            self.n_decode[il] += 1
            if self.n_decode[il] >= self.DECAY:
                for e in range(len(c)):
                    c[e] >>= 1
                self.n_decode[il] = 0

    def _victim(self, g: int, e: int) -> int:
        n = self.n_slots[g]
        key, ref, busy, ne = self.key[g], self.ref[g], self.busy[g], self.tg.n_expert
        for _ in range(2 * n):
            s = self.hand[g]
            self.hand[g] = (s + 1) % n
            if busy[s] == self.gen:
                continue
            if key[s] < 0:
                return s
            if ref[s]:
                ref[s] = 0
                continue
            k = key[s]
            return s if self.count[self.il][e] > self.count[k // ne][k % ne] else -1
        return -1

    def fill(self, e: int) -> str:
        il = self.il
        g = self.tg.group_of[il]
        if self.enabled and self.n_slots[g] > 0:
            ts = self.tier_slot[il][e]
            if ts >= 0:
                self.busy[g][ts] = self.gen
                if self.decode:
                    self.ref[g][ts] = 1
                return "t"
            if self.decode:
                v = self._victim(g, e)
                if v >= 0:
                    old = self.key[g][v]
                    if old >= 0:
                        self.tier_slot[old // self.tg.n_expert][old % self.tg.n_expert] = -1
                    self.key[g][v] = il * self.tg.n_expert + e
                    self.ref[g][v] = 1
                    self.busy[g][v] = self.gen
                    self.tier_slot[il][e] = v
                    return "a"
        if self.ring_next == self.tg.ring_n:
            self.ring_next = 0
            self.gen += 1
            self.n_wrap += 1
        self.ring_next += 1
        return "r"

    def replay(self, misses: list[MissStep]) -> list[list[str]]:
        out = []
        for m in misses:
            self.begin(m.layer, m.decode, m.uniq)
            out.append([self.fill(e) for e in m.fills])
        return out

    def contents(self) -> set[int]:
        return {k for keys in self.key for k in keys if k >= 0}


def static_contents(tg: TierGeometry, profile: dict[int, int], tier_mib: float, expert_bytes: dict[int, int]) -> set[int]:
    """Greedy by profile count x expert bytes per tier byte, across groups, until the tier is full."""
    cap = int(tier_mib * G.MiB)
    ne = tg.n_expert
    items = []
    for key, c in profile.items():
        il = key // ne
        if il in tg.group_of and c > 0:
            sb = tg.slot_bytes[tg.group_of[il]]
            items.append((-(c * expert_bytes[il]) / sb, key, sb))
    items.sort()
    out, used = set(), 0
    for _, key, sb in items:
        if used + sb <= cap:
            out.add(key)
            used += sb
    return out


def static_replay(misses: list[MissStep], contents: set[int], ne: int) -> list[list[str]]:
    return [["t" if m.layer * ne + e in contents else "r" for e in m.fills] for m in misses]


def belady_replay(misses: list[MissStep], tg: TierGeometry, tier_mib: float, counted=None) -> list[list[str]]:
    """Belady's MIN with bypass: on a miss, admit only by evicting experts whose next use is later than the newcomer's,
    furthest first; never admit an expert that is not used again. counted(miss_step) selects the misses the metric
    counts; the others are bypassed, which an optimal policy for that metric is free to do, so the result bounds every
    policy from above (exactly for equal-size experts, approximately across groups of different slot sizes)."""
    ne = tg.n_expert
    ev = [(mi, j, m.layer * ne + e) for mi, m in enumerate(misses) if counted is None or counted(m) for j, e in enumerate(m.fills)]
    nxt = [INF] * len(ev)
    last: dict[int, int] = {}
    for i in range(len(ev) - 1, -1, -1):
        key = ev[i][2]
        nxt[i] = last.get(key, INF)
        last[key] = i
    cap, used = int(tier_mib * G.MiB), 0
    size = lambda key: tg.slot_bytes[tg.group_of[key // ne]]                     # noqa: E731
    cached: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    out = [["r"] * len(m.fills) for m in misses]
    for i, (mi, j, key) in enumerate(ev):
        nu = nxt[i]
        if key in cached:
            out[mi][j] = "t"
            cached[key] = nu
            heapq.heappush(heap, (-nu, key))
            continue
        if nu == INF:
            continue
        sz = size(key)
        while used + sz > cap and heap:
            negnu, k2 = heap[0]
            if cached.get(k2) != -negnu:
                heapq.heappop(heap)             # stale entry
                continue
            if -negnu <= nu:
                break                           # everything cached is needed sooner
            heapq.heappop(heap)
            del cached[k2]
            used -= size(k2)
        if used + sz <= cap:
            cached[key] = nu
            used += sz
            heapq.heappush(heap, (-nu, key))
    return out


def tier_share(trace: Trace, misses: list[MissStep], src: list[list[str]], reqs: set[int] | None, decode_tokens_only: bool) -> dict:
    """Share of VRAM-miss bytes the tier served ('t'), over the misses of the given requests."""
    tot = hit = 0
    for m, srcs in zip(misses, src):
        s = trace.steps[m.step]
        if reqs is not None and s.req not in reqs:
            continue
        if decode_tokens_only and s.n_tokens != 1:
            continue
        tot += len(srcs) * m.expert_bytes
        hit += sum(x == "t" for x in srcs) * m.expert_bytes
    return {"miss_mib": tot / G.MiB, "share": hit / tot if tot else float("nan")}


def fetch_ms(share: float, points=((0.0, 46.0), (75.0, 15.0), (90.0, 8.0))) -> float:
    """Per-token fetch time from the probe's linear model (points given by the M1.1 probe write-up, not by replay):
    piecewise linear in the tier's share of miss bytes, extrapolated with the last slope, never below 0."""
    x = share * 100
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return max(0.0, y1 + (y1 - y0) * (x - x1) / (x1 - x0))


# ----------------------------------------------------------------------------------------- check against a real trace

def check(trace: Trace, run: VramRun, csv_rows: list[dict] | None, fill_rows: list[dict] | None) -> dict:
    """Per-step comparison of the replay with the server's own S lines (row_hits, uniq, uniq_hits, fills, bytes) and
    fill lines (expert and slot of every upload, in order). At the traced budget an exact replay matches every step."""
    res: dict = {}
    by_key = {(s.t_us, s.layer): i for i, s in enumerate(trace.steps) if s.kind == "S"}
    if csv_rows is not None:
        n = ok = missing = 0
        first = None
        for r in csv_rows:
            if r.get("kind") != "S":
                continue
            n += 1
            i = by_key.get((r["t_us"], r["layer"]))
            o = run.out[i] if i is not None else None
            if o is None:
                missing += 1
                continue
            mine = {"n_ids": o.n_ids, "row_hits": o.row_hits, "uniq": len(o.uniq), "uniq_hits": o.n_hit,
                    "fills": len(o.fills), "bytes": o.bytes}
            if all(mine[c] == r[c] for c in mine):
                ok += 1
            elif first is None:
                first = {"t_us": r["t_us"], "layer": r["layer"], "server": {c: r[c] for c in mine}, "replay": mine}
        res["s_lines"] = {"n": n, "match": ok, "not_in_ids_trace": missing, "first_mismatch": first}
    if fill_rows is not None:
        want: dict[tuple, list] = {}
        for r in fill_rows:
            if r.get("kind") == "F":
                want.setdefault((r["t_us"], r["layer"]), []).append((r["expert"], r["slot"]))
        n = ok = missing = 0
        first = None
        for key, fl in want.items():
            n += 1
            i = by_key.get(key)
            o = run.out[i] if i is not None else None
            if o is None:
                missing += 1
                continue
            if [(e, s) for e, s in o.fills] == fl:
                ok += 1
            elif first is None:
                first = {"t_us": key[0], "layer": key[1], "server": fl[:12], "replay": o.fills[:12]}
        res["fill_steps"] = {"n": n, "match": ok, "not_in_ids_trace": missing, "first_mismatch": first}
    return res


def as_server_rows(trace: Trace, run: VramRun) -> tuple[list[dict], list[dict]]:
    """The S lines and F lines a server would have written for this replay (timings left empty): used to test the
    check, and by `synth` to write SYNTHETIC P0/fill traces."""
    s_rows, f_rows = [], []
    for s, o in zip(trace.steps, run.out):
        if o is None:
            continue
        s_rows.append({"kind": "S", "t_us": s.t_us, "layer": s.layer, "n_ids": o.n_ids, "row_hits": o.row_hits,
                       "uniq": len(o.uniq), "uniq_hits": o.n_hit, "fills": len(o.fills), "bytes": o.bytes})
        f_rows += [{"kind": "F", "t_us": s.t_us, "layer": s.layer, "n_ids": o.n_ids, "expert": e, "slot": sl, "src": "m",
                    "bytes": o.expert_bytes} for e, sl in o.fills]
    return s_rows, f_rows


def measured_hit(trace: Trace, csv_rows: list[dict], reqs: set[int] | None, n_layers: int, k: int, phases=None) -> dict:
    """The server's own per-token hit rate from its S lines, with analyse_m10.py's token selection."""
    import bisect
    starts = sorted((r["t0_us"], r["t1_us"], i) for i, r in enumerate(trace.requests))
    t0s = [x[0] for x in starts]

    def req_of(t):
        j = bisect.bisect_right(t0s, t) - 1
        return starts[j][2] if j >= 0 and t <= starts[j][1] else -1
    rows = [r for r in csv_rows if r.get("kind") == "S" and r["n_ids"] <= k]
    by_req: dict[int, list[dict]] = {}
    for r in rows:
        ri = req_of(r["t_us"])
        if ri < 0 or (reqs is not None and ri not in reqs):
            continue
        if phases is not None and trace.requests[ri].get("phase", "all") not in phases:
            continue
        by_req.setdefault(ri, []).append(r)
    hits = []
    for ri in sorted(by_req, key=lambda x: by_req[x][0]["t_us"]):
        cur, prev, toks = [], None, []
        for r in by_req[ri]:
            if prev is not None and r["layer"] <= prev:
                toks.append(cur)
                cur = []
            cur.append(r)
            prev = r["layer"]
        if cur:
            toks.append(cur)
        for t in toks:
            if len(t) >= 0.8 * n_layers:
                hits.append(sum(x["row_hits"] for x in t) / sum(x["n_ids"] for x in t))
    if not hits:
        return {"tokens": 0}
    return {"tokens": len(hits), "hit_median": q(hits, .5), "hit_p10": q(hits, .1), "hit_mean": st.mean(hits)}


# ------------------------------------------------------------------------------------------------------------- report

DEFAULT_P2 = ("core25", "core50", "door", "guard", "core50+door")
TIER_POLICIES = ("current", "warm", "static-sel", "static-miss", "oracle-static", "belady")
DEFAULT_TIERS = (1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384)


def run_all(trace: Trace, geom: G.Geometry, budget_mib: float, *, n_ubatch: int = 512, alignment: int = G.CUDA_ALIGNMENT,
            test_keys: list[str] | None = None, p2: tuple = DEFAULT_P2, tiers: tuple = DEFAULT_TIERS,
            tier_policies: tuple = TIER_POLICIES, ring_mib: float = 64, sector: int = 4096,
            csv_rows: list[dict] | None = None, fill_rows: list[dict] | None = None, progress=None) -> dict:
    say = progress or (lambda *_: None)
    k = geom.n_expert_used
    groups = geom.groups()
    n_layers = len(geom.layers)
    ne = max(g.n_expert for g in groups)
    sp = split_prompts(trace, test_keys)
    rep: dict = {"synthetic": bool(geom.source.get("synthetic")), "budget_mib": budget_mib, "n_ubatch": n_ubatch,
                 "model": geom.model, "n_expert_used": k,
                 "n_layers": n_layers, "steps": len(trace.steps), "truncated_bytes": trace.truncated_bytes,
                 "split": {"train": sp.train_keys, "test": sp.test_keys}}
    say("slru")
    base = run_vram(trace, geom, budget_mib, "slru", None, n_ubatch, alignment)
    rep["slots"] = [{"group": s.group.index, "type": s.group.type, "layers": len(s.group.layers), "slots": s.n_slots,
                     "floor": s.n_floor, "at_floor": s.at_floor, "mib": round(s.mib, 2)} for s in base.slots]
    decode_phases = ("decode", "all")
    tok_all = tokens_of(trace, None, n_layers, k, decode_phases)
    tok_test = tokens_of(trace, sp.test, n_layers, k, decode_phases)
    tok_train = tokens_of(trace, sp.train, n_layers, k, decode_phases)
    test_decode = {i for i in sp.test if trace.requests[i].get("phase", "all") in decode_phases}   # the tokens hit rates use
    rep["slru"] = {"decode_all": hit_stats(base, tok_all), "decode_test": hit_stats(base, tok_test)}
    if csv_rows is not None or fill_rows is not None:
        rep["check"] = check(trace, base, csv_rows, fill_rows)
        if csv_rows is not None:
            meas = measured_hit(trace, csv_rows, None, n_layers, k, decode_phases)
            rep["check"]["server_decode_all"] = meas
            if meas.get("tokens") and rep["slru"]["decode_all"].get("tokens"):
                rep["check"]["median_diff_points"] = 100 * (rep["slru"]["decode_all"]["hit_median"] - meas["hit_median"])
    prof_sel = selection_profile(trace, tok_train, ne)
    rep["p2"] = {}
    ref = rep["slru"]["decode_test"]
    for spec in p2:
        say(f"p2 {spec}")
        if VramPolicy.parse(spec).core > 0 and not prof_sel:
            rep["p2"][spec] = {"error": "no training prompts, so no routing profile"}
            continue
        run = run_vram(trace, geom, budget_mib, spec, prof_sel, n_ubatch, alignment)
        h = hit_stats(run, tok_test)
        if h.get("tokens") and ref.get("tokens"):
            h["delta_median_points"] = 100 * (h["hit_median"] - ref["hit_median"])
            h["delta_p10_points"] = 100 * (h["hit_p10"] - ref["hit_p10"])
            h["delta_mib_per_token"] = h["mib_up_per_token"] - ref["mib_up_per_token"]
        h["core_slots"] = [gr.n_core for gr in run.groups]
        h["transient_slots"] = [gr.n_transient for gr in run.groups]
        rep["p2"][spec] = h
    # RAM tier on the SLRU's miss stream
    tg = TierGeometry(geom, groups, sector, ring_mib)
    misses = miss_stream(trace, base, k)
    eb = {il: geom.layers[il].expert_bytes for il in geom.layers}
    prof_miss = miss_profile(trace, misses, sp.train, ne)
    prof_test = miss_profile(trace, misses, sp.test, ne)
    train_misses = [m for m in misses if trace.steps[m.step].req in sp.train]
    rep["tier"] = {"ring_slots": tg.ring_n, "slot_mib": [round(b / G.MiB, 3) for b in tg.slot_bytes],
                   "miss_mib_decode_test": tier_share(trace, misses, [["r"] * len(m.fills) for m in misses], test_decode, True)["miss_mib"],
                   "by_size": {}}
    for mib in tiers:
        row = {"tier_slots": tg.tier_slots(mib)}
        for pol in tier_policies:
            say(f"tier {mib} {pol}")
            if pol == "current":
                src = CurrentTier(tg, mib).replay(misses)
            elif pol == "warm":
                if not train_misses:
                    continue
                t = CurrentTier(tg, mib)
                t.replay(train_misses)
                src = t.replay(misses)
            elif pol in ("static-sel", "static-miss", "oracle-static"):
                prof = {"static-sel": prof_sel, "static-miss": prof_miss, "oracle-static": prof_test}[pol]
                if not prof:
                    continue
                src = static_replay(misses, static_contents(tg, prof, mib, eb), ne)
            elif pol == "belady":
                src = belady_replay(misses, tg, mib, lambda m: trace.steps[m.step].req in test_decode and trace.steps[m.step].n_tokens == 1)
            else:
                raise ValueError(f"unknown tier policy {pol!r}")
            d = tier_share(trace, misses, src, test_decode, True)
            a = tier_share(trace, misses, src, sp.test, False) if pol != "belady" else {"share": None}
            row[pol] = {"share_decode_test": d["share"], "share_all_test": a["share"],
                        "fetch_ms_model": fetch_ms(d["share"]) if d["share"] == d["share"] else None}
            if pol == "current":
                row[pol]["share_all_session"] = tier_share(trace, misses, src, None, False)["share"]
        rep["tier"]["by_size"][mib] = row
    reach = {}
    for pol in tier_policies:
        for thr in (0.75, 0.90):
            hit = [mib for mib in tiers if (rep["tier"]["by_size"][mib].get(pol) or {}).get("share_decode_test", 0) >= thr]
            reach.setdefault(pol, {})[f"{int(thr * 100)}%"] = min(hit) if hit else None
    rep["tier"]["reaches"] = reach
    return rep


def _pct(x, d=1):
    return "-" if x is None or x != x else f"{100 * x:.{d}f}%"


def to_markdown(rep: dict, label: str = "") -> str:
    L = []
    if rep.get("synthetic"):
        L += ["**SYNTHETIC trace and geometry: every number below is a property of the generator, not of any model.**", ""]
    if label:
        L += [f"**{label}**", ""]
    L += [f"Replay at `--moe-cache-mib {rep['budget_mib']:g}`, n_ubatch {rep['n_ubatch']}, {rep['n_layers']} MoE layers, "
          f"{rep['n_expert_used']} experts per token, {rep['steps']} trace steps"
          + (f", {rep['truncated_bytes']} trailing bytes dropped" if rep['truncated_bytes'] else "") + ".",
          f"Split by prompt: train {', '.join(rep['split']['train']) or '-'}; test {', '.join(rep['split']['test']) or '-'}.", "",
          "| group | type | layers | slots | floor | MiB |", "|---|---|---|---|---|---|"]
    for s in rep["slots"]:
        L.append(f"| {s['group']} | {s['type']} | {s['layers']} | {s['slots']}{' (at floor)' if s['at_floor'] else ''} | {s['floor']} | {s['mib']:.2f} |")
    L += ["", "**Upstream SLRU (exact replay)**", "", "| tokens | set | hit median | hit p10 | byte-weighted | MiB up/token |", "|---|---|---|---|---|---|"]
    for name, h in (("all decode prompts", rep["slru"]["decode_all"]), ("test prompts", rep["slru"]["decode_test"])):
        if h.get("tokens"):
            L.append(f"| {h['tokens']} | {name} | {_pct(h['hit_median'])} | {_pct(h['hit_p10'])} | {_pct(h['byte_hit_mean'])} | {h['mib_up_per_token']:.1f} |")
    if "check" in rep:
        c = rep["check"]
        L += ["", "**Check against the server's own trace**", ""]
        if "s_lines" in c:
            s = c["s_lines"]
            L.append(f"- S lines: {s['match']} of {s['n']} identical (row_hits, uniq, uniq_hits, fills, bytes); {s['not_in_ids_trace']} not in the ids trace")
            if s["first_mismatch"]:
                L.append(f"  - first mismatch: {s['first_mismatch']}")
        if "fill_steps" in c:
            f = c["fill_steps"]
            L.append(f"- fill lines: {f['match']} of {f['n']} steps with identical (expert, slot) uploads; {f['not_in_ids_trace']} not in the ids trace")
            if f["first_mismatch"]:
                L.append(f"  - first mismatch: {f['first_mismatch']}")
        if c.get("server_decode_all", {}).get("tokens"):
            m = c["server_decode_all"]
            L.append(f"- server, all decode prompts: median {_pct(m['hit_median'])}, p10 {_pct(m['hit_p10'])} over {m['tokens']} tokens; "
                     f"replay minus server: {c.get('median_diff_points', float('nan')):+.2f} points (exit check: within 3)")
    L += ["", "**P2 candidates against SLRU at the same budget (test prompts)**", "",
          "| policy | hit median | hit p10 | delta median | delta p10 | MiB up/token | core slots | transient slots |", "|---|---|---|---|---|---|---|---|"]
    for spec, h in rep["p2"].items():
        if not h.get("tokens"):
            L.append(f"| {spec} | {h.get('error', 'no tokens')} | | | | | | |")
            continue
        L.append(f"| {spec} | {_pct(h['hit_median'])} | {_pct(h['hit_p10'])} | {h.get('delta_median_points', 0):+.2f} | "
                 f"{h.get('delta_p10_points', 0):+.2f} | {h['mib_up_per_token']:.1f} | {'/'.join(map(str, h['core_slots']))} | "
                 f"{'/'.join(map(str, h['transient_slots']))} |")
    t = rep["tier"]
    pols = [p for p in TIER_POLICIES if any(p in r for r in t["by_size"].values())]
    L += ["", f"**RAM tier in front of the SSD, serving SLRU misses (share of decode-token miss bytes, test prompts; "
          f"{t['miss_mib_decode_test']:.0f} MiB of such misses)**", "",
          "| tier MiB | " + " | ".join(pols) + " | fetch ms/token (probe model) for current / best |",
          "|---|" + "---|" * len(pols) + "---|"]
    for mib, row in t["by_size"].items():
        cells = [_pct((row.get(p) or {}).get("share_decode_test")) for p in pols]
        best = max((row[p]["share_decode_test"] for p in pols if p in row and p not in ("oracle-static", "belady")
                    and row[p]["share_decode_test"] == row[p]["share_decode_test"]), default=float("nan"))
        cur = (row.get("current") or {}).get("fetch_ms_model")
        L.append(f"| {mib} | " + " | ".join(cells) + f" | {cur:.1f} / {fetch_ms(best):.1f} |" if cur is not None and best == best else
                 f"| {mib} | " + " | ".join(cells) + " | - |")
    L += ["", "Smallest tier size in the sweep that reaches the share (decode tokens, test prompts):", "",
          "| policy | 75% | 90% |", "|---|---|---|"]
    for p in pols:
        r = t["reaches"].get(p, {})
        L.append(f"| {p} | {r.get('75%') or 'not reached'} | {r.get('90%') or 'not reached'} |")
    return "\n".join(L) + "\n"
