"""SYNTHETIC routing traces for testing `marian replay` (no model, no GPU): every number they produce is synthetic.

The session has the shape of run_arm_m11.ps1 with phases warm,decode: the server's 2-token warm-up (outside every
request), a 'warm' request (decode prompt 0, `warm_tokens` tokens), then `n_prompts` decode requests, each a prompt
prefill (ubatches of `n_ubatch`; chunks above 32 tokens are large-batch L steps, the rest slot-map S steps) and
`decode_tokens` single-token steps. Routing per layer: Zipf(`zipf`) weights over a seeded permutation of the experts,
blended with a per-topic permutation (`topic_mix`, topic = prompt index mod `n_topics`); each of the previous token's
experts at a layer is kept with probability `stick`, the rest drawn by weight without replacement. Deterministic for a
seed.
"""
from __future__ import annotations

import bisect
import json
import random
from array import array
from dataclasses import dataclass
from pathlib import Path

from . import geometry as G
from .idstrace import Step, Trace, write_ids


@dataclass
class SynthSpec:
    n_layers: int = 8
    n_expert: int = 64
    k: int = 4
    zipf: float = 1.1
    topic_mix: float = 0.3
    n_topics: int = 3
    stick: float = 0.3
    n_prompts: int = 6
    prompt_tokens: int = 70
    decode_tokens: int = 48
    warm_tokens: int = 16
    n_ubatch: int = 512
    seed: int = 1


def synth_geometry(n_layers: int, n_expert: int, k: int) -> G.Geometry:
    """Two layouts, like Swift's main groups: even layers 'iq2_s'-sized, odd layers 'iq2_xxs'-sized experts."""
    layers = {}
    for il in range(n_layers):
        t = "iq2_s" if il % 2 == 0 else "iq2_xxs"
        nb = 524800 if t == "iq2_s" else 422400
        layers[il] = G.Layer(il, n_expert, [G.ExpertTensor("gate", t, (2560, 640), nb), G.ExpertTensor("up", t, (2560, 640), nb),
                                            G.ExpertTensor("down", "q2_0", (640, 2560), 460800)])
    return G.Geometry(k, layers, "SYNTHETIC", {"synthetic": True})


class _Router:
    def __init__(self, sp: SynthSpec, rng: random.Random):
        self.sp, self.rng = sp, rng
        ranks = [1.0 / (r + 1) ** sp.zipf for r in range(sp.n_expert)]
        self.cdf = {}
        for il in range(sp.n_layers):
            base = list(range(sp.n_expert))
            rng.shuffle(base)
            w0 = [0.0] * sp.n_expert
            for r, e in enumerate(base):
                w0[e] = ranks[r]
            for t in range(sp.n_topics):
                perm = list(range(sp.n_expert))
                rng.shuffle(perm)
                w = [(1 - sp.topic_mix) * w0[e] for e in range(sp.n_expert)]
                for r, e in enumerate(perm):
                    w[e] += sp.topic_mix * ranks[r]
                acc, c = 0.0, []
                for x in w:
                    acc += x
                    c.append(acc)
                self.cdf[il, t] = c
        self.prev: dict[int, list[int]] = {}

    def pick(self, il: int, topic: int, keep_prev: bool) -> list[int]:
        sp, rng, c = self.sp, self.rng, self.cdf[il, topic]
        out: list[int] = []
        if keep_prev:
            out = [e for e in self.prev.get(il, []) if rng.random() < sp.stick]
        while len(out) < sp.k:
            e = bisect.bisect_left(c, rng.random() * c[-1])
            e = min(e, sp.n_expert - 1)
            if e not in out:
                out.append(e)
        self.prev[il] = out
        return out


def synth_trace(sp: SynthSpec) -> tuple[Trace, list[dict], G.Geometry]:
    rng = random.Random(sp.seed)
    router = _Router(sp, rng)
    steps: list[Step] = []
    t = [1_000_000]

    def pass_(n_tokens: int, topic: int, decode: bool):
        for il in range(sp.n_layers):
            rows = [router.pick(il, topic, decode) for _ in range(n_tokens)]
            kind = "S" if n_tokens <= G.MAX_BATCH else "L"
            steps.append(Step(kind, t[0], il, n_tokens, sp.k, array("H", [e for r in rows for e in r])))
            t[0] += 50

    def prompt(n: int, topic: int):
        router.prev.clear()
        while n > 0:
            u = min(n, sp.n_ubatch)
            pass_(u, topic, False)
            n -= u

    requests: list[dict] = []

    def request(phase: str, idx: int, n_prompt: int, n_gen: int, topic: int):
        t[0] += 10_000
        r = {"tag": "synthetic", "phase": phase, "idx": idx, "t0_us": t[0], "ok": True, "prompt_n": n_prompt, "gen_n": n_gen}
        prompt(n_prompt, topic)
        for _ in range(n_gen):
            pass_(1, topic, True)
        r["t1_us"] = t[0]
        requests.append(r)
        t[0] += 10_000

    pass_(2, 0, False)                                      # the server's warm-up run, outside every request
    request("warm", 0, sp.prompt_tokens, sp.warm_tokens, 0)
    for i in range(sp.n_prompts):
        request("decode", i, sp.prompt_tokens + 7 * i, sp.decode_tokens, i % sp.n_topics)
    return Trace(steps), requests, synth_geometry(sp.n_layers, sp.n_expert, sp.k)


def write(outdir: str | Path, sp: SynthSpec) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    trace, reqs, geom = synth_trace(sp)
    write_ids(out / "synthetic.ids.bin", trace.steps, 1_000_000)
    (out / "synthetic.requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs), encoding="utf-8")
    geom.save(out / "synthetic.geometry.json")
    (out / "synthetic.spec.json").write_text(json.dumps(sp.__dict__, indent=1) + "\n", encoding="utf-8")
    return {"steps": len(trace.steps), "requests": len(reqs), "dir": str(out)}
