"""Writes the inputs of the tier cross-check (tier_xcheck.sh) from a SYNTHETIC trace: python -I make_xcheck_inputs.py OUTDIR

Three cases, each a miss stream of the replay's exact SLRU on a synthetic Zipf trace, with expert sizes that are not
sector multiples (so the region rounding matters), a small ring (so steps wrap it) and tiers from tiny to large."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from marian import geometry as G, idstrace, replay, synth  # noqa: E402


def geom(n_layers, n_expert, k):
    layers = {}
    for il in range(n_layers):
        nb = (9000, 9000, 7000) if il % 3 else (5000, 5000, 7000)
        layers[il] = G.Layer(il, n_expert, [G.ExpertTensor(r, "x", (0, 0), b) for r, b in zip(("gate", "up", "down"), nb)])
    return G.Geometry(k, layers, "SYNTHETIC")


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, tier_bytes, ring_bytes, seed in (("tiny", 300_000, 160_000, 3), ("mid", 2_000_000, 100_000, 4), ("big", 9_000_000, 400_000, 5)):
        sp = synth.SynthSpec(n_layers=6, n_expert=256, k=4, n_prompts=5, prompt_tokens=60, decode_tokens=60, seed=seed)
        tr, reqs, _ = synth.synth_trace(sp)
        idstrace.assign_requests(tr, reqs)
        g = geom(6, 256, 4)
        run = replay.run_vram(tr, g, 12, "slru")
        misses = replay.miss_stream(tr, run, 4)
        lines = [f"6 256 {ring_bytes} {tier_bytes}"]
        gof = {il: grp.index for grp in g.groups() for il in grp.layers}
        for il in range(6):
            lines.append(f"{gof[il]} 3 " + " ".join(str(t.nb2) for t in g.layers[il].tensors))
        for m, o in zip(misses, [o for o in run.out if o is not None]):
            lines.append(f"{m.layer} {int(m.decode)} {len(m.uniq)} " + " ".join(map(str, m.uniq)) + f" {len(o.fills)} "
                         + " ".join(f"{e} {s}" for e, s in o.fills))
        (out / f"tier-xcheck-{name}.in").write_text("\n".join(lines) + "\n")
        print(name, len(misses), "steps", sum(len(m.fills) for m in misses), "fills")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
