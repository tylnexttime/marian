# Marian

**Run local mixture-of-experts models by what your machine can actually do, measured, not guessed.**

Marian is named after a popular Polish given name. Status: **pre-release**; M1.1 and M1.2 measured (24/10/2026),
M1.3's calibration v1 built in Python (27/10/2026: calibration gate, cache arms, CPU thread sweep, live confirmation,
and serving through a front proxy with a memory governor), tested offline and run on one machine only; the first
prefill patches (0007-0009) measured the same day (register E51, E54), and the Python side taught the P3 prefill
source and a declared ubatch (offline tests only). The full
story, every measured table and the mistakes are in [docs/JOURNEY.md](docs/JOURNEY.md).

## What it does

Marian has two layers.

1. **M0, a measuring tool** (Python, standard library only). It measures how a model behaves on *this* machine:
   speed and memory for each context size, KV-cache type and GPU/CPU split. Then it picks the fastest configuration
   that fits a request and launches it. It never extrapolates: if nothing measured fits, it says so.
2. **M1, a thin fork of llama.cpp** (pinned at upstream commit [`63b4b303`](https://github.com/ggml-org/llama.cpp/commit/63b4b303)). Mixture-of-experts (MoE) models only use a
   few of their "experts" for each token. M1 keeps the most-used experts in GPU memory (VRAM) **one expert at a time,
   not one whole layer at a time**, and fetches the rest from system RAM or the SSD as fast as the machine allows. It
   builds on upstream llama.cpp's own expert cache (`--moe-cache-mib`). It is **exact by default**: speed-ups that can
   change an answer will only ever be opt-in switches. Self-calibration to the machine at hand is the design goal;
   calibration v1 (milestone M1.3) is built in Python and tested offline. Parts of it have run on one machine only;
   its memory governor has met a real competing allocation once (register E53), which changed its default to
   watch-only. Learned counts are not built.

**Minimum spec we are designing for:** 32 GB of RAM plus a GPU with 8 GB of VRAM (a project decision of
22/10/2026). The design is in [docs/design/DESIGN-M1.md](docs/design/DESIGN-M1.md).

## Why

A modern MoE model is often bigger than the GPU's memory, but each token touches only a few experts (8 of 128 in
Gemma 4 26B-A4B; 10 of 512 in the Qwen3.8-Flash-Next quantisation we test). So the model can run well if the right
experts are in VRAM and the misses arrive quickly. The catch is that **the best split depends on the machine, the
context length and how much RAM the rest of the desktop is using.** A guess that is right at 4K tokens can be badly
wrong at 8K.

Measured on a laptop with an 8 GB GPU: an MoE model ran at 31 tok/s with one GPU/CPU split for short prompts, but the
same split at an 8K context overflowed VRAM and fell to 6 tok/s, because Windows pages GPU memory into system RAM
instead of failing. The cure is not a better guess; it is a measured table per machine, model and context size.

## So what: the honest numbers

These are why a measuring tool matters. Sources and conditions for every figure are in
[docs/JOURNEY.md](docs/JOURNEY.md#results-appendix).

- **The expert cache wins hugely when RAM can hold the experts.** Gemma 4 26B-A4B on a GPU limited to 8 GB, with RAM
  free: **42.6 tok/s** with upstream's cache (2,500 MiB) against **16.8 tok/s** for plain layer offload
  (`-ncmoe 24`).
- **It loses when misses must come from the SSD.** The same model with about 25 GB of RAM taken by desktop work:
  **8.0 tok/s** with the cache against **15.0 tok/s** plain. The misses fault in through memory-mapped pages from the
  SSD (measured at 0.41 GB/s, against 6.6 GB/s from resident RAM and 10.3 GB/s from pinned RAM).
- **Headline: Marian's exact RAM tier roughly doubles decode against upstream's cache.** M1.1's miss source (an
  unbuffered staging ring plus a pinned RAM tier) takes the larger Qwen3.8-Flash-Next model on the 16 GB reference
  laptop from **7.3 / 6.3 tok/s** on upstream's path to **14.4-14.6 tok/s** with a 6 GiB pinned tier, and
  **13.4-13.7 tok/s right after a 16K-token prompt** (all-tokens rate, five interleaved rounds, every one above the
  12 tok/s target). Against stock llama.cpp's best layer offload for the same model (`-ncmoe 34`) that is about
  **1.5x faster warm and 3.4x faster after a long prompt**, in 5 of 5 rounds. The answers are identical to the
  all-in-VRAM reference down to every recorded log-probability (Gemma 4 26B, all three miss sources), which stock
  offload is not. The price, in the default configuration: **prefill 4-5x slower than stock** (about 3x slower
  than upstream's cache path), because prefill reads through the memory map and the pinned tier takes the page cache
  it needs. Two opt-in, exact changes measured on 27/10/2026 close that gap on this laptop: prefill misses through P3
  (16K prefill 35 -> 61 tok/s) and, on top of it, a declared ubatch of 2048 (157 tok/s at 16K, against 142 for stock
  at its default ubatch in the same session). Decode in that session ran 16-18% under the earlier session's figures,
  and the cause is not yet separated (see Limitations).
- **The practical question we were asked** (can that 8 GB / 32 GB laptop, with most of its RAM in use, run a 4-30B
  model at 20 to 30 tok/s?) **was answered yes by stock llama.cpp with good offload settings, not by Marian's cache**:
  Gemma 4 26B-A4B at IQ3_XXS with `-ncmoe 18` ran at 26.3 tok/s, and small dense models that fit in VRAM ran at
  65-90 tok/s. Even with M1.1's ring, the cache reached 13.4 tok/s there against 15.0 for plain offload (measured on
  the M1.0 build): **under
  heavy desktop RAM pressure, plain offload still wins.** A measuring tool that tells you the plain option wins is
  doing its job.

## How it fits together

```
 +------------------------------------------------------------------------------------+
 |  M0 (ours): machine fingerprint -> profile -> plan -> launch llama-server          |
 +------------------------------------------------------------------------------------+
 |  M1 patches (ours), small and off unless asked for:                                |
 |    P0/P0b trace   per-step hits, fills, bytes, host time, expert ids               |
 |    exactness floor  every expert group keeps enough slots for the largest step     |
 |    P3 miss source   unbuffered reads -> pinned staging ring -> pinned RAM tier     |
 |    tier safety      the pinned tier capped, and pinned in chunks of at most 1 GiB  |
 |    P3 prefill       large-batch misses from the same ring and tier (opt-in)        |
 +------------------------------------------------------------------------------------+
 |  upstream llama.cpp at 63b4b303: --moe-cache-mib per-expert VRAM cache (SLRU),     |
 |  MUL_MAT_ID on the cache bank, selective expert copy for large batches             |
 +------------------------------------------------------------------------------------+
```

| Part | Whose | State |
|---|---|---|
| Per-expert VRAM cache, SLRU policy, every expert computed on the GPU | upstream llama.cpp (PRs #29887, #30112, #30637) | used as the base |
| Selective expert copy for large batches | upstream (PR #29943) | used as is by default; patch 0008 serves its misses from P3 when asked (`--moe-prefill-source p3`) |
| M0: profiler, planner, launcher | Marian | done |
| P0 / P0b trace (`MARIAN_MOE_TRACE`, `MARIAN_MOE_TRACE_FILLS`) | Marian | built, used in every M1 run |
| Exactness floor (per-group slot floor, refusal below it) | Marian | built, passed its test |
| P3 miss source (`--moe-miss-source mmap\|ring\|tier`, `--moe-tier-mib`, `--moe-ring-mib`) | Marian | built, measured, opt-in (default stays `mmap`) |
| Tier safety (patch 0007): the tier is capped at min(request, 50% of physical RAM, commit room minus 4 GiB), pinned in chunks of at most 1 GiB, and stops growing at a chunk that cannot be pinned; one log line says what was pinned | Marian | built 27/10/2026, in the default build; no measured cost (register E51). The cap and the failed-chunk branch have never triggered, so they are untested |
| Prefill misses through P3 (patch 0008): batches above 32 tokens take each missed expert from its tier slot, else by unbuffered reads through the pinned ring, never by a memory-map fault | Marian | built 27/10/2026, exact (E51, E54). Swift with the 6 GiB tier: 16K prefill 34.7 -> 61.1 tok/s (+76%, three rounds). Slower than the memory map when the model fits in the page cache (Gemma 4 26B, a 5,428-token prompt: 260 against 573 tok/s, one run each) |
| Prefill source switch (patch 0009): `--moe-prefill-source mmap\|p3` | Marian | built 27/10/2026. Default `mmap`: upstream's uploads, byte for byte. `p3` turns 0008's path on and is refused without a ring or tier miss source. Calibration measures every ring and tier row with both and keeps the faster (Python side built 27/10/2026, offline tests only) |
| Declared ubatch (`-ub N -b N`), a profile parameter: `calibrate --ubatch N`, `serve --ubatch N` | Marian | Python side built 27/10/2026, offline tests only. Default 512 (no flag). A profile records its ubatch (schema 3), and a plan measured at one ubatch is refused at another |
| Calibration v1, Python side (M1.3): cache arms in the profiler (VRAM cliff finder, ring and tier ladder inside a RAM envelope), exact-class per row, calibration gate, launch checks, `serve --stock` rollback, `marian diag` | Marian | built, offline tests pass; one smoke run on one machine |
| Calibration v1, serving side (M1.3): CPU thread sweep, live check and `serve --confirm` (keep-if-faster A/B), front proxy (one sequence, 503 with a reason, streaming pass-through), memory governor (watch by default; restart on a measured slowdown, opt-in), prompt-prefix reuse off by default (register test E52) | Marian | built 27/10/2026, offline tests pass (fake engine and server); see Limitations for what has run live |
| Learned counts (rest of M1.3); ranked core, warm start, TinyLFU doorkeeper in VRAM (P2), drain removal, prefetch | Marian | designed, not built |

The patches are in [docs/m1.0/patches](docs/m1.0/patches) (P0), [docs/m1.1/patches](docs/m1.1/patches) (0001
exactness floor, 0002 fill trace, 0003 P3 minimal), [docs/m1.2/patches](docs/m1.2/patches) (0004 routing-id trace,
0005 its review fixes) and [docs/m1.3/patches](docs/m1.3/patches) (0007 tier safety, 0008 prefill source, 0009 the
prefill-source switch), as `git format-patch` files on top of `63b4b303`, applied in that order. There is no 0006.

## Inspirations and prior art

- **llama.cpp and ggml**, [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Its
  experimental MoE cache arrived in [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) ("llama : add a GPU
  cache for MoE experts kept in host memory"), multi-GPU in [#30112](https://github.com/ggml-org/llama.cpp/pull/30112)
  ("llama: support MoE cache over multiple GPUs"), and the SLRU policy in
  [#30637](https://github.com/ggml-org/llama.cpp/pull/30637) ("Replace MoE LRU cache with a 50/50 SLRU cache"); the
  large-batch selective copy is [PR #29943](https://github.com/ggml-org/llama.cpp/pull/29943) ("ggml: refactor
  selective expert copying to user code"; commit [`6753a033`](https://github.com/ggml-org/llama.cpp/commit/6753a033)).
  All four are merged; titles checked against the PR pages on 24/10/2026.
- **Strata**, [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT): frequency-ranked expert
  residency, learned profiles, VRAM / RAM / NVMe tiers, router look-ahead, a CPU lane for misses, route-tail skip and
  teacher-forced KL quality checks (the method our own blind, paired quality gates build on).
- **Project Maya**, [github.com/mw00/project-maya](https://github.com/mw00/project-maya) (MIT): ranked residency with
  measured GPU slots and a pinned RAM tier, plus a machine-specific calibration run.
- **Edge0**, GitHub `Edge0-AI/Edge0` ([github.com/Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0), Apache-2.0) and
  the Hugging Face model [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview): streams
  experts from the SSD with a trained next-layer routing predictor. Its predictor *replaces* the router, which changes
  answers, so Marian would only ever borrow the idea for prefetch. Its technical report is the first row of the table
  below.

## Research we drew on

Only papers cited in our design documents are listed, as they are cited there. The design pack cites fewer papers
than a survey would; that is the honest count, not a selection. Titles, authors, years and links below were checked
against the primary sources (arXiv abstract pages and API, Crossref for DOIs, the project's own repository) on
24/10/2026.

| Work, as cited | Where cited | What we took from it |
|---|---|---|
| Lin, Wang, Cai, Liu and Zeng, 2026, "[The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063)", arXiv 2609.18063 (the technical report of the Edge0 project above) | research pack, addendum 2 | a trained next-layer predictor is a candidate for exact prefetch, never as the routing itself |
| Einziger, Friedman and Manes, 2017, "[TinyLFU: A Highly Efficient Cache Admission Policy](https://doi.org/10.1145/3149371)", ACM Transactions on Storage 13(4); it introduces W-TinyLFU. A preprint is [arXiv 1512.00727](https://arxiv.org/abs/1512.00727) (2015) | Fable's ideas; DESIGN-M1 section 4 | frequency-gated admission: P3's RAM tier admits a miss only if it is used more than its victim |
| Hwang et al., "[Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/abs/2308.12066)", arXiv 2023; ISCA 2024 ([doi](https://doi.org/10.1109/ISCA59077.2024.00078)). Both years are right: preprint 2023, conference 2024 | ideas files; DESIGN-M1 section 8 | look-ahead: run layer l+1's router on layer l's state to prefetch experts (planned, P6) |
| Song, Zhong, Chen and Chen, 2024, "[ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134)", arXiv 2410.22134 | Claude Tyl's ideas | prefetch of predicted experts (planned, P6) |
| Xue, Fu, Lu, Mai and Marina, 2024, "[MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/abs/2401.14361)", arXiv 2401.14361 (first posted as "Activation-Aware Expert Offloading for Efficient MoE Serving") | Fable's ideas | activation-aware prefetch (planned, P6) |
| Kamahori, Tang, Gu, Zhu and Kasikci, 2024, "[Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033)", arXiv 2402.07033; ICLR 2025 | DESIGN-M1 section 5 | computing missed experts on the CPU; kept opt-in because it changes rounding |
| Leviathan, Kalman and Matias, "[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)", arXiv 2022, [ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html); and Chen, Borgeaud, Irving, Lespiau, Sifre and Jumper, 2023, "[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)", arXiv 2302.01318 | Fable's ideas | verified speculation (class D, off in the exact default) |
| Saxena, 2023, "Prompt lookup decoding": code and write-up in [github.com/apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) (no paper found; the repository has no licence file) | Fable's ideas | n-gram drafting with a row cap (planned, M3) |

## Thanks

- The **llama.cpp and ggml authors**; our design credits name Aman Gupta, Evan Huus and Georgi Gerganov among others.
  On the PR pages, Aman Gupta ([@am17an](https://github.com/am17an)) opened the MoE cache (#29887), its multi-GPU
  support (#30112) and the selective-copy refactor (#29943), and Evan Huus ([@eapache](https://github.com/eapache))
  opened the SLRU policy (#30637).
- **Strata** (Niko1221 and contributors) and **Project Maya** (Peasant Smith / mw00 and needmorevram), both MIT.
- The publishers of the public weights we measured: **Qwen** (Qwen3.5-9B; Qwen3.8-Flash-Next), **UkisAI** (the
  Swift 1.5 IQ2_XS quantisation of Qwen3.8-Flash-Next,
  [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  Swift Open License v1.0 with the Qwen parts under the Qwen Community License 1.0, made with **IST-DASLab**'s GSQ and
  RCO methods), **Google** (Gemma 4 26B-A4B-it, Gemma 4 E4B-it QAT), **NVIDIA** (NVIDIA-Nemotron-3.5-Bolt-E4B-4GB),
  **unsloth** (GGUF quantisations of Gemma 4 26B-A4B-it and Qwen3.5-9B) and **GrootTeam** (Groot-19B-A3B-261023).
- **Astra and Fable**, AI models consulted for design ideas (tagged [AS] and [FB] in the design); **Sol**, the AI model
  that wrote the research pack; and **Sol, Luna and Astra**, who critiqued the first design (24 items, all folded in).
- **Chris (Krystof Tyl)**, the human partner, and **Alder**, who wrote M0 and asked the question that the gate
  measurements answered. Marian M1 is designed and built by Claude Tyl.

## Limitations of this version

Be warned; this is research code.

- **Windows and CUDA only.** The fork has been built and measured for **sm_89** (RTX 40 series) only. An 8 GB
  Blackwell-class (sm_120) laptop is the minimum-spec target, but only M0 with stock llama.cpp has run on one.
  Linux code paths exist (`O_DIRECT`) but have not been run.
- **One machine.** Every M1 number comes from one reference laptop (RTX 4090 Laptop 16 GB, 32 GB RAM). The "8 GB GPU"
  results are a **simulation**: a VRAM ballast on the 16 GB card keeps its kernels, PCIe link and SSD (design item
  C-17), and the desktop's own display memory made it harsher than a real 8 GB card.
- **Few rounds.** Most arms ran twice (some once); only the tier-against-stock comparison has the five interleaved
  rounds the design asks for. Treat other differences under about 10% as noise.
- **By default the tier still costs prefill 4-5x against stock (about 3x against upstream's cache).** With
  `--moe-prefill-source mmap` (the default), prefill batches above 32 tokens go through upstream's selective copy,
  which faults through the memory map, and the 6 GiB pinned tier takes the page cache that path needs. 16K-token
  prefill fell from about 112-119 tok/s to 34-36 tok/s (4K: about 47-53 to 23-26). The ring costs less (16K:
  105-110). This is why neither is a default.
- **Prefill through P3 helps, but is opt-in and drive-bound.** With `--moe-prefill-source p3` (patches 0008 and
  0009), Swift's 16K prefill with the tier rose from 34.7 to 61.1 tok/s (median of three interleaved rounds, register
  E51), short of the 2x the test asked for. Decode did not suffer (15.15 against 14.80 tok/s warm, 14.07 against
  13.17 after a 16K prompt), and the answers are unchanged: Swift's cache arms agree with each other, and Gemma 4
  26B's tier arm matches all-VRAM on every top-3 log-probability. It is still 0.35x stock (174.2 tok/s in the same
  rounds). Each 512-token ubatch re-reads most of each layer's experts: a 16K prompt uploads 509 GiB, the tier holds
  13.8% of it, and the rest comes off the SSD at about 1.95 GB/s. It stays opt-in because it is slower when the
  model fits in the page cache: Gemma 4 26B's 5,428-token prompt ran at 573 tok/s with the memory map and 260 with
  P3 (one run each). Calibration measures both on every ring and tier row and keeps the faster (see How profiling
  works); that has run against a fake engine only.
- **A larger ubatch is exact against a reference at the same ubatch, and is a declared setting.** A ubatch changes
  the compute shape, so all-VRAM at `-ub 2048` and all-VRAM at the default 512 give different answers (Gemma 4 26B,
  ten prompts of 2,009-2,406 tokens: log-probabilities moved by up to 0.668, and 7 of 10 texts differed). At
  `-ub 2048 -b 2048`, the cache arm with memory-mapped misses and the tier arm with P3 prefill both matched all-VRAM
  at the same ubatch exactly (register E54). So, like the context size, the ubatch is part of the reference: a
  deployment that declares 2048 gets answers identical to an all-VRAM run at 2048, not to one at 512. Speed on Swift
  with the tier and P3 prefill (three rounds): 16K prefill 156.7 tok/s against stock's 142.4 at its default ubatch,
  4K prefill 138.3 against 97.1, decode 12.67 against 10.57, and 11.47 against 4.61 after a 16K prompt. Decode was
  16-18% under the earlier session's figures (15.15 and 14.07), which failed that test's decode bound. That
  comparison crosses sessions: stock ran about 10% slower in the same session, the hit rate was unchanged, and the
  2,048-row compute buffer takes about 300 MiB more VRAM. Until a same-session run at 512 settles it, the default
  stays at 512. The ubatch is a declared profile parameter (`--ubatch`, see How serving works), tested offline
  only. Stock at
  `-ub 2048` reached 279.6 tok/s at 16K (one run), but stock changes answers.
- **The pinned tier competes with the desktop.** It takes RAM the OS and other programs cannot page; pinning can fail
  (it then falls back to pageable memory and says so).
- **The decode target is met on one machine, one model, one prompt set.** Five interleaved rounds (design 9.3) put
  the 6 GiB tier at 14.4-14.6 tok/s warm and 13.4-13.7 after a 16K prompt, all above 12
  ([JOURNEY A21](docs/JOURNEY.md#a21-the-five-round-confirmation)). It has not been measured on a real 8 GB card or a
  second laptop.
- **Exactness is proved at log-probability level, not yet bitwise over the full logits.** Against the true
  reference, every expert in VRAM, Gemma 4 26B-A4B's cache arms with memory-mapped, ring and tier misses match on all
  1,280 generated tokens and on every top-3 log-probability to full float precision (largest difference 0; the
  reference run against itself also 0). Stock offload of the same model, by contrast, changes every one of the 10
  answers and moves log-probabilities by up to 0.59, because it computes some experts on the CPU. Until 24/10/2026 the
  probability half of this check was empty (the client read field names this server does not send), so earlier
  claims of identical probabilities rested on token identity alone; see
  [JOURNEY A22](docs/JOURNEY.md#a22-exactness-with-real-probabilities). Not yet done: full-logit comparison, the
  single-layer harness and the kernel-boundary parity matrix; for the Swift model a literal all-in-VRAM reference
  cannot fit on this GPU (C-09), so its cache arms are checked against each other (identical, difference 0).
- **The miss-path probe overstated its baseline.** The replay probe's memory-mapped pass ran 23% to 1.9x slower than
  the real server, so its ratios flattered the ring and the tier.
- **No routing profile, no held-out split, no learning yet.** The hit rates are upstream SLRU's, warming over a fixed
  prompt set.
- **One sequence at a time, by design.** Concurrent requests change the kernels and memory peaks, so `serve`'s proxy
  runs one request, queues one more and answers the rest with 503. Multi-sequence serving is M2 work.
- **Upstream moves fast.** The cache is experimental upstream and changed three times in two weeks. On 27/10/2026 the
  fork was rebased from `a37cc461` to `63b4b303` (107 upstream commits, mechanical fixes only) and re-checked (E48):
  the Gemma cache arms are still log-probability-identical to all-VRAM, and the Swift tier still meets the decode exit
  in one round (15.1 tok/s warm, 12.6 after a 16K prompt, against 14.4-14.6 and 13.4-13.7 at the old pin). Three of
  those commits change the Swift model's numerics, so Swift's texts differ from the runs at `a37cc461`; measurements
  dated before 27/10/2026 were taken at the old pin.
- **The memory governor only watches by default.** A live test (register E53) found that restarting on a memory
  reading cost more than the pressure did: a plain llama-server slowed by at most about 4% with 150 MiB of VRAM or
  under 1 GiB of RAM left, while the old governor moved to a 9.5-11.7 tok/s row for about 240 s and never stepped
  back up. So `--governor watch` (the default) warns on low VRAM or RAM and pauses admission only on low commit
  headroom; `--governor restart` (opt-in) restarts only on a measured slowdown and steps back up, and has run offline
  only. E53 did not reproduce the paging cliff at its depth of pressure, so no mode has met one yet.
- **Calibration v1's serving side has run live once, in part.** E53 ran the front proxy, the live check and the
  old governor on the reference laptop: every answer, across two restarts, was token- and log-probability-identical
  to all-VRAM, and every refusal was a 503 with a reason. `--confirm`, the CPU thread sweep (both built 27/10/2026)
  and the governor's new `restart` mode have run only against a fake engine, a fake llama-server, fake memory
  readings and a fake clock (41 tests in `tests/test_serve.py`; 17 deliberately broken versions of the governor and
  the live check each fail at least one of them). The thread ladder has not run on the GPU. The live check compares with the profile, and serve's own loop ran 10-16% under calibrate's figure on a
  clean machine (E53), so it reports "below profile" rather than "slow". Memory is read through `nvidia-smi`. The
  DXGI budget the design asks for turned out to be per-process (it does not see another program's VRAM), so it is
  not used. `--confirm` runs 4 rounds with a fresh server per window, far fewer than the 64 windows of Strata's rule
  it follows, so it can only catch large differences.
- **Calibration v1 measures, then trusts its measurement.** The planner restores cache budgets, miss sources, tier
  sizes and thread counts into the launch configuration, and `serve` checks the chosen row against the VRAM and RAM
  free at launch. The cliff finder assumes decode speed rises with the cache budget until the cliff. The cache arms
  have run once on one machine (the reference laptop), with one model, and with the cache capped at 2,500 MiB so that
  the miss sources had misses to serve: the bisection itself has only run against synthetic speed and VRAM curves in
  the tests, not against a real paging cliff. That smoke ran the M1.2 build, at the old pin (`a37cc461`), not the
  rebased fork.
- **Prompt-prefix reuse changes answers, so it is off in exact mode.** Register test E52 (27/10/2026) asked
  whether an answer computed on a reused prompt prefix equals the answer computed fresh.
  - **The test:** Gemma 4 26B-A4B IQ4_XS on the rebased build, with a 1,608-token instruction block (1,616 with the chat template) and ten
    documents.
  - **The result:** none of the 10 reused answers matched its fresh twin, either all-VRAM or with the cache and
    tier; log-probabilities moved by 1.08-4.66, and 5 of 10 differed in tokens. A fresh prompt run twice matched
    exactly, so the difference comes from the reuse.
  - **What is still exact:** the cache and tier arm matched all-VRAM on every request, reused prefixes included. The
    miss path is exact; the reuse is not.
  - **What it would buy:** 83% of prompt tokens saved, so it remains available as an opt-in, answer-changing switch.
- The build carries no C++ exception unwinding (`/EHsc` is off, inherited from the measured builds).

## Status

| Milestone | State | Evidence |
|---|---|---|
| M0 | done, 21/10/2026 | profiler, planner, launcher; first measured profile in `profiles/` |
| M1.0 | measured, 22-24/10/2026 | upstream cache vs stock on Swift 1.5 IQ2_XS; copy probe; 8 GB floor arms; gate and control models |
| M1.1 | in progress | exactness floor PASS; trace-replay probe GO; P3 verify run PASS (11/11); P3 server arms measured 24/10/2026 (two rounds): the tier doubles decode, exact, but costs prefill about 3x; controls and the Maya drive test done; replay = server exactly (+0.00 points, the M1.1 exit check); five-round confirmation: decode exit met (E44); log-probability exactness against all-VRAM (E45) |
| M1.2 | tooling done, policy deprioritised | the routing-id trace and the exact replay (A19); the ranked policy lost to upstream's SLRU on the real trace ([JOURNEY section 7](docs/JOURNEY.md#7-what-is-next)) |
| M1.3 | Python side built, 27/10/2026 | `marian calibrate`, cache arms (cliff finder, miss sources, RAM envelope), calibration gate, launch checks, `serve --stock`, `--exact-only`, `marian diag`; 26 new offline tests, and the whole offline suite passes. One GPU smoke on the reference laptop (27/10/2026, the M1.2 build at the old pin: Gemma 4 26B-A4B IQ3_XXS, context 4096, 218 s): the gate refused before calibration, the walk, two cache budgets, the ring and two tier sizes ran, `plan`, `serve --dry-run` and `serve --stock --dry-run` passed. Serving side built the same day: CPU thread sweep, live check and `serve --confirm`, front proxy (one sequence, 503 with a reason, streaming), memory governor with supervised restart, prompt-prefix reuse off by default; 30 more offline tests (the whole suite: 118). Register test E52: prefix reuse is not exact (0 of 10 reused answers matched), so it stays off in exact mode. **The M1.3 exit is not met:** in E53 the governor kept every answer exact across two restarts, but restarting on a memory reading cost more than the pressure did, so its default is now `watch` and `restart` (on a measured slowdown, with a step-up) is opt-in and has run offline only. The fork's pinning rules: chunked pinning and the RAM cap are in patch 0007; the cap has never triggered. Not built: learned counts |
| Prefill track | first patches measured, 27/10/2026; calibration rows built (offline) | patch 0008 (prefill misses through P3, opt-in via 0009's `--moe-prefill-source p3`): +76% at 16K with the tier, exact (E51). A declared ubatch of 2048 is exact against all-VRAM at the same ubatch and, with P3 prefill, beat stock's default prefill at 16K (156.7 against 142.4 tok/s, three rounds); its decode was 16-18% under the earlier session's, cause not yet separated (E54). Defaults unchanged: `mmap` and ubatch 512. The Python side (27/10/2026) measures ring and tier rows with both prefill sources and keeps the faster, and carries a declared ubatch through calibrate, plan, serve and diag, refusing a plan at another ubatch; 26 offline tests (the whole suite: 155), each behaviour broken once on purpose and caught |
| M1.4 onwards | not started | real-hardware validation (see [DESIGN-M1 section 10](docs/design/DESIGN-M1.md)) |

**M1 exit criteria, and where each stands** (reference laptop, Swift 1.5 IQ2_XS):

| Criterion | Target | Where it stands |
|---|---|---|
| VRAM lookup hit rate, median per decode token | at least 85% | **met** at a 9,400 MiB cache: 90.8% (10th percentile 80.4%), with upstream's SLRU |
| Exactness | answers independent of placement | **met at log-probability level**: cache arms vs the all-VRAM reference (Gemma 26B) identical on 1,280 tokens and every top-3 log-probability, all three miss sources; Swift cache arms identical to each other. Full-logit check, harness and parity matrix not built |
| Decode, warm | at least 12 tok/s | **met, five rounds**: P3 tier 14.4-14.6 tok/s (all-tokens rate; medians 14.5-15.1); stock `-ncmoe 34` 8.2-10.8 in the same rounds; upstream's path 7.3 / 6.3 |
| Decode after a 16K prompt | at least 12 tok/s | **met, five rounds**: P3 tier 13.4-13.7 tok/s (medians 14.4-14.7); stock 3.9-8.5 in the same rounds; upstream's path 5.0 / 4.6 |

## Measured results (short)

Gemma 4 26B-A4B-it, GPU limited to 8 GB on the reference laptop (decode tok/s, median of 10 prompts):

| RAM situation | Configuration | Decode | Prefill, 4K prompt |
|---|---|---|---|
| RAM free | upstream cache 2,500 MiB (IQ4_XS) | 42.6 | 756.9 |
| RAM free | stock `-ncmoe 24` (IQ4_XS) | 16.8 | 700.8 |
| about 25 GB in use | upstream cache 2,500 MiB (IQ4_XS) | 8.0 | 97.8 |
| about 25 GB in use | upstream cache 2,500 MiB + P3 ring (IQ4_XS, M1.1 build, one round) | 13.4 | 48.1 |
| about 25 GB in use | stock `-ncmoe 24` (IQ4_XS) | 15.0 | 129.1 |
| about 25 GB in use | stock `-ncmoe 18` (IQ3_XXS) | **26.3** | 775.4 |
| reference, whole 16 GB GPU | all experts on GPU (IQ4_XS) | 79.4 | 3,645.7 |

Swift 1.5 IQ2_XS (a quantisation of Qwen3.8-Flash-Next, 33 GiB of experts) on the 16 GB reference laptop, two rounds
(tok/s, median per request, round 1 / round 2; requests that stop within 8 tokens are left out):

| Configuration | Decode | Decode after a 16K prompt | Prefill, 16K prompt |
|---|---|---|---|
| stock `-ncmoe 34` (first campaign) | 10.6 / 10.9 | 5.3 / 5.1 | 152.0 / 147.9 |
| upstream cache 9,400 MiB, memory-mapped misses | 7.3 / 6.3 | 5.0 / 4.6 | 119.3 / 112.0 |
| + P3 ring | 12.4 / 12.0 | 10.7 / 11.7 | 105.0 / 109.9 |
| + P3 6 GiB pinned tier | 14.7 / 14.0 | 14.8 / 14.4 | 34.0 / 35.8 |

The confirmation, five interleaved rounds (T S | S T | T S | S T | T S), all-tokens rate in tok/s:

| Round | Tier decode | Stock decode | Tier after 16K | Stock after 16K | Prefill 16K, tier / stock |
|---|---|---|---|---|---|
| 1 | 14.5 | 10.8 | 13.4 | 3.9 | 29.7 / 144.4 |
| 2 | 14.4 | 9.6 | 13.6 | 4.2 | 34.4 / 148.6 |
| 3 | 14.6 | 8.2 | 13.4 | 3.9 | 31.2 / 155.0 |
| 4 | 14.5 | 9.5 | 13.7 | 3.9 | 34.4 / 141.8 |
| 5 | 14.6 | 8.2 | 13.5 | 8.5 | 30.3 / 154.8 |

Everything else, with conditions and what each table taught us: [docs/JOURNEY.md](docs/JOURNEY.md).

## Quick start (M0 and M1.3; standard library only, Python 3.10+)

```
python -m marian machine                                   # GPU, VRAM, RAM, driver; a fingerprint, no hostname
python -m marian inspect model.gguf                        # architecture, layers, experts, MoE or dense
python -m marian calibrate model.gguf --engine path/to/llama-server     # the quick profile: one context, one KV type
python -m marian calibrate model.gguf --engine path/to/llama-server --ubatch 2048   # ... for a deployment that declares -ub 2048
python -m marian profile model.gguf --engine path/to/llama-server --ctx 4096,8192,16384 --kv f16,q8_0   # the full one
python -m marian plan model.gguf --need 6000               # the fastest measured configuration for 6,000 tokens
python -m marian plan model.gguf --need 6000 --fit-now --json   # ... that also fits the VRAM free right now, as JSON
python -m marian plan model.gguf --need 6000 --exact-only  # ... computing every expert on the GPU (answers unchanged)
python -m marian serve model.gguf --engine path/to/llama-server --need 6000     # refuses until this machine is calibrated
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --calibrate   # calibrate first, then serve
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --stock       # rollback: plain upstream flags
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --ubatch 2048 # only rows measured at 2048
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --confirm     # A/B the plan against the runner-up first
python -m marian diag diag.zip model.gguf --engine path/to/llama-server   # a bundle to share: no prompts, home as ~
```

`serve` will not launch on a machine that has not run its pre-tuning tests: without a profile for this machine, model
and engine it refuses and says how to calibrate (or calibrates first with `--calibrate`). It then checks the chosen
row before launching: the model file is the one profiled, the engine is the build that was measured (and has every
flag the row passes), and the VRAM free now covers the row's share plus a margin, and the RAM available covers any
pinned tier plus a floor (6 GiB by default). A failure is a reason and exit code 2 (`--json`: one object;
`--dry-run` checks without launching). `--stock` serves only plain upstream placement (`-ncmoe` / `-ngl`, no expert
cache, no Marian flag) from the same profile, and only warns if the engine is another build.

`plan` refuses when nothing measured fits: it never extrapolates. Each profiled point records the GPU memory other
programs were already using, so `--fit-now` (or `--vram-free MIB`) checks the model's own share against what is free
at launch: a busier desktop than at profiling time cannot push a plan into the paging cliff. It cannot recover points
that overflowed on a busy desktop; profile again on a quieter one for those. `--json` prints one object for other
programs (`ok`, `row`, `config`, `server_args`, `vram_basis`, or `ok: false` with a `reason`; exit code 2 on failure).

Install from a clone or from git (`pip install git+<this repository's URL>`): an unrelated package already uses the
name `marian` on PyPI. Profiles are kept in `$MARIAN_PROFILES` if set, in the clone's `profiles/` when run from a
clone, and in `~/.marian/profiles` when installed.

## How profiling works

For each context size and KV type, Marian walks the GPU/CPU split from the most conservative to the most aggressive
(for MoE models: experts of the first N layers kept in system RAM, `--n-cpu-moe N`; for dense models: layers on the
GPU, `-ngl N`). Each point starts a real server bound to 127.0.0.1, fills about 75 % of the context, generates a fixed
number of tokens, and records prompt and generation speed plus peak VRAM and the lowest free RAM. A point is an
**overflow** when VRAM is within a margin of full or speed collapses against the best point so far; the walk stops
there. Profiles are JSON files named by machine fingerprint, model id and engine.

On a Marian build (its `--help` lists `--moe-miss-source`, `--moe-tier-mib` and `--moe-ring-mib`) and a MoE model,
each context and KV type also gets **cache arms** (M1.3). A cliff finder bisects `--moe-cache-mib` between the
exactness floor (below it the fork refuses to start, which is recorded as a refusal, not a crash) and the budget that
caches every expert, capped by the VRAM left beside the rest of the model. At the best budget it measures the staging
ring and a small ladder of pinned RAM tier sizes, never pinning more than the RAM available minus a floor, because
pinned memory is RAM the desktop cannot page. Each row records whether it is **exact-class** (every expert computed
on the GPU: full offload or any cache arm) or not (stock `-ncmoe` above 0, which changes answers), the RAM available
before and at its lowest, commit headroom, the server's peak working set and commit, and what the fork's log said.
`--max-cache-mib` caps the cache, `--ram-floor` sets the floor, `--no-cache` skips the arms.

On a build with patch 0009 (its `--help` lists `--moe-prefill-source`), every ring and tier row is measured twice:
with the default prefill source (`mmap`, upstream's path) and with `p3` (prefill misses from the same ring and tier).
Both are exact, and which is faster depends on the machine (register E51: +76% at 16K under memory pressure, under
half the speed when the model fits in the page cache). The pair is judged on prefill and decode together, with
differences within 10% counted as ties: a source faster on one and not slower on the other keeps its row, and its
twin is recorded as `dominated`, which the planner skips; a tie keeps the default; a split keeps both, and the
planner's objective (`--objective tg` or `pp`) chooses. `p3` is never measured or launched beside memory-mapped
misses, which the fork refuses. Each profile's `prefill_pairs` says what was kept and why.

`--ubatch N` (default 512) measures every row at `-ub N -b N`. A ubatch is part of the exactness reference (register
E54), so it is declared by the deployment, never picked by calibration. The profile records it (schema 3) and, at a
value other than 512, is written beside the default profile (`...-ub2048.json`) rather than over it. Calibration then
prints the best decode and the best prefill of each context at that ubatch, beside the same figures from this
machine's profiles at other ubatches, marked as another session, so the trade is visible; nothing in the code
assumes what it is.

Rows that compute on the CPU (stock `-ncmoe` above 0, or a dense model with layers left on the CPU) get a **CPU thread
sweep**: the fastest such point of each context and KV type is measured again at physical cores, cores - 2,
cores / 2, 8 and 4 threads, and every result is kept as a row, so the planner picks the fastest. CPU expert work is
bound by memory bandwidth, and Project Maya reported 15 -> 41 tok/s with fewer threads (theirs). `--threads` fixes
the count instead, `--no-thread-sweep` skips it. Rows that compute every expert on the GPU skip it.

## How serving works

**The declared ubatch.** `plan` and `serve` take `--ubatch N` (default 512, no flag), and use only rows measured at
that ubatch: a plan measured at one ubatch is never served at another, just as an exact-only plan never takes a
non-exact row. Without a calibration at the declared value, `serve` refuses (exit 2, reason `ubatch`) and names the
ubatches that were measured; `--calibrate` measures at the declared value first. The filter sits where the planner,
`--confirm`'s runner-up and the governor's restarts all choose rows, and the launch checks repeat it. Profiles written
before the declared ubatch (schema 1 and 2) were measured without `-ub`, so they count as 512; a profile of a newer
schema than this version reads is refused. `--dry-run` prints both settings in one line (`--json`: `ubatch` and
`prefill_source`), and `marian diag --ubatch N` records them in the bundle's plan. `--stock` keeps its rows and rules;
it too serves only rows measured at the declared ubatch.

`serve` puts llama-server on an internal port (`--port` + 1, 127.0.0.1 only) and a small standard-library proxy on
`--port`. The proxy serves **one request at a time** with one more waiting (concurrent requests change the kernels and
the memory peaks, design item C-23); past that it answers HTTP 503 with `Retry-After` and a JSON reason. Answers pass
through as they arrive, streamed ones included.

A **memory governor** reads free VRAM, available RAM and commit headroom every 2 s, against the floors the launch check
used. It has three modes (`--governor`):

- `watch` (the default): each crossing is one warning line with the live decode rate. Only low commit headroom
  pauses admission (503, reason `memory_pressure`), because below it allocations fail; the running request finishes.
  Low VRAM or RAM only warns. Nothing restarts on a reading.
- `restart` (opt-in): if decode stays under 70% of the configuration's own baseline (the median of its first 8
  answers) for 10 s while a memory floor is crossed, it restarts llama-server on the next smaller measured
  configuration that fits, and says why in one line. An exact configuration is only replaced by an exact one unless
  `--allow-non-exact` is given. When memory has fitted the planned configuration again for 60 s (`--step-up`), and
  no request is queued, it steps back up. After three restarts (step-ups count), or when nothing smaller fits, it
  stops with the reason (upstream allocates the expert cache once, so a running server cannot shrink it; design item
  C-02).
- `off`: no governor.

The proxy reads the timings of the first 8 answers and says "below profile" when live decode is more than 15% under
the profiled rate (serve's loop runs somewhat under calibrate's figure, so this is a note, not a fault). `--confirm`
goes further before serving: the plan and the runner-up (the next best configuration of the same
exactness) run four interleaved rounds of short windows, and the runner-up replaces the plan only if it wins at least
3 of 4 rounds by a median of more than 2% (after Strata's keep-if-faster rule). The outcome is written into the
profile, and the planner prefers the winner from then on.

**Prompt-prefix reuse is off.** llama-server reuses the shared start of consecutive prompts by default. A reused
prefix does not give the same answer as a fresh one (register test E52, see Limitations), so the proxy sets `cache_prompt: false` on every generation request and llama-server starts with `--no-cache-prompt`
where the build has it. `--cache-prompt on` opts in, and `--gate-prefix FILE` then primes the slot with a fixed
instruction block after every start. The file must hold the exact text the requests begin with, after any chat
template.

## Roadmap

The living roadmap, with the evidence behind each item and the dev debt, is
[docs/ROADMAP.md](docs/ROADMAP.md) (updated 27/10/2026 from scans of upstream llama.cpp and the wider ecosystem). In
short: rebase onto current upstream (done); make prefill the main target (expert page faults were the cost, not
Swift's embedding table; prefill misses can now come from P3, opt-in, and a declared ubatch of 2048 reached stock's
prefill exactly; next, fewer bytes per prompt); calibration v1 with Windows pinning and RAM-safety rules; then the
8 GB laptop under RAM pressure. The milestones from
[DESIGN-M1 section 10](docs/design/DESIGN-M1.md), reshaped by what M1.0 measured:

- **M1.2:** a ranked policy (ranked core, warm start, doorkeeper, transient slots, budgets by need), with a bitwise
  invariance test.
- **M1.3:** calibration v1. Built (Python): the VRAM cliff finder, miss sources in a RAM safety envelope, the CPU
  thread sweep, the plan, the calibration gate, launch checks, the `--stock` rollback, the diagnostics bundle, live
  confirmation, and serving through a proxy with a memory governor (watch by default since register E53). Still to
  do: learned counts, a live run of the governor's `restart` mode, a test that makes 0007's tier cap trigger, and a
  live run of the calibration rows for the prefill source and a declared ubatch (built 27/10/2026, offline tests
  only).
- **M1.4:** real hardware: a real 8 GB GPU, Linux, a slower link and drive.
- **M2:** the full expert source (prefill included), removing the per-layer drain, look-ahead prefetch.
- **M3 to M5:** verified speculation, opt-in answer-changing switches with blind quality gates, and prefill work.

## Licence

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The llama.cpp patches modify MIT-licensed
upstream code: that code stays MIT, and our changes in the patches are offered under MIT as well, so they can go
upstream. The four milestone tags made before the relicence on 24/10/2026 (`m0`, `design-m1`, `m1.0-results`,
`m1.1`) still carry the earlier MIT LICENSE file; the project is Apache-2.0 from commit `afed698` onwards.
