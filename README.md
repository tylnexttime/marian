# Marian

**Run local mixture-of-experts models by what your machine can actually do, measured, not guessed.**

Marian is named after a popular Polish given name. Status: **pre-release**; M1.1 and M1.2 measured (24/10/2026), the
Python side of M1.3's calibration v1 built (27/10/2026). The full
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
   the Python side of calibration v1 (milestone M1.3) is built and tested offline, and has run on one machine only;
   its memory governor, live confirmation and learned counts are not built yet.

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
  offload is not. The price: **prefill 4-5x slower than stock** (about 3x slower than upstream's cache path),
  because prefill still reads through the memory map and the pinned tier takes the page cache it needs (see
  Limitations).
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
 +------------------------------------------------------------------------------------+
 |  upstream llama.cpp at 63b4b303: --moe-cache-mib per-expert VRAM cache (SLRU),     |
 |  MUL_MAT_ID on the cache bank, selective expert copy for large batches             |
 +------------------------------------------------------------------------------------+
```

| Part | Whose | State |
|---|---|---|
| Per-expert VRAM cache, SLRU policy, every expert computed on the GPU | upstream llama.cpp (PRs #29887, #30112, #30637) | used as the base |
| Selective expert copy for large batches | upstream (PR #29943) | used as is; P3 does not cover it yet |
| M0: profiler, planner, launcher | Marian | done |
| P0 / P0b trace (`MARIAN_MOE_TRACE`, `MARIAN_MOE_TRACE_FILLS`) | Marian | built, used in every M1 run |
| Exactness floor (per-group slot floor, refusal below it) | Marian | built, passed its test |
| P3 miss source (`--moe-miss-source mmap\|ring\|tier`, `--moe-tier-mib`, `--moe-ring-mib`) | Marian | built, measured, opt-in (default stays `mmap`) |
| Calibration v1, Python side (M1.3): cache arms in the profiler (VRAM cliff finder, ring and tier ladder inside a RAM envelope), exact-class per row, calibration gate, launch checks, `serve --stock` rollback, `marian diag` | Marian | built, offline tests pass; one smoke run on one machine |
| Memory governor with supervised restart, live confirmation, learned counts (rest of M1.3); ranked core, warm start, TinyLFU doorkeeper in VRAM (P2), drain removal, prefetch | Marian | designed, not built |

The patches are in [docs/m1.0/patches](docs/m1.0/patches) (P0) and [docs/m1.1/patches](docs/m1.1/patches) (0001
exactness floor, 0002 fill trace, 0003 P3 minimal), as `git format-patch` files on top of `63b4b303`.

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
- **The tier costs prefill 4-5x against stock (about 3x against upstream's cache).** P3 serves decode misses only:
  prefill batches above 32 tokens still go through
  upstream's selective copy, which faults through the memory map, and the 6 GiB pinned tier takes the page cache
  that path needs. 16K-token prefill fell from about 112-119 tok/s to 34-36 tok/s (4K: about 47-53 to 23-26). The
  ring costs less (16K: 105-110). This is why neither is a default.
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
- **One sequence at a time.** Concurrent requests change the kernels and memory peaks; multi-sequence serving is not
  supported.
- **Upstream moves fast.** The cache is experimental upstream and changed three times in two weeks. On 27/10/2026 the
  fork was rebased from `a37cc461` to `63b4b303` (107 upstream commits, mechanical fixes only) and re-checked (E48):
  the Gemma cache arms are still log-probability-identical to all-VRAM, and the Swift tier still meets the decode exit
  in one round (15.1 tok/s warm, 12.6 after a 16K prompt, against 14.4-14.6 and 13.4-13.7 at the old pin). Three of
  those commits change the Swift model's numerics, so Swift's texts differ from the runs at `a37cc461`; measurements
  dated before 27/10/2026 were taken at the old pin.
- **Calibration v1 measures, then trusts its measurement.** The planner now restores cache budgets, miss sources and
  tier sizes into the launch configuration, and `serve` checks the chosen row against the VRAM and RAM free at
  launch. But nothing watches memory after launch yet (no governor, no supervised restart), no live confirmation run
  checks the winner against the runner-up, and the cliff finder assumes decode speed rises with the cache budget
  until the cliff. The cache arms have run once on one machine (the reference laptop), with one model, and with
  the cache capped at 2,500 MiB so that the miss sources had misses to serve: the bisection itself has only run
  against synthetic speed and VRAM curves in the tests, not against a real paging cliff. That smoke ran the M1.2
  build, at the old pin (`a37cc461`), not the rebased fork.
- The build carries no C++ exception unwinding (`/EHsc` is off, inherited from the measured builds).

## Status

| Milestone | State | Evidence |
|---|---|---|
| M0 | done, 21/10/2026 | profiler, planner, launcher; first measured profile in `profiles/` |
| M1.0 | measured, 22-24/10/2026 | upstream cache vs stock on Swift 1.5 IQ2_XS; copy probe; 8 GB floor arms; gate and control models |
| M1.1 | in progress | exactness floor PASS; trace-replay probe GO; P3 verify run PASS (11/11); P3 server arms measured 24/10/2026 (two rounds): the tier doubles decode, exact, but costs prefill about 3x; controls and the Maya drive test done; replay = server exactly (+0.00 points, the M1.1 exit check); five-round confirmation: decode exit met (E44); log-probability exactness against all-VRAM (E45) |
| M1.2 | tooling done, policy deprioritised | the routing-id trace and the exact replay (A19); the ranked policy lost to upstream's SLRU on the real trace ([JOURNEY section 7](docs/JOURNEY.md#7-what-is-next)) |
| M1.3 | Python side built, 27/10/2026 | `marian calibrate`, cache arms (cliff finder, miss sources, RAM envelope), calibration gate, launch checks, `serve --stock`, `--exact-only`, `marian diag`; 26 new offline tests, and the whole offline suite passes. One GPU smoke on the reference laptop (27/10/2026, the M1.2 build at the old pin: Gemma 4 26B-A4B IQ3_XXS, context 4096, 218 s): the gate refused before calibration, the walk, two cache budgets, the ring and two tier sizes ran, `plan`, `serve --dry-run` and `serve --stock --dry-run` passed. Not built: governor with supervised restart, live confirmation, learned counts |
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
python -m marian profile model.gguf --engine path/to/llama-server --ctx 4096,8192,16384 --kv f16,q8_0   # the full one
python -m marian plan model.gguf --need 6000               # the fastest measured configuration for 6,000 tokens
python -m marian plan model.gguf --need 6000 --fit-now --json   # ... that also fits the VRAM free right now, as JSON
python -m marian plan model.gguf --need 6000 --exact-only  # ... computing every expert on the GPU (answers unchanged)
python -m marian serve model.gguf --engine path/to/llama-server --need 6000     # refuses until this machine is calibrated
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --calibrate   # calibrate first, then serve
python -m marian serve model.gguf --engine path/to/llama-server --need 6000 --stock       # rollback: plain upstream flags
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

## Roadmap

The living roadmap, with the evidence behind each item and the dev debt, is
[docs/ROADMAP.md](docs/ROADMAP.md) (updated 27/10/2026 from scans of upstream llama.cpp and the wider ecosystem). In
short: rebase onto current upstream; make prefill the main target (Swift's 26.8 GiB per-layer embedding table shares
the page cache with the experts, and prefill misses still fault through the memory map); calibration v1 with Windows
pinning and RAM-safety rules; then the 8 GB laptop under RAM pressure. The milestones from
[DESIGN-M1 section 10](docs/design/DESIGN-M1.md), reshaped by what M1.0 measured:

- **M1.2:** a ranked policy (ranked core, warm start, doorkeeper, transient slots, budgets by need), with a bitwise
  invariance test.
- **M1.3:** calibration v1. Built (Python side): the VRAM cliff finder, miss sources in a RAM safety envelope, the
  plan, the calibration gate, launch checks, the `--stock` rollback and the diagnostics bundle. Still to build: a
  live confirmation, a memory governor with supervised restart, learned counts.
- **M1.4:** real hardware: a real 8 GB GPU, Linux, a slower link and drive.
- **M2:** the full expert source (prefill included), removing the per-layer drain, look-ahead prefetch.
- **M3 to M5:** verified speculation, opt-in answer-changing switches with blind quality gates, and prefill work.

## Licence

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The llama.cpp patches modify MIT-licensed
upstream code: that code stays MIT, and our changes in the patches are offered under MIT as well, so they can go
upstream. The four milestone tags made before the relicence on 24/10/2026 (`m0`, `design-m1`, `m1.0-results`,
`m1.1`) still carry the earlier MIT LICENSE file; the project is Apache-2.0 from commit `afed698` onwards.
