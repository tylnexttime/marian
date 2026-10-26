# Marian roadmap

A living document: what we build next, why, and what evidence put it there. Updated 27/10/2026 after a scan of
upstream llama.cpp and of the wider ecosystem (other engines and 2025-2026 papers). Milestone numbers follow
[DESIGN-M1 section 10](design/DESIGN-M1.md); this page reorders them by what we have measured since.

Every item names its class, as in the design: **E** exact (answers cannot change), **D** verified speculation (the
target model's choice, computed in a different batch shape), **A** answer-changing (opt-in only, behind a blind quality
gate). Evidence marked *theirs* comes from another project's own machine and has not been reproduced here.

## Where we stand

- **Decode is solved on the reference laptop:** the P3 pinned tier meets the M1 exit, exactly (README, JOURNEY A21-A22).
- **Prefill is the obstacle.** With the tier, a 16K prompt runs 4-5x slower than stock. Large batches still fault
  experts through the memory map, and the pinned tier takes the page cache they need.
- **A new suspect for the prefill cost** (found 27/10/2026, not yet measured as a cause): Swift's per-layer embedding
  table `per_layer_token_embd` is **26.8 GiB**, beside 33.0 GiB of routed experts, on a 32 GB machine. It is read row
  by row through the same memory map, so its pages compete with the experts' pages in the page cache.
  ik_llama.cpp found the same table evicting expert pages and fixed it (rows prefetched per batch: prompt speed
  154.7 -> 203.0 tok/s, *theirs*).
- **The use case that drives the work:** a PII/sensitivity gate on 4-30B models, on an 8 GB GPU beside a desktop that
  uses most of 32 GB. Prefill matters as much as decode there, and today stock offload wins that case.

## Next, in order

### 1. Rebase onto current upstream (in progress)

We are 107 upstream commits behind the pin `a37cc461`. The stack re-applies with mechanical fixes (the cache
constructor gained `no_alloc`), but three upstream changes alter Swift's numerics (qwen4exp attention selection, a
scale move, `softplus`). Exactness is therefore re-proved from scratch at the new base. Test E48 is pre-registered
before the build.

### 2. Prefill track (the main target)

| Item | Class | Why | Evidence | Effort |
|---|---|---|---|---|
| **Measure first: page-cache residency by tensor kind** (PLE, experts, other) during pp16K, tier on and off | test | Decide whether PLE pollution or tier pinning causes the 4-5x | our tensor sizes; ik #2389 | S |
| **PLE row source**: read the batch's PLE rows explicitly (prefetch or unbuffered), keep the table out of expert-sized readahead | E | Swift's 26.8 GiB table; Gemma 4 E-models have PLE too | ik_llama.cpp #2389, #2692, #2563 (*theirs*) | S-M |
| **Prefill misses through P3**: route the large-batch selective copy through the unbuffered ring and tier, not mmap faults | E | P3 serves decode only today | our E40/E44 | M |
| **Prefill expert prefetch**: read layer l+1's experts while layer l computes | E | The fault stalls are serial today | ik_llama.cpp #2101, #2667: 1.9-2.7x on models larger than RAM (*theirs*, Linux) | M |
| **Pin mapped expert ranges** (upstream #31122, closed by a bot, not on merit) as a measured arm | E | Prompt speed +27-44% (*theirs*) | upstream PR #31122 | S |
| **Layer-major chunked prefill**, then refill the cache with the prompt's hottest experts | E at a fixed ubatch | Moves M5 forward: a 12 GB card prefilled 231 tok/s at 8K this way | Strata v0.2.4; MoE-CORE (arXiv 2610.01950); Maya #352 (*theirs*) | L |

### 3. Calibration v1 (M1.3, Python side in progress)

Built from the design, plus safety and method rules other projects paid for:

- A first-launch calibration gate: no serving until this machine has been measured (a project rule).
- Cache-aware profiles and plans: the cache budget, miss source and tier size are restored at launch.
- A VRAM cliff finder, a RAM safety envelope, `serve --stock` rollback, launch checks, `marian diag`.
- **Pin in chunks under a cap** (E). On Windows the driver limits pinned memory per allocation and per process; a
  one-shot pin fails and silently falls back to pageable memory. *Theirs*: ik #2604 (10.3 vs 3.6 GB/s), Maya #417.
- **Cap the tier at about half of RAM, and copy all-or-none with a commit-room check** (E). *Theirs*: upstream #31122's
  Windows note, Maya v1.1.27, Strata v0.2.6. This means re-testing our partial 6 GiB tier.
- **Keep a faster path only if a live A/B confirms it** (Strata v0.2.5), and **sweep CPU threads**: Maya #433 saw
  15 -> 41 tok/s with fewer threads (*theirs*).
- **Fixed-prefix KV reuse for the gate** (E at a qualified boundary): a gate re-reads the same instructions every
  time. Strata v0.2.3 reused 91% of a shared prompt (*theirs*). Proven exact here before it becomes a default.
- Build on upstream's **#30966** (`--moe-cache-cpu-share` + `moe-cache-tune.py`) rather than duplicate it. Computing
  misses on the CPU changes rounding, so it is class A here: opt-in.

### 4. The next milestone: the 8 GB laptop under RAM pressure

Its pass rule is written before any run. The proposal is to match or beat stock's best offload, on decode and on
prefill, with about 25 GB of RAM in use, for one named gate model. Open question: whether opt-in class A switches may
count, since stock offload itself changes answers.

### 5. M1.2, reshaped: evaluation and exactness rather than a new VRAM policy

On the real trace every candidate policy lost to upstream's SLRU (JOURNEY A19), so M1.2's work is now:

- **E47, predictability bounds** (pre-registered 27/10): token persistence, layer-to-layer itinerary, prompt-to-decode
  working set, and what perfect prediction is worth by source and horizon. This gates any prefetch predictor.
- **Replay methodology**: per-access replay inflated recency policies by 27-29% and inverted rankings, and
  single-template prompt sets moved results by 19-32 points (arXiv 2608.07911). Check our replay against both.
- **Exactness to the end**: golden bit-compare kernel tests at real shapes (Strata's pattern) as the single-layer
  harness E0, the kernel-boundary parity matrix, and a full-logit comparison (today: top-3 log-probabilities only).
- **Residency bonus for predicted experts** (Edge0's `pin_bonus` idea): exact, cheap, only if E47 says routing is
  predictable.

### 6. Later milestones

| Milestone | What | Notes |
|---|---|---|
| M1.4 | Real hardware: an 8 GB sm_120 laptop, Linux, a slower link and drive | Linux `O_DIRECT` path never run yet |
| M2 | Overlap copies with routing; compute resident experts while misses arrive (E?) | Strata 0d385178 +13.5%, Maya v1.1.21 (*theirs*) |
| M2 | Look-ahead prefetch, the predictor used for transfers only, training-free first (E) | SpecPrefetch, Pre-gated, ProMoE. Caution: a saturated bus gains nothing (Budgeting Bytes, arXiv 2609.04238) |
| M3 | Speculation (D): MTP for qwen4exp is upstream (#29761); SpecFit verify trees (arXiv 2610.15417, Apache-2.0); drafts cut by cumulative probability | Swift + MTP + the expert cache currently aborts at start (upstream #30860). Untested idea: define the reference as the verify-shaped computation |
| M4 | Class A switches with blind gates: CPU share of misses (#30966), a MoE-aware batch threshold (ik #520), a bit-faithful CPU kernel spike | Strata/Maya "exact" CPU lanes are exact to ggml-cpu, not to our GPU reference |
| M5 | Quantise activations once per call (E?, +13% decode *theirs*); dequantise hot experts to FP16 for prefill (A) | INT8 matmuls are 60-77% of kernel time here (E46) |

## Not adopting, and why

- **Answer-changing:**
  - **A predictor that replaces the router.** Edge0 does this, and its 35B tier is 3.9 benchmark points below its base.
  - **Fewer experts per token.** Strata's "Faster" mode routes 5 of 6.
  - **Skipping layers on long prompts.** In Strata's version the top token changed at 0.5% of positions; a PII gate
    rides on single tokens.
  - **Expert deferral** (KTransformers).
- **Network surfaces:** launchers, update feeds, downloaders, self-updating engines.
- **Unlicensed code:** SpecPrefetch, prompt-lookup decoding. The ideas only.
- **FP16/BF16 tensor-rate unlock** (gpu-unlock): measured and parked (E46). The ceiling is under 10% in 7 of 8 cases.

## Model note

Swift 1.5 is published by UkisAI under the Swift Open License v1.0, with the Qwen parts under the Qwen Community
License 1.0. Commercial use is licensed below US$1M gross revenue. It stays a research test model here. A production
gate should use a model whose licence fits the deploying organisation.

## Dev debt

| # | Debt | State |
|---|---|---|
| 1 | 107 commits behind upstream | rebase + E48 in progress |
| 2 | M0 drops the cache budget when it launches a plan | fixed on the M1.3 branch (in progress) |
| 3 | Prefill 4-5x behind stock with the tier | prefill track above |
| 4 | Exactness proved at top-3 log-probability, not full logits; no E0 harness or parity matrix | M1.2 above |
| 5 | Most arms ran one or two rounds | five rounds only for the decode exit; repeat any result that changes a default |
| 6 | The build has no C++ exception unwinding (`/EHsc` off) | to fix with the rebase build |
| 7 | One sequence at a time | by design until M1.3's serving rules |
| 8 | Linux paths never run; one machine only | M1.4 |
| 9 | Swift's publisher missing from the thanks | fixed 27/10 (UkisAI) |
