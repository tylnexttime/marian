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
- **Where the prefill time goes (E49, 27/10/2026): expert page faults, not the embedding table.**
  - **The suspect:** Swift's per-layer embedding table is 26.8 GiB, beside 33.0 GiB of experts on a 32 GB machine.
  - **Measured with `wsprobe`** (tools/wsprobe, no admin needed): the table held at most 0.48 GiB of the server's
    memory in any arm. A 16K prompt can touch at most 1.18 GiB of it (16 rows of 90 bytes per token), and upstream
    already prefetches those rows.
  - **What does hurt:** with the 6 GiB tier, prefill runs about 38,700 single-page disk reads per second, which is
    expert pages faulting in under memory pressure. That makes the P3 prefill source below the first fix.
- **The use case that drives the work:** a PII/sensitivity gate on 4-30B models, on an 8 GB GPU beside a desktop that
  uses most of 32 GB. Prefill matters as much as decode there, and today stock offload wins that case.

## Next, in order

### 1. Rebase onto current upstream (done 27/10/2026)

We are 107 upstream commits behind the pin `a37cc461`. The stack re-applies with mechanical fixes (the cache
constructor gained `no_alloc`), but three upstream changes alter Swift's numerics (qwen4exp attention selection, a
scale move, `softplus`). Exactness is therefore re-proved from scratch at the new base. Test E48 is pre-registered
before the build.

### 2. Prefill track (the main target)

| Item | Class | Why | Evidence | Effort |
|---|---|---|---|---|
| ~~Measure first: residency by tensor kind~~ **done (E49)**: the embedding table is ruled out; expert page faults are the cost | test | | `wsprobe` | done |
| **Prefill misses through P3** (patches 0007 + 0008, test E51, in progress): route the large-batch selective copy through the unbuffered ring and tier, not memory-map faults; pin the tier in chunks under a RAM cap | E | P3 serves decode only today; E49 measured about 38,700 4 KiB faults per second during the tier's prefill | our E40/E44/E49 | M |
| **Prefill expert prefetch**: read layer l+1's experts while layer l computes | E | The fault stalls are serial today | ik_llama.cpp #2101, #2667: 1.9-2.7x on models larger than RAM (*theirs*, Linux) | M |
| **Pin mapped expert ranges** (upstream #31122, closed by a bot, not on merit) as a measured arm | E | Prompt speed +27-44% (*theirs*) | upstream PR #31122 | S |
| **Layer-major chunked prefill**, then refill the cache with the prompt's hottest experts | E at a fixed ubatch | Moves M5 forward: a 12 GB card prefilled 231 tok/s at 8K this way | Strata v0.2.4; MoE-CORE (arXiv 2610.01950); Maya #352 (*theirs*) | L |

### 3. Calibration v1 (M1.3): built in Python 27/10/2026 except learned counts; tested offline; one machine

Built from the design, plus safety and method rules other projects paid for:

- A first-launch calibration gate: no serving until this machine has been measured (a project rule).
- Cache-aware profiles and plans: the cache budget, miss source, tier size and thread count are restored at launch.
- A VRAM cliff finder, a RAM safety envelope, `serve --stock` rollback, launch checks, `marian diag`.
- **CPU thread sweep** (E), built 27/10: rows that compute on the CPU are measured again at physical cores, cores - 2,
  cores / 2, 8 and 4 threads; the planner picks the fastest. Maya #433 saw 15 -> 41 tok/s with fewer threads
  (*theirs*). Not yet run on the GPU.
- **Keep a faster path only if a live A/B confirms it** (Strata v0.2.5), built 27/10: a passive live check on the
  first 8 answers (says so when decode is more than 15% below the profile), and `serve --confirm`, 4 ABBA rounds
  against the runner-up of the same exactness; the runner-up wins only with 3 of 4 rounds and a median above +2%.
  The outcome is written into the profile.
- **Serving envelope and memory governor** (design 7.7 after C-02, 7.8 after C-23), built 27/10: a front proxy runs
  one request at a time with one queued, answers 503 with a reason, and streams; the governor pauses admission when
  free VRAM, available RAM or commit headroom cross the launch floors, and after 10 s restarts llama-server on the
  next smaller measured row (exact stays exact), giving up after 3 restarts. Tested offline against a fake server
  and fake readings; **not yet run against a real competing allocation** (the M1.3 exit asks for exactly that).
- **Fixed-prefix KV reuse for the gate**: a gate re-reads the same instructions every time. Strata v0.2.3 reused 91%
  of a shared prompt (*theirs*). Plumbed (`--cache-prompt on`, `--gate-prefix FILE`), **off by default** until test
  E52 shows it exact. E52's registered run is still to be made (the GPU was held past its 90-minute wait); its
  instrument check (27/10, all-VRAM Gemma, 3 reused pairs) found every reused answer different from its fresh twin
  (max |d logprob| 1.28), so the likely outcome is class A. On Gemma 4 (sliding-window attention) llama-server
  reuses a prefix through `/completion` only after a short previous prompt, and through `/v1/chat/completions`
  always (it checkpoints the window at the user turn).

Still to build:

- **Pin in chunks under a cap** (E, in the fork). On Windows the driver limits pinned memory per allocation and per
  process; a one-shot pin fails and silently falls back to pageable memory. *Theirs*: ik #2604 (10.3 vs 3.6 GB/s),
  Maya #417.
- **Cap the tier at about half of RAM, and copy all-or-none with a commit-room check** (E, in the fork). *Theirs*:
  upstream #31122's Windows note, Maya v1.1.27, Strata v0.2.6. This means re-testing our partial 6 GiB tier.
- Learned counts (P4), and the governor reading the DXGI budget rather than `nvidia-smi`.
- Build on upstream's **#30966** (`--moe-cache-cpu-share` + `moe-cache-tune.py`) rather than duplicate it. Computing
  misses on the CPU changes rounding, so it is class A here: opt-in.

### 4. The next milestone: the 8 GB laptop under RAM pressure

The pass rule was written 27/10/2026, before any run. Gemma 4 26B-A4B IQ4_XS, with the GPU capped at 8 GB and about
25 GB of RAM in use. **Exact Marian, with its configuration chosen by its own calibration, must match or beat stock's
best `-ncmoe` offload on median decode AND median 4K prefill over five interleaved rounds**, and be at least as fast
on decode in 4 of the 5 rounds.

- **Exact-only, decided:** stock offload changes answers. Opt-in class A switches are measured and reported in their
  own rows, but they cannot pass this milestone.
- **If Marian loses,** its planner choosing stock on that machine is the right behaviour, and is reported as such.
- IQ3_XXS is reported too, not judged: today's fastest gate configuration across quants.

### 5. M1.2, reshaped: evaluation and exactness rather than a new VRAM policy

On the real trace every candidate policy lost to upstream's SLRU (JOURNEY A19), so M1.2's work is now:

- **E47, predictability bounds: done 27/10** (`marian predict`, [docs/m1.2/E47.md](m1.2/E47.md)).
  - **Token persistence is real:** 37.7% of a token's experts were used by the previous token at that layer, against
    20.1% by chance.
  - **Predicting the next layer's experts from this layer's ids is dead:** 19.2%, below plain persistence.
  - **A prompt's prefill does not predict its decode experts** better than global frequency.
  - **Routing is near-uniform across prompts.** The global frequency prior is 3.1% against 2.0% chance, so ranking
    experts by frequency has little to work with.
  - **Perfect one-layer-ahead prediction from pinned RAM would be worth +41%** (20.5 against 14.5 tok/s): real
    headroom, but in the miss path. **So M2's overlap work comes before any predictor,** and a predictor must first be
    scored on misses, not on all activations.
- **Replay methodology**: per-access replay inflated recency policies by 27-29% and inverted rankings, and
  single-template prompt sets moved results by 19-32 points (arXiv 2608.07911). Check our replay against both.
- **Exactness to the end**: golden bit-compare kernel tests at real shapes (Strata's pattern) as the single-layer
  harness E0, the kernel-boundary parity matrix, and a full-logit comparison (today: top-3 log-probabilities only).
- **Residency bonus for predicted experts** (Edge0's `pin_bonus` idea): exact and cheap, but deferred. E47 found
  routing near-uniform across prompts and layer-ahead id prediction dead.

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
| 1 | 107 commits behind upstream | **done 27/10**: rebased onto `63b4b303`, E48 passed (exact, decode exit met) |
| 2 | M0 drops the cache budget when it launches a plan | **done 27/10**: plans restore cache budget, miss source and tier |
| 3 | Prefill 4-5x behind stock with the tier | prefill track above |
| 4 | Exactness proved at top-3 log-probability, not full logits; no E0 harness or parity matrix | M1.2 above |
| 5 | Most arms ran one or two rounds | five rounds only for the decode exit; repeat any result that changes a default |
| 6 | The build has no C++ exception unwinding (`/EHsc` off) | to fix with the rebase build |
| 7 | One sequence at a time | by design: since 27/10 `serve`'s proxy enforces it (one running, one queued, 503 with a reason); multi-sequence is M2 |
| 10 | E52 (prefix-reuse exactness) registered run not made | to run when the GPU is free; client and inputs are ready (outside this repository) |
| 11 | Governor, `--confirm` and thread sweep tested offline only | a live competing-allocation test is the M1.3 exit |
| 8 | Linux paths never run; one machine only | M1.4 |
| 9 | Swift's publisher missing from the thanks | fixed 27/10 (UkisAI) |
