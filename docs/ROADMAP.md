# Marian roadmap

A living document: what we build next, why, and what evidence put it there. Updated 27/10/2026 after a scan of
upstream llama.cpp and of the wider ecosystem (other engines and 2025-2026 papers), and again the same afternoon
after register tests E51-E54 (the Python side of their two settings was built that evening, tested offline). Milestone numbers follow
[DESIGN-M1 section 10](design/DESIGN-M1.md); this page reorders them by what we have measured since.

Every item names its class, as in the design: **E** exact (answers cannot change), **D** verified speculation (the
target model's choice, computed in a different batch shape), **A** answer-changing (opt-in only, behind a blind quality
gate). Evidence marked *theirs* comes from another project's own machine and has not been reproduced here.

## Where we stand

- **Decode is solved on the reference laptop:** the P3 pinned tier meets the M1 exit, exactly (README, JOURNEY A21-A22).
  Since the middle of 27/10 the laptop itself runs about 20% slower (dev debt 17, not the code: E60), and on 28/10
  the exit was missed by the old and the new build alike (E58: 11.6-11.9 warm); it is re-measured when the machine's
  state is understood.
- **Prefill was the obstacle, and is now closed on this laptop by two opt-in, exact changes (E51, E54).** In the
  default configuration a 16K prompt with the tier still runs 4-5x slower than stock: large batches fault experts
  through the memory map, and the pinned tier takes the page cache they need. With P3 prefill
  (`--moe-prefill-source p3`) it runs 61 tok/s instead of 35; with a declared ubatch of 2048 on top, 157 tok/s,
  against 142 for stock at its default ubatch in the same session. A same-session check (E56) then measured the
  ubatch's own decode cost: warm decode **+6%** at 2048 (12.63 against 11.87, faster in 4 of 4 rounds), decode
  after a 16K prompt **-6%** (11.12 against 11.82, from two requests a run), and prefill about 3x. The earlier
  session's 16-18% decode gap was the machine, which ran slower that afternoon. So the ubatch is a trade: for a
  gate (long prompt, short answer) 2048 is the obvious declaration; the default stays 512.
- **Where the prefill time goes (E49, 27/10/2026): expert page faults, not the embedding table.**
  - **The suspect:** Swift's per-layer embedding table is 26.8 GiB, beside 33.0 GiB of experts on a 32 GB machine.
  - **Measured with `wsprobe`** (tools/wsprobe, no admin needed): the table held at most 0.48 GiB of the server's
    memory in any arm. A 16K prompt can touch at most 1.18 GiB of it (16 rows of 90 bytes per token), and upstream
    already prefetches those rows.
  - **What does hurt:** with the 6 GiB tier, prefill runs about 38,700 single-page disk reads per second, which is
    expert pages faulting in under memory pressure. That makes the P3 prefill source below the first fix.
- **The use case that drives the work:** a PII/sensitivity gate on 4-30B models, on an 8 GB GPU beside a desktop that
  uses most of 32 GB. Prefill matters as much as decode there, and today stock offload wins that case (not yet
  re-measured with the prefill patches; that is E50).

## Next, in order

### 1. Rebase onto current upstream (done 27/10/2026)

We are 107 upstream commits behind the pin `a37cc461`. The stack re-applies with mechanical fixes (the cache
constructor gained `no_alloc`), but three upstream changes alter Swift's numerics (qwen4exp attention selection, a
scale move, `softplus`). Exactness is therefore re-proved from scratch at the new base. Test E48 is pre-registered
before the build.

**The rebase rule (decided 27/10/2026).**
- **When:** rebase onto upstream weekly, or at once when an upstream commit touches what Marian runs: the MoE
  cache, the CUDA matrix, scale and attention kernels, the Gemma and Qwen model code, or a file our patches touch.
- **Every rebase re-proves exactness** with an E48-style check: the cache arms identical to all-VRAM, the stock
  contrast still differing.
- **A daily upstream watch** (git only, no model calls) does three things each morning: it lists the new upstream
  commits, flags the ones that touch those paths, and trial-applies our whole patch stack onto upstream master. It
  only says *when*; it never rebases.
- **First reading (27/10, evening):** 13 commits behind `63b4b303`, and the patch stack applies cleanly. Three
  commits touch what we run: a CUDA scale-kernel change (non-contiguous views), a KV-cache clean-up, and MoE
  rescaling for another model family. So the next rebase is due soon, with its exactness check.
- **The second rebase (28/10/2026, register E58):** the morning watch flagged 11 of 32 new commits, three able to
  move numerics (the CUDA scale kernel, qwen4exp's k-pool selection, the KV cache's n_pad). The nine patches
  cherry-picked onto `338fce651` (b11936) with no conflict and are byte-identical. E58 passed on exactness (every
  Gemma cache arm, P3 prefill included, identical to all-VRAM; the stock contrast still 0.590) and on regression (the
  new build 1.02x the old warm, 0.98x after 16K, same session). The decode exit was missed by **both** builds that
  morning (11.6-11.9 warm), so by the registered branch the machine, not the rebase, missed it (dev debt 17). None of
  the 32 commits moved an answer on either model.

### 2. Prefill track (the main target)

| Item | Class | Why | Evidence | Effort |
|---|---|---|---|---|
| ~~Measure first: residency by tensor kind~~ **done (E49)**: the embedding table is ruled out; expert page faults are the cost | test | | `wsprobe` | done |
| ~~Prefill misses through P3~~ **done (E51), shipped opt-in**: patch 0008 routes the large-batch selective copy through the tier and the unbuffered ring, never memory-map faults; patch 0009 adds `--moe-prefill-source mmap\|p3`, default `mmap` (upstream's path, byte for byte); patch 0007 pins the tier in chunks under a RAM cap | E | 16K prefill with the tier +76% (34.7 -> 61.1 tok/s, three rounds), decode unchanged or better, exact (Gemma's tier arm identical to all-VRAM; Swift's cache arms identical to each other). Short of the registered 2x, and 0.35x stock: drive-bound. Opt-in because it is slower when the model fits in the page cache (Gemma 4 26B, 5,428-token prompt: 260 against 573 tok/s, one run each) | our E51 | done |
| ~~Is a larger ubatch exact?~~ **done (E54): a declared ubatch is class E against a reference at the same ubatch** | E, declared | At `-ub 2048 -b 2048` the memory-mapped cache arm and the tier arm with P3 prefill match all-VRAM at 2048 on ten 2,009-2,406-token prompts, while all-VRAM at 512 differs from all-VRAM at 2048. With the tier and P3 prefill, Swift's 16K prefill is 156.7 tok/s against stock's 142.4 at its default ubatch, exactly. Ships as a declared profile parameter, default 512: calibration measures prefill and decode at the declared value and reports the trade. **Python side built 27/10** (offline tests only): `calibrate --ubatch N`, `serve --ubatch N`, profile schema 3, a plan at another ubatch refused | our E54 | done; calibration rows built |
| **Same-session decode check: T8 at ubatch 512 against 2048** | test | E54's decode was 16-18% under E51's (12.67 against 15.15 warm, 11.47 against 14.07 after 16K), but the comparison crossed sessions: stock also ran about 10% slower, the hit rate was unchanged (0.906), host time per token rose, and E51's one `-ub 2048` run decoded at 14.9. Before any claim that the cost is intrinsic | our E51, E54 | S |
| **Compute-buffer sizing for a large ubatch** | E | The 2,048-row compute buffer took about 300 MiB more VRAM beside an unchanged 9,400 MiB cache (peak 15,294 against 14,992 MiB). Size it for the large batch without shrinking the cache's share | our E54 | S-M |
| **Prefill expert prefetch**: read layer l+1's experts while layer l computes | E | The fault stalls are serial today | ik_llama.cpp #2101, #2667: 1.9-2.7x on models larger than RAM (*theirs*, Linux) | M |
| **Pin mapped expert ranges** (upstream #31122, closed by a bot, not on merit) as a measured arm | E | Prompt speed +27-44% (*theirs*) | upstream PR #31122 | S |
| **Layer-major chunked prefill**, then refill the cache with the prompt's hottest experts | E at a fixed ubatch | Moves M5 forward: a 12 GB card prefilled 231 tok/s at 8K this way. **Note after E51:** the next lever is fewer bytes per prompt, not a faster source. At the default ubatch a 16K prompt uploads 509 GiB of experts, because each 512-token ubatch re-reads most of each layer's experts; the point of layer-major prefill is to read them once per larger chunk | Strata v0.2.4; MoE-CORE (arXiv 2610.01950); Maya #352 (*theirs*); our E51 | L |

**A correction.** On the morning of 27/10 we classed a larger ubatch as A "until shown otherwise", because it
changes the compute shape (one of that day's logged decisions; E51 reported its `-ub 2048` arm that way, as its text differed
from the default ubatch's). That compared two different references. Marian's exactness claim is against an all-VRAM
run with the same settings, as for the context size, and DESIGN-M1 already said "E at a fixed `n_ubatch`". E54
measured it that way: a larger ubatch is class E relative to a reference at the same ubatch. It is a declared
setting, not an answer-changing switch, and a deployment that declares it gets the all-VRAM answers at that ubatch,
not at 512.

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
  one request at a time with one queued, answers 503 with a reason, and streams. **The governor's default changed
  after its first live run (register E53, 27/10).** With a VRAM hog leaving 150 MiB free, then a RAM hog leaving
  under 1 GiB available, a plain llama-server slowed by at most about 4%. The old governor paused, restarted on a
  smaller row (9.5-11.7 tok/s, from 71-76) for about 240 s, and never stepped back up, though 11 GiB of VRAM was
  free again. Answers stayed exact across its restarts. Now `--governor watch` (default) reads the same floors, warns
  with the live decode rate, and pauses admission only on low commit headroom (the crash risk); it never restarts
  on a reading. `--governor restart` (opt-in) restarts on the next smaller row only when decode stays under 70% of
  the row's own baseline (the median of its first 8 answers) for 10 s while a floor is crossed, and steps back up
  after 60 s of memory that fits the planned row, never with a queue; step-ups count against the 3 restarts.
  `--governor off` runs none. The new modes are tested offline only.
- **Fixed-prefix KV reuse for the gate: measured, NOT exact, so class A** (register E52, 27/10/2026).
  - **Why it was tried:** a gate re-reads the same instructions every time. Strata v0.2.3 reused 91% of a shared
    prompt (*theirs*).
  - **The test:** Gemma 4 26B-A4B IQ4_XS, a 1,608-token instruction block (1,616 with the chat template) plus ten documents.
  - **The result:**
    - Reused answers matched their fresh twins in 0 of 10 pairs, both all-VRAM and with the cache and tier;
      log-probabilities moved by 1.08-4.66.
    - A fresh prompt run twice matched exactly, and the reuse was real: only the document's tokens were evaluated.
    - Reuse would have saved 83% of prompt tokens (2.8-4.3x prompt time).
  - **Status:** plumbed (`--cache-prompt on`, `--gate-prefix FILE`) as an opt-in answer-changing switch, off in
    exact mode.
  - **Endpoint note:** on this model (sliding-window attention), llama-server reuses a prefix through `/completion`
    only after a short previous prompt, and through `/v1/chat/completions` always.

Still to build:

- ~~Pin in chunks under a cap~~ **built in the fork, patch 0007 (27/10), in the default build.** On Windows the
  driver limits pinned memory per allocation and per process; a one-shot pin fails and silently falls back to
  pageable memory. *Theirs*: ik #2604 (10.3 vs 3.6 GB/s), Maya #417. 0007 pins in chunks of at most 1 GiB and stops
  growing at a chunk that cannot be pinned, keeping what is pinned (not all-or-none). It caps the tier at
  min(request, 50% of physical RAM, commit room minus 4 GiB), after upstream #31122's Windows note, Maya v1.1.27 and
  Strata v0.2.6. No measured cost (E51), but in E51 and E54 every request sat below the cap and no chunk failed, so
  the cap and the failed-chunk branch are untested (dev debt 14).
- ~~Calibration rows for prefill~~ **built 27/10 in Python, tested offline only** (decided after E51 and E54):
  - On a 0009 build every ring and tier row is measured with `--moe-prefill-source mmap` and `p3`. The pair is judged
    on prefill and decode together, with differences within 10% counted as ties: the source faster on one and not
    slower on the other keeps its row, its twin is recorded as `dominated` (skipped by the planner), a tie keeps
    `mmap`, and a split keeps both for the planner's objective. `p3` is never paired with memory-mapped misses.
  - A declared ubatch is a profile parameter (`calibrate --ubatch N`, `serve --ubatch N`; default 512, no flag;
    profile schema 3, older profiles count as 512). Every row is measured at it, a profile at another value sits
    beside the default one, and calibration prints decode and prefill beside other ubatches' profiles, marked as
    another session. Plan, `--confirm`'s runner-up and the governor choose only rows at the declared value, and serve
    refuses a plan measured at another, as it refuses mixing exactness classes. E56 measured the trade at 2048 (about 3x
    prefill, +6% warm decode, -6% decode after a 16K prompt); calibration measures it per machine and the code
    encodes no result.
  - Not yet run against a real engine.
- Learned counts (P4), and a system-wide VRAM reading for the governor. The design asked for the DXGI budget, but a
  live read (27/10) showed `QueryVideoMemoryInfo` reports only the calling process's budget and usage: with another
  process holding 15 GB it still read 15,280 MiB free, against `nvidia-smi`'s 1,090. So the governor keeps
  `nvidia-smi` and the next candidate is the adapter-wide counter `GPU Adapter Memory(*)\Dedicated Usage`.
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
- **A declared-ubatch row is reported beside it** (decided with E54); the rule stays as written. With a long prompt
  and a short answer, a declared `-ub 2048` row with P3 prefill is the likely contender: on the 16 GB laptop it beat
  default stock on prefill and decode, exactly.

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

| # | Debt | State | Ends when |
|---|---|---|---|
| 1 | 107 commits behind upstream | **done 27/10**: rebased onto `63b4b303`, E48 passed (exact, decode exit met); **again 28/10** onto `338fce651`, E58 passed on exactness and regression (decode exit missed by both builds: the machine) | ended 27/10; the rebase rule keeps it ended |
| 2 | M0 drops the cache budget when it launches a plan | **done 27/10**: plans restore cache budget, miss source and tier | ended 27/10 |
| 3 | Prefill 4-5x behind stock with the tier | **paid down 27/10, opt-in, and calibration now measures both prefill sources and a declared ubatch (offline):** P3 prefill +76% at 16K, exact (E51); with a declared ubatch of 2048, 156.7 tok/s against stock's 142.4 at its default ubatch, exact against all-VRAM at 2048 (E54). The defaults (`mmap`, ubatch 512) are unchanged, so a default tier row is still 4-5x behind; prefill track above | a default (not opt-in) configuration prefills within 10% of stock's best on the reference laptop, exactly |
| 4 | Exactness proved at top-3 log-probability, not full logits; no E0 harness or parity matrix | M1.2 above | an E0 single-layer harness passes bit-compare at real shapes, and a full-logit comparison is logged for the exactness arms |
| 5 | Most arms ran one or two rounds | five rounds only for the decode exit; repeat any result that changes a default | every result that changed a default today has a confirming run of at least three rounds |
| 6 | The build has no C++ exception unwinding (`/EHsc` off) | **done 27/10 (E57)**: built both ways from the same sources, every answer identical (max difference 0 on every exactness arm, the stock contrast still 0.59); `docs/m1.3/tools/build_m13.bat` turns it on | ended 27/10 (E57) |
| 7 | One sequence at a time | by design: since 27/10 `serve`'s proxy enforces it (one running, one queued, 503 with a reason); multi-sequence is M2 | by design; ends only if M2 lifts it |
| 8 | Linux paths never run; one machine only | M1.4 | one Linux run passes the exactness suite (M1.4) |
| 9 | Swift's publisher missing from the thanks | fixed 27/10 (UkisAI) | ended 27/10 |
| 10 | Prefix reuse: is it exact? | answered 27/10 (E52): not exact (0 of 10 pairs); class A, off in exact mode | ended 27/10 (E52) |
| 11 | Governor, `--confirm` and thread sweep tested offline only | governor run live once (E53, 27/10): restarting on a memory reading did harm, so the default is now `watch`; `--governor restart` (slowdown trigger, step-up), `--confirm` and the thread sweep are still offline only | `--governor restart`, `--confirm` and the thread sweep each have one registered live run (E55 for the governor; designed and registered 28/10 with a compute-hog phase that must NOT restart and an oversubscription hog that must, waiting for those two tools) |
| 12 | Serve's request loop decodes 10-16% under calibrate's figure for the same row on a clean machine (E53) | not investigated; the live check now says "below profile", and the restart trigger uses the row's own measured baseline, not the profile | the gap is explained or closed: a measured cause, or calibration measuring through the same request loop serve uses |
| 13 | E53's paging-cliff scenario (a model loaded beyond VRAM) was not reproduced by pressure on a running server | a launch-time event, guarded by the launch checks against the VRAM and RAM free at launch; the governor does not cover it | a launch-time test loads a model beyond VRAM and the launch checks refuse it |
| 14 | 0007's cap branch never triggered | in E51 and E54 every tier request (4,096 and 6,144 MiB) sat below both limits and no chunk failed, so the cap and the stop-at-a-failed-chunk branch have never run on the GPU; the Linux branch has only been compiled. Needs a planted test: a request above the cap, and a forced failed chunk | a planted test drives a request above the cap and a forced failed chunk, and both behave as written |
| 15 | E54's decode figures were judged against E51's, from another session | **done 27/10 (E56)**: a same-session run of T8 at 512 against 2048 showed the machine, not the ubatch, caused the gap | ended 27/10 (E56): the 16-18% gap was the machine; the ubatch trade is +6% warm decode, -6% after 16K, about 3x prefill |
| 16 | E53's (d) clause was mis-specified: an absolute threshold (shared GPU memory at most 256 MiB) with no baseline | every all-VRAM llama-server read about 854 MiB of shared GPU memory from model load, on a quiet machine, so (d) could not pass and its "could not fail" escape could not trigger. The FAIL stands as written. E55 (registered 28/10, before any run) defines it, as its clause (f), as a rise above the arm's own P0 median, and says that definition was written after seeing E53's data; not yet run | E55 is registered with (d) as a rise above each arm's own P0 baseline, and run |
| 17 | The whole machine ran about 22% slower on the afternoon of 27/10 than that morning (the same T8 configuration: 11.87 against 15.15 tok/s decode; stock fell too) | **ended 28/10 by its second condition; cause narrowed, not named.** Forensics: the GPU side and host side slowed together (same bytes, CPU perf, GPU share, free RAM); the 2x slump ran with the CPU below base clock; power source, Defender scans and memory ruled out. E60: 27/10 morning's own binary runs 0.96x today's build, so not the code. GPU logs show the GPU idling at 800-1,200 MHz between decode bursts: the platform's power state is the suspect; testing it needs a driver power-management change on the reference laptop. `tools/prerun/prerun_check.py` and a GPU clock log run beside every registered speed run since E58 | ended 28/10 (pre-run check in every registered speed run) |
| 18 | Calibration's prefill probe fills 75% of the measured context, so a 4K calibration judges the P3 prefill source on about 3K tokens, while its gain was measured at 16K | **done 28/10**: `calibrate --probe-prompt N` probes at the deployment's prompt length (refused if it does not fit the context); the profile records the probe; default unchanged; offline tests | ended 28/10 |
| 19 | Is the logical batch `-b` part of the reference? The default is `-ub 512 -b 2048`, a declared N is `-ub N -b N`; unmeasured | **answered 28/10 (E59): yes.** At a fixed ubatch, `-b 512` against `-b 2048` matched 0/10 records (0.603); the cache path at `-b 512` still equals all-VRAM at `-b 512`. Profiles now record `-b`; launch refuses a different `-b` and `LLAMA_ARG_BATCH`/`LLAMA_ARG_UBATCH` | ended 28/10 (E59) |
| 19 | The default ubatch means llama-server's own `-ub 512 -b 2048`; a declared ubatch N passes `-ub N -b N` | only the ubatch should set the compute shape; not measured | one exactness run shows `-b` does not change answers at a fixed ubatch |