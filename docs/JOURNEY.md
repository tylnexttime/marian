# The Marian journey

How this repository came to be, in order, with the options we rejected and the mistakes we made. Written for Chris,
for the agents who work with us, and for a future contributor who wants to know why things are the way they are.

The story comes first (sections 1 to 8). Every measured table so far is in the [Results appendix](#results-appendix)
at the end, grouped by date, each with one line on its conditions and one on what it taught us. Results still
running have marked placeholders there.

**Conventions.** tok/s figures are medians over an arm's requests unless stated. An "arm" is one configuration run
once with a fresh server; "r1 / r2" are two counterbalanced rounds. "Stock" means upstream llama.cpp with whole
expert layers offloaded to the CPU (`-ncmoe N` keeps the experts of N layers in system RAM). "Swift" is the Swift 1.5
IQ2_XS quantisation of Qwen3.8-Flash-Next (48 layers, 512 experts, 10 used per token, 33.02 GiB of routed experts).
"The reference laptop" has an RTX 4090 Laptop GPU with 16 GB of VRAM, an i7-13700HX, 32 GB of DDR5-4800, Windows 11
and an NVMe model drive (`docs/design/RESEARCH-PACK.md`).

**Where the numbers come from.** Files in this repository are cited by path. The result tables are copied into the
repository as `docs/m1.0/*-summary.md`, `docs/m1.1/*-summary.md` (the gate pair on the M1.1 build is
`docs/m1.1/gate11-summary.md`) and `docs/m1.1/probe/*.txt`, so a citation such as `m1.1/p3-summary.md` means
`docs/m1.1/p3-summary.md`. Register entries (E34 to E41) are in the project's private test register; every number they
carry is also in a table here. Raw per-request logs and traces stay in the research workspace. Anything not verified is
marked.

---

## 1. Origins: big models on a 32 GB laptop

Marian started as a side question in a longer run of local-model experiments on the reference laptop. In June 2026 a
first proof of concept, through Ollama, ran a 30B-A3B MoE at about 61 tok/s and a 35B-A3B at about 48 tok/s
(June 2026 notes, not in this repo). The same notes recorded the lesson that shaped everything after it: a best-case
57 tok/s for a tuned 35B-A3B setup fell to 25-35 tok/s, and at times to single digits, once ordinary desktop
programs were open, because free RAM while the model was loaded dropped to 4-6.5 GB. **RAM headroom, not raw
capacity, decides how these models behave on a real desk.**

On 21 and 22 October four register entries asked how far that goes (full numbers in [A1](#a1-e34-glm-53-flash-through-project-maya)
and [A3 to A5](#a3-e35-strata-and-swift)):

- **E34 (21/10):** could a 321B MoE (GLM-5.3-Flash, as Project Maya's 94.7 GB Maya-S24 pack) run at all? Yes, at
  about 2.1 tok/s at best (1.7 on Maya's own benchmark). It was **SSD-bound**: 67-75% of each decode token was spent
  waiting on the SSD, which read about 2 GB/s for the whole run. 32 GB of RAM cannot hold the experts.
- **E35 (22/10):** Strata with Swift (68 GB) reached **35.5 tok/s median decode** with an 85.4% VRAM expert hit rate.
  My prediction before the run ("disk-bound like Maya") was wrong for decode. **Prefill is where 32 GB bites**:
  155 tok/s at 4K, 193 at 16K, and 83 s to the first token of a 16K prompt.
- **E36 (22/10):** Strata's own tuner found no faster setting, and the model passed Strata's 12 sanity checks. Its
  repeat gave the laptop's noise floor: about plus or minus 10% on decode and 15-18% on prefill between identical runs.
- **E37 (22/10):** a blind three-way quality comparison (GLM-S24, Qwen3.8-27B dense, Flash-Next) was a **near-tie**:
  a panel of three judges ranked GLM first and Chris ranked Flash-Next first, so the pre-registered rule said "no
  consensus", and speed and footprint decided. One check could not have failed (it parsed the reasoning instead of the
  answer); it was caught and the answers were checked by hand.

The same Swift files ran at **7-10.6 tok/s warm (5-6 cold) on stock llama.cpp** against Strata's 35.5
(`docs/design/RESEARCH-PACK.md`, "Same GGUF"). Strata gets there partly with switches that change answers (route-tail
skip, a CPU/GPU split that rounds differently, speculation), so 35.5 is prior art, not an exact-mode target. The gap
between 7-10 and 35 tok/s on identical files is the reason Marian exists: **llama.cpp places whole expert layers;
Strata and Maya rank individual (layer, expert) pairs by how often they are used.** Chris's idea was to bring that
per-expert tiering to any GGUF MoE as a thin patch set on llama.cpp.

## 2. The name, and M0 (21/10/2026)

Marian is named after a popular Polish given name (`README.md`).

M0 came first and was written by Alder (credited in `docs/design/DESIGN-M1.md`, section 13). It is a standard-library
Python tool: a machine fingerprint without the hostname, GGUF header inspection, a profiler that walks the GPU/CPU
split for each context size and KV type on a real server, a planner that never extrapolates, and a launcher. The
first commit landed on 21/10 at 12:56; the first real profile crashed it (a sampler flag shadowed `Thread._stop`),
was fixed and tested 10 minutes later, and the first measured profile was committed at 13:48 (git history).

That profile ([A2](#a2-m0s-first-profile-gemma-4-26b-a4b-on-an-8-gb-laptop-gpu)) measured Gemma 4 26B-A4B IQ4_XS on
a laptop with 8 GB of VRAM and 32 GB of RAM from 4K to 32K tokens. Two findings set the tone:

- **f16 KV beat q8_0** (30.3 against 22.3 tok/s at 4K): the smaller KV cache freed too little VRAM to move another
  expert layer onto the GPU, so the "obvious" memory saving lost.
- **The overflow cliff is sharp.** One more layer on the GPU than fits turned 29-30 tok/s into 6.6-8.3 tok/s, because
  Windows pages GPU memory into system RAM instead of failing. At 16K and 32K the overflow showed in prefill first.

## 3. Chris's decisions (22/10/2026)

After the Strata-against-stock comparison, Chris decided (project decisions note, 22/10):

1. **Shape: a thin fork on the latest llama.cpp**, not on the older pin the stock baseline used ("let's not waste our
   time, and move to the latest upfront"), borrowing ideas from Strata and Maya (both MIT, credited).
2. **Exact by default.** Speed-ups that can change an answer (route-tail skip, CPU-versus-GPU rounding and the like)
   are opt-in switches, each measured blind for quality before it could ever become a default.
3. **Built here, by Claude Tyl.**
4. **Machine-adaptive.** Marian must calibrate itself on whatever machine it runs on, and must not be tuned for the
   reference laptop. Every machine runs its own tests on first launch, before any model is served.
5. **Minimum spec: 32 GB of RAM plus an 8 GB GPU**, "work from there".
6. **Design process:** draft the design, ask two outside models (Astra and Fable) for out-of-the-box ideas, then a
   external critique asking specifically for performance improvements as well as defects.

The design ended with eight open decisions. Chris delegated them ("you decide on the best course here"), and the calls
were (`docs/m1.0/M1.0-STATUS.md`, section 1, and the decisions note): build on upstream's cache rather than port
Strata; speculation off by default, allowed later only in a named non-exact preset after a blind quality bar; ship
routing profiles in the repo; no cache clears that need administrator rights (warm runs and interleaved repeats
instead); validate on a real 8 GB card and on Linux later; no rented 40 GB GPU until M1 shows gains here; offer
patches upstream once measured.

## 4. Designing M1 (22/10/2026)

**The research pack** (`docs/design/RESEARCH-PACK.md`) was written by Sol, an AI model, from line-numbered sources:
llama.cpp at the pin, Strata, Maya, M0 and our baseline measurements. Two addenda followed: Swift's expert bytes by
layout (three layouts in 34, 11 and 3 layers; about 660 MiB of experts per token under uniform routing; a correction
that the baseline's "32 IQ2_S layers" is really 34) and Edge0 as prior art.

**The finding that changed the plan:** two weeks earlier, upstream llama.cpp had merged its own GPU cache for MoE
experts held in host memory (`--moe-cache-mib`, [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887), multi-GPU [#30112](https://github.com/ggml-org/llama.cpp/pull/30112), SLRU policy [#30637](https://github.com/ggml-org/llama.cpp/pull/30637)). It already had
per-expert slots, a slot map, `MUL_MAT_ID` on the slot bank, and computes **every expert on the GPU**, so its output
should not depend on what is cached: that is Marian's exact mode almost for free. The design became "a small patch
set on upstream's cache plus a calibration layer on top", not a new cache (`DESIGN-M1.md`, section 1).

**Three idea files**, saved verbatim in `docs/design/ideas/`:

- **Mine, written before reading the others** (`claude-tyl-ideas-before-consult.md`): ranked warm start, frequency
  admission with a pinned core, per-layer budgets, next-layer prefetch by pre-gating, removing the 48 blocking syncs
  per token, a bit-faithful CPU twin of the GPU kernel (so CPU help could be exact), and treating the OS page cache
  as an unmanaged RAM tier that should be managed.
- **Astra** (`astra-ideas.md`): optimise **exposed stall time per token across VRAM, RAM and NVMe**, not VRAM hit
  rate alone ("an 85% hit rate whose misses fault from NVMe loses to an 82% hit rate whose misses come from pinned
  RAM"); transient slots so a declined admission still computes on the GPU; one expert-source interface for both copy
  paths; a route "itinerary" predictor; bounded RAM with a small pinned ring. Astra explicitly rejected two options:
  replacing upstream execution with a Strata-style engine, and pinning the whole host complement.
- **Fable** (`fable-ideas.md`): sync-free slot resolution on the GPU with zero-copy misses; a TinyLFU doorkeeper; a
  large pinned RAM arena with an unbuffered NVMe tail; a row cap that keeps speculation on the fast kernels; and the
  falsifier that mattered most: **if upstream's plain SLRU already reaches 84%, ranking buys little.** Fable guessed
  SLRU would land at 70-80% and decode at 20-28 tok/s.

**The critique** (`docs/design/critique/CRITIQUE-M1-v1.md`): three critics (Sol, Luna and Astra) raised 24 items; 0
were refuted, 13 survived the refuter, 3 were "unverifiable" because of an apparatus quirk and were decided by hand.
The refuters were all from one model family, which `DESIGN-M1.md` section 14 records as a weakness. Every item was
folded into DESIGN-M1 v2 (section 14 maps each to its change). The ones that changed the design most, judged with
hindsight:

| Code | What it said | What it changed, and did it matter? |
|---|---|---|
| C-03 | bound the expert source before the M1 exit | an endurance check in M1.0, and "minimal P3 moves into M1 if decode after a 16K prompt misses 12 tok/s". **It missed, and P3 moved in.** |
| C-24 | a ranked core needs a capacity invariant per group | slots per group must cover the largest step. **M1.0 hit exactly this bug** (section 5). |
| C-10 | a cache-off or overflowing arm may compute on the CPU | arms must be shown GPU-only before logits are compared. **The same bug.** |
| C-06 | distinct-expert counters are not lookup hits | the 85% target became a row-level, per-token definition |
| C-07 | no valid timestamp for exposed wait | GPU-event timing and defined "drain" |
| C-13 | the parity gate skipped the kernel row-count boundaries | a parity matrix at rows 1, 7, 8, 9, 32, 33 per layout (not built yet) |
| C-20 | fresh processes do not equalise the page cache | named states (warm, post-prefill, cold) and recorded physical reads |
| C-02 | the governor cannot shrink a fixed bank | pause admission, then refuse or restart at a smaller plan |
| C-16, C-23 | class D default unresolved; concurrency undefined | speculation off in the exact default; one sequence at a time |
| C-09, C-17 | no literal Swift oracle; a ballast is not a smaller GPU | both stated as limits; real hardware moved to M1.4 |
| C-05, C-15, C-21, C-08, C-11, C-01, C-22, C-14, C-12, C-04 | trace coverage, profile invalidation, probe safety, budgets, warm fill, sidecar checks, crash-safe writes, direct I/O, ordering | each folded into the relevant patch or milestone |
| C-19, C-18 | keep upload coalescing; measure 8 verify rows | parked, adopted cheaply |

DESIGN-M1 v2 also ranked the levers by expected gain (section 15): **drain removal first, miss sourcing fourth.**
M1.0 reversed that order within hours.

## 5. M1.0: measuring before building (22/10/2026)

The fork was built at `a37cc461` with one patch, **P0**, a trace of every cache lookup and every large-batch copy
(row-level hits before the step's fills, fills, bytes, host time waiting in and after the per-layer sync). Then the
Swift campaign: stock `-ncmoe 34` against upstream's cache at 4,000, 6,500 and 9,400 MiB, two counterbalanced rounds,
the same 10 decode prompts, a 4K and a 16K prompt, then decode again after the long prompt
(`docs/m1.0/M1.0-STATUS.md`; tables [A6 to A8](#a6-m10-swift-campaign-upstream-cache-against-stock)).

The surprises:

1. **Upstream's plain SLRU already met the hit-rate target**: 84.4% at 6,500 MiB and 90.8% at 9,400 MiB. Fable's
   falsifier fired, and Fable's own 70-80% estimate was wrong in our favour. The ranked policy (P2) is worth a few
   points at most here.
2. **And decode was about 30% slower than stock** (7-8 against 10.6-10.9 tok/s). Hit rate was not the bottleneck.
3. **The time was in fetching misses, not in the syncs.** Time inside the 48 per-layer syncs was about 0.8 ms per
   token, under the 1.5 ms threshold the design set, so drain removal (lever 1) dropped. The **71-88 ms per token
   after the sync** was the host issuing uploads from the memory-mapped model file, about 0.95 GB/s effective.
4. **The copy probe showed why** ([A7](#a7-copy-probe)): one expert copied from a faulting memory-mapped page moves at
   0.41 GB/s, from resident pageable memory at 6.6 GB/s, and from pinned memory at 10.3 GB/s. Free RAM sat at
   0.06-0.18 GiB, so misses mostly faulted to the SSD. Stock is faster because the CPU computes its host-side experts
   in place instead of copying them over PCIe. **Where misses come from is the lever.**
5. **Every arm collapsed after the 16K prompt** (3.2-5.1 tok/s): prefill flushes the page cache that decode needs.
6. **The cache broke exactness at 4,000 MiB.** Arms at 6,500 and 9,400 MiB gave identical greedy text and top-3
   probabilities (10/10), as designed. At 4,000 MiB only 7/10 matched. The trace showed why: the three IQ1_M layers
   form their own small cache group, which got 180 slots; three prompt chunks of 19-28 tokens select more experts than
   that, so upstream's capacity check sent them off the cache path, and below 32 tokens a host-resident expert is
   computed **on the CPU**. Three CPU-computed steps, three changed answers. This is C-24 and C-10 happening for real.

What we rejected or reshaped as a result (`M1.0-STATUS.md`, section 3): M1 was **reshaped around the miss path, not
the residency policy**; minimal P3 moved into M1; the lever ranking flipped (miss sourcing first, drain removal down);
and the CPU lane (opt-in, answer-changing) looked stronger than estimated on this machine.

Mistakes on 22/10: a test-runner script was corrupted for a few minutes by a Python string escape (`\r` in a Windows
path became a carriage return); the arm running then had already parsed the file, and it was repaired and
parse-checked. A process stop at the end also killed the background task's wrapper shells. Two other checks that day:
the Gyro-S model uses ggml type ids llama.cpp does not define, so it cannot load; and Edge0 was read as prior art (its
trained predictor replaces the router, so it changes answers). Chris called the stop at 15:20 during the first 8 GB
floor arm.

## 6. Saturday 24/10/2026: floors, the gate, exactness, the probe, P3

**The M1.1 code was built before the GPU was free** (commit `7fa8f50`, "built, not yet run on the GPU";
`docs/m1.1/M1.1-NOTES.md`): an exactness floor (0001), a fill trace with expert ids (0002), a trace-replay probe, and
P3 minimal (0003), an opt-in miss source with three modes: `mmap` (upstream, the default), `ring` (unbuffered
overlapped reads into a small pinned staging ring) and `tier` (a pinned RAM tier with TinyLFU-style admission in
front of the ring). A CPU-only self-test against the real Swift shards checked 42,069 expert tensors with 0
mismatches before any GPU run.

**The 8 GB floor and the desktop-RAM simulations.** The reference laptop stands in for the minimum-spec laptop by
holding about 8 GB of VRAM with a ballast; holding about 25 GB of RAM simulates a desktop full of everyday work. Both
are capacity simulations only (C-17). Results:

- **Swift at 8 GB** ([A9](#a9-swift-at-8-gb-the-floor-arms)): 4.9-5.9 tok/s with a 2,400 MiB cache, 7.5-7.7 with
  stock `-ncmoe 45`. Far below the floor; Swift is not an 8 GB model.
- **Gemma 4 26B-A4B at 8 GB with RAM free** ([A10](#a10-controls-gemma-4-26b-a4b-at-8-gb-ram-free)): **42.6 tok/s
  with the cache against 16.8 stock**. When RAM can hold the experts, the per-expert cache wins hugely.
- **The same model with 25 GB of RAM taken** ([A11](#a11-gate-arms-8-gb-gpu-and-about-25-gb-of-desktop-ram)):
  **8.0 tok/s with the cache against 15.0 stock**. When misses must come from the SSD, the cache loses.

**The gate question.** Alder asked whether a laptop with 8 GB of VRAM and 32 GB of RAM, most of it taken by everyday
desktop work, could run a 4-30B model at **20 to 30 tok/s**, with prefill mattering as much as decode because the intended use
reads long inputs and writes little. A research-only survey that morning (`m1.0/gate-candidates.md`, from Hugging
Face model cards and the fork's architecture table; nothing benchmarked) shortlisted Gemma 4 26B-A4B-it (with an
IQ3_XXS fallback), Qwen3.5-9B as the dense bar to beat, Gemma 4 E4B-it with NVIDIA's post-trained
Nemotron-3.5-Bolt-E4B-4GB beside it, and Groot-19B-A3B-261023 as an unverified lead published the day before.

**The honest answer: yes, and stock llama.cpp did it, not Marian's cache.** Gemma 4 26B-A4B at UD-IQ3_XXS with
`-ncmoe 18` ran at **26.3 tok/s** decode and 756-775 tok/s prefill; the small dense models that fit wholly in VRAM
ran at 65.4 (Qwen3.5-9B), 79.7 (Bolt E4B) and 89.7 tok/s (E4B QAT). Every cache arm in that setting lost to the
best stock placement (8.0 against 15.0 for Gemma IQ4_XS, 12.7 against 26.3 for IQ3_XXS, 5.0 against 19.2 for Groot).
A measuring tool that reports this is doing its job.

**A bug that silently skipped three arms.** On the first pass, three gate arms produced nothing and the campaign
carried on. The arm runner declares its server-arguments parameter as a mandatory string, and PowerShell refuses an
empty string for a mandatory parameter, so arms that needed no placement flags could not start. My reading is that
these were the three whole-on-GPU small models, which are now in the gate table; that reading comes from the runner's
code (`docs/m1.0/tools/run_arm.ps1`, line 3), not from the run log, which I have not checked. The lesson is the usual
one: a campaign script must fail loudly when an arm produces no output.

**The exactness floor fixed the answers** (E38, [A13](#a13-e1-the-exactness-floor)). Every cache group now first
gets enough slots for the largest step on the slot-map path, `min(min(n_ubatch, 32) x n_expert_used, n_expert)`
(320 for Swift), and budgets below the sum of the floors are refused with a message instead of silently computing on
the CPU. At 4,000 MiB the answers became **10/10** identical to 9,400 MiB (M1.0: 7/10), the log carried the floor
warning, every layer showed the same number of multi-token steps, and the new binary matched M1.0's at 9,400 MiB
10/10. Speed was unchanged (7.1 and 7.5 tok/s).

**The replay probe said GO, with an honest caveat** (E39, [A14](#a14-the-trace-replay-probe)). It replays a real
trace's misses through three sources. The pre-registered rule was GO if the ring or the tier halves the median host
time per token under memory pressure. The tier did (0.32 with synthetic ids, 0.27 with real ids); the ring was
borderline (0.51 and 0.36). But the probe's memory-mapped pass was 23% slower than the server with synthetic ids and
1.9x slower with real ids, because the probe itself held 14 GiB of memory and left less page cache. So its baseline
was harsher than reality and its ratios flattered the new sources: against the server's own 69-73 ms, the honest
gains were about 0.65x for the ring and 0.40-0.51x for the tier. The register records the GO and that it is weaker
than it looks, and it pre-registered predictions for the server arms.

**P3 on the GPU: the exact RAM tier roughly doubles Swift decode** (register E40,
[A16 to A18](#a16-p3-verify-run)). First a correctness run with every read checked against the memory map: 11/11
requests answered, unbuffered reads, nothing fell back to pageable memory. Then two rounds of four arms on Swift with
a 9,400 MiB cache:

- **Decode:** upstream's path 7.1 / 6.1 tok/s, the ring 11.7 / 11.5, the 6 GiB tier **14.2 / 13.8**; after a 16K
  prompt 3.5 / 3.3, 7.6 / 8.4 and 10.9 / 10.3. Stock was 8.2 / 10.4 in the same campaign. **All three cache arms give
  identical answers (10/10 texts and top-3 probabilities) within and across rounds.**
- **Every speed prediction came in too low.** E39 predicted 8.3-9.5 tok/s for the ring and 9.5-11.5 for the tier.
  From upstream's path to the tier, host time per token fell by about 40 ms but token time by about 70 ms. Why the
  gain exceeds the host-time saving is unexplained; E40 records a guess, not a finding, that page faults stall more
  than the upload path itself.
- **The predicted prefill cost was far worse than predicted:** "equal or slightly worse" became **about 3x slower**
  (16K prompt: 34-36 tok/s against 112-119), because prefill still faults through the memory map and the 6 GiB
  pinned tier takes the page cache it faults through. Server commit rose to 26.3 GiB.
- **Gemma in the 8 GB and 25 GB setting, M1.1 build, one round each:** the ring took decode from 8.1 to **13.4
  tok/s** with identical answers. E39 predicted the ring would not help here (7-9 tok/s): wrong, because memory-mapped
  misses under desktop pressure were not running at the ring's rate after all. But plain `-ncmoe 24` (15.0 on the
  M1.0 build, same conditions) is still ahead, and 4K prefill fell from 122.6 to 48.1 tok/s.

The decision (E40): the M1 exit's decode floor of 12 tok/s is **met warm by the tier** (two rounds, indicative) and
**not met after the 16K prompt** (10.3-10.9, though tripled from 3.3-3.5). Ring and tier ship as opt-in, exact miss
sources. The tier is not a default while it costs prefill 3x. Under heavy desktop RAM pressure, plain offload still
wins the gate case.

## 7. What is next

From `DESIGN-M1.md` section 10, reshaped by what M1.0 and M1.1 measured:

- **Finish M1.1:** the remaining controls and the Maya drive comparison (placeholders below). Then the prefill path:
  P3 covers decode only, and the tier's 3x prefill cost has to go before it can be a default; tier sizing by need
  (M1.3) and the M1.2 replay to raise the tier's share of misses come with it (E40).
- **M1.2, policy:** a ranked core, warm start, a TinyLFU doorkeeper, transient slots and budgets by need, with the
  bitwise invariance test E1 at several budgets and policies. Expected to be worth less than designed, because SLRU
  already reaches 84-91%.
- **M1.3, calibration v1:** probes with a safety envelope, a VRAM cliff finder, the plan, a live confirmation, a memory
  governor with supervised restart, one-sequence serving with clear refusals, learned counts, and `marian serve
  --stock` as the rollback. This is where "self-calibrating" stops being a goal and becomes code.
- **M1.4:** real hardware: a real 8 GB GPU, Linux, a slower link and drive.
- **M2 to M5:** the full expert source including prefill, drain removal, look-ahead prefetch (M2); verified
  speculation (M3); opt-in answer-changing switches with blind quality gates and the bit-faithful CPU kernel spike
  (M4); prefill work, including bank lending and layer-major prefill (M5).

**The open question for Chris.** M1's exit asks for at least 12 tok/s on Swift on the 16 GB reference laptop, warm
and after a 16K prompt. The P3 tier meets the warm half (two rounds) and misses the post-prefill half. But the use case
that now drives the work is different: an 8 GB GPU beside a desktop that uses most of the RAM, with 4-30B models and
prefill as important as decode. On that laptop stock llama.cpp already meets 20-30 tok/s. **Do we keep the M1 exit as
written, or re-aim it at the laptop use case** (for example: beat the best stock placement on decode *and* prefill,
in the 8 GB and 25 GB setting, for one named model)?

## 8. Mistakes and rejected options, in one place

| When | What | What we did |
|---|---|---|
| June 2026 | a tuned 57 tok/s result did not survive normal desktop use | re-ran under load; RAM headroom became a first-class measurement |
| 21/10 | M0's first real profile crashed on a shadowed `Thread._stop` | fixed with a test within the hour |
| 22/10 | prediction "Strata will be disk-bound" (E35) | wrong for decode; recorded |
| 22/10 | a judging check that could not fail (E37) | caught, answers checked by hand, rule written |
| 22/10 | rejected: port Strata's engine; pin the whole host complement | built on upstream's cache; bounded, measured pinning |
| 22/10 | lever ranking put drain removal first | measured 0.8 ms; demoted |
| 22/10 | Fable's 70-80% SLRU estimate | measured 84-91%; the policy work shrank |
| 22/10 | 4,000 MiB arm changed 3/10 answers | exactness floor, now refusing too-small budgets |
| 22/10 | a runner script corrupted by a string escape | repaired, parse-checked |
| 24/10 | the probe's baseline was harsher than the server | GO kept, caveat recorded, honest ratios given |
| 24/10 | three arms silently skipped on empty server arguments | re-run; a loud failure is needed |
| 24/10 | E39's predictions for the ring and tier, and "the ring will not help Gemma" | all too low on decode, and the gate one wrong; the prefill cost far worse than "equal or slightly worse" (E40) |

---

## Results appendix

### 21/10/2026

#### A1. E34: GLM-5.3-Flash through Project Maya

Conditions: Maya-S24 pack (321B MoE, 94.7 GB, 3 shards), reference laptop, Maya's own calibration (PCIe shares
0.00-0.75, 4-24 CPU threads) and `--bench`. Single measurements per setting. Source: register E34.

| Measure | Result |
|---|---|
| best decode (engine's split, PCIe share 0.33, all threads) | 2.1 tok/s |
| other splits / thread counts | 1.7-2.0 / 1.3-1.7 tok/s |
| Maya `--bench` decode (3 x 256 tokens, greedy) | 1.7 / 1.8 / 1.4 tok/s, mean 1.7 |
| prefill | 39 tok/s at 2,048 tokens; 295 tok/s at 8,192 |
| share of each decode token waiting on the SSD | 67-75% (305-531 ms of 454-708 ms) |
| VRAM expert hit rate | 42-54% |
| SSD reads | median 2.0 GB/s, peak 2.7 GB/s, sustained |
| VRAM / RAM peak / commit peak | 14,790 MiB flat / 25.8 GB / 25.9 GB |
| calibration time | 80+ minutes against an estimate of 10-15 |

Taught us: it runs, but SSD-bound; batching helps prefill a lot (39 to 295 tok/s); RAM capacity is the lever.

#### A2. M0's first profile: Gemma 4 26B-A4B on an 8 GB laptop GPU

Conditions: a laptop with 8 GB of VRAM and 32 GB of RAM, stock llama.cpp (build 11760), Gemma 4 26B-A4B-it
UD-IQ4_XS, 8 threads, each point fills 75% of the context and generates 64 tokens. Best "ok" point and the first
overflow point per context and KV type. Source: `profiles/examples/gemma4-26b-a4b-iq4xs-8gb-blackwell-laptop.json` (anonymised).

| Context | KV | Best ok: `-ncmoe`, decode, prefill tok/s, VRAM peak MiB | First overflow: `-ncmoe`, decode, prefill tok/s |
|---|---|---|---|
| 4K | f16 | 18: 30.29, 813.2, 7,622 | 16: 8.32, 116.4 |
| 8K | f16 | 18: 29.21, 888.7, 7,830 | 16: 6.55, 106.7 |
| 16K | f16 | 20: 26.58, 843.9, 7,310 | 18: 27.37, 781.1 (VRAM near full) |
| 32K | f16 | 20: 21.06, 857.5, 7,646 | 18: 22.59, 253.2 |
| 4K | q8_0 | 18: 22.25, 803.4, 7,446 | 16: 22.71, 278.8 |
| 8K | q8_0 | 18: 22.16, 870.2, 7,616 | 16: 23.49, 283.8 |
| 16K | q8_0 | 18: 19.88, 881.8, 7,708 | 16: 12.02, 102.0 |
| 32K | q8_0 | 20: 19.03, 851.8, 7,206 | 18: 19.29, 893.4 (VRAM near full) |

At the most conservative split (`-ncmoe 30`) decode was 15.5-22.9 tok/s. Taught us: f16 KV beats q8_0 here; 4K to
32K costs about 30% of decode; the overflow cliff can show in decode (8K: 29 to 6.6 tok/s) or in prefill first (32K:
858 to 253 tok/s), so both must be measured.

### 22/10/2026

#### A3. E35: Strata and Swift

Conditions: Strata 0.1.47 (v0.2.2), defaults, no tuning; Swift 1.5 IQ2_XS pinned and hashed; reference laptop.
Single runs per condition. Source: register E35.

| Measure | Result |
|---|---|
| decode, median (range, n = 9) | **35.5 tok/s** (16.5-53.7); run 3 at 46-54 as the page cache warmed |
| prefill | 155 tok/s at 4K, 193 at 16K |
| first token | 26 s at 4K, 83 s at 16K |
| agent turns | 22.7 tok/s decode, 24.8 s to first token |
| decode VRAM expert hit rate | 85.4% (6,661 missed experts skipped by route-tail skip, an answer-changing default) |
| VRAM / RAM used / available median / commit peak | 14.96 GB / 31.6 GB / 0.24 GB / 35.5 GB |

Taught us: decode can be fast on 32 GB with good per-expert residency; prefill is where RAM bites. My "disk-bound"
prediction was wrong.

#### A4. E36: Strata's tune and standard protocol

Conditions: `strata tune auto --mode quick`, then Strata's standard suite (thinking on), then E35's bench repeated.
Source: register E36.

| Measure | Result |
|---|---|
| tune | "already fastest" (0 setting pairs tried; a control repeat varied by up to 42%) |
| standard suite quality | 12/12 (instruction, retrieval, tool call 4/4 each) |
| decode at short / 4K / 16K | 29.2 / 22.6 / 21.2 tok/s |
| first answer token at short / 4K / 16K | 6.5 / 39.0 / 96.7 s |
| repeat bench against E35 | decode 33.3 against 35.5; prefill +16-18% |

Taught us: the noise floor on this laptop is about plus or minus 10% decode and 15-18% prefill; comparisons need more
than that, or interleaved repeats.

#### A5. E37: blind quality, three large local models

Conditions: pre-registered and hash-frozen; 10 prompts, thinking off, temperature 0.3; three blind judges plus Chris.
Source: register E37.

| Model | Panel mean rank | Chris's mean rank | Objective checks |
|---|---|---|---|
| GLM-S24 | 1.4 (6 firsts; p = 0.045, W = 0.96) | 2.3 | 5/5 |
| Qwen3.8-27B dense | 2.1 | 1.9 | 5/5 |
| Flash-Next (Swift) | 2.5 | 1.8 | 4/5 |

Without one prompt where every judge called the answers identical, GLM had 5 firsts and p = 0.097. Taught us: a
near-tie, so speed and footprint decide; a unanimous ranking can still be arbitrary; one check could not fail and was
caught.

#### A6. M1.0 Swift campaign: upstream cache against stock

Conditions: reference laptop, M1.0 fork at `a37cc461` + P0; context 20,480, `-fa on -t 16 --parallel 1`, greedy,
no prompt cache; per arm a 64-token warm-up, 10 decode prompts x 256 tokens, a 5,313-token and a 19,361-token
prompt, then 3 prompts x 128 tokens after the long prompt. Another process held about 8 GB of RAM, so free RAM before
each arm was about 16 GB. Source: `docs/m1.0/swift-summary.md`, `docs/m1.0/M1.0-STATUS.md`.

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Hit median / p10 | MiB uploaded per token | Host ms per token: sync wait / after | VRAM peak MiB | Free RAM min GiB | Peak working set / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| r1-stock34 | 10.6 | 5.1 | 111.6 | 152.0 | - | - | - | 15,200 | 0.16 | 22.28 / 20.83 |
| r2-stock34 | 10.9 | 4.7 | 113.2 | 147.9 | - | - | - | 15,200 | 0.18 | 22.26 / 20.83 |
| r1-c4000 | 7.1 | 3.2 | 34.6 | 50.2 | 74.8% / 56.7% | 175.6 | 0.76 / 83.42 | 9,058 | 0.17 | 23.03 / 14.85 |
| r2-c4000 | 6.9 | 3.3 | 33.6 | 51.5 | 74.8% / 56.7% | 175.6 | 0.85 / 87.71 | 9,058 | 0.16 | 22.83 / 14.85 |
| r1-c6500 | 6.9 | 3.7 | 45.1 | 80.4 | 84.4% / 69.8% | 112.7 | 0.82 / 78.31 | 11,562 | 0.18 | 22.54 / 17.29 |
| r2-c6500 | 7.1 | 3.6 | 40.9 | 86.1 | 84.4% / 69.8% | 112.7 | 0.83 / 80.07 | 11,562 | 0.15 | 22.45 / 17.29 |
| r1-c9400 (*) | 7.9 | 4.0 | 54.1 | 107.2 | 90.8% / 80.4% | 69.3 | 0.79 / 72.98 | 14,462 | 0.06 | 22.5 / 20.13 |
| r2-c9400 | 7.3 | 3.5 | 51.6 | 107.9 | 90.8% / 80.4% | 69.3 | 0.79 / 70.73 | 14,462 | 0.11 | 22.43 / 20.13 |

(*) overlapped the end of a 13.6 GB download and its hash read; r2 is the clean run. The 22/10 partial floor arm
(5.6 tok/s) was superseded by the full floor arms in [A9](#a9-swift-at-8-gb-the-floor-arms).

Taught us: SLRU already hits 84-91%; decode is still about 30% slower than stock; the host time after the sync
(71-88 ms per token) is the cost; every arm collapses after a 16K prompt; rounds agree within about 5% on decode.

#### A7. Copy probe

Conditions: 400 random 1,510,400-byte expert blocks from Swift's second shard (26.4 GiB), host to device, reference
laptop. Source: `docs/m1.0/copy-probe.txt`.

| Source | Median per expert | p10 / p90 us | Rate (median) |
|---|---|---|---|
| mmap, first touch (page faults to the SSD) | 3,650.7 us | 2,446.1 / 5,195.6 | 0.41 GB/s |
| mmap again (resident) | 228.3 us | 222.7 / 248.6 | 6.62 GB/s |
| malloc, pageable | 225.2 us | 221.8 / 233.8 | 6.71 GB/s |
| `cudaMallocHost`, pinned | 147.2 us | 144.8 / 166.9 | 10.26 GB/s |
| submit only, pinned / pageable | 5.1 / 122.5 us | | (pageable blocks the host) |
| 64 B device-to-host + sync (the ids read) | 7.4 us | 6.9 / 23.9 | |

Taught us: pinned is 1.55x resident pageable, and resident pageable is 16x a faulting page. Where misses come from is
the lever.

#### A8. Answer identity in M1.0

Conditions: greedy text and top-3 probabilities on the 10 decode prompts, pairs of A6 arms. Source:
`docs/m1.0/swift-summary.md`.

| Pair | Identical |
|---|---|
| any arm with itself across rounds | 10/10 (deterministic) |
| cache 6,500 against 9,400 MiB | 10/10 |
| cache 4,000 against 6,500 or 9,400 MiB | **7/10** |
| stock against any cache arm | 2/10 (expected: CPU and GPU round differently) |

Taught us: the cache is placement-independent where every step stays on the GPU, and not otherwise (the IQ1_M
capacity bug, section 5).

### 24/10/2026

#### A9. Swift at 8 GB: the floor arms

Conditions: as A6, plus a VRAM ballast leaving about 8 GB (on 22/10 the ballast left 6,890 MiB free because the
display holds about 1.3 GB); M1.0 build, so before the exactness floor. The RAM setting on 24/10 is not recorded in the
summary. Source: `m1.0/swift-summary.md` (24/10 version, research workspace).

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB (incl. ballast) | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| f1-c2400 | 4.9 | 3.2 | 28.1 | 38.3 | 63.7% / 45.0% | 242.8 | 1.42 / 131.64 | 15,488 | 0.03 | 23.07 / 13.25 |
| f2-c2400 | 5.9 | 3.6 | 33.9 | 42.4 | 63.7% / 45.0% | 242.8 | 0.87 / 103.85 | 15,766 | 0.11 | 23.24 / 13.26 |
| f1-stock45 | 7.7 | 5.6 | 27.0 | 30.5 | - | - | - | 15,623 | 0.08 | 22.66 / 13.01 |
| f2-stock45 | 7.5 | 5.8 | 27.9 | 27.7 | - | - | - | 15,881 | 0.10 | 22.67 / 13.05 |

Answer identity: each arm 10/10 with its other round; cache against stock 4/10. Against the A6 arms, compared over
the 128 tokens both generated (the A6 arms generated 256): cache 2,400 against 4,000 MiB 5/10 and against 6,500 or
9,400 MiB 2/10; stock 45 against stock 34 6/10. (A first version of this table said 1/10 for all of these: the
analysis compared whole texts of different length. Fixed 24/10; see register E41.)
Taught us: Swift is not an 8 GB model on either path; the 2,400 MiB arms predate the floor, so their differences from
the 16 GB cache arms are the same capacity bug (prompt steps on the CPU), not a new one.

#### A10. Controls: Gemma 4 26B-A4B at 8 GB, RAM free

Conditions: Gemma 4 26B-A4B-it UD-IQ4_XS, M1.0 build, same request set (16K prefill not measured); "floor" arms with
the 8 GB VRAM ballast and no RAM ballast (free RAM minimum about 8 GiB); "full" on the whole 16 GB GPU. Source:
`m1.0/controls-summary.md`.

| Arm | Decode | After 16K prompt | Prefill 4K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|
| gemma-floor-c2500 | **42.6** | 40.5 | 756.9 | 87.5% / 72.9% | 93.8 | 0.34 / 8.52 | 15,086 | 8.05 | 14.69 / 8.61 |
| gemma-floor-ncmoe24 | 16.8 | 19.8 | 700.8 | - | - | - | 14,802 | 8.06 | 14.73 / 8.33 |
| gemma-full-ncmoe0 | 79.4 | 79.7 | 3,645.7 | - | - | - | 14,439 | 7.77 | 15.09 / 16.44 |

Answer identity: the cache arm against the all-on-GPU arm, over the 128 tokens both generated (the full arm
generated 256): **1,280 of 1,280 tokens identical, and every top-3 id and probability identical**, on all 10 prompts.
The two M1.1 gate arms (memory-mapped and ring misses, ctx 20K, under 25 GiB of desktop pressure) match it the same
way. Plain `-ncmoe 24` differs from both within 0-73 tokens (CPU rounding, as design section 3.2 predicts).
Taught us: with RAM to hold the experts, the cache is 2.5x stock at 8 GB and only 8.5 ms of host time per token, and
it is **exact against the true reference** (design E2) at token and top-3 level; bitwise logits are not logged, so
bitwise is not claimed. A first version of this entry said "0/10, unexamined": the analysis compared whole texts of
different length, a check that could not pass. Found because the result contradicted the design, then fixed (E41).

#### A11. Gate arms: 8 GB GPU and about 25 GB of desktop RAM

Conditions: 8 GB VRAM ballast; about 25 GB of RAM held to simulate desktop work for the arms tagged `desk25` (register
E39 confirms this for the Gemma cache arm). The summary does not record the RAM setting for the other arms; their low
free-RAM minimum is consistent with the same setting (not verified). M1.0 build; same request set as A6. Source:
`m1.0/gate-summary.md`.

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate-e4b-qat-full (Gemma 4 E4B-it QAT Q4_0) | **89.7** | 86.0 | 6,108.7 | 5,643.6 | - | - | - | 12,377 | 3.30 | 4.01 / 5.02 |
| gate-bolt-e4b-full (Nemotron-3.5-Bolt-E4B Q4_K_M) | **79.7** | 68.3 | 5,488.8 | 5,212.8 | - | - | - | 12,703 | 2.99 | 4.38 / 5.39 |
| gate-qwen35-9b-full (Qwen3.5-9B UD-Q4_K_XL) | **65.4** | 65.6 | 3,838.2 | 3,829.7 | - | - | - | 15,129 | 0.38 | 6.4 / 9.58 |
| gate-gemma26-iq3xxs-ncmoe18 | **26.3** | 27.0 | 775.4 | 756.1 | - | - | - | 15,833 | 0.13 | 6.5 / 10.25 |
| gate-gemma26-iq3xxs-c2500 | 12.7 | 11.3 | 146.3 | 350.2 | 91.7% / 78.8% | 58.2 | 0.58 / 32.14 | 14,651 | 0.12 | 6.42 / 9.1 |
| gate-gemma26-ncmoe24-desk25 (IQ4_XS) | 15.0 | 15.7 | 129.1 | 322.5 | - | - | - | 14,886 | 0.06 | 8.55 / 9.38 |
| gate-gemma26-ncmoe30-desk25 (IQ4_XS) | 10.0 | 10.1 | 21.5 | 20.3 | - | - | - | 12,791 | 0.07 | 7.05 / 7.2 |
| gate-gemma26-c2500-desk25 (IQ4_XS) | 8.0 | 7.4 | 97.8 | 169.3 | 87.5% / 72.9% | 93.8 | 0.65 / 65.87 | 15,154 | 0.25 | 6.38 / 9.64 |
| gate-groot-ncmoe28 (Groot-19B-A3B Q4_K_M) | 19.2 | 20.6 | 267.3 | 447.9 | - | - | - | 13,985 | 0.20 | 6.59 / 8.6 |
| gate-groot-c2500 | 5.0 | 6.1 | 149.6 | 246.8 | 75.6% / 60.6% | 150.9 | 1.27 / 124.98 | 13,633 | 0.17 | 7.06 / 8.25 |

Answer identity: 0/10 between different models, as expected; Gemma IQ4_XS `-ncmoe 24` against `-ncmoe 30` 4/10 and
Groot cache against stock 2/10 (stock placements compute different layers on the CPU, so stock answers depend on
placement). Taught us: the 20-30 tok/s question is met by stock placements and by small dense models; every cache arm
lost when misses came from the SSD (the Gemma cache arm pulled misses at about 1.4 GB/s, 93.8 MiB in 65.9 ms). A
surprise not yet investigated: `-ncmoe 30` collapsed prefill to about 21 tok/s.

#### A12. The gate candidates (survey, not a measurement)

Conditions: research only, from Hugging Face model cards and API listings fetched on 24/10 and the fork's
architecture table; nothing downloaded over 50 MB, nothing benchmarked; fit arithmetic includes an assumed 0.5 GB of
compute buffers. Source: `m1.0/gate-candidates.md`, section 1.

| Rank | Model | Class | Quant | File size, bytes | Estimated fit at 16K |
|---|---|---|---|---|---|
| 1 | Gemma 4 26B-A4B-it | MoE 25.2B / 3.8B active, 128 experts, top-8 | unsloth UD-IQ4_XS (fallback UD-IQ3_XXS, 11,416,548,832) | 13,597,177,568 | about 14.8 of about 15 GB: at the limit |
| 2 | Qwen3.5-9B | dense 8.95B | unsloth UD-Q4_K_XL | 5,966,095,584 | about 7.0 GB |
| 3 | Gemma 4 E4B-it, with NVIDIA-Nemotron-3.5-Bolt-E4B-4GB | dense 4.5B effective | Google QAT Q4_0 / Q4_K_M 6,180,446,208 | 5,154,941,280 | about 5.9 / 7.0 GB |
| 4 | Groot-19B-A3B-261023 (lead, unverified) | MoE about 19B / 3B active, 128 + 1 shared, top-8 | Q4_K_M | 11,657,618,560 | about 12.5 GB |

Taught us: the survey's ranking was by specs and vendor numbers; the measurements in A11 then put the dense models
and a stock MoE placement ahead of the cache.

#### A13. E1: the exactness floor

Conditions: M1.1 build (floor `aa381ca4`), Swift, `-ncmoe 48`, warm-up plus the 10 decode prompts only; reference
laptop, no ballast. Sources: `m1.1/e1-summary.md`, `m1.1/e1-cross-summary.md`, register E38.

| Arm | Decode | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|
| e1-c4000 | 7.1 | 74.4% / 56.9% | 178.7 | 0.87 / 81.40 | 9,460 | 0.23 | 22.8 / 13.06 |
| e1-c9400 | 7.5 | 90.8% / 80.4% | 69.2 | 0.91 / 69.30 | 14,860 | 0.25 | 22.58 / 18.34 |

| Pair | Decode texts identical | Top-3 probabilities identical |
|---|---|---|
| e1-c4000 against e1-c9400 | **10/10** (M1.0: 7/10) | 10/10 |
| e1-c9400 (M1.1 build) against r2-c9400 (M1.0 build) | 10/10 | 10/10 |

Also: the floor warning appeared in the 4,000 MiB log only; 23 multi-token steps in each of layers 0-46 in both arms
(layer 47 has none because only the output row is computed there). Taught us: the floor makes the budget
answer-invariant, and the M1.1 binary moved no answer, so M1.0's arms stay valid references.

#### A14. The trace-replay probe

Conditions: `replay_probe` replays a trace's decode misses through three sources: `mmap` (pageable copies from the
mapped file), `ring` (64 MiB pinned ring of unbuffered reads, queue depth 8) and `tier` (a 6 GiB pinned tier in front
of the ring). "Pressure" holds memory until about 2 GiB is available. Synthetic ids use a Zipf 1.2 model; the real-id
run uses E1's fill trace, ranks the tier on the first half and measures the second. Sources: `m1.1/probe/*.txt`,
register E39.

Run 1, M1.0's `r2-c9400` trace, synthetic ids, under pressure (2,396 decode tokens, 61.6 MiB per token):

| Source | ms per token median | p10 / p90 | GB/s | Tier share of fills / bytes | Read GiB | Prompt steps: GiB, s, GB/s |
|---|---|---|---|---|---|---|
| mmap | 90.09 | 30.97 / 194.23 | 0.72 | - | 0.00 | 33.79, 55.4, 0.65 |
| ring | 45.87 | 15.69 / 96.14 | 1.41 | - | 199.99 | 33.79, 23.7, 1.53 |
| tier | 28.52 | 9.55 / 61.04 | 2.26 | 43% / 43% | 113.68 | 33.79, 15.0, 2.41 |

Run 2, the same trace, synthetic ids, no pressure (15.97 GiB available at the start of the mmap pass):

| Source | ms per token median | p10 / p90 | GB/s | Tier share of fills / bytes | Read GiB | Prompt steps: GiB, s, GB/s |
|---|---|---|---|---|---|---|
| mmap | 83.15 | 25.85 / 185.06 | 0.78 | - | 0.00 | 33.79, 50.1, 0.72 |
| ring | 46.26 | 15.81 / 95.31 | 1.40 | - | 199.99 | 33.79, 23.4, 1.55 |
| tier | 28.13 | 9.63 / 60.19 | 2.30 | 43% / 43% | 113.68 | 33.79, 14.7, 2.46 |

Run 3, E1's `e1-c9400` trace with real ids, under pressure (1,074 measured decode tokens, 63.0 MiB per token; 1,488
trace lines still used synthetic ids):

| Source | ms per token median | p10 / p90 | GB/s | Tier share of fills / bytes | Read GiB | Prompt steps: GiB, s, GB/s |
|---|---|---|---|---|---|---|
| mmap | 132.05 | 46.63 / 286.15 | 0.50 | - | 0.00 | 12.46, 34.9, 0.38 |
| ring | 46.96 | 18.03 / 94.78 | 1.41 | - | 88.61 | 12.46, 8.7, 1.54 |
| tier | 35.94 | 13.99 / 71.67 | 1.84 | 26% / 26% | 65.26 | 12.46, 6.6, 2.03 |

The probe's linear what-if (ms per token against the tier's share of miss bytes):

| Run | 0% | 25% | 50% | 75% | 90% | 100% |
|---|---|---|---|---|---|---|
| 1, pressure | 45.5 | 35.8 | 26.2 | 16.5 | 10.7 | 6.9 |
| 2, no pressure | 45.5 | 35.6 | 25.7 | 15.8 | 9.8 | 5.8 |
| 3, real ids | 46.7 | 36.0 | 25.4 | 14.8 | 8.4 | 4.1 |

Tier fill rate 1.46-1.64 GB/s including pinning; ring reads at about 1.42 GB/s (the SSD at queue depth 8); 0 failed
reads. Taught us: GO on the tier; the ring is borderline; the probe's mmap pass was 23% (synthetic) to 1.9x (real ids)
slower than the server's 69-73 ms, so the honest gains are about 0.65x (ring) and 0.40-0.51x (tier). The tier's hit
share with real ids (26%) was lower than the synthetic model's (43%).

#### A15. M1.1 P3 CPU self-test (T0)

Conditions: `p3_selftest` drives the P3 source with the ggml CPU backend over both Swift shards. Sources:
`docs/m1.1/M1.1-NOTES.md` section 4, `m1.1/t0-selftest.txt`, register E38.

| Run | Expert tensors checked | Mismatched | Detail |
|---|---|---|---|
| development self-test, three configurations | 42,069 | 0 | 1,754 tier hits, 351 admissions, 11,918 ring reads, 2,242 ring wraps |
| T0 before the GPU runs | 13,005 | 0 | 4,335 misses: 705 from the tier (16.3%), 96 admitted, 3,534 through the ring, 541 ring wraps; PASS |

Taught us: the source delivers the same bytes as the memory map; it cannot see a slot rewritten while its DMA is in
flight, which only the GPU identity test can.

#### A16. P3 verify run

Conditions: M1.1 build, Swift, 9,400 MiB cache, 6 GiB tier, `MARIAN_MOE_SOURCE_VERIFY=all` (every read compared with
the memory map before its upload; a mismatch aborts the server), warm-up and decode only, not timed. Pass condition
(`docs/m1.1/M1.1-NOTES.md` section 7): every request answered, no abort. Source: register E40.

| Check | Result |
|---|---|
| requests answered | 11/11 |
| load line | "unbuffered reads, verified against the memory map" |
| fallback to pageable memory | none (no PAGEABLE) |

Taught us: every byte the ring and tier delivered matched the memory map on the GPU path too; the verify run could
have failed (it aborts on any mismatch).

#### A17. P3 server arms (A/B/C/D, two rounds)

Conditions: reference laptop, no ballast; Swift, 9,400 MiB cache with `-ncmoe 48`; A = upstream mmap, B = ring,
C = 6 GiB pinned tier (all on the M1.1 build), D = stock `-ncmoe 34` (M1.0 build); full phases as A6; order
A B C D D C B A. Sources: `m1.1/p3-summary.md`, register E40.

**Pre-registered prediction (register E39):** decode A 7.0-7.8, B 8.3-9.5, C 9.5-11.5 tok/s, against stock's
10.6-10.9; prefill (4K and 16K) for C equal to or worse than A; B and C against A 10/10 in both rounds.

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|
| p3-r1-c9400 (A) | 7.1 | 3.5 | 53.3 | 119.3 | 0.94 / 75.37 | 14,991 | 0.23 | 22.07 / 20.14 |
| p3-r2-c9400 (A) | 6.1 | 3.3 | 47.2 | 112.0 | 1.06 / 83.91 | 14,990 | 0.21 | 21.57 / 20.13 |
| p3-r1-c9400-ring (B) | 11.7 | 7.6 | 45.0 | 105.0 | 0.77 / 48.01 | 15,021 | 0.14 | 21.78 / 20.26 |
| p3-r2-c9400-ring (B) | 11.5 | 8.4 | 41.0 | 109.9 | 0.76 / 47.91 | 15,020 | 0.13 | 21.75 / 20.25 |
| p3-r1-c9400-tier6g (C) | **14.2** | 10.9 | 25.5 | 34.0 | 0.77 / 39.11 | 15,021 | 0.11 | 21.48 / 26.27 |
| p3-r2-c9400-tier6g (C) | **13.8** | 10.3 | 23.0 | 35.8 | 0.76 / 39.00 | 15,020 | 0.10 | 20.73 / 26.27 |
| p3-r1-stock34 (D) | 8.2 | 4.1 | 104.1 | 145.2 | - | 15,863 | 0.16 | 21.81 / 20.84 |
| p3-r2-stock34 (D) | 10.4 | 3.7 | 96.1 | 136.4 | - | 15,728 | 0.11 | 21.77 / 20.83 |

Hit rate 90.8% / 80.4% and 69.3 MiB uploaded per token in every cache arm (the miss source changes where misses come
from, not how many there are).

| Pair | Decode texts identical | Top-3 probabilities identical |
|---|---|---|
| every pair of A, B and C arms, within and across rounds | 10/10 | 10/10 |
| D with itself across rounds | 10/10 | 10/10 |
| D against any cache arm | 2/10 | 2/10 |

Scored against the prediction (E40): A in range for r1 and below it for r2; **B and C beat their ranges** (about
11.6 against 8.3-9.5; about 14.0 against 9.5-11.5); C's prefill about 3x slower at 16K, far worse than "equal or
slightly worse". Host time fell about 40 ms per token from A to C, token time about 70 ms: unexplained. D is noisy
(8.2 against 10.4, beyond the 5% seen on 22/10), so "C beats stock" holds against both rounds but by how much is
uncertain. Not stated in E40: the fill-trace counts of tier hits, admissions and ring reads per arm.

#### A18. Gemma gate: mmap vs ring on the M1.1 build

Conditions: Gemma 4 26B-A4B-it UD-IQ4_XS, `-ncmoe 30 --moe-cache-mib 2500`, 8 GB VRAM ballast plus about 25 GiB of
desktop RAM held, M1.1 build, one round each; same request set as A6. Sources: `m1.1/gate-summary.md`, register E40.

**Pre-registered prediction (register E39):** the ring gives no gain in the gate scenario, because the M1.0 cache arm
already pulled misses at about 1.4 GB/s, the ring's own rate: 7-9 tok/s against 8.0.

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate11-gemma26-c2500-mmap | 8.1 | 12.0 | 122.6 | 199.9 | 83.3% / 66.2% | 117.3 | 0.61 / 59.92 | 15,274 | 0.13 | 6.8 / 9.64 |
| gate11-gemma26-c2500-ring | **13.4** | 12.9 | 48.1 | 218.7 | 83.3% / 66.2% | 117.3 | 0.47 / 42.99 | 15,264 | 0.21 | 7.17 / 9.72 |

Answer identity: 10/10 texts and top-3. Taught us: the prediction was wrong; memory-mapped misses under desktop
pressure were not running at the ring's rate after all. But plain `-ncmoe 24` (15.0 on the M1.0 build, same
conditions) still wins decode, and the ring more than halved 4K prefill. The hit rate is 83.3% against 87.5% on the
M1.0 build at the same budget; a lead, not checked: the exactness floor changes how the budget splits between groups.

### Controls and the Maya drive test (24/10, afternoon)

These two sections were placeholders while the runs were going; both are filled in from the finished runs.

#### Controls: Coder IQ1_M, Qwen3.6-35B-A3B, Qwen3.8-27B dense

Conditions: M1.0 build; "floor" = 8 GB VRAM ballast, no RAM ballast; "full" = the whole 16 GB GPU. Coder (the
Qwen3.8-Flash-Next GSQ-RCO IQ1_M download) ctx 20K with all phases; Qwen3.6-35B-A3B UD-Q4_K_XL and Qwen3.8-27B dense
UD-Q3_K_XL ctx 16K without the 16K prompt. Floor arms 128 decode tokens, full arms 256. No pre-registered prediction.
Source: `m1.0/controls-summary.md`; register E41.

| Arm | Decode | After 16K prompt | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | Free RAM min GiB |
|---|---|---|---|---|---|---|---|---|
| coder-full-c9400 | 13.0 | 6.4 | 97.8 | 181.2 | 91.0% / 80.0% | 96.6 | 0.87 / 42.11 | 0.22 |
| coder-floor-c2400 | **7.1** | **7.5** | 52.8 | 122.5 | 57.9% / 41.0% | 402.8 | 0.92 / 95.71 | 0.15 |
| coder-floor-ncmoe45 | 6.5 | 2.8 | 88.7 | 103.7 | - | - | - | 0.21 |
| q36-full-c9000 | 43.3 | 37.3 | 367.3 | - | 95.9% / 88.4% | 31.6 | 0.58 / 4.75 | 0.30 |
| q36-floor-c2500 | **15.5** | 18.2 | 181.6 | - | 70.6% / 51.2% | 181.4 | 0.81 / 38.07 | 0.23 |
| q36-floor-ncmoe32 | 13.1 | **23.8** | **270.2** | - | - | - | - | 0.24 |
| dense27-full | 29.5 | 29.8 | 1,234.2 | - | - | - | - | 4.77 |
| dense27-floor-ngl40 | 2.2 | 2.2 | 415.4 | - | - | - | - | 2.18 |

Answer identity (shared prefix): cache against stock at the floor 3/10 for both Coder and Qwen3.6 (stock computes
CPU-layer experts on the CPU, so it is not a reference; these cache arms are the M1.0 build, before the floor).
Taught us: **the cache's value at 8 GB depends on the model.** Coder: cache ahead on decode and far ahead after a long
prompt (7.5 against 2.8). Qwen3.6: cache 18% ahead on decode, but plain offload wins after the long prompt and on
prefill. Dense 27B: not an 8 GB model at all (2.2 tok/s), which is the case mixture-of-experts models exist to avoid.
So the right setting is a per-machine, per-model measurement, which is M1.3's calibration job.

#### Maya C: vs E:

Conditions: the same Maya `--bench` (3 answers x 256 tokens greedy, prefill at 2K and 8K) with the 89 GB pack on the
model drive (E:) and on a copy on the system drive (C:, copied at 581 MB/s), in the order E, C, C, E; the copy deleted
afterwards. No pre-registered prediction. Register E42.

| Run | Decode tok/s | Prefill 2K tok/s | Prefill 8K tok/s | Maya's decode disk wait, ms per token |
|---|---|---|---|---|
| E1 | 1.5 | 31 | 161 | 522-526 |
| C1 | 1.1 | 66 | 248 | 377-765 |
| C2 | 1.2 | 67 | 248 | 364-732 |
| E2 | 1.0 | 30 | 124 | 784-1,028 |

Taught us: the drive is **not** the decode lever for Maya on this laptop. The two runs on the same drive differ by 50%,
more than the drives differ from each other, and decode is still dominated by small random reads on both. It **is** a
prefill lever: the system drive reads prompts 1.5-2.2x faster. The decode lever remains RAM (E34). Two runs per drive
cannot resolve a decode difference under about 50%, so "no difference" here means "none this test could see".
