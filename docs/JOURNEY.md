# The Marian journey

How this repository came to be, in order, with the options we rejected and the mistakes we made. Written for Chris,
for the agents who work with us, and for a future contributor who wants to know why things are the way they are.

The story comes first (sections 1 to 8). Every measured table so far is in the [Results appendix](#results-appendix)
at the end, grouped by date, each with one line on its conditions and one on what it taught us. Results still
running have marked placeholders there.

**Conventions.** tok/s figures are medians over an arm's requests unless stated, leaving out requests that stop
within 8 tokens (one decode prompt ends at once in every arm; section 6b, point 7); where two figures are given, the
second is the all-tokens rate (total tokens over total generation time). An "arm" is one configuration run
once with a fresh server; "r1 / r2" are two counterbalanced rounds. "Stock" means upstream llama.cpp with whole
expert layers offloaded to the CPU (`-ncmoe N` keeps the experts of N layers in system RAM). "Swift" is the Swift 1.5
IQ2_XS quantisation of Qwen3.8-Flash-Next (48 layers, 512 experts, 10 used per token, 33.02 GiB of routed experts).
"The reference laptop" has an RTX 4090 Laptop GPU with 16 GB of VRAM, an i7-13700HX, 32 GB of DDR5-4800, Windows 11
and an NVMe model drive (`docs/design/RESEARCH-PACK.md`).

**Where the numbers come from.** Files in this repository are cited by path. The result tables are copied into the
repository as `docs/m1.0/*-summary.md`, `docs/m1.1/*-summary.md` (the gate pair on the M1.1 build is
`docs/m1.1/gate11-summary.md`), `docs/m1.2/results/` and `docs/m1.1/probe/*.txt`, so a citation such as `m1.1/p3-summary.md` means
`docs/m1.1/p3-summary.md`. Register entries (E34 to E54) are in the project's private test register; every number they
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
5. **Every arm slowed sharply after the 16K prompt** (4.2-5.4 tok/s, down by about a third to a half): prefill
   flushes the page cache that decode needs (corrected 24/10).
6. **The cache broke exactness at 4,000 MiB.** Arms at 6,500 and 9,400 MiB gave identical greedy text (10/10), as
   designed. At 4,000 MiB only 7/10 matched. The trace showed why: the three IQ1_M layers
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

- **Swift at 8 GB** ([A9](#a9-swift-at-8-gb-the-floor-arms)): 5.1-6.1 tok/s with a 2,400 MiB cache, 7.6-7.8 with
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
10/10. Speed was unchanged (7.2 and 7.8 tok/s, medians; corrected 24/10).

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

- **Decode** (median per request, rounds 1 / 2; corrected 24/10 as explained in section 6b, point 7): upstream's path
  7.3 / 6.3 tok/s, the ring 12.4 / 12.0, the 6 GiB tier **14.7 / 14.0**; after a 16K prompt 5.0 / 4.6, 10.7 / 11.7
  and **14.8 / 14.4**. Stock was 8.6 / 11.0 warm and 4.2 / 3.8 after the long prompt. (First published as 14.2 / 13.8
  warm and 10.9 / 10.3 after the long prompt, before a one-token request was excluded.) **All three cache arms give
  identical answers (10/10 texts) within and across rounds.** (Their probabilities were not actually recorded; section
  6b, point 11.)
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

The decision (E40, as first recorded): the decode floor of 12 tok/s **met warm by the tier** and **not met after the
16K prompt**. That second half was the analysis artefact of section 6b, point 7: corrected, the tier is above 12 in
both states on both rounds, which made a proper five-round confirmation the next step. Ring and tier ship as opt-in,
exact miss sources. The tier is not a default while it costs prefill 3x. Under heavy desktop RAM pressure, plain
offload still wins the gate case.

## 6b. Saturday afternoon: from M1.1 to the version we publish

Chris asked for this part to be told in detail, because it is where most of what we now believe was either found or
corrected. It happened in one afternoon, mostly while the GPU ran unattended, and it is a story of instruments
being checked as hard as the code.

**1. Getting ready to publish.** The README was rewritten for strangers (what Marian does, why, how the parts fit,
prior art, research, thanks, limitations), and every citation was checked against its primary source: the
arXiv ids, DOIs, the four upstream pull requests behind llama.cpp's MoE cache, and the repositories of Strata, Maya
and Edge0. One claim did not survive: we had called Strata's quality method "blind-KL"; Strata calls it
teacher-forced KL, and "blind" was our own gate's word. The licence became Apache-2.0, with a NOTICE that keeps the
MIT licence of the upstream code our patches touch. Then the tree was scrubbed for publication: local paths became
placeholders, machine names and the details of one laptop that is not ours to describe came out, and two tool
ledgers that listed internal credential names were deleted. Because old commits still held the original files, the
**history was rewritten** with `git filter-repo` and checked blob by blob: the final tree came out byte-identical to
the cleaned one (same tree hash), every file version in every commit was scanned, and the repository was deleted and
recreated so that GitHub stops serving the old commits by hash (verified: they now answer "No commit found"). A full
local backup of the pre-rewrite history was kept off GitHub.

**2. The first analysis artefact: "Gemma 0/10" was an instrument fault** (register E41). The controls said Gemma's
cache arm matched the all-on-GPU arm 0/10, which would have contradicted the design's central exactness claim. The
design predicted identity, so the contradiction was worth reading token by token: the full arm had generated 256
tokens and the cache arm 128, and the analysis compared whole texts. Over the shared prefix the cache arms (M1.0
build, and M1.1 with memory-mapped and ring misses, at a different context size and under desktop RAM pressure)
match the all-VRAM reference on **1,280 of 1,280 tokens**. (We also wrote "and every top-3 probability"; point 11
shows those had never been recorded, and then measures them.) This is the design's E2 test,
the first evidence against the true reference rather than across budgets. The analyser now compares a shared prefix
by token id and says so.

**3. The remaining controls and the Maya drive test** (E41, E42). The cache's value at 8 GB depends on the model
(decisive for Gemma and for the Coder after long prompts, mixed for Qwen3.6, irrelevant for the dense 27B), which is
exactly why calibration has to be per machine and per model. Maya's pack read prompts 1.5-2.2x faster from the
system drive, while decode varied more between runs on one drive than between drives.

**4. The routing-id trace, and a review that saved the GPU run** (E43). The cloud session (Linux, no GPU) had built
patch 0004 (every routed expert id of every MoE step) and `marian replay` (a simulator of upstream's cache, checked
step by step against the fork's own C++ on synthetic data). Before it ran on the GPU, an independent review read the
hooks against the fork and found a **blocker**: the S records took the token count and the expert count from a
tensor the graph had already flattened to one dimension, so every multi-token step was written with the two swapped.
Decode (one token) was unaffected, which is why the writer's unit test and the size estimate passed. The first
28-token step would have made the reader lose its place and silently drop three of the five test prompts. Patch 0005
fixes it, and the reader now refuses any record whose expert count is not the model's. The lesson is on the test,
not the code: the unit test exercised the writer, not the arguments the hook passes it.

**5. The replay matches the server exactly, and answers the M1.2 question** (E43, [A19](#a19-m12-the-routing-id-trace-and-the-replay)).
With the trace on, answers stayed 10/10 identical. The replay reproduced every one of 107,008 cache lookups and
58,110 uploads, and the server's hit rate to **+0.00 points**: M1.1's exit check ("replay predicts the server within
3 points") is met, exactly. Then the replay answered the questions the design had left open:
- **P2 (a smarter VRAM policy) is not worth it here.** Every candidate (a ranked core, a doorkeeper, a guard, their
  combination) was 2 to 11 points *below* upstream's SLRU at the same budget. The design had a falsifier for exactly
  this ("SLRU at 84% or more means ranking buys little"); it fired.
- **A bigger or smarter RAM tier is not the lever either.** On the real trace a 6 GiB tier serves only 18-27% of
  decode miss bytes, and no realistic policy reaches 75% even at 16 GiB, which a 32 GB machine cannot pin.

**6. Then why did P3 double the speed?** If the tier serves only a quarter of the misses, its gain must come from
somewhere else. A per-phase split of the P3 traces ([A20](#a20-where-the-post-prefill-time-went)) showed it: after a
16K prompt the VRAM hit rate barely moves (90.8% to 89.0%), but memory-mapped misses take **twice** the host time,
because reading the prompt evicts the model's pages from RAM. The ring and the tier read misses without the page
cache, so their cost hardly changes after a long prompt. P3's gain is mostly that: misses whose cost no longer
depends on what the page cache happens to hold.

**7. The second analysis artefact: the post-prefill figure measured one slow prompt** (E43). The per-phase split also
showed the tier arm running post-prefill tokens at a median of 70-74 ms, about 14 tok/s, while the summary said
10.3-10.9. One of the ten decode prompts (a train timetable puzzle) makes the model stop immediately, with one token
and an empty answer, in every arm; the summary counted it as 0 tok/s. With only three post-prefill prompts, a median
of [18.7, 10.9, 0] reports the slower real prompt. The analyser now leaves out requests that stop within 8 tokens
(and lists them), and reports a second figure that one prompt cannot swing: total tokens over total time. **Corrected,
the 6 GiB tier measured 14.0-14.7 tok/s warm and 14.4-14.8 after a 16K prompt (median; 13.2-14.2 all-tokens), on two
rounds: above the M1 exit's 12 on both.** Two rounds are not a claim under design 9.3, so a five-round interleaved
confirmation against stock was started the same afternoon ([A21](#a21-the-five-round-confirmation)). It passed
(point 10).

**8. An outside user found two gaps in M0** (the planner). Alder, wiring Marian into another project as an optional
planner, reported that a profile's VRAM peak includes whatever the desktop was using at the time, so a plan made on a
quiet desktop could overflow on a busy one, and that there was no machine-readable output. Profiles now record the
other programs' VRAM per row and the model's own share; `marian plan --fit-now` checks that share against the VRAM
free at launch; `plan --json` returns one object with the exact server arguments. A third gap turned up while checking
the install line: the repository had no `pyproject.toml`, so `pip install git+...` could not work, and an installed copy
would have written profiles into its own package directory. Both fixed, with tests (48 pass).

**9. Small traps, each cost minutes, each now written down:** a `git -C` worktree path that resolved inside the fork;
a shell substitution that collapsed Windows backslashes (redone with literal strings); and a PowerShell loop counter
`$r` that silently overwrote the research root `$R`, because PowerShell names are case-insensitive (the confirmation
started ten empty runs in one second before it was caught; nothing reached the GPU).

**10. The five-round confirmation: the decode exit is met** (E44, [A21](#a21-the-five-round-confirmation)). The rule
was written into the register before any result was read: the tier meets the exit in a state only if every one of
five rounds is at or above 12 tok/s, all-tokens rate. Interleaved against stock (T S | S T | T S | S T | T S), the
6 GiB tier ran **14.4-14.6 tok/s warm and 13.4-13.7 right after a 16K prompt**, all ten values above 12, with a
spread of 1-2% between its own rounds. Stock `-ncmoe 34` in the same rounds: 8.2-10.8 warm, 3.9-8.5 after the long
prompt. So the tier is faster than stock in 5 of 5 rounds, about 1.5x warm and 3.4x after a long prompt. Its cost is
just as consistent: stock reads a 16K prompt **4-5x faster** (141.8-155.0 tok/s against 29.7-34.4). The "3x" we had
been quoting was against upstream's memory-mapped cache, not against stock.

**11. The third analysis artefact: no probability had ever been recorded** (E44, E45,
[A22](#a22-exactness-with-real-probabilities)). Reading the confirmation's exactness table, tier against stock showed
2/10 identical, the same as every cache arm against stock since 22/10: stock computes 34 layers' experts on the CPU,
which rounds differently, so it is not the exactness reference. But the per-token data showed something else: every
"top-3" entry, in every run since 22/10, was `[[null], null]`. The request script read `top_probs` and `prob`; this
llama.cpp sends `top_logprobs` and `logprob` (log-probabilities before sampling) unless asked otherwise, and in
PowerShell `$null | ForEach-Object` runs once on null, so each empty answer was stored as a well-formed-looking
record. Every "top-3 probabilities identical" in this journal and the README had compared nulls with nulls, a check
that could not fail. Token ids were recorded correctly, so every token-identity result stands; the probability half
had never been measured. The client now reads both shapes and stops with an error if a response carries no top-k;
the analyser says "not measured" instead of "identical" and reports the largest log-probability difference. Then we
measured it, with the rule registered first (E45). **The cache is exact to full float precision:** Gemma 4 26B's
cache arms, with memory-mapped, ring and tier misses (hit rate 83%, so misses on every token), match the all-in-VRAM
reference on all 1,280 tokens and every top-3 log-probability, largest difference 0; the reference against its own
repeat, 0. Stock offload of the same model changes all 10 answers and moves log-probabilities by up to 0.59. That
contrast matters twice: it shows the instrument can now fail, and it shows that "exact" is something Marian's cache
has and stock layer offload does not.

## 6c. Monday 26 and Tuesday 27/10: the question answered, the rebase, and the prefill hunt

Chris answered the open question at the end of section 7 and then, on Tuesday morning, handed the day over: "you are
in charge, C. kindly make all decisions by yourself". Every decision below was made and logged that way, and each
test was written into the register before it ran.

**1. The open question, answered.** The M1 exit stays as written, so M1 is met (A21, A22). The laptop use case becomes
the next milestone, with its own pass rule written before any run (point 4).

**2. A tempting unlock, measured and parked (26/10, register E46, A23).** Chris asked whether the FP16/BF16 tensor-rate
unlock for GeForce cards (Zagitta/gpu-unlock) was worth installing for prefill, at least to know that we gain
nothing by it. It doubles the rate of FP16/BF16 matmuls that accumulate in FP32, and nothing else, and it needs
Linux or Secure Boot off. So we did not install it: a CUPTI trace recorded every kernel of a prefill, and we summed the
time the unlock could touch. The ceiling was under 10% of kernel time in 7 of 8 cases (11.4% at most), and about 1% of
wall time in the 8 GB gate case, which is CPU-bound. The quantised INT8 matmuls, which the unlock does not touch, take
60-77% of kernel time. Parked.

**3. Leads first, then a roadmap.** Two scans ran in the morning: upstream llama.cpp since our pin (107 commits), and
the wider ecosystem (other engines, and 2025-2026 papers). Each lead was weighed by its class (exact, verified
speculation, answer-changing) and by evidence (ours, or *theirs*), and the result became the living
[ROADMAP](ROADMAP.md). The scan also corrected our README: Swift's publisher (UkisAI) and the IST-DASLab authors were
missing from the thanks. And Swift's licence bounds commercial use, so it stays a research test model.

**4. The next milestone, written before any run (register E50).** Gemma 4 26B-A4B IQ4_XS on an 8 GB GPU, with about
25 GB of RAM taken by a desktop. Exact Marian, configured by its own calibration with no hand tuning, must match or
beat stock's best `-ncmoe` on median decode and median 4K prefill over five interleaved rounds. The milestone is
exact-only, by decision: stock offload changes answers (A22). Answer-changing switches are measured and reported, but
they cannot pass it. If Marian loses, the planner choosing stock on that machine is the right behaviour, and is
reported as such. Our own prediction for today's code is a FAIL, on prefill.

**5. The rebase (E48, A25).** 107 upstream commits re-applied with mechanical fixes. But three upstream commits change
Swift's numerics, so exactness was proved again from scratch rather than assumed. Gemma's numbers did not move: the
all-VRAM reference is bit-identical to the old base's, every cache arm still matches it with a difference of exactly
0, and the stock contrast differs by the same 0.59 as before. Swift's text did change (6 of 10 prompts diverge from
the old base within 128 tokens), as those commits predicted. The decode exit still holds, narrowly after a 16K prompt
(12.6 against 12.0, one round). The fork now sits on `63b4b303`.

**6. How predictable is the routing? (E47, A24).** Before anyone builds a router predictor or a prefetcher, we bounded
what prediction could buy, offline, on traces we already had (`marian predict`).
- **Persistence is real:** a token reuses 38% of the previous token's experts at a layer, against 20% by chance.
- **No cross-layer or prompt signal:** predicting the next layer's experts from this layer's is dead (19%, below plain
  persistence), a prompt's prefill does not predict its decode, and routing is close to uniform across prompts.
- **The headroom is real, but in the miss path:** perfect one-layer-ahead prediction, served from pinned RAM, would be
  worth +41% (20.5 against 14.5 tok/s). So overlapping copies (M2) comes before any predictor, and a predictor will be
  scored on misses only.

**7. Where the prefill time goes (E49, A26).**
- **The suspect:** Swift's per-layer embedding table, 26.8 GiB, read through the same memory map as the experts on a
  32 GB machine. Another project had seen that very table evict experts.
- **The probe:** we wrote a small residency probe (`tools/wsprobe`, no admin rights needed). It counts, page by page,
  how much of each tensor kind a process holds, and we tested it against a synthetic model with known answers before
  trusting it.
- **The suspect was innocent:** the table never held more than 0.48 GiB.
- **What the probe saw instead:** about 38,700 single-page disk reads per second during the tier's prefill. Expert
  pages were faulting in one by one under memory pressure: 4 KiB at a time, the slowest way to read an SSD.
- **Our prediction** (pollution, about 55%) was wrong. The fix the data points at was already being built (point 8).

**8. Prefill through P3 (E51, A27).** Two fork patches:
- **Chunked pinning (0007):** pin the tier in chunks under a RAM cap. Windows quietly falls back to slow pageable
  memory when one large pin fails.
- **Large-batch misses through P3 (0008):** serve them through the unbuffered ring and the pinned tier instead of
  memory-map faults.
- **The pass rule:** at least twice the 16K prefill speed, decode within 5%, and answers unchanged.
- **The result: a FAIL on the speed clause, and a useful one.** 16K prefill went from 34.7 to 61.1 tok/s, 1.76x, not
  2x. Decode did not suffer, and the answers did not change: Gemma's tier arm matched all-VRAM on every top-3
  log-probability, and every Swift cache run matched every other. We had predicted about 3x.
- **Why it fell short, measured:** the drive. A 16K prompt uploads 509 GiB of experts, because each 512-token ubatch
  re-reads most of each layer's experts. The tier held 13.8% of those bytes, and the rest came off the SSD at about
  1.95 GB/s. A faster source cannot fix that; fewer bytes per prompt can.
- **What shipped:** the rule had no branch for "speed fails, decode and exactness pass", so the call was made
  outside it, and logged as such. 0007 goes into the default build: it is a safety patch with no measured cost, though its cap never triggered.
  0008 ships opt-in, behind a new switch, `--moe-prefill-source mmap|p3` (patch 0009, default `mmap`, the old path
  byte for byte), because on Gemma, whose file sits in the page cache, P3 prefill was slower than the memory map.
- **A side result we labelled too quickly:** one run at `-ub 2048` reached stock's 16K prefill (176.8 tok/s). Its
  text differed from the default ubatch's, so we called it class A. Point 13 corrects that.

**9. Calibration v1 (M1.3) becomes code.**
- **Calibration:** `marian calibrate` measures a machine before anything is served: cache arms with a VRAM-cliff
  bisection, a RAM safety envelope, and CPU thread counts (other projects saw fewer threads win by 2-3x).
- **Plans:** they now restore the cache budget, miss source, tier and threads at launch, which fixes a debt from M0.
- **Serving:** `marian serve` runs behind a small front proxy on 127.0.0.1.
  - **Admission:** one request at a time, one queued; the rest get a 503 with a reason.
  - **Checks:** a passive check that the first answers run as fast as the profile said, and `--confirm` for a live A/B
    against the runner-up.
  - **Memory governor:** pauses admission under pressure and restarts on the next smaller measured row, keeping an
    exact plan exact.
  - **Rollback:** `--stock`.
- **Tests:** 118, all offline. The governor had not yet faced a real competing allocation, and that is M1.3's exit
  (point 12).

**10. Is reusing a prompt prefix exact? (E52, A28). No.**
- **Why it matters:** a gate re-reads the same instructions on every call, so reusing their KV would turn most of its
  prefill into a lookup.
- **Why it may not be exact:** the reused KV was computed in a different batch split.
- **The client path was fixed first:** an uncounted instrument check found that on Gemma 4 (sliding-window
  attention) whether llama-server reuses a prefix at all depends on the endpoint. So the registered run's client path
  was fixed before it ran, and the register says it was fixed after the check. That check had also found 0 of 3
  reused answers identical to their fresh twins, so prefix reuse already shipped **off** in `serve`.
- **The result:** none of the 10 reused answers matched its fresh twin, all-VRAM or with the cache and tier.
  Log-probabilities moved by 1.08 to 4.66, and half the answers changed tokens. The reuse was real (only the
  document's tokens were evaluated), and a fresh prompt run twice matched exactly, so the difference is the reuse.
  We predicted "not exact in at least one pair" (about 60%); it was 20 of 20.
- **What it settled:** prefix reuse is class A. It stays off in exact mode, as an opt-in switch. It would have saved
  83% of prompt tokens, so the gate's prefill gain has to come from somewhere else (point 13).
- **A side result worth keeping:** the cache-and-tier arm matched all-VRAM on all 68 requests, reused prefixes
  included. The miss path is exact; the reuse is not.

**11. A mistake about time.** Three pre-registration stamps in the register were written as guessed clock times, not
read from the clock, and they fell after the work had begun. The E47 worker noticed its own entry was stamped too
late. The stamps were corrected from file evidence, and a rule was added: read the clock before stamping. A
pre-registration is worth only as much as its timestamp.

**12. The governor meets a real competitor (E53, A29), and makes things worse.** M1.3's exit was the memory governor
surviving a real competing allocation. Gemma 4 26B served through `marian serve` for 600 s while a VRAM hog took all
but 150 MiB of the GPU, and later a RAM hog pushed available RAM 2 GiB under the row's floor. A plain llama-server ran
the same schedule as the contrast.
- **What held, in two full runs:** every answer, across two restarts on three different rows, was token- and
  log-probability-identical to all-VRAM at the same context (409 of 409). Every refusal was a 503 with a reason, and
  answers resumed within 22-37 s of each crossing.
- **What failed, as written:** the clause that the server's shared GPU memory never exceeds 256 MiB. Every all-VRAM
  llama-server reads about 854 MiB there from the moment its model loads, on a quiet machine, the contrast included.
  The clause was an absolute threshold with no baseline, so it could never pass, and its "could not fail" escape
  could never trigger. That was a design fault of ours; the worker had flagged the risk before the run. The FAIL
  stands as written: it is not re-read after seeing the data.
- **The finding that matters:** the plain server never slowed by more than about 4%, even with 150 MiB of VRAM or
  under 1 GiB of RAM left. Marian's governor restarted onto a smaller row at 9.5-11.7 tok/s, about 240 s in all, and
  never stepped back up when the hog left and 11 GiB of VRAM was free again. We had predicted the contrast would page
  and halve its decode; it kept 96-97% of it.
- **What changed:** the M1.3 exit is not met. The governor now only watches by default: it warns, and pauses
  admission only when commit headroom crosses its floor. Restarting is opt-in, triggered by a measured slowdown
  rather than a memory reading, and it steps back up. A new test (E55), with the shared-memory clause defined as a
  rise above the arm's own baseline, is to be registered before any rerun, and will say that definition was written
  after seeing E53's data.
- **Not validated:** the paging cliff of 21/10 (42.7 -> 5.5 tok/s) happened when a model was loaded beyond VRAM. A
  competitor arriving after load is a different event, and on this driver it was mild.

**13. A larger ubatch, judged against the right reference (E54, A30).** E51 showed that prefill is bound by bytes
per prompt, and its one `-ub 2048` run reached stock's speed. We had called that class A because its text differed
from the default ubatch's. But that compared two different compute shapes. Marian's promise is exactness against an
all-VRAM run *with the same settings*, as E53 had just ruled for the context size. So E54 asked the right question:
at `-ub 2048`, does the cache path equal all-VRAM at `-ub 2048`?
- **Exactness: yes.** On ten prompts of 2,009-2,406 tokens, the cache arm with memory-mapped misses and the tier
  arm with P3 prefill both matched all-VRAM at 2048 on every token and every top-3 log-probability. All-VRAM at the
  default 512 differed from all-VRAM at 2048 (up to 0.668 on the chosen token, 7 of 10 texts), so the check could
  fail. A larger ubatch is class E relative to a reference at the same ubatch. That corrects the morning's decision
  to call it class A.
- **Prefill: level with stock and past it.** With the tier and P3 prefill, Swift's 16K prefill was 156.7 tok/s
  against stock's 142.4 at its default ubatch (three rounds), 2.56x E51's 61.1.
- **Decode: a fail on the registered comparison, cause not established.** Warm decode was 12.67 tok/s and 11.47
  after a 16K prompt, 0.837 and 0.816 of E51's figures, under the 0.95 bound. We had predicted decode unaffected.
  But E51's figures came from another session. Stock also ran about 10% slower in this one, the hit rate was the
  same as E51's, host time per token rose and the drive read slower, and E51's one run at 2048 decoded at 14.9. Only
  a same-session run at 512 can separate the machine from the ubatch, and the protocol did not include one.
- **What it settled:** the ubatch becomes a declared profile parameter, default 512. A deployment that declares
  2048 gets answers identical to all-VRAM at 2048. For a gate (long prompt, short answer), that row beat default
  stock on 16K and 4K prefill, warm decode and decode after 16K, exactly, which makes it the likely contender for the
  8 GB milestone.

## 7. What is next

The live list is [ROADMAP.md](ROADMAP.md). In short, as of 27/10:

- **Done:**
  - M1: the decode exit is met and kept as written, exactly (A21, A22).
  - The rebase onto `63b4b303` (E48).
  - The predictability bounds (E47).
  - The prefill diagnosis (E49).
  - Calibration v1's Python side and the serving rules (M1.3, tested offline).
  - Prefill misses through P3, exact, shipped opt-in (E51: +76% at 16K), and the tier's pinning rules in the fork
    (0007).
  - A declared ubatch is exact against a reference at the same ubatch (E54), and with it the tier's 16K prefill
    passed stock's default.
  - Prefix reuse is not exact, so it stays off in exact mode (E52).
- **Prefill, next:**
  - done the same day (E56, A31): the ubatch's decode cost, measured in one session, is +6% warm and -6% after a
    16K prompt, with prefill about 3x;
  - sizing the compute buffer for a large ubatch without shrinking the cache's share;
  - calibration rows that measure the prefill source and a declared ubatch;
  - then fewer bytes per prompt: layer-major prefill, prefetching the next layer's experts while this one computes,
    and pinning the mapped expert ranges as a measured arm (upstream #31122).
- **M1.3's exit, not met (E53):**
  - E55, the governor's opt-in `restart` mode against a real competitor, with the shared-memory clause measured
    against the arm's own baseline;
  - a test that makes 0007's tier cap trigger;
  - learned counts.
- **Rebases:** weekly, or at once when upstream touches what we run, each with an E48-style exactness check. A
  daily upstream watch says when (ROADMAP section 1). The first reading, on the evening of 27/10, flagged three
  commits, one of them a CUDA scale-kernel change, so the next rebase comes soon.
- **The build:** `/EHsc` (C++ exception unwinding) is on from M1.3. E57 (A32) found every answer identical with and
  without it.
- **The 8 GB milestone (E50)** runs once the prefill track lands. Its rule is already written, and so is our
  prediction that the code as it stood that morning fails it. A declared-ubatch row is reported beside it.
- **Exactness, the rest of the way:**
  - the single-layer harness (E0);
  - the kernel-boundary parity matrix;
  - full-logit comparison, where today only the top 3 are compared.
- **Later milestones:**
  - M1.4: real hardware.
  - M2: overlap copies with routing, which E47 says comes before any predictor.
  - M3: verified speculation.
  - M4: opt-in answer-changing switches.
  - M5: prefill compute.

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
| 24/10 | analysis compared whole texts of different lengths ("Gemma 0/10") | shared-prefix comparison; Gemma is exact against all-VRAM (E41) |
| 24/10 | analysis counted a one-token answer as 0 tok/s; the post-prefill median became one slow prompt | short requests excluded and listed; an all-tokens rate added; P3 figures corrected (E43) |
| 24/10 | patch 0004 wrote S record heads with token count and expert count swapped | caught by review before the GPU run; patch 0005; a strict reader (E43) |
| 24/10 | M1.2's plan to raise the tier's share of misses | the exact replay showed no realistic policy gets there on 32 GB; the lever was the page cache, not the tier's share |
| 24/10 | M0 profiles counted the desktop's VRAM as the model's; no packaging | per-row baseline, `--fit-now`, `--json`, `pyproject.toml` (found by an outside user) |
| 24/10 | a PowerShell loop counter `$r` overwrote `$R` (names are case-insensitive) | caught in the first second, nothing ran; renamed |
| 22-24/10 | no run ever recorded a probability (the client read field names this server does not send), so every "top-3 identical" compared nulls | client reads both shapes and refuses an empty top-k; analyser says "not measured"; measured: identical to full precision (E44, E45) |
| 24/10 | "the tier costs prefill 3x" was against upstream's cache, not stock | against stock it is 4-5x (E44) |
| 26-27/10 | three pre-registration stamps were guessed clock times, later than the work they preceded | corrected from file evidence; rule: read the clock before stamping (section 6c, point 11) |
| 27/10 | the first roadmap offered a fix for the embedding table's page granularity before checking whether upstream already prefetches its rows | corrected the same morning; E49 then ruled the table out |
| 27/10 | prediction: "the embedding table crowds the experts out" (E49, about 55%) | wrong: it held under 0.5 GiB; the cost is single-page expert faults |
| 27/10 | E47's frequency-prior check failed the 2x bar its worker declared | judged not voiding (the ids reproduce exactly, and the null differs); the flat prior is reported as a finding |
| 27/10 | a larger ubatch called class A "until shown otherwise" because it changes the compute shape; E51 then labelled its `-ub 2048` run that way because its text differed from the default ubatch's | the wrong reference: E54 compared it with all-VRAM at the same ubatch and found it class E, a declared setting (section 6c, point 13) |
| 27/10 | prediction: P3 prefill about 3x at 16K, "about 60%" on the 2x clause (E51) | 1.76x, so wrong on the size and on the pass; it had assumed reads near the drive's peak and a larger tier share, and the drive bounds it |
| 27/10 | E53's shared-GPU-memory clause: an absolute 256 MiB limit with no baseline (an instrument-design fault of Claude Tyl's; the worker flagged the risk before the run) | every all-VRAM server reads about 854 MiB from load, so the clause could not pass and its escape could not trigger; the FAIL kept as written; E55 to measure a rise over the arm's own baseline |
| 27/10 | prediction: under E53's VRAM hog the plain server would page and halve its decode | it kept 96-97% of it; the governor's restart cost more than the pressure, so the default became `watch` |
| 27/10 | prediction: a larger ubatch costs nothing in decode, "free" at about 65% (E56) | a trade: +6% warm, -6% after a 16K prompt; the reasons held (the machine was slower, hit rate unchanged), the conclusion did not |
| 27/10 | prediction: a larger ubatch leaves decode unaffected (E54) | decode 16-18% under E51's figures on the registered comparison, a fail; the comparison crosses sessions, so the cause is open until a same-session run at 512 |

| 27/10-28/10 | row 17 compared "the same T8 configuration" across sessions, but not the same binary: the fast morning runs used builds that were replaced at 12:29 | E60 ran the morning's own binary beside today's: 0.96x, so not the code; the comparison rule now names binaries |
| 28/10 | the same PowerShell `$r`/`$R` collision as on 24/10, in E60's loop; two launches ran only their pre-run checks | renamed; every chain's variables listed and checked for case collisions before launch |
| 28/10 | two more guessed stamps (E59 "07:53", E55 "08:05"), each corrected from the clock within a minute | every stamp is now copied from a clock read in the same command that writes it |
| 28/10 | a recursive search of mine ran during E58's first exactness arms; E58's pre-run drive baseline came from another model's file | both disclosed in E58 (exactness unaffected); the second exposed a real drive finding (one file reads 2.0 GB/s, another 3.1, every time) |
| 28/10 | prediction: the logical batch `-b` is not part of the reference, about 80% (E59) | wrong: 0/10 records at `-b 512` against 2048; profiles now record `-b` and launch refuses a mismatch |
| 28/10 | prediction: E60 gives MACHINE (65%); its threshold 0.85 x 15.15 = 12.9 | the run gave 12.98: NO DECISION by 0.08, with CODE ruled out; a threshold set from the outlier, not from the slow state |

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

| Arm | Decode: median / all tokens | After 16K prompt: median / all tokens | Prefill 4K | Prefill 16K | Hit median / p10 | MiB uploaded per token | Host ms per token: sync wait / after | VRAM peak MiB | Free RAM min GiB | Peak working set / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| r1-stock34 | 10.6 / 8.8 | 5.3 / 5.3 | 111.6 | 152.0 | - | - | - | 15,200 | 0.16 | 22.28 / 20.83 |
| r2-stock34 | 10.9 / 9.3 | 5.1 / 5.1 | 113.2 | 147.9 | - | - | - | 15,200 | 0.18 | 22.26 / 20.83 |
| r1-c4000 | 7.3 / 6.9 | 4.2 / 4.0 | 34.6 | 50.2 | 74.8% / 56.7% | 175.6 | 0.76 / 83.42 | 9,058 | 0.17 | 23.03 / 14.85 |
| r2-c4000 | 7.0 / 6.7 | 4.4 / 4.1 | 33.6 | 51.5 | 74.8% / 56.7% | 175.6 | 0.85 / 87.71 | 9,058 | 0.16 | 22.83 / 14.85 |
| r1-c6500 | 7.0 / 6.9 | 4.3 / 4.2 | 45.1 | 80.4 | 84.4% / 69.8% | 112.7 | 0.82 / 78.31 | 11,562 | 0.18 | 22.54 / 17.29 |
| r2-c6500 | 7.1 / 6.8 | 4.4 / 4.2 | 40.9 | 86.1 | 84.4% / 69.8% | 112.7 | 0.83 / 80.07 | 11,562 | 0.15 | 22.45 / 17.29 |
| r1-c9400 (*) | 8.2 / 7.1 | 5.4 / 5.0 | 54.1 | 107.2 | 90.8% / 80.4% | 69.3 | 0.79 / 72.98 | 14,462 | 0.06 | 22.5 / 20.13 |
| r2-c9400 | 7.3 / 7.1 | 4.5 / 4.3 | 51.6 | 107.9 | 90.8% / 80.4% | 69.3 | 0.79 / 70.73 | 14,462 | 0.11 | 22.43 / 20.13 |

Speeds leave out the one decode request and the one after-16K request per arm that stop within 8 tokens (a first
version of this table counted them as 0 tok/s; section 6b, point 7). All tokens = total tokens over total generation
time.

(*) overlapped the end of a 13.6 GB download and its hash read; r2 is the clean run. The 22/10 partial floor arm
(5.6 tok/s) was superseded by the full floor arms in [A9](#a9-swift-at-8-gb-the-floor-arms).

Taught us: SLRU already hits 84-91%; decode is still about 30% slower than stock; the host time after the sync
(71-88 ms per token) is the cost; every arm slows by about a third to a half after a 16K prompt; rounds agree within
about 5% on decode.

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

Conditions: greedy text on the 10 decode prompts, pairs of A6 arms (top-3 probabilities were requested but not
recorded; A22). Source:
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

| Arm | Decode: median / all tokens | After 16K prompt: median / all tokens | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB (incl. ballast) | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| f1-c2400 | 5.1 / 4.7 | 4.3 / 4.0 | 28.1 | 38.3 | 63.7% / 45.0% | 242.8 | 1.42 / 131.64 | 15,488 | 0.03 | 23.07 / 13.25 |
| f2-c2400 | 6.1 / 5.9 | 5.0 / 4.6 | 33.9 | 42.4 | 63.7% / 45.0% | 242.8 | 0.87 / 103.85 | 15,766 | 0.11 | 23.24 / 13.26 |
| f1-stock45 | 7.8 / 7.3 | 6.7 / 6.5 | 27.0 | 30.5 | - | - | - | 15,623 | 0.08 | 22.66 / 13.01 |
| f2-stock45 | 7.6 / 7.6 | 6.9 / 6.7 | 27.9 | 27.7 | - | - | - | 15,881 | 0.10 | 22.67 / 13.05 |

Speeds leave out the one decode request and the one after-16K request per arm that stop within 8 tokens (a first
version of this table counted them as 0 tok/s; section 6b, point 7). All tokens = total tokens over total generation
time.

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

| Arm | Decode: median / all tokens | After the 4K prompt: median / all tokens | Prefill 4K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|
| gemma-floor-c2500 | **42.6 / 42.7** | 40.5 / 43.5 | 756.9 | 87.5% / 72.9% | 93.8 | 0.34 / 8.52 | 15,086 | 8.05 | 14.69 / 8.61 |
| gemma-floor-ncmoe24 | 16.8 / 16.9 | 19.8 / 19.5 | 700.8 | - | - | - | 14,802 | 8.06 | 14.73 / 8.33 |
| gemma-full-ncmoe0 | 79.4 / 80.5 | 79.7 / 78.8 | 3,645.7 | - | - | - | 14,439 | 7.77 | 15.09 / 16.44 |

Answer identity: the cache arm against the all-on-GPU arm, over the 128 tokens both generated (the full arm
generated 256): **1,280 of 1,280 tokens identical**, on all 10 prompts. (This entry first said "and every top-3 id
and probability identical"; those were never recorded. A22 measures them on the M1.1 build: identical too.)
The two M1.1 gate arms (memory-mapped and ring misses, ctx 20K, under 25 GiB of desktop pressure) match it the same
way. Plain `-ncmoe 24` differs from both within 0-73 tokens (CPU rounding, as design section 3.2 predicts).
Taught us: with RAM to hold the experts, the cache is 2.5x stock at 8 GB and only 8.5 ms of host time per token, and
it is **exact against the true reference** (design E2) at token level, and A22 adds the top-3 log-probabilities;
full logits are not logged, so bitwise is not claimed. A first version of this entry said "0/10, unexamined": the analysis compared whole texts of
different length, a check that could not pass. Found because the result contradicted the design, then fixed (E41).

#### A11. Gate arms: 8 GB GPU and about 25 GB of desktop RAM

Conditions: 8 GB VRAM ballast; about 25 GB of RAM held to simulate desktop work for the arms tagged `desk25` (register
E39 confirms this for the Gemma cache arm). The summary does not record the RAM setting for the other arms; their low
free-RAM minimum is consistent with the same setting (not verified). M1.0 build; same request set as A6. Source:
`m1.0/gate-summary.md`.

| Arm | Decode: median / all tokens | After 16K prompt: median / all tokens | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate-e4b-qat-full (Gemma 4 E4B-it QAT Q4_0) | **89.7 / 89.6** | 86.0 / 85.9 | 6,108.7 | 5,643.6 | - | - | - | 12,377 | 3.30 | 4.01 / 5.02 |
| gate-bolt-e4b-full (Nemotron-3.5-Bolt-E4B Q4_K_M) | **79.7 / 79.7** | 68.3 / 69.5 | 5,488.8 | 5,212.8 | - | - | - | 12,703 | 2.99 | 4.38 / 5.39 |
| gate-qwen35-9b-full (Qwen3.5-9B UD-Q4_K_XL) | **65.4 / 65.2** | 65.6 / 65.6 | 3,838.2 | 3,829.7 | - | - | - | 15,129 | 0.38 | 6.4 / 9.58 |
| gate-gemma26-iq3xxs-ncmoe18 | **26.3 / 25.7** | 27.0 / 28.1 | 775.4 | 756.1 | - | - | - | 15,833 | 0.13 | 6.5 / 10.25 |
| gate-gemma26-iq3xxs-c2500 | 12.7 / 13.1 | 11.3 / 11.7 | 146.3 | 350.2 | 91.7% / 78.8% | 58.2 | 0.58 / 32.14 | 14,651 | 0.12 | 6.42 / 9.1 |
| gate-gemma26-ncmoe24-desk25 (IQ4_XS) | 15.0 / 14.9 | 15.7 / 15.2 | 129.1 | 322.5 | - | - | - | 14,886 | 0.06 | 8.55 / 9.38 |
| gate-gemma26-ncmoe30-desk25 (IQ4_XS) | 10.0 / 10.2 | 10.1 / 10.3 | 21.5 | 20.3 | - | - | - | 12,791 | 0.07 | 7.05 / 7.2 |
| gate-gemma26-c2500-desk25 (IQ4_XS) | 8.0 / 7.5 | 7.4 / 7.4 | 97.8 | 169.3 | 87.5% / 72.9% | 93.8 | 0.65 / 65.87 | 15,154 | 0.25 | 6.38 / 9.64 |
| gate-groot-ncmoe28 (Groot-19B-A3B Q4_K_M) | 19.2 / 18.2 | 20.6 / 21.3 | 267.3 | 447.9 | - | - | - | 13,985 | 0.20 | 6.59 / 8.6 |
| gate-groot-c2500 | 5.0 / 4.8 | 6.1 / 6.2 | 149.6 | 246.8 | 75.6% / 60.6% | 150.9 | 1.27 / 124.98 | 13,633 | 0.17 | 7.06 / 8.25 |

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

| Arm | Decode: median / all tokens | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|
| e1-c4000 | 7.2 / 7.0 | 74.4% / 56.9% | 178.7 | 0.87 / 81.40 | 9,460 | 0.23 | 22.8 / 13.06 |
| e1-c9400 | 7.8 / 7.3 | 90.8% / 80.4% | 69.2 | 0.91 / 69.30 | 14,860 | 0.25 | 22.58 / 18.34 |

Speeds leave out the one decode request per arm that stops within 8 tokens (a first version of this table counted it
as 0 tok/s; section 6b, point 7). All tokens = total tokens over total generation time.

| Pair | Decode texts identical |
|---|---|
| e1-c4000 against e1-c9400 | **10/10** (M1.0: 7/10) |
| e1-c9400 (M1.1 build) against r2-c9400 (M1.0 build) | 10/10 |

(A first version had a column "top-3 probabilities identical: 10/10"; it compared nulls, A22.)

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

| Arm | Decode: median / all tokens | After 16K prompt: median / all tokens | Prefill 4K | Prefill 16K | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|
| p3-r1-c9400 (A) | 7.3 / 6.9 | 5.0 / 4.5 | 53.3 | 119.3 | 0.94 / 75.37 | 14,991 | 0.23 | 22.07 / 20.14 |
| p3-r2-c9400 (A) | 6.3 / 6.2 | 4.6 / 4.2 | 47.2 | 112.0 | 1.06 / 83.91 | 14,990 | 0.21 | 21.57 / 20.13 |
| p3-r1-c9400-ring (B) | 12.4 / 12.0 | 10.7 / 9.8 | 45.0 | 105.0 | 0.77 / 48.01 | 15,021 | 0.14 | 21.78 / 20.26 |
| p3-r2-c9400-ring (B) | 12.0 / 12.0 | 11.7 / 10.8 | 41.0 | 109.9 | 0.76 / 47.91 | 15,020 | 0.13 | 21.75 / 20.25 |
| p3-r1-c9400-tier6g (C) | **14.7 / 14.2** | **14.8 / 13.8** | 25.5 | 34.0 | 0.77 / 39.11 | 15,021 | 0.11 | 21.48 / 26.27 |
| p3-r2-c9400-tier6g (C) | **14.0 / 14.1** | **14.4 / 13.2** | 23.0 | 35.8 | 0.76 / 39.00 | 15,020 | 0.10 | 20.73 / 26.27 |
| p3-r1-stock34 (D) | 8.6 / 7.7 | 4.2 / 4.2 | 104.1 | 145.2 | - | 15,863 | 0.16 | 21.81 / 20.84 |
| p3-r2-stock34 (D) | 11.0 / 8.9 | 3.8 / 3.8 | 96.1 | 136.4 | - | 15,728 | 0.11 | 21.77 / 20.83 |

Speeds exclude decode prompt 2, which stops after one token in every arm (a first version of this table counted it
as 0 tok/s, which put "after 16K prompt" at 10.3-10.9 for C; see section 6b, point 7). "All tokens" = total tokens
over total generation time.

Hit rate 90.8% / 80.4% and 69.3 MiB uploaded per token in every cache arm (the miss source changes where misses come
from, not how many there are).

| Pair | Decode texts identical |
|---|---|
| every pair of A, B and C arms, within and across rounds | 10/10 |
| D with itself across rounds | 10/10 |
| D against any cache arm | 2/10 (D computes 34 layers' experts on the CPU) |

(A first version had a top-3 probability column; it compared nulls, A22.)

Scored against the prediction (E40, on the first, uncorrected figures): A in range for r1 and below it for r2; **B and
C beat their ranges** (about 11.6 against 8.3-9.5; about 14.0 against 9.5-11.5; higher still after the correction); C's prefill about 3x slower at 16K, far worse than "equal or
slightly worse". Host time fell about 40 ms per token from A to C, token time about 70 ms. Partly explained since (A20): the
non-fetch part of each token also fell, from about 43 to 26 ms, so page faulting cost time outside the cache
callback's measured window too. D is noisy
(8.2 against 10.4, beyond the 5% seen on 22/10), so "C beats stock" holds against both rounds but by how much is
uncertain. Not stated in E40: the fill-trace counts of tier hits, admissions and ring reads per arm.

#### A18. Gemma gate: mmap vs ring on the M1.1 build

Conditions: Gemma 4 26B-A4B-it UD-IQ4_XS, `-ncmoe 30 --moe-cache-mib 2500`, 8 GB VRAM ballast plus about 25 GiB of
desktop RAM held, M1.1 build, one round each; same request set as A6. Sources: `m1.1/gate-summary.md`, register E40.

**Pre-registered prediction (register E39):** the ring gives no gain in the gate scenario, because the M1.0 cache arm
already pulled misses at about 1.4 GB/s, the ring's own rate: 7-9 tok/s against 8.0.

| Arm | Decode: median / all tokens | After 16K prompt: median / all tokens | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | VRAM peak MiB | Free RAM min GiB | Peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate11-gemma26-c2500-mmap | 8.1 / 8.1 | 12.0 / 10.6 | 122.6 | 199.9 | 83.3% / 66.2% | 117.3 | 0.61 / 59.92 | 15,274 | 0.13 | 6.8 / 9.64 |
| gate11-gemma26-c2500-ring | **13.4 / 14.1** | 12.9 / 14.2 | 48.1 | 218.7 | 83.3% / 66.2% | 117.3 | 0.47 / 42.99 | 15,264 | 0.21 | 7.17 / 9.72 |

Answer identity: 10/10 texts (probabilities not recorded; A22). Taught us: the prediction was wrong; memory-mapped misses under desktop
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

| Arm | Decode: median / all tokens | After the long prompt (16K for the Coder, 4K for the others): median / all tokens | Prefill 4K | Prefill 16K | Hit median / p10 | MiB up per token | Host ms per token: wait / after | Free RAM min GiB |
|---|---|---|---|---|---|---|---|---|
| coder-full-c9400 | 13.0 / 11.8 | 6.4 / 7.4 | 97.8 | 181.2 | 91.0% / 80.0% | 96.6 | 0.87 / 42.11 | 0.22 |
| coder-floor-c2400 | **7.1 / 6.9** | **7.5 / 6.5** | 52.8 | 122.5 | 57.9% / 41.0% | 402.8 | 0.92 / 95.71 | 0.15 |
| coder-floor-ncmoe45 | 6.5 / 5.9 | 2.8 / 3.2 | 88.7 | 103.7 | - | - | - | 0.21 |
| q36-full-c9000 | 43.3 / 41.2 | 37.3 / 40.1 | 367.3 | - | 95.9% / 88.4% | 31.6 | 0.58 / 4.75 | 0.30 |
| q36-floor-c2500 | **15.5 / 14.7** | 18.2 / 17.8 | 181.6 | - | 70.6% / 51.2% | 181.4 | 0.81 / 38.07 | 0.23 |
| q36-floor-ncmoe32 | 13.1 / 10.2 | **23.8 / 25.0** | **270.2** | - | - | - | - | 0.24 |
| dense27-full | 29.5 / 29.5 | 29.8 / 29.8 | 1,234.2 | - | - | - | - | 4.77 |
| dense27-floor-ngl40 | 2.2 / 2.2 | 2.2 / 2.2 | 415.4 | - | - | - | - | 2.18 |

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

#### A19. M1.2: the routing-id trace and the replay

Conditions: build `build-m1.2` (M1.1 + patch 0004 + review fixes 0005); Swift IQ2_XS, `-ncmoe 48 --moe-cache-mib
9400`; `i1-c9400` warm and decode (the acceptance run), `i2-c9400` all phases. Replay on the test prompts (decode 1,
3, 5, 7, 9), split by prompt. Sources: `docs/m1.2/results/`, `docs/m1.2/NOTES.md` section 10, register E43.

| Acceptance check | Result |
|---|---|
| answers, trace on (M1.2) against trace off (M1.1, e1-c9400) | **10/10 texts** (top-3 not recorded; A22) |
| replay against the server, S lines / fill steps | **107,008 / 107,008** and **58,110 / 58,110** identical |
| replay minus server, median per-token hit | **+0.00 points** (M1.1 exit check: within 3) |
| trace health | 4.10 MB, 107,221 records, 0 truncated, k = 10 throughout |
| speed against e1-c9400 | 6.6 against 7.8 tok/s: outside +-5%, attributed to page-cache state (4.5 GiB less free RAM at start), not isolated |

| P2 candidate (same 9,400 MiB) | Hit median | Delta against SLRU |
|---|---|---|
| upstream SLRU | 90.8% | - |
| guard (transient slots for prompt steps) | 88.8% | -2.1 |
| core25 (a quarter of slots ranked by training-prompt use) | 88.5% | -2.3 |
| core50 | 85.4% | -5.4 |
| door (TinyLFU-style doorkeeper) | 85.2% | -5.6 |
| core50 + door | 79.4% | -11.5 |

| RAM tier | current | warm | static-miss | oracle-static | Belady (equal sizes) |
|---|---|---|---|---|---|
| 6 GiB | 21.1% | 21.6% | 26.9% | 40.7% | 48.0% |
| 16 GiB | 56.9% | 59.4% | 61.7% | **79.3%** | 66.7% |

(Share of decode miss bytes served by the tier, test prompts; 75% is reached only by the oracle, at 16 GiB.)
Taught us: the replay is exact, so its answers can be trusted: a smarter VRAM policy and a bigger RAM tier are both
the wrong levers on this machine.

#### A20. Where the post-prefill time went

Conditions: per-token split of the E40 traces (single-token steps grouped by layer 0; tokens of the decode phase
against tokens of the three post-prefill requests). Host = time in the cache callback; other = the rest of the
interval between tokens.

| Arm | Decode: hit / MiB up / host ms | After 16K prompt: hit / MiB up / host ms | Token interval: decode / after (ms) | Other: decode / after (ms) |
|---|---|---|---|---|
| A, mmap (r1) | 90.8% / 60.2 / 75.4 | 89.0% / 73.3 / 144.4 | 118.8 / 202.6 | 43.4 / 58.2 |
| B, ring (r1) | 90.8% / 60.2 / 47.9 | 89.0% / 73.3 / 62.5 | 76.5 / 102.6 | 28.5 / 40.1 |
| C, tier 6 GiB (r1) | 90.8% / 60.2 / 39.1 | 89.1% / 72.5 / 44.0 | 65.4 / 70.2 | 26.2 / 26.2 |
| C, tier 6 GiB (r2) | 90.8% / 60.2 / 39.0 | 89.0% / 73.3 / 46.1 | 65.1 / 74.0 | 26.1 / 27.9 |

Taught us: after a long prompt the VRAM cache misses only a little more (+22% bytes), but memory-mapped misses cost
twice as much, because the prompt evicted the model's pages. The ring and the tier take the page cache out of the
miss path. Medians of token intervals; the per-request speeds also vary by prompt (24 tok/s for decode prompt 0, 10
for prompt 1 in the tier arm), which is why the post-prefill figure needs more than three prompts to be stable.

#### A21. The five-round confirmation

Plan (design 9.3): tier 6 GiB (M1.1 build) against stock `-ncmoe 34` (M1.0 build), five rounds in the order
T S | S T | T S | S T | T S, full phases, fresh process per run, resource log per run. Claim rule: the tier meets the
12 tok/s exit in a state only if every round's all-tokens figure is at or above 12 (register E44, written before any
result was read). Source: `docs/m1.2/results/confirm-summary.md`. All 10 servers came up and all 160 requests were
answered; in every run one decode and one post-prefill request stop at once (EOS) and are left out, as the rule says.

| Round | Tier decode | Stock decode | Ratio | Tier after 16K | Stock after 16K | Ratio | Prefill 16K, tier / stock |
|---|---|---|---|---|---|---|---|
| 1 (T S) | 14.5 | 10.8 | 1.34 | 13.4 | 3.9 | 3.44 | 29.7 / 144.4 |
| 2 (S T) | 14.4 | 9.6 | 1.50 | 13.6 | 4.2 | 3.24 | 34.4 / 148.6 |
| 3 (T S) | 14.6 | 8.2 | 1.78 | 13.4 | 3.9 | 3.44 | 31.2 / 155.0 |
| 4 (S T) | 14.5 | 9.5 | 1.53 | 13.7 | 3.9 | 3.51 | 34.4 / 141.8 |
| 5 (T S) | 14.6 | 8.2 | 1.78 | 13.5 | 8.5 | 1.59 | 30.3 / 154.8 |

(tok/s, all-tokens rate; the tier's medians per request are 14.5-15.1 warm and 14.4-14.7 after 16K.) Hit rate 90.8%
(10th percentile 80.4%) and 69.3 MiB uploaded per token in every tier run; the tier's runs agree with each other on
all 10 answers.

Taught us: **the M1 decode exit is met in both states**, in every round, with the tier's own spread at 1-2%. The tier
beats stock in 5 of 5 rounds (median ratio 1.53 warm, 3.44 after 16K). Stock is the noisy arm (8.2-10.8 warm; after
16K 3.9-4.2, except 8.5 in round 5, unexplained and kept). Prefill: stock is 4-5x faster at 16K and 4x at 4K.

#### A22. Exactness with real probabilities

Conditions: build-m1.1, no ballast, warm plus 10 decode prompts of 128 tokens, 3 log-probabilities per token
(pre-sampling, recorded at full float precision), fixed client. Gemma 4 26B-A4B-it UD-IQ4_XS: F1 all experts in VRAM
(`-ncmoe 0`); M cache `-ncmoe 30 --moe-cache-mib 2500` with memory-mapped misses; T the same with a 4 GiB tier; R the
same with the ring; C stock `-ncmoe 24`; F2 = F1 again. Swift: SM cache 9,400 MiB memory-mapped; ST with the 6 GiB
tier; SS stock `-ncmoe 34`. Rule registered first (E45). Sources: `docs/m1.2/results/exact/`.

| Pair | Texts identical | Top-3 log-probabilities identical | Largest log-probability difference, chosen token |
|---|---|---|---|
| Gemma F1 against F2 (the reference against itself) | 10/10 | 10/10 | 0 |
| Gemma M (memory-mapped misses) against F1 | 10/10 | 10/10 | 0 |
| Gemma T (tier misses) against F1 | 10/10 | 10/10 | 0 |
| Gemma R (ring misses) against F1 | 10/10 | 10/10 | 0 |
| Gemma C (stock, CPU experts) against F1 | 0/10 | 0/10 | 0.59 |
| Swift ST (tier) against SM (memory-mapped) | 10/10 | 10/10 | 0 |
| Swift SS (stock) against SM | 4/10 | 0/10 | 0.337 |

The Gemma cache arms ran at hit rate 83.3% (10th percentile 66.2%) and 117 MiB uploaded per token, so every token
went through the miss path being tested. A side result, from these unpressured runs: with RAM free and the page cache
warm, Gemma decoded at 34.0 tok/s with memory-mapped misses, 23.5 with the tier and 12.2 with the ring (82.8 all in
VRAM, 12.9 stock); the ring's unbuffered reads skip a page cache that, here, already held the experts. That is why the
ring and tier are opt-in and calibration has to choose them per machine.

Taught us: **the cache is exact against the true reference to full float precision, for every miss source**; the
prediction (difference exactly 0) was right, and the stock contrast was 30-60x larger than the 0.01 we guessed. The
contrast is also what makes the result worth something: the same instrument that reads 0 for the cache reads up to
0.59 for stock. Not shown: the full logit vector (only the top 3 are recorded).

### 26-27/10/2026

#### A23. E46: what the FP16/BF16 unlock could speed up at most

Conditions: a CUPTI activity trace of every kernel (loaded through `CUDA_INJECTION64_PATH`, no admin) on build-m1.2's
`llama-bench`, flash attention on, prompts of 4,096 and 16,384 tokens, two repeats, four gate cases (whole on the
GPU, and offloaded at 8 GB). The ceiling counts every kernel whose tensor-core instructions take FP16/BF16 inputs with
FP32 accumulation, the only ones the unlock speeds up, as if it made them free.

| Measure | Result |
|---|---|
| Ceiling, share of kernel time | under 10% in 7 of 8 cases; 11.4% at most (Gemma 4 E4B, 16K) |
| Ceiling, 8 GB offloaded gate case | about 1% of wall time (the run is bound by the CPU) |
| What the time is | the addressable time is almost all FlashAttention; INT8 quantised matmuls take 60-77% of kernel time, untouched |

Taught us: not worth a Secure Boot change or a Linux boot for this workload. Parked; if prefill compute ever matters,
the lever is the INT8 matmuls (M5).

#### A24. E47: predictability bounds of Swift's routing

Conditions: offline, CPU only. `marian predict` read the M1.2 routing-id traces (Swift 1.5 IQ2_XS,
`--moe-cache-mib 9400`). Training prompts were the even decode prompts and testing the odd ones (1,204 test tokens).
Source: [docs/m1.2/E47.md](m1.2/E47.md).

| Measure | Result |
|---|---|
| Determinism: two runs' decode ids | 2,210 of 2,210 tokens identical |
| Persistence: share of a token's experts also used by the previous token, same layer | 37.7% (shuffled null 20.1%; 100 shuffles 19.5-20.7%) |
| Next layer's experts from this layer's ids | 19.2%, below persistence |
| Global frequency prior, out of sample | 3.1% (chance 1.95%) |
| Perfect one-layer-ahead prediction served from pinned RAM | 20.5 against 14.5 tok/s, +41% |

Taught us: the headroom is real and sits in the miss path. Overlap (M2) comes before any predictor, and a predictor is
to be scored on misses, not on all activations.

#### A25. E48: the rebase onto `63b4b303`

Conditions: the six-patch stack re-applied (patches 1, 3 and 6 byte-identical; 2, 4 and 5 changed only for upstream's
`no_alloc` plumbing and moved context). Build: MSVC 19.44 and CUDA 12.8, sm_89. `ctest -L main`: 16 of 16 passed.
Gemma: the E45 arms; Swift: one interleaved round of the E44 protocol, with both arms on the new build.

| Gemma 4 26B-A4B IQ4_XS, against all-VRAM F1 | Texts | Top-3 log-probabilities | Largest difference, chosen token |
|---|---|---|---|
| F2 (the reference again) | 10/10 | 10/10 | 0 |
| M cache 2,500 MiB, memory-mapped misses | 10/10 | 10/10 | 0 |
| T cache + tier 4 GiB | 10/10 | 10/10 | 0 |
| R cache + ring | 10/10 | 10/10 | 0 |
| C stock `-ncmoe 24` (contrast) | 0/10 | 0/10 | 0.590 |
| F1 at the new base against F1 at the old base | 10/10 | 10/10 | 0 |

| Swift 1.5 IQ2_XS | Tier 6 GiB | Stock `-ncmoe 34` | E44 tier, 5 rounds |
|---|---|---|---|
| Warm decode, all tokens | 15.1 | 11.7 | 14.4-14.6 |
| After a 16K prompt, all tokens | 12.6 | 9.5 | 13.4-13.7 |
| Prefill 4K / 16K, tok/s | 25.9 / 44.6 | 109.0 / 167.2 | 18.9-25.8 / 29.7-34.4 |

Taught us: the rebase kept exactness where there is a true reference (Gemma), and Swift's text moved as the upstream
numerics commits predicted (6 of 10 prompts diverge from the old base within 128 tokens). The exit after 16K holds by a
thin margin, on one round.

#### A26. E49: residency by tensor kind during a 16K prompt

Conditions: build-m1.2, Swift 1.5 IQ2_XS, E44's request sequence, one round per arm. `tools/wsprobe` sampled the
server's working set every 500 ms (stride 64 pages, checked against an exact count first; the self-test passed 13 of
13 checks on a synthetic model). A: cache 9,400 MiB, memory-mapped misses. B: A plus the 6 GiB tier. C: stock
`-ncmoe 34`.

| Arm | Embedding table resident at the end of the 16K prefill | Experts resident | Prefill 16K, tok/s | Decode, all tokens |
|---|---|---|---|---|
| A | 0.36 GiB | 21.10 GiB | 125.7 | 6.6 |
| B | 0.03 GiB | 15.56 GiB | 37.0 | 14.0 |
| C | 0.35 GiB | 21.36 GiB | 170.1 | 12.6 |

During B's prefill the drive served about 38,700 page reads a second at about 151 MB/s: 4 KiB per read. B took
18.65 million page faults in 523 s. B holds about 5.5 GiB fewer expert pages than A or C, roughly its pinned tier.

Taught us: the embedding table is not the cause (at most 0.48 GiB in any arm, at any moment). The tier takes the RAM,
and the prefill pays for it in single-page faults. The fix is to give prefill misses the same unbuffered and pinned
sources decode already has (E51).

#### A27. E51: prefill through P3

Conditions: the rebased build plus patches 0007 (tier safety) and 0008 (prefill source). Swift 1.5 IQ2_XS on the
reference laptop, E44's request sequence and rate definitions, cache 9,400 MiB. Arms: T, the 6 GiB tier on the
rebased build without the new patches; T8, T plus 0007 and 0008; R8, the ring only plus 0008; S, stock `-ncmoe 34`.
Three interleaved rounds (T T8 R8 S, rotated), a fresh process per run. T8u (T8 at `-ub 2048`) ran once after round
3, reported only. Gemma 4 26B-A4B IQ4_XS: E45's exactness check on the same build. The rule was registered before any
code was written: T8's 16K prefill at least 2x T's, T8's decode within 5% of T's or better, answers unchanged.

| Swift, median of three rounds (tok/s) | Prefill 16K | Prefill 4K | Decode, all tokens | After a 16K prompt, all tokens |
|---|---|---|---|---|
| T (tier, memory-mapped prefill) | 34.7 | 27.1 | 14.80 | 13.17 |
| **T8 (tier, P3 prefill)** | **61.1** | 56.6 | **15.15** | **14.07** |
| R8 (ring, P3 prefill) | 54.7 | 51.1 | 12.5 | 11.2 |
| S (stock) | 174.2 | 110.8 | 11.9 | 10.0 |
| T8u (`-ub 2048`, one run, reported) | 176.8 | 159.2 | 14.9 | 13.2 |

| Clause | Result | Verdict |
|---|---|---|
| T8's 16K prefill at least 2x T's | 1.76x (rounds 1.53, 1.85, 1.74; T8 60.6-61.3, T 33.2-40.0) | **fail** |
| T8's decode within 5% of T's or better | 1.024 warm, 1.068 after a 16K prompt | pass |
| Answers unchanged | Swift: T, T8 and R8 identical on all 9 runs, largest difference 0. Gemma's tier arm with 0008 against all-VRAM: 10/10 texts and top-3 log-probabilities, largest difference 0 over 1,280 tokens. Stock contrast: 0.59 | pass |
| Stretch: T8 at stock's prefill | 0.35x | not met |

Where the 16K prefill went, from the traces: every cache arm uploads about 509 GiB of experts per 16K prompt through the
selective copies. T8 spent 281 of its 319 s inside them, with the tier supplying 13.8% of the bytes and the rest read
unbuffered from the SSD at about 1.95 GB/s (T, through memory-map faults: 1.30 GB/s; T8u: 173 GiB at 2.11 GB/s). A
verify run compared every one of about 600,000 uploads with the memory map: no difference. Single runs, not in the
rule: Gemma's 5,428-token prompt ran at 3,626 tok/s all-VRAM, 573 with memory-mapped misses, 260 with T8, 193 with R8
and 680 stock; its 13 GB file stays in the page cache, where unbuffered reads are slower than page faults. T8's
lowest free RAM was 0.3-1.4 GiB, against about 0.03 GiB for T.

Taught us: prefill through P3 is exact and 76% faster at 16K, and decode does not pay for it, but it is bound by the
drive, because each 512-token ubatch re-reads most of each layer's experts. The next lever is fewer bytes per prompt
(a larger ubatch, layer-major prefill), not a faster source. Two checks could not fail: 0007's cap and its
failed-chunk branch never triggered, and on Swift only one decode prompt is long enough to take 0008's path. 0008's
exactness rests on Gemma, the verify run and the CPU self-tests. 0008 is opt-in (`--moe-prefill-source p3`, patch
0009) because of Gemma's result.

#### A28. E52: is prefix reuse exact?

Conditions: the rebased build (no prefill patches). Gemma 4 26B-A4B IQ4_XS, `-c 16384`, two arms: A all-VRAM, and T
cache 2,500 MiB with a 4 GiB tier. A fixed instruction block (1,608 tokens; 1,616 with the chat template) and ten
invented documents of 206-599 tokens, each answered with 64 greedy tokens and 3 log-probabilities per token. F: fresh,
with prompt caching off. R: reused, after a different document had primed the same block. Registered client path
`/completion` with the templated text, the shortest other document as the primer, thinking off; the chat endpoint was
also run and reported. An A-A control (two fresh runs) and a contrast (one character changed in the block).

| Arm, path | Pairs identical in tokens | Pairs with largest difference 0 | Largest difference, chosen / any top-3 | Reuse real | A-A control | Prompt-time speed-up |
|---|---|---|---|---|---|---|
| A, `/completion` (registered) | 5 of 10 | **0 of 10** | 1.29 / 4.66 | 10 of 10 | identical | 2.84x |
| T, `/completion` (registered) | 5 of 10 | **0 of 10** | 1.29 / 4.66 | 10 of 10 | identical | 3.82x |
| A, chat endpoint (reported) | 6 of 10 | 0 of 10 | 0.95 / 2.34 | 10 of 10 | identical | 3.14x |
| T, chat endpoint (reported) | 6 of 10 | 0 of 10 | 0.95 / 2.34 | 10 of 10 | identical | 4.26x |

The smallest per-pair largest difference was 1.08. Each reused request evaluated exactly its document's tokens (1,616
reused), so 82.7% of prompt tokens were saved. The contrast re-evaluated all 1,962 tokens and differed (1.73). Every
divergence came at token 11 or later, so the one-word verdict was the same in all 40 pairs, but the stated reason
changed in some. Across arms, T gave exactly A's tokens and log-probabilities on all 68 requests.

Taught us: a reused prefix does not give the fresh answer, so prefix reuse is class A, off in exact mode, and an
opt-in switch. The miss path, by contrast, stayed exact even with reuse. Our prediction (not exact in at least one
pair, about 60%) was right and too timid: 20 of 20. The likely mechanism, a different batch split for the reused KV,
fits what we saw but was not isolated.

#### A29. E53: the governor under a real competing allocation

Conditions: Marian's serving code at the time (governor restarting on memory floors), the rebased build, Gemma 4
26B-A4B IQ4_XS, a profile from `marian calibrate`, `serve` with its defaults and `--need 4096`. One client looped over
E45's 10 decode prompts (128 greedy tokens, 3 log-probabilities) for 600 s. P0 (0-120 s) clean; P1 (120-240 s) a VRAM
hog leaving 150 MiB free; R1 (240-360 s) released; P2 (360-480 s) a RAM hog pushing available RAM 2 GiB under the
row's 3,240 MiB floor; P3 (480-600 s) clean. A once-per-second monitor read free VRAM, available RAM, commit headroom
and the server's shared GPU memory. Contrast: plain llama-server on the same starting row, no proxy, no governor. Two
full runs; answers judged against all-VRAM at the same context.

| Clause | Run 1 | Run 2 |
|---|---|---|
| Every answer identical to all-VRAM | 200 of 200, largest difference 0, three rows | 209 of 209, largest difference 0 |
| Refusals only as a 503 with a reason; nothing open over 300 s | 14 refusals; longest request 17.6 s | 17 refusals; longest 14.0 s |
| Answers resume within the bound after each crossing; nothing in P0 | resumed after 37 s and 26 s | resumed after 31 s and 22 s |
| Shared GPU memory at most 256 MiB while serving | **854-918 MiB: fail** | **854 MiB: fail** |

| Median decode by phase (tok/s) | P0 | P1 | R1 | P2 | P3 |
|---|---|---|---|---|---|
| Marian, run 1 | 71.2 | 9.5 | 11.6 | 73.9 | 75.5 |
| Marian, run 2 | 76.3 | 11.7 | 11.5 | 76.6 | 76.5 |
| Contrast, run 1 | 77.2 | 74.8 | 76.8 | 75.3 | 76.9 |
| Contrast, run 2 | 79.2 | 76.0 | 76.6 | 76.3 | 75.8 |

In P1 the governor paused within about a second of the crossing and restarted on a smaller cache row (`-ncmoe 30`,
866 MiB cache). It stayed there through R1, with 11 GiB of VRAM free, until P2's RAM pressure forced a second restart,
which happened to choose all-VRAM again. The contrast lost 64-208 MiB of VRAM to shared memory in P1, at no measurable
cost, and kept running with 0.7-1.0 GiB of RAM available in P2.

Taught us: the governor kept answers exact and refused cleanly, but on this machine restarting on a memory reading
cost far more than the pressure it guarded against, and with no step-up it never recovered. So the default became
`watch`, and restarting became opt-in, on a measured slowdown, with a step-up. The shared-memory clause measured the
counter's floor, not paging: every all-VRAM server sits near 854 MiB from load. It needed a baseline. The 21/10 paging
cliff was not reproduced: it came from loading a model beyond VRAM, not from a competitor arriving later.

#### A30. E54: a declared ubatch

Conditions: the rebased build plus patches 0007-0009. All arms at `-ub 2048 -b 2048` unless named. Gemma 4 26B-A4B
IQ4_XS, `-c 16384`: ten public-domain passages of 2,009-2,406 tokens, so the ubatch size matters (each runs in two
ubatches at 2048, in four or five at 512), 64 greedy tokens, 3 log-probabilities. F-U all-VRAM; M-U cache 2,500 MiB with
memory-mapped misses; T8-U cache plus a 4 GiB tier with `--moe-prefill-source p3`; F-U2 = F-U again. Contrast: F-512,
all-VRAM at the default ubatch. Swift 1.5 IQ2_XS: three interleaved rounds of T8-U (E51's T8 at `-ub 2048`) against
S (stock `-ncmoe 34` at its default ubatch), E44's sequence; R8-U and stock at `-ub 2048` once each afterwards. The
decode clause was ruled one-sided before any run (at least 0.95 of E51's T8).

| Gemma, against F-U | Texts | Top-3 log-probabilities | Largest difference, chosen / any top-3 |
|---|---|---|---|
| M-U (memory-mapped misses) | 10/10 | 10/10 | 0 / 0 |
| T8-U (tier, P3 prefill) | 10/10 | 10/10 | 0 / 0 |
| F-U2 (the reference again) | 10/10 | 10/10 | 0 / 0 |
| F-512 (contrast) | 3/10 | 0/10 | 0.668 / 12.0 |

| Swift, median of three rounds (tok/s) | T8-U | S, default ubatch | E51's T8, default ubatch, earlier session |
|---|---|---|---|
| Prefill 16K | **156.7** | 142.4 | 61.1 |
| Prefill 4K | 138.3 | 97.1 | 56.6 |
| Decode, all tokens | 12.67 | 10.57 | 15.15 |
| After a 16K prompt, all tokens | 11.47 | 4.61 | 14.07 |

Reported once each: R8-U 151.1 / 135.2 / 10.0 / 9.2; stock at `-ub 2048` 279.6 / 123.3 / 8.8 / 10.7 (and stock
changes answers). T8-U really used P3 in prefill (24,724 tier and 45,229 ring uploads on Gemma) and saw ubatches of
2,044 tokens. Swift's T8-U rounds and R8-U agreed exactly. Patch 0009's own checks passed: with the default `mmap`, 0
P3 uploads over 1,102 large copies; with `p3` set through the environment, 4,392; `p3` without a ring or tier refused. Gemma's prefill on these
prompts: 4,564 tok/s all-VRAM, 1,209 with memory-mapped misses, 660 with the tier and P3.

The decode comparison crosses sessions. In this one, stock's decode was 10.57 against 11.9 in E51 and its 16K
prefill 142 against 174; T8-U's hit rate equalled E51's (0.906) while host time per token rose from 36-38 ms to
44-46 ms and the drive read at about 790 against 1,370 MB/s; and E51's one `-ub 2048` run decoded at 14.9. The
2,048-row compute buffer also took about 300 MiB more VRAM (peak 15,294 against 14,992 MiB) beside the same cache.

Taught us: a larger ubatch is exact against all-VRAM at the same ubatch, so it is a declared setting, like the
context, not an answer-changing switch. With it, the tier's 16K prefill passed stock's default, exactly. The
registered decode clause failed (0.837 and 0.816), and we had predicted decode unaffected; whether the ubatch or the
machine caused the drop needs a same-session run at 512. Until then the default stays 512, and the ubatch becomes a
declared profile parameter.

#### A31. E56: the ubatch's decode cost, measured in one session

Conditions: Swift 1.5 IQ2_XS, build-p4 (0007-0009), cache 9,400 MiB, tier 6 GiB, `--moe-prefill-source p3`; T8
(`-ub 512 -b 2048`) and T8-U (`-ub 2048 -b 2048`) in four interleaved rounds on the afternoon of 27/10, E44's request
sequence, fresh process per run; both arms at `-lv 4`, which prints the compute-buffer sizes. Rule registered first.

| Measure (median of four rounds) | T8 (512) | T8-U (2048) | Ratio |
|---|---|---|---|
| Warm decode, tok/s | 11.87 | 12.63 | 1.064 (T8-U faster in 4 of 4 rounds) |
| Decode after a 16K prompt | 11.82 | 11.12 | 0.941 |
| 4K prefill | 46.4 | 140.0 | 3.0 |
| 16K prefill | 51.7 | 154.5 | 3.0 |
| GPU compute buffer, MiB | 351 | 728 | |

Hit rate 0.906 in every run of both arms; each arm's records equal its earlier runs' (E51's T8, E54's T8-U). The
post-16K figure rests on two requests a run (the third post prompt ends after one token).

Taught us: the rule called it a trade (the post-16K clause missed 0.95 by a hair), and it is one: about 3x prefill
and +6% warm decode, against -6% after a long prompt, which the 377 MiB larger compute buffer probably explains. E54's
16-18% decode gap was the machine: the same T8 configuration decoded at 11.87 that afternoon against 15.15 that
morning. We had predicted "free" at about 65%; the reasons held, the conclusion did not. A gate declares 2048; the
default stays 512.

#### A32. E57: C++ exception unwinding (`/EHsc`) in the build

Conditions: the same sources as the measured build (upstream `63b4b303` plus patches through 0009), built twice:
without `/EHsc` (every earlier Marian build) and with it, in the same toolchain. Then E48's Gemma 4 26B-A4B IQ4_XS
exactness arms (10 prompts × 128 tokens, top-3 log-probabilities) and one Swift T8 run. Rule registered first.

| Check | Result |
|---|---|
| Unit tests (`ctest -L main`, the trace-id test) | 16/16 and 1/1 |
| All-VRAM, cache (mmap), cache + tier, cache + ring: with `/EHsc` against without | 10/10 identical on each, max difference 0 over 1,280 tokens |
| Stock contrast against all-VRAM, with `/EHsc` | still differs: 0/10, 0.590, with the same divergence points as before |
| Swift T8 decode records against four earlier runs | 10/10 identical, 2,189 tokens |
| Flag evidence | `/EHsc` on every C++ compile line (458); the unwinding warning gone (197 before, 0 after); the binaries changed (+65 KB, +200 KB) |

Taught us: exception unwinding changes nothing we can measure in the answers, so it is the build default from M1.3
(`docs/m1.3/tools/build_m13.bat`). Speed was unchanged within this laptop's own noise, and that noise turned out to
be large: in one back-to-back check the old build's stock arm ran at 16.5 and then 7.8 tok/s. That is now a debt row
of its own, to settle before the 8 GB milestone's rounds.

### 28/10/2026

#### A33. E58: the second rebase, onto `338fce651`

Conditions: upstream `338fce651` (b11936, 32 commits after `63b4b303`, 11 of them touching what Marian runs), the
nine patches cherry-picked with no conflict and byte-identical, built with `/EHsc`. E48's Gemma 4 26B-A4B IQ4_XS arms
plus two new P3-prefill arms, then Swift T8 on the old and new builds interleaved in one session. Rule registered
before the build; pre-run check and GPU clock log beside every run.

| Check | Result |
|---|---|
| Unit tests | `ctest -L main` 16/16, trace-id test 1/1, Python 155/155 |
| Every cache arm (mmap, tier, tier + P3 prefill, ring, ring + P3 prefill) against all-VRAM | 10/10 identical each, max difference 0 over 1,280 tokens |
| Stock contrast against all-VRAM | still differs: 0/10, 0.590, same divergence points |
| New build against the old, same arm (Gemma) and Swift T8 records | identical everywhere: none of the 32 commits moved an answer |
| Swift T8, new/old, same session | 1.02 warm, 0.98 after a 16K prompt |
| Decode exit (12.0 warm and after 16K) | missed by both builds: new 11.86 / 11.14, old 11.63 / 11.31 |

Taught us: the rebase is clean, and by its registered branch it lands; the exit miss belongs to the machine (A35).

#### A34. E59: the logical batch is part of the reference

Conditions: all-VRAM Gemma on E54's ten long prompts (2,009-2,406 tokens), 64 tokens, at four (ubatch, batch) pairs,
plus the cache with the tier and P3 prefill at `-ub 512 -b 512`.

| Comparison | Records identical | Max difference |
|---|---|---|
| `-ub 512 -b 512` against `-ub 512 -b 2048` (the default) | 0/10 | 0.603 |
| `-ub 2048 -b 4096` against `-ub 2048 -b 2048` | 1/10 | 0.416 |
| cache + tier + P3 prefill at `-b 512` against all-VRAM at `-b 512` | 10/10 | 0 |
| `-ub 2048` against `-ub 512` (the contrast) | 0/10 | 0.668 |

Taught us: the reference is the (ubatch, batch) pair, not the ubatch alone. Marian already fixed both, so no earlier
result moves, but the default leaned on llama-server's own default `-b`. Profiles now record `-b`, and launch refuses
a different one or a batch variable in the environment.

#### A35. Row 17 and E60: the machine, not the code

Forensics on 27/10's own logs found two effects. In the afternoon drift the GPU side and the host side slowed
together, with the same bytes moved, the same CPU performance and GPU share, and the same free memory. In the 2x
slump the CPU was held below its base clock. E60 then ran 27/10 morning's own binary with its own arguments beside
today's build, interleaved:

| Arm | Warm decode (two runs) | After 16K |
|---|---|---|
| 27/10 morning's binary (build-rb), E51's T arguments | 11.92, 12.96 | 10.57, 10.55 |
| today's build, T8 | 12.98, 12.99 | 11.65, 11.87 |

Taught us: nothing we built slowed decode; the morning of 27/10 was the fast outlier. New GPU logs show the GPU idle
between decode bursts at 800-1,200 MHz of 3,105, about 30% busy, far under its power limit: the platform's power
state is the suspect, and testing it needs a driver setting changed on the reference laptop. The registered verdict
is still NO DECISION (a threshold missed by 0.08 tok/s). Since 28/10, `tools/prerun/prerun_check.py` and a GPU clock
log run beside every registered speed run, and comparisons are made inside one session.
