# Fable's ideas for Marian M1 (22/10/2026)

Design consult, read-only. I read `brief-ideas.md`, `RESEARCH-PACK.md` in full (including the 22/10 byte addendum), and
then the upstream llama.cpp sources the pack cites, plus Strata's dispatcher and look-ahead. I did not open the other
consultants' answers. Where a number is derived I show the arithmetic; where it is a guess I say "speculation".

## 0. What the code says that the pack only hints at

Five findings from the source that shape every idea below. Line numbers are the pinned llama.cpp tree at
`<research>\llama.cpp` and Strata 0.2.2 at `<strata-build>\src\Strata-0.2.2`.

1. **One hard GPU stall per cached layer per ubatch.** `llama_moe_cache::impl::copy` reads the selected IDs back with
   `ggml_backend_tensor_get_async` and then calls `ggml_backend_synchronize(backend)` (`src/llama-moe-cache.cpp:583-584`).
   The GPU has nothing queued behind that point (the slot-map `GET_ROWS` opened a new split, `src/llama-graph.cpp:2609-2619`),
   so for Swift that is 48 full pipeline drains per decode token, each followed by host work (SLRU plan, upload issue,
   slot-map upload) before the expert matmuls can start. The selective copier for larger batches does the same thing
   (`src/llama-context.cpp:2807-2808`). Nothing in the material times this on Windows (pack open question 2).

2. **Miss uploads come from pageable, possibly non-resident memory.** The fill copies from `w->data`
   (`src/llama-moe-cache.cpp:602`), which is the mmap of the GGUF wrapped as a CPU buffer (`src/llama-model.cpp:1937-1951`).
   CUDA only pins that region if `GGML_CUDA_REGISTER_HOST` is set (`ggml/src/ggml-cuda/ggml-cuda.cu:166-169`), which
   nobody sets, and which would pin all 33 GiB anyway. A `cudaMemcpyAsync` from pageable memory is a staged, host-blocking
   copy at a fraction of link speed, and if the page is not resident the copy page-faults to NVMe in the middle of the
   token. This is the mechanism behind the "4.86 tok/s after two pp4096 passes" collapse in BASELINE (pack §1).

3. **The budget split is by bytes, not by routing need.** Each layout group gets `size * host_bytes_group / host_bytes_device`
   (`src/llama-moe-cache.cpp:350`), so Swift's three groups (34 : 11 : 3 layers) each cache the same *fraction* of their
   experts. A frequency profile will want something else, and the bank sizes are fixed at construction, so the policy has to
   decide the group sizes before the banks exist.

4. **A kernel-path cliff sits at 9 tokens for the three IQ1_M layers.** For `MUL_MAT_ID`, quantised types take `mmvq` up to
   `get_mmvq_mmid_max_batch` rows (`ggml-cuda.cu:2080-2086`; `mmvq.cuh:3` sets the ceiling at 8; on Turing and later IQ2_S
   is capped at 7, `mmvq.cu:177-188`), then MMQ if the type is supported. MMQ has Q2_0, IQ2_XXS and IQ2_S (`mmq.cuh:63-88`)
   but **no IQ1_M**, so a 9-row verify window sends layers 8, 13 and 37 into the fallback path that does two
   `cudaStreamSynchronize` calls (`ggml-cuda.cu:2134-2156`) and switches CUDA graphs off for the whole graph
   (`ggml-cuda.cu:2738-2747`). Speculation windows must stay at 8 rows or fewer (7 drafted tokens) on this GGUF.

5. **Strata pays the same dependency but hides the host work.** Its dispatcher publishes the GPU's share first and starts
   PCIe fetches "beside the CPU's work" (`src/core/expert_source.cpp:4827-4961`), its look-ahead runs layer l+1's router on
   layer l's activation on a worker thread (`expert_source.cpp:3153-3212`) and only warms bytes, and its tail skip
   (`expert_source.cpp:4854-4876`) is the answer-changing part worth 13-16% (`docs/DETAILS.md:1899-1907`). Marian cannot
   take the skip as a default, so it must win the 13-16% elsewhere.

**Arithmetic I use throughout.** One decode token selects 480 experts, about 660 MiB under uniform routing (pack
addendum); the IQ2_S layout expert is 1,510,400 B. Dense per-layer weights are 2.98 GiB and are read every forward pass.
At a realistic 450-500 GB/s of VRAM bandwidth on the 4090 Laptop, an all-resident token costs about (2.98 + 0.64) GiB /
475 GB/s = 8 ms before launch overhead, so the ceiling on this laptop is of the order of **100-125 tok/s** (speculation
from bandwidth arithmetic; the hybrid attention and delta-net layers add some). Strata's 35.5 tok/s is 28 ms per token,
so roughly 20 ms of every Strata token is miss traffic, host orchestration and CPU-lane time. That 20 ms is the prize.

## 1. Ranked ideas

Ranking is by expected decode gain on the reference laptop (16 GB RTX 4090 Laptop, 32 GB RAM, Windows 11), measured
against an upstream `--moe-cache-mib` baseline that has itself not yet been measured on Swift (pack open question 1).
Percentages compound, roughly, when stacked; I give a stacked target at the end.

### Idea 1. Sync-free residency: resolve expert slots on the GPU, serve misses by zero-copy, admit between tokens

**What.** Replace the host round trip per layer with an in-stream GPU op. Keep upstream's banks, slot maps and bindings
(`llama-moe-cache.cpp:187-250`) but move the slot map to device memory and add one CUDA op, `moe_resolve(ids, slot_map_dev,
host_tier_base, bank, scratch)`: for each selected ID, if the slot is valid write it out; if not, copy the expert's bytes
from the **pinned, device-mapped host tier** (Idea 4) into one of a few scratch slots at the end of the bank with a
device-side gather (threads reading mapped host memory over PCIe), and write the scratch slot. The existing
`ggml_mul_mat_id(bank, cur, slots)` (`llama-graph.cpp:1714-1716`) then runs unchanged on identical bytes. No split, no
`ggml_backend_synchronize`, no slot-map upload, one CUDA graph per token.

Admission and eviction leave the critical path: the resolve op appends the token's miss list to a small device ring, the
host drains it once per token (one async download, no stall on the compute stream), runs the policy (Idea 2), and promotes
misses from scratch into real slots with device-to-device copies on a second stream, publishing the new slot map with a
tiny kernel at the token boundary. Experts that are not in the pinned tier (the NVMe tail) cannot be read by the GPU; for
them the resolve op spins on a mapped flag that the host sets after an unbuffered NVMe read into pinned staging (Strata's
spin kernels do the same wait on CPU rows, `DETAILS.md:1939`), with a bounded timeout so a stuck host cannot trip WDDM's
2-second TDR. With a decent RAM tier and Idea 5's prefetch that wait is rare.

**Effect.** Removes 48 × (drain + host plan + pageable staging + launch gap) per token. I have no Windows measurement of
the drain; typical WDDM stream-sync round trips are 50-150 µs with hardware-accelerated GPU scheduling on and 2-5× worse
with it off (speculation from experience, not from the pack). Adding the host plan and the upload issue gives 100-250 µs
per layer, 5-12 ms per token, against a 28 ms Strata token. **Decode +20% to +45%** over an upstream cache with the same
hit rate; prefill unchanged (the large-batch path keeps the selective copier); hit rate unchanged. Speculation until the
first experiment below is run.

**Exact or answer-changing.** Exact: same kernel, same bytes, same slot semantics.

**Where.** llama.cpp patch: one CUDA op (about 200-300 lines), the graph wiring in `build_moe_cache_slots` replacing the
`GET_ROWS` split, and a device-side slot map in `llama-moe-cache.cpp`. The policy thread and the NVMe path are the layer
on top (they only touch the cache through a small API: "promote these", "slot map version").

**Cost.** 8-12 days including the parity rig (Idea 10). Stage A (all host experts pinned, no NVMe tail) is 5-6 days and
already works on a machine whose RAM holds every expert.

**Smallest experiment that proves it wrong.** Instrument `copy()` with a steady clock around lines 582-614 and around the
whole callback, run 500 decode tokens with `--moe-cache-mib 9380` on Swift, and sum per token. If the 48 callbacks total
under 1.5 ms per token on this laptop, the idea is not worth its build and should be dropped in favour of Idea 2 alone.
Second check: `GGML_CUDA_GRAPH_DEBUG=1` to confirm whether CUDA graphs are already off at decode because of the splits.

**Source.** Mine, assembled from three known pieces: Strata's GPU plan that is published first and consumed by GPU groups
(`expert_source.cpp:4886-4961`), CUDA mapped pinned memory (zero-copy) which is standard, and the TinyLFU/async-admission
pattern from Idea 2.

### Idea 2. Ranked warm start, protected region, TinyLFU admission, and group budgets by marginal gain

**What.** Keep upstream's SLRU as the *adaptive* region but put a *ranked* region in front of it, sized from a stored
routing profile: at construction, Marian supplies per-group slot counts derived from the top-N `(layer, expert)` pairs
of the profile (so the 34 : 11 : 3 group split follows routing heat instead of bytes), fills those slots before the first
token, and marks them pinned until the policy thread demotes them. The remainder stays SLRU, but admission into it is
gated by a TinyLFU doorkeeper (a 4-bit count-min sketch over `layer*512+expert`, halved periodically): a one-off expert
seen during a scan or a long prompt does not evict a repeat-use expert on its first miss. The policy thread also maintains
the live frequency table and writes it back as the next start's profile, Strata-style (resident first, then heat), via
temp-file-and-rename.

**Effect.** Strata reached 85.4% from 6,836 pairs (27.8%) with a frequency rank plus swaps (pack §1). A pure recency SLRU
on a heavy-tailed routing distribution usually lands several points lower; my estimate for upstream SLRU at the same
9.16 GiB is **70-80%** (speculation; open question 1 measures it). Ranked plus TinyLFU should reach **86-90%** in-sample
and about 84-88% on a held-out workload. W-TinyLFU beats LRU-family policies by 3-8 hit-rate points on Zipfian traces in
the Caffeine results (borrowed estimate). Going from 75% to 87% cuts misses per token from 120 to 62, roughly 85 MiB of
PCIe traffic and, under upstream's current path, 60 fewer per-token fills. **Decode +15% to +30%** on its own; more if
Idea 1 is not built because every avoided miss also avoids host work. Prefill: small gain, the large-batch copier takes
more experts from the bank (`llama-context.cpp:2835-2846`). Warm start also removes the cold first minute that Strata's own
run shows (16-23 tok/s in run one, 46-54 in run three, pack §1).

**Exact or answer-changing.** Exact.

**Where.** Patch: a policy interface in `llama-moe-cache.cpp` (SLRU becomes one implementation; `plan()` grows a
"pinned" bit and a doorkeeper check), a constructor argument for per-group slot counts and an initial fill list, a getter
for the live frequency table. Upstreamable as "warm fill from a frequency file". Profile files, ranking, group sizing and
the write-back live in the layer on top.

**Cost.** 3-5 days.

**Smallest experiment that proves it wrong.** Run upstream SLRU at 9,380 MiB on three 2,000-token generations (code,
prose, multilingual) and read the `ubatch <= 8` line at destruction (`llama-moe-cache.cpp:625-628`). If it already reports
84% or better, ranking buys at most one or two points and only warm start survives. Second, replay the same routing
trace offline through SLRU, ranked, and ranked plus TinyLFU: a 30-line simulator settles the policy question before any
C++.

**Source.** Strata's STRP profile and learned ranking (`src/core/expert_cache.hpp:40-73`, `expert_cache.cpp:143-211`),
W-TinyLFU (Einziger, Friedman and Manes, 2017), my own for the group sizing and the two-region bank.

### Idea 3. Bounded speculation: n-gram first, a quantised MTP head second, window capped at 8 rows

**What.** Turn on upstream's drafting (`common/speculative.cpp` lists simple, EAGLE3, MTP, DFlash, DSpark and n-gram
choices, pack §4) in this order of preference: (a) prompt-lookup / n-gram drafting, which costs no memory and is exact in
the output distribution when the verifier uses proper rejection sampling (identical tokens under greedy); (b) the
separate qwen4exp MTP head that upstream can load as its own tensors (`src/models/qwen4exp.cpp:178-180`), requantised to
Q4 or lower so that Strata's 5.2 GB draft becomes about 1.3-1.8 GB of VRAM; (c) a small same-vocabulary dense draft as a
fallback. Cap the verify window at **8 rows** (7 drafted tokens) because of finding 4 above; Marian's calibrator checks the
per-layout `mmvq`/MMQ eligibility and lowers the cap on GPUs where the tables differ (`mmvq.cu:150-194`).

**Effect.** Each forward pass reads 2.98 GiB of dense weights (about 6 ms), does the 48 stalls of finding 1, and launches a
hybrid model's worth of kernels; all of that is paid once per window, not once per token, so the gain is (fixed share) ×
(accepted - 1) / accepted. Miss bytes scale with distinct experts in the window, roughly linearly in tokens, so they do
not amortise; that is why the gain is larger before Idea 1 lands and smaller after. Strata's MTP accepted 93 of 130
drafts in its first request (pack §5). With 2.5 accepted tokens per window: **decode +40% to +80% on code and agent
workloads, +10% to +30% on prose** (speculation; the prose figure assumes n-gram acceptance around 1.3 per window). The
expert cache interacts in three places: the `uniq > n_slots` abort (`llama-moe-cache.cpp:124-126`) is safe at 8 × 10 = 80
distinct experts; `victim()` already protects every expert of the current request (`:88-99`); the per-token stats bucket
flips to `ubatch > 8` only above 80 IDs (`:608`), so keep telemetry per row, not per bucket.

**Exact or answer-changing.** Exact under greedy, exact in distribution under sampling with upstream's rejection sampler;
confirm with the parity rig (Idea 10) because the pack notes the sampling PRs do not themselves establish token identity.

**Where.** Layer on top (server flags, draft management, the window cap) plus a tiny patch if the cache's `max_batch`
or stats need the row-level view.

**Cost.** 2-4 days for n-gram and the cap; 3-5 more for the quantised MTP head if the GGUF converter needs coaxing.

**Smallest experiment that proves it wrong.** Run `llama-server` with and without n-gram drafting on three fixed prompts
and read accepted-per-window. Below 1.3 on prose and 2.0 on code, speculation goes opt-in. Then run a 9-row window on
Swift with `GGML_CUDA_GRAPH_DEBUG=1`: if CUDA graphs do not switch off, finding 4 is wrong and the cap can rise.

**Source.** Upstream's speculative module; Strata's MTP and prompt-lookup (`docs/HOW_IT_WORKS.md:35-46`); speculative
decoding (Leviathan et al.; Chen et al. 2023); prompt lookup decoding (Saxena, 2023). The 8-row cap is mine, from the code.

### Idea 4. A ranked, bounded, pinned RAM tier with an unbuffered NVMe tail and pinned staging

**What.** Stop treating the OS page cache as the RAM tier. From the same routing profile, the next-hottest experts after
the VRAM set go into a `cudaMallocHost` arena sized from measured headroom (available physical RAM and Windows commit
minus a margin, the check Strata makes at `expert_source.cpp:3512-3564`), filled by **unbuffered** reads
(`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) so the page cache is not doubled up. The coldest experts stay on NVMe and are
read on demand, also unbuffered, into a pinned staging ring (a few dozen 1.44 MiB slots). All uploads to the GPU then
come from pinned memory at full link speed and are truly asynchronous. Prefer `cudaHostRegister` on the hot ranges of the
mmap itself where the platform allows it (saves the copy; speculation that WDDM accepts file-backed registrations, so the
arena is the fallback). The three tiers are assigned by a single greedy over the profile with two cut points, because
each tier has a constant cost per expert: VRAM about 0, RAM 1.44 MiB / link rate, NVMe latency + 1.44 MiB / NVMe rate,
all from the probes in Idea 7.

**Effect.** On the reference laptop, 33.02 - 9.16 = 23.9 GiB of experts are not in VRAM and only about 19 GiB of RAM was
free (pack §1), so today's mmap thrashes after any long prompt. A 12-14 GiB pinned tier holds the next 8,500-9,700 pairs;
on Strata's heat curve those take most of the remaining selections, leaving perhaps 2-4% of selections (10-20 per token)
for NVMe (speculation; the exact curve comes from the profile). Pinned versus pageable upload: expect 20-25 GB/s versus
6-12 GB/s on a PCIe 4.0 x16 laptop link, so miss transfer time halves (**decode +10% to +25%** steady state), and the
post-prefill collapse (10.59 to 4.86 tok/s in BASELINE) disappears because nothing on the decode path page-faults. Prefill:
the layer-ahead read in Idea 6 uses the same unbuffered reader. On a 16 GB RAM machine the tier shrinks to 4-6 GiB and
decode becomes NVMe-bound at roughly 30-40 reads per token, 15-20 tok/s at best with deep queues (speculation); the job
there is not to collapse. Required by Idea 1 (device-mapped bytes).

**Exact or answer-changing.** Exact.

**Where.** Layer on top for the tier manager, the readers and the budget; a small patch so the cache's fill source can be
"a pointer the tier gives me" instead of `w->data`, and so experts whose host copy is the arena are still recognised as the
same tensor (bindings are keyed by the host tensor, `llama-moe-cache.cpp:241`).

**Cost.** 4-6 days. Windows unbuffered I/O alignment rules (sector-aligned offsets and lengths) cost one of those days;
GGUF expert offsets are 32-byte aligned, so reads over-fetch to the sector and trim.

**Smallest experiment that proves it wrong.** Copy 1,000 random experts host-to-device from the mmap'd tensor and from a
`cudaMallocHost` copy of the same bytes, 1.44 MiB each, and compare bandwidth and host blocking time. If pinned is within
15% of pageable on this driver, the tier is only a page-cache fix and ranks lower. Then run a 16K prefill followed by
decode with and without the tier and watch `Pages Input/sec`.

**Source.** Strata's arena and budgeted complement (`src/core/expert_source.hpp:724-749`, `expert_source.cpp:3493-3564`),
Maya's bounded 7.03 GB pinned tier (pack §6), my own for the per-expert registration and the single greedy with two cuts.

### Idea 5. Router look-ahead that prefetches into VRAM and into pinned staging, one layer ahead

**What.** Borrow Strata's RouterLookahead (`expert_source.cpp:3153-3212`): apply layer l+1's router (a 512 × 2560 matrix,
2.6 MB in BF16) to layer l's hyper-connection-mixed state and take the top-k as a prediction. Then do two things Strata
does not: (a) on the GPU, on a second stream, run the predicted misses through the same zero-copy gather as Idea 1 into
the bank's prefetch slots, so by the time the exact router of layer l+1 runs, most misses are already in VRAM; (b) on the
host, read the prediction from a mapped buffer without stalling the stream and issue unbuffered NVMe reads for predicted
tail experts into pinned staging. Prediction never changes which experts compute; a wrong guess only wastes bandwidth,
and the policy thread tracks precision per layer and switches the look-ahead off for layers where it is poor.

**Effect.** Pre-gated MoE style predictors report 70-90% top-k precision one layer ahead on dense-residual models;
hyper-connections mix four streams, so I would expect the lower end (speculation). At 70% precision, 70% of miss bytes
move off the critical path; with Idea 1 in place that is **decode +5% to +15%**, without Idea 1 it is +10% to +25%
because each hidden miss also removes a fill from the critical host section. Prefill: none. Hit rate: unchanged by
definition (a prefetched expert that is used is a hit the policy did not have to earn). On the NVMe tail the gain is much
larger in tail-heavy configurations (16 GB RAM, 8 GB GPU): a predicted NVMe read gets about 0.6 ms of lead time, roughly
one layer's compute at 35 tok/s, and queue depth becomes usable.

**Exact or answer-changing.** Exact (bytes only).

**Where.** Patch: a small GPU matmul plus top-k inserted in the graph per layer on a side stream and the prefetch gather
from Idea 1; the host side and the per-layer on/off live in the layer on top.

**Cost.** 5-8 days on top of Idea 1.

**Smallest experiment that proves it wrong.** Offline, from a 2,000-token routing trace plus the saved per-layer router
inputs, compute precision@10 of "layer l+1's router applied to layer l's state" against the real selection. Below 50% the
idea mostly wastes PCIe and should be shelved. Costs an afternoon with a Python script and a GGUF reader.

**Source.** Strata's look-ahead (warm bytes only, `expert_source.hpp:296-325`); Pre-gated MoE (Hwang et al., 2023),
MoE-Infinity's activation-aware prefetch; mine for the VRAM prefetch slots and the GPU-side variant.

### Idea 6. Opt-in answer-changing lanes: CPU miss lane, route-tail skip, and an iGPU lane

**What.** Three switches, each off by default and each only eligible for default after a blind quality test (teacher-forced
KL against the exact path on a fixed prompt set, plus task checks, the design Strata used in `DETAILS.md:1880-1897`).
(a) **CPU miss lane**: compute a measured share of each token's misses on the CPU from the pinned tier while the GPU does
the hits, Strata's design, but with the share fitted on this machine from the two timings Marian already has (per-expert
CPU time from Idea 7, link rate from Idea 7) instead of Strata's constant 1.85 fitted to three other machines. In ggml
terms it is a second `mul_mat_id` on the host expert tensor with a masked ID set, added to the bank result; the scheduler
will split there, which is acceptable for an opt-in. (b) **Route-tail skip** exactly as Strata (`expert_source.cpp:4854-
4876`), gated off above 90% hit rate as Strata learnt (`DETAILS.md:1875-1877`). (c) **iGPU lane**: the i7-13700HX's Intel
iGPU reads system RAM at 50-70 GB/s, more than the PCIe link; a Vulkan or SYCL backend could compute misses from the pinned
tier. The cache constructor already admits `GGML_BACKEND_DEVICE_TYPE_IGPU` devices (`llama-moe-cache.cpp:259`). Speculation,
and the iGPU's low compute may make it slower than PCIe for 1.44 MiB experts; it is the kind of thing the brief asked for
and nobody in the material has tried.

**Effect.** CPU lane: Strata's own data has the pool at 32-115 µs per expert depending on the CPU (`DETAILS.md:1928-1932`);
on a 16-core i7 with 60 misses per token after Idea 2, taking 20-30% of them on the CPU saves roughly 1-2 ms of link time,
**decode +5% to +12%** (estimate from Strata's measured share effects). Tail skip: Strata measured 1.13-1.16× at 75-90% hit
rates (`DETAILS.md:1904-1906`), so **+13% to +16%** while hit rate stays below 90%, falling to nothing above. iGPU: unknown;
I would not promise more than +0-10% (speculation).

**Exact or answer-changing.** All three change answers: CPU and GPU round differently (Strata's floor KL 0.0006-0.0013),
the skip zeroes a contribution without renormalising, the iGPU's kernels round differently again.

**Where.** CPU lane and iGPU lane are graph-build patches (moderate); the skip is a few lines in the resolve or plan step.
Blind-test harness is the layer on top.

**Cost.** CPU lane 6-10 days; skip 1 day plus the test; iGPU 3 days to a yes/no measurement.

**Smallest experiment that proves it wrong.** For the CPU lane: time one IQ2_S expert (gate + up + down) on 14 threads with
the existing CPU `mul_mat_id`; if it exceeds 1.44 MiB / link rate (about 70 µs at 20 GB/s), the lane cannot beat the link
and only matters where the link is slow (x8 laptops). For the skip: Strata's numbers already exist; the Marian experiment
is the blind KL run. For the iGPU: `llama-bench` a single expert-sized `mul_mat_id` on the Vulkan backend of the iGPU.

**Source.** Strata (`DETAILS.md:1858-1941`); the iGPU lane is mine.

### Idea 7. Self-calibration that measures the cliffs instead of guessing them, and a two-file profile model

**What.** At first start, Marian runs a short probe suite (about 90 seconds, no model load needed for most of it) and
stores a **machine profile**: (1) VRAM cliff finder: allocate device buffers in 256 MiB steps and after each step time a
64 MiB device-to-device copy and a small matmul; the step where bandwidth collapses by more than 3× is WDDM spilling to
system memory (the 16,018 to 16,040 MiB cliff in BASELINE, pack §1); safe budget = last good step minus 512 MiB and minus
the display's share; (2) link rate, pinned and pageable, host-to-device and back, at 1.44 MiB granularity; (3) sync
round-trip cost (event record + wait, 1,000 reps) and whether hardware-accelerated GPU scheduling is on (registry
`HwSchMode`), with a one-line recommendation to the user if off; (4) RAM: total, available, commit limit, and a pin test
in 1 GiB steps until available falls below a floor; (5) NVMe: unbuffered 1.44 MiB random reads at queue depth 1 and 8;
(6) CPU: per-expert `mul_mat_id` time per layout for the opt-in lane; (7) CUDA-graph eligibility per layout per window size
(finding 4), by building a tiny graph and asking the backend. Separately, a **model routing profile** keyed by a content
hash of the tensor-info table and expert offsets (not M0's first 64 MiB plus size, pack §8), holding the frequency table
of Idea 2. The two are deliberately separate: routing heat is a property of model and workload and is **shareable** across
machines (ship a starter profile with Marian; exchange them freely); the machine profile is local. A **plan** is a function
of the two plus the requested context: VRAM budget for the bank and its group sizes, pinned tier size, NVMe tail, ubatch
for prefill, speculation window.

Re-measure: free VRAM and RAM at every start (cheap); the cliff and link probes when the driver version or the GPU
changes; after any observed collapse (decode below 0.6× of the plan's expectation, M0's rule) or any CUDA OOM; and when the
laptop's power state changes (battery versus mains changes the GPU clock, which changes the all-resident floor and so the
right CPU/GPU split). Live, a cheap VRAM watchdog (`cudaMemGetInfo` every few seconds) shrinks the bank's SLRU region by
whole slots if another process starts eating VRAM, and grows it back later; Strata only does this on API request
(`VRAM_ELASTIC.md`, pack §5).

**Effect.** No speedup by itself. It is what keeps every other idea off the 3.56 tok/s cliff and off Maya's 95-minute
calibration (pack §6). Decode 0%, prefill 0%, hit rate 0% directly; it decides the budgets that produce the gains above.

**Exact or answer-changing.** Exact.

**Where.** Layer on top (Python M0 plus a small C++ probe binary that links ggml, or the existing `test-backend-ops`
harness with new cases).

**Cost.** 5-8 days.

**Smallest experiment that proves it wrong.** Predict the safe bank budget from the cliff finder, then run decode at that
budget and at budget + 512 MiB; if neither collapses, the finder is too conservative; if the predicted-safe one collapses,
it is wrong. One evening.

**Source.** M0's `machine.py`, `profile.py` and `plan.py` (pack §8) for the skeleton; Strata's RAM and commit checks; the
cliff finder, the sync probe and the two-file split are mine.

### Idea 8. Prefill without re-reading the expert set: layer-major mega-ubatch, bank lending, layer-ahead reads

**What.** llama.cpp already computes a ubatch layer-major (all tokens through layer l before layer l+1), and the selective
copier uploads a layer's selected experts once per ubatch (`llama-context.cpp:2822-2857`). For prompts beyond about 300
tokens a layer's selection covers nearly all 512 experts, so each ubatch pass costs one full 33 GiB walk over the expert
set, from a host that cannot hold it, so from NVMe. The number of passes is the whole story: Strata's 8,192-token chunk
(pack §5) is two passes at 16K; stock `-ub 512` is 32. Marian's prefill plan: (a) choose the largest ubatch the VRAM plan
allows, up to the whole prompt; (b) **lend the bank**: for prompts above a calibrated threshold, release the SLRU region
of the expert bank (keep the ranked region) to the compute allocator for the duration of the prefill, then re-fill it from
the pinned tier and from the last layer's uploaded weights (device-to-device, nearly free) when decode resumes; (c)
**layer-ahead read**: while layer l's experts upload, the tier manager issues an unbuffered sequential read of layer l+1's
expert tensors (contiguous in the file) into the pinned staging, so the upload never faults; (d) keep the selective copier's
device-to-device path for ranked-region experts (`llama-context.cpp:2835-2846`), which already saves about 28% of the bytes.

**Effect.** At 16K with one pass: 33 GiB from NVMe at 3-3.5 GB/s is about 10 s, PCIe at 20 GB/s about 1.7 s, MoE compute
of the order of 2-3 s on this GPU, attention and the delta-net layers a few seconds more, all partly overlapped. Call it
15-25 s, which is **650-1,100 tok/s at 16K against Strata's 193** and 4K at roughly 300-500 tok/s against 155
(speculation; the two measured stock points, 113.8 tok/s at `-ub 512` and 130.4 at `-ub 512 -ncmoe 34`, do not isolate
the pass count because they sat on the VRAM cliff, pack §1). Intermediate tensors for a 16K ubatch are large (the
`[n_ff, 10, n_tokens]` activations and the `[n_embd, 10, n_tokens]` expert output are about 0.4 and 1.7 GiB in F32), so
without bank lending the mega-ubatch does not fit on 16 GB; with lending it does. Decode: a re-fill after a long prefill
costs about 0.3-0.5 s once, which is why the threshold matters (chat turns under it never lend). Hit rate: the ranked
region survives prefill untouched, so decode after a long prompt starts warm rather than evicted, which is exactly the
case upstream's `max_batch` rule was protecting (`llama-moe-cache.cpp:502-503`).

**Exact or answer-changing.** Exact at a given ubatch. Changing the ubatch can change last bits because the CUDA backend
picks different kernels by row count (finding 4), which is a configuration choice llama.cpp users already make; Marian
should record the chosen ubatch in the plan so a parity run can match it, and must not call a cross-ubatch comparison
"identical" without checking.

**Where.** Bank lending is a patch (the bank buffer becomes two allocations, the SLRU half resizable through the scheduler's
reserve); the prefill plan, the thresholds and the layer-ahead reader are the layer on top.

**Cost.** 4-7 days.

**Smallest experiment that proves it wrong.** With the cache off and `-ncmoe` raised so VRAM has 3 GiB of headroom, run
`pp4096` at `-ub 512` and `-ub 4096` and record wall time and disk bytes read. If the time does not fall by anything like
the pass ratio, the host read is not the bottleneck and the gain is mostly from fewer kernel launches, which is smaller.

**Source.** Strata's chunking and its note that Windows unbuffered reads keep the chunk (pack §5); the PLE row prefetch
pattern in upstream for the layer-ahead idea (`qwen4exp.cpp:1292-1301`); bank lending is mine.

### Idea 9. PLE row care: a hot-row cache and draft-token prefetch for the 26.82 GiB table

**What.** Swift's per-layer token embedding table is lazily read by row (`qwen4exp.cpp:195-215`, `TENSOR_READ_LAZY`), and
each decoded token gathers its rows for all layers from a table that is bigger than RAM. Under the memory pressure the
baseline shows, every token's rows are candidate page faults to NVMe, about 100-200 µs each, outside any cache Marian
controls. Keep a small pinned cache of PLE rows for the hottest token IDs (a token-frequency table is cheap to learn and
to ship), and when speculation is on, prefetch the rows of the drafted tokens before the verify pass (upstream already
calls `llama_prefetch_rows` for the current rows, `qwen4exp.cpp:1295-1298`; extend the ID list to the draft).

**Effect.** **Decode +1% to +4%** (speculation; depends on page-cache state), more on 16 GB RAM machines where the table
competes with everything else. Prefill: a long prompt's rows are sequential-ish by position, not by ID, so little change.

**Exact or answer-changing.** Exact.

**Where.** Layer on top for the hot-row set; a 20-line patch to pass draft IDs into the prefetch.

**Cost.** 1-2 days.

**Smallest experiment that proves it wrong.** Count hard page faults attributable to the PLE mapping per decode token
(ETW or `Pages Input/sec` with and without a `--no-ple-prefetch` toggle). Under 5 per token, skip it.

**Source.** Mine, from reading the qwen4exp input path.

### Idea 10. Per-token telemetry and a parity rig that can actually say "identical"

**What.** The pack's open question 9 asks for a synchronised per-token series; nothing in upstream, Strata or Maya gives
one. Marian records, per token or per verify window: hits by tier, misses by tier, bytes moved by link, stall µs in the
resolve wait, prefetch precision, VRAM and RAM free, and wall time, into a ring buffer exposed on the local `/metrics`
(M0 already binds `llama-server` to 127.0.0.1). The parity rig has two legs: (a) on a MoE model that fits entirely in 16 GB
(a 30B-A3B class GGUF at 2-3 bits), run the all-resident reference and the cached engine with a deliberately tiny bank
forcing misses every layer, and require token-ID identity over 4,096 generated tokens and logit identity at sampled
positions; (b) on Swift, where no all-resident reference exists on this laptop, require identity between the cached path
and stock's selective-copy path (same kernels, both GPU) over the same 4,096 tokens, at the same ubatch. Every opt-in
switch in Idea 6 is then measured against leg (b) with teacher-forced KL on a fixed 28-prompt set, blind to the switch.

**Effect.** None directly. It is the "every test records its decision" discipline the brief's constraints need, and it
turns the warm-up, routing-drift and page-cache noise that confused the Strata and stock runs (±10% decode between
identical runs, pack §1) into separable series.

**Exact or answer-changing.** Not applicable; it is the instrument.

**Where.** Layer on top, plus a few counters in the cache that already half exist (`llama-moe-cache.cpp:224-236`).

**Cost.** 2-4 days.

**Smallest experiment that proves it wrong.** Run leg (b) on stock at the pin with `--moe-cache-mib` set versus unset,
before any Marian change. If those two already differ in token IDs, the rig has found a real exactness problem in
upstream's cache path (or a non-determinism in the kernels) and the whole "exact by default" claim needs a footnote.

**Source.** Mine; the blind-KL design is Strata's (`DETAILS.md:1880-1897`).

## 2. The six numbered points, answered directly

**1. Per-expert residency inside the stacked `ffn_*_exps` tensors.** Build on upstream's cache, do not replace it. Its
bank-and-slot-map storage, bindings keyed by the host tensor, layout grouping, budget apportioning across devices and the
large-batch device-to-device copier are sound and already upstream (`llama-moe-cache.cpp:187-250, 304-365, 514-547`).
What to change is narrow: the policy (Idea 2: ranked region, TinyLFU doorkeeper, SLRU remainder, group sizes from the
profile), the lookup path (Idea 1: resolve on the GPU, no split, no drain), the fill source (Idea 4: pinned), and the
telemetry (Idea 10). From wherever SLRU lands to 85% and beyond: warm start from a stored profile is the big step, the
doorkeeper adds a few points, and per-layer budgets fall out of a global rank over `(layer, expert)` pairs automatically
because a layer with concentrated routing simply has more pairs above the cut. Predictive prefetch (Idea 5) does not raise
the hit rate; it lowers the cost of a miss, which is the thing that matters.

**2. Misses.** Upload over PCIe is exact and should remain the default miss path; the fixes are to upload from pinned
memory (Idea 4), to hide the host work (Idea 1) and to start the transfer early (Idea 5). CPU compute is opt-in (Idea 6a)
and should be sized from this machine's measured per-expert time, not Strata's constant. Pinned staging is a ring of a few
dozen expert-sized slots, enough for one layer's misses plus one layer of prefetch. On a machine whose RAM cannot hold all
experts, the RAM tier is a ranked, budgeted pinned arena, and the tail is read unbuffered so the OS page cache is never
the tier (Idea 4). NVMe reads that do not thrash: unbuffered, sector-aligned, issued one layer ahead from the look-ahead
prediction, queue depth from the probe.

**3. MTP and other speculation.** Idea 3. Use upstream's n-gram drafting first (free, exact), add the separately-loadable
MTP head requantised small, cap the verify window at 8 rows because the three IQ1_M layers fall into the synchronising,
graph-disabling fallback at 9 (finding 4). A window of T tokens touches up to 10T experts per layer; the cache handles 80
distinct experts per layer fine, misses scale with T, fixed costs do not, so speculation pays most while the fixed costs
are large (before Idea 1) and on repetitive workloads.

**4. Prefill.** Idea 8. The pass count over the expert set is the lever; the ubatch should be as large as the VRAM plan
allows, the bank's adaptive half lent to the compute allocator for long prompts, the next layer's experts read sequentially
and unbuffered one layer ahead, and the ranked region kept so decode after a long prompt starts warm. CPU share for
short prompts stays opt-in (Idea 6).

**5. Self-calibration and adaptation.** Idea 7. Measure the cliff, the link, the sync, the RAM headroom, the NVMe and the
CPU at first start; re-measure on driver, power or collapse events; keep machine and routing profiles in separate files
so routing profiles travel. Avoid the WDDM cliff by finding it rather than by a fixed margin, by keeping a live watchdog
that shrinks the adaptive region, and by telling the user about "Prefer No Sysmem Fallback" and hardware-accelerated GPU
scheduling once, plainly.

**6. What the material has not tried.** The sync-free GPU-side resolve with zero-copy misses (Idea 1), the TinyLFU
doorkeeper in front of the SLRU (Idea 2), bank lending during prefill (Idea 8), the PLE hot-row cache (Idea 9), the iGPU
lane (Idea 6c), and the cliff finder as a measurement rather than a margin (Idea 7). Honest estimates and the evidence
each rests on are in the idea entries; the first two are the ones I would bet on, the iGPU lane is the one I would bet
against but would still spend three days measuring, because nobody has.

## 3. Stacked target and the order to build

Against an unmeasured upstream baseline I will guess at 20-28 tok/s with `--moe-cache-mib 9380` (speculation: it has the
hit-rate deficit, the 48 drains and the pageable uploads all at once). Ideas 2 + 4 are the cheapest and together should
give +30% to +60%; Idea 1 adds +20% to +45% on top; Idea 5 another +5% to +15%. That lands the **exact path at roughly 45-60
tok/s** on the reference laptop against Strata's 35.5 answer-changing, with Idea 3 taking code and agent sessions to 70-90
(all speculation, bounded above by the 100-125 tok/s all-resident floor). On the 8 GB GPU with 32 GB RAM the bank holds
only 1,500-2,000 pairs; hit rate perhaps 45-55%, PCIe becomes the limit at about 240 misses × 1.44 MiB per token, so
15-25 tok/s, and the CPU lane earns its keep there. On 16 GB RAM the aim is 10-20 tok/s without collapse.

Build order, by information gained per day: Idea 10 (the instrument) and the three one-evening experiments (time the
drain, SLRU hit rate at 9,380 MiB, pinned versus pageable copy) in week one; Idea 2 and Idea 4 in weeks two and three; a
decision on Idea 1 from the drain measurement; Idea 3's n-gram switch any time after the parity rig exists; Idea 8 when a
long-prompt workload is on the table; Idea 7 grows alongside all of it. Everything in Idea 6 waits for a blind test.

## 4. Things I would not do

Do not pin the whole mmap (`GGML_CUDA_REGISTER_HOST`): it reproduces Strata's 0.24 GB free RAM. Do not let the ubatch
climb toward the VRAM cliff without the finder; the baseline's `-ub 2048` result (53.5 tok/s) is a cliff artefact, not a
property of large ubatches. Do not take `ggml_backend_dev_memory`'s free figure as a budget on Windows; it does not know
about WDDM's spill. Do not make any CPU-lane, tail-skip or prefill-share default on the strength of a KL number alone; the
brief's bar is a blind quality test, and Strata's own tables show the floor and the effect are within a factor of ten of
each other. And do not report a hit rate from upstream's destruction log as if it were per-token; it is two buckets split
at 80 IDs (`llama-moe-cache.cpp:608`), which a 9-row verify window straddles.
