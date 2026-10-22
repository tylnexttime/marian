# Marian M1 design: per-expert VRAM residency on upstream llama.cpp

| | |
|---|---|
| Version | **v2**, 22/10/2026: v1 plus the foreign critique's 24 items folded in (section 14 lists every change) |
| Author | Claude Tyl (Claude Tyl, the reference laptop), for Chris |
| Decisions it implements | Chris, 22/10/2026: thin fork on the latest llama.cpp; exact by default, answer-changing speed-ups opt-in and measured blind; built here; self-calibrating to any machine |
| Upstream pin | llama.cpp `a37cc46104eb50900d5ab4a9ba02a0abdac039db` (21/10/2026, master) |
| Evidence | `RESEARCH-PACK.md` (Sol, cited, plus my addendum on expert bytes); the three idea files in `ideas/` |
| Status | Design only. Nothing here has run on the new pin. Every number marked *estimate* is arithmetic or a consultant's judgement, not a measurement |

**Credit tags:** **[UP]** upstream llama.cpp (Aman Gupta, Evan Huus, Georgi Gerganov and others, MIT) · **[ST]**
Strata (Niko1221 and contributors, MIT) · **[MY]** Project Maya (Peasant Smith / mw00 and needmorevram, MIT) ·
**[M0]** Alder's Marian M0 · **[CT]** Claude Tyl, written before the consults
(`ideas/claude-tyl-ideas-before-consult.md`) · **[AS]** Astra (`ideas/astra-ideas.md`) · **[FB]** Fable
(`ideas/fable-ideas.md`) · **[LIT]** published work, named. Where several sources reached an idea independently, all
are named, the earliest first.

## 1. The design in one paragraph

Two weeks before this design, upstream llama.cpp merged its own GPU cache for MoE experts kept in host memory
(`--moe-cache-mib`, `src/llama-moe-cache.cpp`, #29887, multi-GPU #30112, SLRU #30637) [UP]. It already has the
plumbing Marian needed: per-expert slots in banks shaped like the stacked `ffn_*_exps` tensors, a per-layer slot map,
`MUL_MAT_ID` on the bank with slot ids, and selective copying for large batches. It computes **every expert on the
GPU** (a miss is uploaded, then computed), so its output should not depend on what is cached: that is Marian's exact
mode. What it lacks is what made Strata reach 85%: a routing-frequency ranking, a warm start, learning that persists,
budgets by need, a managed RAM tier, prefetch, and a machine-fitted plan. It also pays a host round trip per layer
and uploads misses from pageable, possibly non-resident mmap pages [FB, verified below]. **So Marian M1 is a small
patch set on `llama-moe-cache.cpp` (trace, policy, warm start) plus a calibration layer on top (M0 extended), not a
new cache.** The objective is not the VRAM hit rate alone but **the exposed stall time per token across VRAM, RAM and
NVMe** [AS]: an 85% hit rate whose misses fault from NVMe loses to an 82% hit rate whose misses come from pinned RAM.

## 2. Goals, targets and non-goals

**M1 goal (Chris):** the VRAM expert hit rate on the Swift 1.5 IQ2_XS files on the reference laptop goes from about
33% (stock whole-layer placement, `-ncmoe 32`: 16 of 48 layers resident, so arithmetic, not measured) to about **85%**
per decode token on held-out prompts, in exact mode, with a calibration that would make the right choice on any
machine from that machine's own measurements.

**The 85% is defined precisely** (critique brief challenge): per single-token decode step, **lookup hit rate** =
(router selections whose expert was resident in the bank before that step's fills) / (`n_layer` x `n_expert_used`,
480 on Swift), counted per row, so a duplicate expert across rows counts once per row (C-06; upstream's `plan()`
counts distinct experts, `llama-moe-cache.cpp:117-136`, and is not used for this). Reported as the median and 10th
percentile over the held-out decode set, at a stated budget, warm state (section 9.3) and `n_ubatch`, beside the
byte-weighted hit rate. Neither stock's 33% (arithmetic) nor Strata's 85.4% (an aggregate under different settings)
is the same measurement, so both are context, and M1 compares against arm (b) measured the same way.

**Targets:** M1 decode **>= 12 tok/s floor, aim 20+** (`project_marian_decisions_2026-10-22`). Strata's 35.5 tok/s
median uses MTP, route-tail skip and a CPU/GPU split, so it is not an exact-mode figure. The consultants' stacked
*estimates* for the exact path after M2-M3 are 35-50 tok/s [AS] and 45-60 tok/s [FB], against an all-resident ceiling
of about 100-125 tok/s on this GPU (3.6 GiB of weights per token at 450-500 GB/s) [CT, FB]. Upstream's SLRU baseline
is unmeasured; Fable guesses 20-28 tok/s [FB]. M1.0 replaces all of these guesses with numbers.

**Reference laptop** (measurement only, never tuning): the reference laptop, RTX 4090 Laptop 16 GB (16,376 MiB), i7-13700HX AVX2,
32 GB DDR5-4800, Windows 11, model on the E: NVMe. RAM upgrade approved 22/10; after it, the 32 GB profile is kept as a
separate machine class.

**Supported matrix, not "any machine"** (C-17, brief challenge): a ballast allocation and a RAM cap on this laptop
test **capacity only**; they keep the 4090's kernels, its PCIe 4.0 link, Windows I/O and this NVMe. M1 therefore
claims only what it measures: (1) the reference laptop; (2) the same laptop with an 8 GB VRAM ballast and a 16 GB RAM
cap, labelled "capacity simulation". Any other class (a real 8 GB GPU, Linux, PCIe 3.0, a slow drive, another quant
layout) is **unsupported until measured on real hardware**, and the calibration's job there is to refuse or degrade
honestly (say what it predicts and why it will be slow), never to promise. Real-hardware validation is M1.4 below,
outside the M1 exit.

**Non-goals for M1:** experts over the LAN (M0 roadmap M2), multi-GPU tuning, AMD/Intel speed (they must not break),
a server or UI of our own (Marian launches upstream `llama-server` on 127.0.0.1), new quant types.

## 3. What "exact" means

1. **Exact mode** = for the same model files, prompt, sampler, seed, context, `n_batch` / `n_ubatch`, speculation
   setting **and therefore the same rows per `MUL_MAT_ID` call**, Marian produces **bitwise identical logits and tokens
   to the same build with every routed expert resident in VRAM of the same GPU**. Placement, cache size, policy, warm
   start, prefetch and the RAM/NVMe tier are invisible to the output. **Adaptation may move bytes; it may never change
   a numerical path.** Under memory pressure Marian pauses admission or refuses a request; it never silently falls
   back to CPU compute [AS].
2. Why this reference and not "stock llama.cpp": stock `-ncmoe N` computes CPU-layer experts on the CPU, which rounds
   differently [ST, `DETAILS.md:1933-1936`], so stock output already depends on placement. All-on-GPU is the only
   placement-independent reference.
3. **Kernel choice depends on rows, and that bounds every claim** [FB, verified in the pinned source]: CUDA
   `MUL_MAT_ID` uses MMVQ up to `MMVQ_MAX_BATCH_SIZE` = 8 rows (`ggml-cuda/mmvq.cuh:3`), capped lower per type on
   Turing and later: **IQ2_S at 7** (`mmvq.cu:177-180`). Above that it uses MMQ, which covers Q2_0, IQ2_XXS and IQ2_S
   but **not IQ1_M** (`mmq.cuh:60-96`), so IQ1_M layers (blocks 8, 13, 37) fall to a fallback that synchronises the
   stream twice (`ggml-cuda.cu:2134-2156`) and switches CUDA graphs off. Consequences: (a) exactness is defined per row
   count, so a speculative verify window is a different numerical path from single-token decode (class D below);
   (b) on Swift, windows of **7 rows or fewer** keep every layer on MMVQ, 8 moves the 34 IQ2_S layers to MMQ, 9 puts
   the IQ1_M layers on the synchronising fallback; (c) the calibration reads these tables per GPU, never assumes them.
4. **Why upstream's cache should already satisfy item 1** (*to be proved, not assumed*): the bank holds the same bytes
   as the host tensor (`llama-moe-cache.cpp:591-605`), `MUL_MAT_ID` computes each (token, expert) row independently
   and the row count is the same, and the router weights and their weighted sum are untouched by slot ids
   (`llama-graph.cpp:1706-1717`, `2352-2357`). Risks: dispatch depending on the bank's `ne[2]` or on the set of ids,
   and the extra MMQ slot (`llama-moe-cache.cpp:304-312`).
5. **Three classes** for every mechanism:
   - **E (exact):** bitwise invariant under item 1. May be default.
   - **D (distribution-preserving, not bitwise):** verified speculation (n-gram / prompt lookup, MTP) changes the
     verify pass's row count and so its kernels; greedy text is the model's own up to near-ties, sampled text keeps the
     distribution under proper rejection sampling. **Class D is off in the exact default** (C-16): it is not bitwise,
     so it cannot be part of "exact by default". Whether a named non-exact preset (for example `--preset
     fast-verified`) may switch it on after its gate is Chris's decision (section 12).
   - **A (answer-changing):** route-tail skip, CPU compute of misses with ggml's CPU kernels, CPU prefill share, an
     iGPU lane, reduced-precision misses. Opt-in only, each with a blind quality gate (section 9.4).
6. **Tests, in this order** [AS first-test rule, FB parity rig]:
   - **E0, single-layer harness** (first thing built): one Swift layer of each layout with frozen input activations;
     the bank path with every expert resident, one forced miss, permuted slots, and eviction then reuse, against the
     all-resident operation, bit for bit. One mismatch rejects the exactness claim for that path.
   - **E1, invariance:** same build, cache budgets {minimum, 25%, 50%, maximum}, policies (SLRU, ranked), prefetch
     on/off, RAM tier on/off: bitwise logits on 20 held-out prompts x 512 teacher-forced tokens, plus token-ID identity
     over 4,096 generated tokens.
   - **E2, reference:** a MoE GGUF that fits wholly in 16 GB (M0's Gemma 4 26B-A4B IQ4_XS, or a 30B-A3B class model):
     all-VRAM against Marian with a deliberately tiny bank, bitwise.
   - **E2b, Swift:** where no all-resident reference fits, the cached path against upstream's selective-copy path
     (both GPU, same row counts) [FB], and upstream's cache at two different budgets. **Both arms must be shown
     GPU-only from backend traces before logits are compared** (C-10): with the cache off, a host-resident expert
     layer at a small batch may run on the CPU rather than through the GPU selective copy, and a CPU-vs-GPU last-bit
     difference would not prove a bank defect. If two verified GPU-only arms differ, the rig has found an upstream
     exactness problem before Marian changed anything.
   - **The literal oracle does not exist for Swift on this laptop** (C-09): its 33.02 GiB of routed experts cannot all
     sit in 16 GB. E0 (every layout, one layer), E2 (another model, whole) and E2b (GPU paths) are surrogates, and the
     design says so wherever it claims exactness for Swift. A whole-Swift all-resident check needs a GPU with about
     40 GB free (rented or borrowed); it is listed in M1.4, not required for the M1 exit.
   - **The parity matrix enumerates the dispatch boundaries** (C-13). For each of the three Swift layouts: rows 1, 7,
     8 and 9 (the MMVQ cap for IQ2_S, the MMQ switch, the IQ1_M fallback), 32 and 33 (upstream's `max_batch` cut,
     `llama-moe-cache.cpp:230`), the capacity fallback (selected set larger than the slots), and two large prefill
     ubatches through the selective-copy path. Each cell records the kernel actually dispatched and an
     intermediate-output hash; a pass at one row does not qualify another.
   - E0-E2b gate every merge into Marian's main branch.

## 4. Architecture

```
          +----------------------- Marian layer on top (Python, M0 extended) ------------------------+
          |  probe -> routing profile -> offline policy replay -> plan -> launch -> trace -> learn       |
          +------------------------------------------+----------------------------------------------+
                                                     | flags, profile files, trace ring
+----------------------------------------------------v----------------------------------------------------+
| llama.cpp at the pin + Marian patches (P1-P9)                                                             |
|  GPU   dense weights, KV, compute buffers | moe-cache bank per layout group:                              |
|                                           |   [ranked core | adaptive SLRU behind a TinyLFU doorkeeper |    |
|                                           |    transient slots]  <- uploads on a copy stream            |
|  RAM   pinned staging ring (always) + bounded RAM tier (ranked; pinned only where measured to pay)       |
|  NVMe  GGUF shards: cold experts read unbuffered (optional expert-major sidecar), the lazy PLE table       |
+-----------------------------------------------------------------------------------------------------------+
```

- **All routed-expert layers host-resident** (`-ngl 99 --n-cpu-moe <n_layer>`), which makes every layer eligible
  for the cache (`llama-moe-cache.cpp:275-298`) [UP]. VRAM holds dense weights, KV, compute buffers and one cache
  budget set by calibration. A per-expert cache beats whole-layer residency once routing is skewed (Strata: 85.4%
  from 9.16 GiB against stock's 33.3% from 11.36 GiB) [ST].
- **Banks per layout group as upstream builds them** (Swift: three groups, IQ2_S / IQ2_XXS / IQ1_M gate-up, each with
  Q2_0 down; 34 : 11 : 3 layers) [UP]. Marian changes how slots are chosen and kept, and how big each group's bank is
  (decided before the banks exist, from the profile [FB]), not how slots are stored.
- **Three slot classes in each bank:**
  - a **ranked core** filled from the routing profile before the first token and not evicted until the policy
    demotes it with hysteresis [ST `STRP` profile `expert_cache.hpp:40-73`; MY; CT; AS; FB];
  - an **adaptive region**: upstream's SLRU, with a **TinyLFU doorkeeper** (a 4-bit count-min sketch over
    `layer*n_expert + expert`, halved periodically) so a one-off expert seen in a long prompt does not evict a
    repeat-use expert on its first miss [FB; LIT W-TinyLFU, Einziger, Friedman and Manes 2017; AS names TinyLFU
    too];
  - **transient slots** that serve a miss the policy declines to admit: the expert is still uploaded and computed on
    the GPU in the same bank `MUL_MAT_ID`, then the slot is reused. Declining admission never means CPU compute or
    skipping [AS].
  - **Capacity invariant** (C-24): upstream checks only that a step's distinct experts fit the group's total slots
    (`llama-moe-cache.cpp:124-126`) and aborts otherwise (`:587-589`). With a ranked core that is not evictable, the
    check must be on **evictable plus transient slots**: for every group, `n_slots - n_core >= max distinct experts one
    layer can select in one step` = min(`n_expert_used` x max rows on the slot-map path, `n_expert`), 10 x 32 = 320 on
    Swift at upstream's `max_batch`, or 80 if Marian caps the slot-map path at 8 rows. The plan sizes the core under
    this bound; a step that would still exceed it takes the selective-copy path instead of aborting.
  - **Staged warm fill** (C-11): only a small synchronous core (the top slots of each group, sized so the fill takes
    about a second at the probed read rate) is filled before the first token; the rest of the ranked fill streams in
    the background at low priority behind demand reads. A cold start never waits on an 8-9 GiB read [critique; CT].
- **Value of a slot** = expected requests x measured exposed miss cost / VRAM bytes, so a cheap-to-miss expert (in
  pinned RAM) is worth less VRAM than one that would come from NVMe, and group budgets follow marginal saved time,
  not source bytes (`llama-moe-cache.cpp:350` today splits by bytes) [AS; FB; CT].
- **Under the cache, a bounded RAM tier and an unbuffered NVMe tail.** Today the fill copies from `w->data`, the mmap
  (`llama-moe-cache.cpp:602`), which CUDA pins only with `GGML_CUDA_REGISTER_HOST` (`ggml-cuda.cu:166-169`), off by
  default and all-or-nothing [FB, verified]. A pageable copy is staged and host-blocking, and a non-resident page
  faults to NVMe mid-token: the likely mechanism behind stock's 10.59 to 4.86 tok/s drop after two pp4096 passes
  [FB]. Marian adds an expert-source interface used by **both** the decode fill and the large-batch selective copy
  [AS], backed by a small pinned staging ring, a ranked bounded RAM tier and unbuffered reads for the tail
  (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) [FB; AS; MY; ST budgeted complement `expert_source.hpp:724-749`]. **How much
  of the RAM tier to pin is measured, not assumed:** the consultants disagree (Fable: a 12-14 GiB pinned arena;
  Astra: ordinary bounded RAM plus a 64-256 MiB pinned ring, pinning more only where it pays). M2.0's pinned-versus-
  pageable copy test decides per machine.
- **Telemetry first:** the copy callback already reads each layer's ids to the host (`llama-moe-cache.cpp:582-584`)
  [UP], so a per-token, per-layer ring of ids, hits by tier, bytes, exposed wait and prefetch precision costs little
  [CT; AS; FB]. The routing profile is built from the same trace. Upstream's destruction-time log is two buckets
  split at 8 x `n_expert_used` ids (`:608`), which a verify window straddles, so it is never reported as a per-token
  hit rate [FB].

## 5. What gets patched in llama.cpp, and what does not

Rule: each patch small, self-contained, off unless asked for, written so it could go upstream. Sizes are estimates.

| # | Patch | Where | Class | Est. size | When |
|---|---|---|---|---|---|
| P1 | **Trace and stats**: per ubatch and layer: selected ids per row, row-level hits (section 2), misses by source, bytes, upload-run count; callback in `llama_context_params` plus an env-gated binary ring for tools; an instrumentation-off control arm [AS]. **Covers both callbacks** (C-05): the slot-map path in `copy()` and the large-batch path in `sched_copy_experts` (`llama-context.cpp:2778-2859`), which also reads the ids, so prefill routing reaches the profile. **Timing from GPU events, not host clocks** (C-07): events recorded at ids-ready, before and after the fills, after the slot-map upload and at the consuming `MUL_MAT_ID`'s start, read back asynchronously; "exposed wait" = consumer start minus the end of the previous GPU work, and "drain" = host time inside `ggml_backend_synchronize` minus the GPU work it waited for. The callback's own duration is neither | `llama-moe-cache.cpp` `copy()` `:549-617`, `llama-context.cpp`, `llama.h` | E (observes) | ~250 lines | M1.1 |
| P2 | **Policy interface + ranked core + doorkeeper + transient slots**: SLRU becomes one policy; a policy receives the step's ids and returns hits, admissions, victims, slots; constructor takes per-group slot counts and an initial fill list (`--moe-cache-profile FILE`), checks the capacity invariant (section 4) and stages the warm fill. **Keeps upload coalescing** (C-19): victims for sorted misses are chosen as consecutive slots where possible, because upstream merges an upload only when experts and slots are both consecutive (`llama-moe-cache.cpp:597-605`); P1 counts upload runs per step so a fragmentation regression is visible | `llama-moe-cache.cpp` (`moe_cache_slru`, constructor `:255-486`) | E | ~400 lines | M1.2 |
| P3 | **Expert-source interface**: fills and the large-batch copier read bytes from a source the layer on top provides (staging ring, RAM tier, unbuffered file reader), keyed so bindings still match the host tensor (`:241`) [AS; FB]. **Direct-I/O contract** (C-14): offsets and lengths rounded out to the volume's sector size (queried, not assumed) and buffers aligned to it, the expert trimmed out of the larger read; a failed unbuffered open or read falls back to a **bounded** buffered read with a logged reason and a counter in the trace, never a silent switch. **Sidecar binding** (C-01): an expert-major sidecar, if built, carries the sha256 of every source shard and a per-record checksum of the expert's bytes, verified on first read of each record and in full at build; a mismatch disables the sidecar, it never serves bytes | `llama-moe-cache.cpp`, `llama-context.cpp` `sched_copy_experts` `:2778-2859` | E | ~300 lines | minimal form in M1 if C-03's check fails (section 10), full in M2 |
| P4 | **Learned counts out**: decayed per-(layer, expert) counts, separate for prefill and decode [AS], dumped on exit for the next warm start [ST learned-profile writer `expert_cache.cpp:143-211`; FB]. **Crash-safe** (C-22): written to a temporary file, flushed, then renamed over the old one (Strata's pattern), with a schema version, the build's pin and the patch-set version; a file from an incompatible build is kept but not loaded; one writer at a time (a lock file) | `llama-moe-cache.cpp`, layer on top | E | ~100 lines | M1.3 |
| P5 | **Remove the per-layer drain.** Today: 48 `ggml_backend_synchronize` per token (`:583-584`), each followed by host planning before the matmuls start. Two candidate designs, chosen by M1.0's drain measurement: **(a) precise dependencies** [AS]: ids into a pinned buffer, a routing-complete event, a worker plans and fetches, uploads on a copy stream, the bank op waits on a copy-complete event; **(b) sync-free resolve** [FB]: the slot map moves to the GPU and one CUDA op per layer resolves ids to slots in-stream, gathering any miss from the device-mapped pinned tier into scratch slots by zero-copy; admission happens between tokens from a device-side miss ring; tail misses wait on a mapped flag with a bounded timeout (never near WDDM's 2 s TDR). (b) keeps one CUDA graph per token; it needs the pinned tier. **Ordering rules from the critique:** for (a), the slot-map `GET_ROWS` (`llama-graph.cpp:2609-2619`) must wait on the same map generation as the bank it indexes, not only the bank `MUL_MAT_ID`: map and bank carry one generation number and a slot is not overwritten until every reader of the old generation has completed (C-12). For (b), a cold tail miss needs **intra-token** host service: a host I/O thread polls the device miss ring (mapped memory) continuously during the token and serves demands as they appear; a between-token drain alone would deadlock the waiting layer. Until that service exists and its failure behaviour is exact (a timeout aborts the request with an error, it never computes with stale or missing bytes), P5b runs only when every expert is in VRAM or the pinned tier (C-04) | `llama-moe-cache.cpp`, `llama-graph.cpp` `build_moe_cache_slots` `:2582-2621`, a CUDA op | E | (a) ~400, (b) ~700 lines | M2 |
| P6 | **Look-ahead prefetch**: layer l+1's router applied to layer l's mixed state, top-k as a prediction; predicted misses uploaded on the copy stream (or gathered into prefetch slots under P5b), predicted tail experts read into staging; per-layer precision tracked and the look-ahead turned off where it is poor. Alternative or complement: a route-transition "itinerary" table that predicts the next layers' experts from the current route, with no router copy [AS] | `llama-graph.cpp`, cache | E | ~300 lines | M2 |
| P7 | **Bank lending in long prefill**: for prompts above a calibrated threshold, release the adaptive region to the compute allocator for a larger ubatch, keep the ranked core, refill device-to-device after [FB] | cache, scheduler reserve | E at a fixed ubatch | ~250 lines | M5 |
| P8 | **Zero-slot route-tail skip** (opt-in): a missed expert that every row of the step ranks 7th or lower maps to a bank slot of zeros, so it contributes exactly zero with the other weights untouched: Strata's semantics [ST `DETAILS.md:1858-1897`] in one slot instead of new kernels [CT]; off above 90% hit rate as Strata learnt [ST, FB] | `llama-moe-cache.cpp` plan step | A | ~60 lines | M4 |
| P9 | **CPU (or iGPU) lane for misses** (opt-in): a second `MUL_MAT_ID` on the host tensor with a masked id set, summed in the router's order; share fitted from this machine's measured per-expert CPU time and link rate, not Strata's constant 1.85 [ST CPU pool `DETAILS.md:1913-1941`; LIT Fiddler, Kamahori et al. 2024; AS; FB]; iGPU variant via Vulkan/SYCL, which the cache constructor already admits as `IGPU` (`:259`) [FB] | `llama-graph.cpp`, scheduler | A (E only if the bit-faithful kernel, section 8, works) | ~800+ lines | M4 |

**Not patched:** model architectures, quant kernels, the server, the sampler, the loader, `-ot` / `--n-cpu-moe`
semantics, the GGUF format. **The routing profile needs no patch to start:** before P1, the eval callback on
`ffn_moe_topk-<il>` (the mechanism `imatrix` uses) records the ids [UP].

**Rebase policy:** the cache is EXPERIMENTAL upstream and changed three times in two weeks (LRU to SLRU on 19/10).
Marian rebases onto master fortnightly, keeps each patch one commit, and offers P1 and P2's warm fill upstream first.

## 6. The layer on top (M0 extended)

### 6.1 Components

- **`marian machine`** [M0], extended: the WDDM local-memory budget via DXGI `QueryVideoMemoryInfo` (not
  `cudaMemGetInfo` / `ggml_backend_dev_memory`, which do not see WDDM spill) [CT; AS; FB]; commit headroom; RAM;
  hardware-accelerated GPU scheduling state and the "Prefer No Sysmem Fallback" setting, reported once in plain words
  [FB]; power source (battery vs mains changes clocks) [FB; AS]; the model drive.
- **`marian-probe`** (new, about 300-400 lines C++ linking ggml; or new cases in `test-backend-ops`) [CT; FB]:
  1. **VRAM cliff finder**: allocate in 256 MiB steps and time a device copy and a small matmul after each; the step
     where bandwidth collapses more than 3x is the spill; safe budget = last good step minus a margin [FB];
  2. host-to-device rate for expert-sized copies, pinned and pageable, with latency distributions [FB; AS];
  3. the cost of one stream synchronise / event round trip on this driver [CT; FB];
  4. RAM: available, commit, and a pin test in 1 GiB steps until available falls below a floor [FB];
  5. NVMe: unbuffered expert-sized random reads at queue depth 1 and 8 [FB; AS];
  6. per-layout CPU expert time (for the opt-in lane only) [FB];
  7. MMVQ/MMQ eligibility per layout and row count on this GPU (section 3.3) [FB].

  **Safety envelope for the probes** (C-21): every probe has a wall-clock timeout and is cancellable; the VRAM finder
  stops at the first collapse, or at the DXGI budget minus 1 GiB, whichever comes first, and frees everything before
  the next probe; the pin test stops at the earliest of available RAM below max(2 GiB, 10% of total), Windows commit
  headroom below 4 GiB (Strata's floor, `DETAILS.md:1829`), or a cgroup limit on Linux; probes never run while a
  model server is up; a probe that crashes or times out records a "probe failed" row and the plan uses the
  conservative default (cache off, stock placement), never a guess.

  **The probe bounds the budget; it does not certify it** (C-08): a model-free 256 MiB-step probe cannot reproduce a
  loaded server's peak. The live confirmation in section 7 step 5 therefore runs **at the requested context, the
  planned peak `n_ubatch` and with a controlled competing allocation** (a ballast the size of the margin), sampling
  the DXGI local-budget usage; only a budget that survives all three is written to the machine profile.
- **`marian inspect`** [M0], extended: every shard's tensor directory (types, shapes, `nb[2]`, role offsets, fused
  roles), so nothing is hard-coded for Swift's three layouts [AS].
- **`marian trace`**: runs a calibration corpus with P1 on (or the eval callback before P1) and writes the routing
  profile.
- **`marian replay`** (new): replays a routing trace offline through candidate policies and budgets (SLRU, ranked,
  ranked + doorkeeper, transient slots) with the probed costs, so bad combinations die in seconds before any live
  sweep; Maya's 95-minute calibration is the warning [AS; FB "a 30-line simulator"].
- **`marian plan`** [M0], extended: from probes + profile + geometry + requested context, picks bank size per group,
  ranked-core share, RAM tier size and pinning, staging ring, `n_ubatch` for prefill, speculation window cap.
- **`marian serve`** [M0]: launches `llama-server` (127.0.0.1, as M0 does) with the plan; at exit merges learned
  counts (P4) into the routing profile.

### 6.2 Two profile files

- **Machine profile** (local, never shared): probe results keyed by M0's fingerprint plus driver; re-probed on driver,
  GPU, RAM or power-mode change, after any observed collapse or CUDA OOM, or on request [FB; AS].
- **Routing profile** (a model property, shareable): decayed per-(layer, expert) counts for prefill and decode, route
  transitions for the itinerary predictor, byte-weighted hit curves, corpus hash, token counts, age, workload category
  [AS]; keyed by **a content hash of every shard's tensor-info table and expert offsets** (not M0's first 64 MiB +
  size, which two quants could share) [FB; AS; CT] **plus a hash of the router weights' bytes** (`ffn_gate_inp` of
  every layer: small, and they are what routing depends on; a fine-tune with the same shapes changes them) (C-15).
  The machine profile is keyed by the M0 fingerprint, driver, **build pin and patch-set version, KV type, model
  location (drive)** and power source; any change invalidates the timing rows it affects (C-15). A shipped profile
  lets any machine start warm. **Shared profiles
  are untrusted priors**: sizes and ids validated, allocations capped, local observations override them; routing
  statistics can reveal what a user works on, so sharing is explicit and offline [AS].
- **Hold-out rule:** hit rates are always reported on prompts not in the corpus that built the profile.

## 7. Calibration and adaptation loop

1. **Probe** (section 6.1; about 90 s, mostly without loading the model) [FB]. Reuse the machine profile when nothing
   changed.
2. **Inspect** the model's tensor directory; find or build its routing profile (trace a mixed corpus of about 20k
   tokens: code, chat, reasoning, several languages, a long document).
3. **Replay** the trace through candidate policies at candidate budgets with the probed costs; keep the finalists.
4. **Plan:** VRAM for experts = cliff-finder budget - dense - KV(context) - workspace peak(`n_ubatch`) - transfer
   buffers - reserve [AS; FB; CT]. Assign VRAM / RAM / NVMe by one greedy over the ranked profile with two cut points,
   since each tier has a near-constant cost per expert [FB], and with the slot-value rule of section 4.
5. **Confirm the finalist live** (one short run, interleaved against the next-best) [AS].
6. **Run** with the trace ring on; the adaptive region learns within the session; P4 keeps decayed counts; rejected
   speculative branches get less promotion weight than committed tokens [AS].
7. **Govern:** a cheap VRAM and RAM watchdog. **Upstream allocates each group's bank once, at construction**
   (`llama-moe-cache.cpp:437-447`), so invalidating slots frees no VRAM (C-02). M1's response to a competing
   allocation is therefore: (i) pause admission (no new fills beyond transient slots) at once; (ii) if the DXGI
   budget falls below the plan's floor, **finish or refuse the current request, then restart the server at a smaller
   plan** (a supervised restart by `marian serve`: the KV and prompt cache are lost, the ranked core refills from the
   profile, and the user sees one line saying why). Live physical shrinking (a bank split into segments that can be
   freed, as Strata's `--vram-elastic` does on request [ST]) is M2 work behind its own parity test. Never change a
   numerical path to cope [AS; FB].
8. **Serving envelope** (C-23): concurrent requests change rows per `MUL_MAT_ID`, the cache path and peak workspace,
   so M1 serves **one sequence at a time** (`llama-server --parallel 1`, continuous batching off); further requests
   queue in arrival order with a configurable queue limit and timeout; past the limit, or while the governor has
   paused for memory, a request gets HTTP 503 with a `Retry-After` and a reason; a cancelled request releases its
   transient slots at the next step boundary. Multi-sequence serving is M2+ and needs its own parity cells.
9. **Learn:** compare predicted with measured hit rate, exposed wait and decode; if VRAM peaked inside the margin or
   decode collapsed (M0's overflow rule: within 300 MiB of full, or below 0.6x the expected point [M0]), shrink one
   step for next time; merge learned counts.
10. **Re-plan** on a change of context class, machine, driver, power source, or a drift of more than five points
   between predicted and measured hit rate over a session.

**Never tuned for one machine:** every quantity above is measured on the machine at hand or is a margin the loop
corrects. M1.3 validates the plan on two machine classes. On 8 GB VRAM and 16 GB RAM the same mechanism runs with
smaller tiers; Marian publishes the measured capacity curve and predicted disk demand rather than promising 85% [AS].
Fable's estimate for an 8 GB card with 32 GB RAM: 45-55% hit, 15-25 tok/s; for 16 GB RAM: 10-20 tok/s without
collapse [FB].

## 8. Exact mode versus opt-in switches

| Mechanism | Class | Default | Source |
|---|---|---|---|
| Per-expert cache, all experts computed on the GPU | E | on | UP |
| Ranked core and warm start, learned counts | E | on | ST, MY; CT, AS, FB |
| TinyLFU doorkeeper; transient slots; slot value by exposed cost; budgets by marginal gain | E | on once E1 passes | FB (doorkeeper); AS (transient slots, value); CT, FB, AS (budgets) |
| Expert-source interface, staging ring, bounded RAM tier, unbuffered tail, optional expert-major sidecar | E | on when RAM allows | AS (sidecar, both copy paths); FB; MY; ST |
| Drain removal (P5a or P5b) | E | on once E1 passes | AS (a); FB (b) |
| Look-ahead prefetch; itinerary predictor | E | on where per-layer precision pays | ST (byte warming); LIT Pre-gated MoE (Hwang et al.); CT, FB (into VRAM); AS (itinerary) |
| PLE coordination: PLE rows scheduled with expert I/O, a small hot-row cache, draft-token row prefetch | E | on if PLE faults are measured to matter | AS; FB |
| Exact prompt-boundary checkpoints for branches | E at qualified boundaries | on | AS; ST `PROMPT_CACHE_TAIL.md` |
| Layer-major ("weight-stationary") prefill at a fixed ubatch | E in principle, experimental until parity | off until M5 parity | AS |
| Larger `n_ubatch`, bank lending in long prefill | E at a fixed `n_ubatch` (changing it changes prefill bits: a planned, recorded setting) | on, recorded | FB; CT |
| n-gram / prompt-lookup speculation, window <= 7 rows on Swift (8 rows measured as a separate arm, C-18), length chosen by milliseconds per committed token | D | **off in exact default**; a non-exact preset is Chris's call | UP; ST; AS (cost-aware window); FB (row cap); critique (8-row arm) |
| MTP head for Swift (base model's MTP layer, requantised small) | D | **off**; download is Chris's call | UP #29761; ST; FB (requantise) |
| Zero-slot route-tail skip | A | off | ST semantics; CT mechanism |
| CPU miss lane; iGPU miss lane | A | off | ST; LIT Fiddler; AS; FB (iGPU) |
| CPU share of short prompts | A | off | ST |
| **Bit-faithful CPU twin of the GPU expert kernel** | would turn the CPU lane into E | research spike, M4 | CT |

**The bit-faithful CPU kernel** [CT]: the GPU's MMVQ path quantises the activation to q8_1, sums integer block dot
products (exact in any order) and accumulates floats in a fixed tree; Strata documents such a tree lane by lane
(`EXPERTDEC.md:44-53`) [ST]. A CPU kernel reproducing the q8_1 rounding, the tree order and the FMA contraction would
give the GPU's bytes and make the hybrid exact. It is a spike: the tree depends on the kernel the GPU picks (section
3.3) and on its architecture, and it may be slower than ggml's CPU kernels.

## 9. How we measure it

### 9.1 Fixed inputs

- The same Swift 1.5 IQ2_XS shards (`strata-build\SWIFT-SHA256SUMS`), on the same drive.
- Arms: (a) upstream at the pin, no cache, `-ncmoe 32` (stock re-baselined on the new pin); (b) upstream at the pin,
  `--n-cpu-moe 48 --moe-cache-mib <budget>` (upstream SLRU); (c) Marian. Same compiler and CUDA (12.8.93 here).
- No other GPU work during runs (check `nvidia-smi`; another session holds the GPU today, so M1.0 waits for it).

### 9.2 Workloads and metrics

- **Decode:** `llama-bench -n 128` at depth 0 for continuity with `BASELINE.md`; 10 held-out server prompts (chat,
  code, reasoning, two non-English, a 4K document), greedy, 256 tokens each; a recorded agent session with tool calls.
- **Prefill:** pp512, pp4096 and a 16K prompt through the server (stock 16K was never measured).
- **Per token** (from P1's ring): VRAM hit rate (hits / 480 lookups on this model), **byte-weighted** hit rate
  [AS], combined VRAM-or-RAM coverage [AS], misses and bytes by source, exposed wait, **late misses** (a prefetch that
  arrived after it was needed) and prefetch precision [AS; FB]. With speculation: **expert bytes and milliseconds per
  committed token** [AS]. **The M1 claim is the median per-token hit rate over the held-out decode set, with its 10th
  percentile**, not a run total.
- **Per run:** decode, prefill, first-token time, VRAM peak, RAM available minimum, hard faults per second, disk reads
  (M0's sampler [M0]; `marian-baseline\tools\reslog.ps1`).

### 9.3 Beating the noise floor

Floors on this laptop: Strata about +-10% decode and +15-18% prefill run to run; stock about 2x on decode with
page-cache state.

1. **A-A first:** every comparison begins with two runs of one arm to measure today's floor, and A-A pairs are
   interleaved again during the series, not only at its start.
2. **Interleaved A-B-B-A rounds, at least 5**, fresh process per run, counterbalanced. **A fresh process does not
   reset the page cache** (C-20): pages stay cached after exit, and one arm can warm or evict the next arm's bytes.
   So the protocol names the state it measures and records it rather than assuming it: at the start of and during
   each run, physical model-file reads, available RAM and hard faults are logged (`reslog.ps1`), and a run whose
   starting state differs from its pair's (for example, more than 1 GiB more model data already resident) is
   flagged. Three named states: **warm** (a fixed 64-token decode discarded first; the default), **post-prefill**
   (a pp16K immediately before; the state that collapsed stock), and **cold** (standby list purged with an admin
   tool before the run, when Chris allows it; otherwise "not measured").
3. **Report** the median of per-round ratios and how many rounds favour each arm; **a difference is claimed only if at
   least 4 of 5 rounds agree and the median ratio lies outside the A-A floor**, else NO DECISION.
4. Hit rate barely depends on the page cache, so M1's exit uses hit rate first and tok/s second.
5. **Every test records its decision** in `docs/TEST-REGISTER.md` (Chris's rule), including NO DECISION and tests
   that could not have failed.

### 9.4 Quality gates for class A and D switches

Blind and paired, before any switch may become a default: teacher-forced KL and top-1 agreement against exact mode
on the held-out set (Strata's method, `DETAILS.md:1858-1897` [ST]); perplexity on a fixed text; task checks (a
HumanEval subset, reasoning, instruction, tool calls); a blind A/B read of 30 paired answers by a judge who does not
know the arms. A KL number alone never decides: Strata's own tables put the floor and the effect within a factor of
ten [FB]. Thresholds: section 12.

## 10. Milestones

| | What | Exit criterion | Est. days |
|---|---|---|---|
| **M1.0** | Build the fork at the pin. E0 single-layer harness with the parity matrix (section 3.6). Measurements, each a falsifier: stock re-baseline; **upstream SLRU hit rate and decode at 3 budgets** (>= 84% means ranking buys little) [FB]; **the per-layer drain, timed with GPU events** (< 1.5 ms of avoidable drain per token means P5 drops in priority) [FB; C-07]; **pinned vs pageable expert copy** (within 15% means the RAM tier is a page-cache fix only) [FB]; **E2b on verified GPU-only arms** [C-10]; **the sourcing endurance check**: decode at matched budgets alone and straight after pp4096 and pp16K, with physical reads, faults and available RAM (C-03) | numbers in the register; M1.1+ re-scoped from them. **If decode after pp16K falls below the 12 tok/s floor, a minimal P3 (pinned staging ring + bounded unbuffered reads for misses) moves into M1** (C-03) | 2-4 |
| **M1.1** | P1 trace; `marian trace`; routing profile; `marian replay` | per-token hit rates for arm (b); replay predicts (b) within 3 points | 3-4 |
| **M1.2** | P2 policy: ranked core, doorkeeper, transient slots, budgets by need; E1 | E1 bitwise pass; hit-rate gain over (b) outside noise | 4-5 |
| **M1.3** | Calibration v1: probes with their safety envelope, cliff finder, plan, live confirmation, governor with supervised restart, serving envelope, P4 learned counts. **Deployment deliverables** (brief challenge): launch checks (model identity, profile compatibility, free VRAM/RAM against the plan), health and error semantics (503 with a reason), a one-command diagnostics bundle (machine profile, plan, last trace summary, no prompts), and **rollback**: `marian serve --stock` launches plain upstream flags from M0's whole-layer profile | the plan, run fresh here and on the capacity simulation, picks its sizes with no hand tuning; a competing allocation produces a pause, then a refused request or a supervised restart, never a collapse [AS falsifier; C-02] | 5-6 |
| **M1 exit** | **median per-token VRAM lookup hit >= 85% (from about 33%; definition in section 2) on held-out prompts; exact (E0-E2b and the parity matrix pass; Swift's literal oracle stated as absent); decode >= 12 tok/s on the reference laptop in the warm and post-prefill states, per 9.3** | | **13-18 total** |
| M1.4 | Real-hardware validation outside the exit: a real 8 GB GPU, a Linux machine, a PCIe 3.0 or x8 link, a slower drive; whole-Swift all-resident parity on a borrowed or rented GPU with about 40 GB free | each class moves from "unsupported" to "supported" only by its own measured run | as hardware allows |
| M2 | P3 expert source, staging ring, RAM tier, unbuffered tail, sidecar; P5 drain removal; P6 prefetch | decode 20+ exact; no post-prefill collapse | 12-18 |
| M3 | Speculation: n-gram with the row cap and cost-aware window; MTP if Chris approves the download | class D gate passed; ms per committed token down | 4-8 |
| M4 | Class A switches with blind gates; bit-faithful CPU kernel spike; iGPU yes/no | each switch measured blind | 8-12 |
| M5 | Prefill: calibrated `n_ubatch`, bank lending, layer-ahead reads; layer-major prefill prototype (two-layer parity first); upstream offers | 16K prefill above Strata's 193 tok/s | 8-15 |

## 11. Risks

| Risk | Likelihood | Effect | Mitigation |
|---|---|---|---|
| Upstream's cache changes under us (EXPERIMENTAL, three changes in two weeks) | high | rebase work | small separate patches; offer P1/P2 upstream; fortnightly rebase |
| E0-E2b fail: dispatch depends on bank shape or id set | medium | exact mode needs a fix | harness first (M1.0); padding banks per group if needed; configurations that change dispatch stay experimental [AS] |
| Upstream SLRU already near 85% | medium | M1's policy work shrinks | good news: then M1 is warm start + telemetry and M2 starts early (Fable's falsifier) |
| RAM, not VRAM, is the limit on 32 GB | high on Swift | decode bound by NVMe even at 85% | RAM tier, unbuffered tail, misses reported by source |
| WDDM sysmem fallback eats the margin silently | high on Windows | decode to ~3.5 tok/s | cliff finder, DXGI budget, governor |
| Slot reused before its readers finish (async fills, P5) | medium | silent corruption | generation-tagged leases, completion events, E1 under stress [AS] |
| Spin-wait on a tail read trips WDDM TDR (P5b) | low | driver reset | bounded timeout well under 2 s; fall back to P5a [FB] |
| Pinned memory starves the OS | medium | paging, WDDM trouble | pin only what M2.0 shows pays; measured floor [AS; FB] |
| Prefetch competes with demand I/O | medium | p95 latency up while "hits" rise | demand first, a byte budget for speculative reads, late-miss accounting [AS] |
| Telemetry costs what it measures | low | biased results | buffered ring, an instrumentation-off control [AS] |
| A ranked core leaves too few evictable slots, and a cold step aborts (`llama-moe-cache.cpp:587-589`) | medium | crash mid-request | the capacity invariant, the selective-copy fallback (C-24) |
| A stale or mismatched profile or sidecar is loaded | medium | wrong ranks (profile: slower only) or wrong bytes (sidecar: breaks exactness) | router-weight hash in the profile key; per-record checksums in the sidecar (C-15, C-01) |
| A crash mid-write corrupts the learned profile | low | lost learning | temp file and rename, schema version, lock (C-22) |
| A second request changes rows and peak memory | high without a rule | non-exact or out of budget | one sequence at a time, a queue, 503 with a reason (C-23) |
| Routing profile overfits the corpus | medium | hit below prediction | hold-out reporting; learned counts; adaptive region |
| Swift has no MTP tensors | certain | MTP waits | n-gram first; Chris decides the 5.2 GB download |
| GPU shared with another session | certain today | no measurements | design only now; M1.0 waits for a free GPU |
| Borrowed Strata code imports its network surfaces | low | security | borrow algorithms, not launcher, downloader or server (`SECURITY-READ.md`) |

## 12. Open decisions for Chris

1. **Build on upstream's `--moe-cache-mib` (recommended; Astra and Fable agree)** rather than port Strata's engine.
2. **Class D (verified speculation):** it is off in the exact default (v2, C-16). May a named non-exact preset
   (`fast-verified`) switch it on once its gate passes, and should Marian suggest that preset to users?
3. **The MTP layer for Swift:** fetch the base model's MTP tensors (about 5.2 GB, the pinned revision Strata uses),
   requantise, convert to GGUF?
4. **Quality-gate thresholds** for class A/D (proposal: mean KL <= 0.005, top-1 >= 98%, task scores within run-to-run
   spread, blind A/B indistinguishable at 30 pairs).
5. **Upstream offers** of P1 and P2's warm fill: yes, and under whose GitHub identity?
6. **Shipping routing profiles** for known models in the repo (statistics of our corpus; no personal data).
7. **A cold page-cache state** needs the standby list purged with an admin tool (for example Sysinternals RAMMap)
   before each cold run. Allowed on the reference laptop, or do we measure warm and post-prefill only?
8. **Hardware for M1.4:** which real 8 GB / Linux / PCIe 3.0 machines may Marian be measured on, and is a rented
   40 GB GPU for the whole-Swift oracle worth paying for?

## 13. Credits

Strata (github.com/Niko1221/Strata, MIT) and Project Maya (github.com/mw00/project-maya, MIT): frequency-ranked
residency, ranked warm start, learned profiles, the GPU/RAM/NVMe tiers, router look-ahead, the CPU lane and its PCIe
share, route-tail skip and the blind-KL method. Upstream llama.cpp (MIT): the cache Marian builds on. Alder: M0. Sol:
the research pack. Astra and Fable: the ideas tagged [AS] and [FB]. Every borrowed file keeps its MIT notice in
`NOTICE.md`.

## 14. What changed from v1 to v2 (the foreign critique)

Critique: `critique/CRITIQUE-M1-v1.md` (+ `.json`, `.yaml`), critique harness v3.2 on 22/10/2026, critics
Sol (correctness), Luna (correctness, max) and Astra (operations); the brief asked for performance improvements as
well as defects. Run stamp: v1 `DESIGN-M1.md` sha256 `3a27a5a4...`. 24 items, **0 refuted**, 13 survived the refuter, 3
"unverifiable" (the refuter's manifest said it had not opened `DESIGN-M1.md`, a solution file it was given: an
apparatus quirk, so I decided those three by hand). The refuters were all one model family (OpenAI GPT); no Claude-side
twin was run (the Max plan's weekly window is at 85%).

| # | Item | Editor's action | What v2 does | Where |
|---|---|---|---|---|
| 1 | C-07 no valid bank-ready / exposed-wait timestamp | fix now | GPU-event timing; drain and exposed wait defined | P1, M1.0 |
| 2 | C-13 parity gate skips the row-count boundaries | fix now | parity matrix: rows 1, 7, 8, 9, 32, 33, capacity fallback, large ubatches, per layout, kernel recorded | 3.6 |
| 3 | C-20 fresh processes do not equalise the page cache | fix now | named states (warm, post-prefill, cold), recorded physical reads, flagged mismatches, interleaved A-A | 9.3 |
| 4 | C-02 the governor cannot shrink a fixed bank | fix now | pause admission, then refuse or supervised restart at a smaller plan; live shrink moved to M2 | 7.7 |
| 5 | C-03 PERF: bounded expert source before the M1 exit | fix now | endurance check in M1.0; a minimal P3 moves into M1 if decode after pp16K misses the floor | M1.0, P3 |
| 6 | C-16 class D default unresolved | fix now | class D off in the exact default; a non-exact preset is Chris's call | 3.5, 8, 12.2 |
| 7 | C-23 serving concurrency undefined | fix now | one sequence, queue, 503 with reason, cancellation rule | 7.8 |
| 8 | C-09 no literal Swift oracle | fix now | stated as absent; surrogates named; whole-Swift check in M1.4 | 3.6, M1.4 |
| 9 | C-10 cache-off arm may run on the CPU | fix now | arms verified GPU-only from backend traces first | 3.6 |
| 10 | C-05 trace misses large-batch routing | fix now | P1 hooks `sched_copy_experts` too | P1 |
| 11 | C-15 profiles lack content/config invalidation | fix now | router-weight hash; build, KV, drive, power in the machine key | 6.2 |
| 12 | C-12 P5a must order the slot-map lookup | fix next | map/bank generation numbers, no overwrite before old readers finish | P5 |
| 13 | C-04 P5b has no intra-token host service | fix next | host I/O thread polling the miss ring during the token; P5b limited to VRAM+pinned until then | P5 |
| 14 | C-24 transient-capacity invariant | fix now | `n_slots - n_core >= max distinct per step`; selective-copy fallback | 4 |
| 15 | C-17 ballast is not unlike hardware | fix next | supported matrix; capacity simulation labelled; M1.4 real hardware | 2, M1.4 |
| 16 | C-21 probes lack a safety envelope | fix now | timeouts, stop rules, commit/cgroup floors, conservative default on failure | 6.1 |
| 17 | C-08 probe cannot certify the serving budget (unverifiable) | decided by hand: accept | live confirmation at requested context, peak ubatch and a competing ballast | 6.1, 7.5 |
| 18 | C-11 PERF: warm fill as a first-token barrier | fix next | staged warm fill: small synchronous core, the rest in the background | 4 |
| 19 | C-01 sidecar bound to metadata only (unverifiable) | decided by hand: accept | shard hashes and per-record checksums, verified; mismatch disables the sidecar | P3 |
| 20 | C-22 profile writes not crash-safe | fix next | temp file, rename, schema version, lock | P4 |
| 21 | C-14 direct-I/O alignment and fallback | fix next | sector-rounded reads, aligned buffers, bounded logged fallback | P3 |
| 22 | C-19 PERF: keep upload coalescing | park | adopted cheaply anyway: consecutive victims, upload-run counter | P2, P1 |
| 23 | C-06 distinct-expert counters are not lookup hits (unverifiable) | reject (as a defect) | adopted as a definition: the 85% is row-level, upstream's counter not used | 2 |
| 24 | C-18 PERF: measure 8 verify rows before capping at 7 | park | 8 rows kept as a measured arm in M3 | 8 |
| - | Brief challenges: define the 85%; a supported matrix; deployment and rollback | | 85% defined; supported matrix; launch checks, diagnostics, `--stock` rollback | 2, M1.3 |

Not run here, and needed: every "proposed read-only data check" in the critique needs the GPU, so all of them are folded
into M1.0 (C-03, C-07, C-10, C-13, C-20) or the milestone that owns them (C-08 in M1.3, C-11 in M1.2, C-18 and C-19 in
M3 and M1.2, C-23 in M1.3).

## 15. Levers ranked by expected decode gain (estimates, to be replaced by M1.0's numbers)

Gains are relative to upstream's SLRU cache at the same budget on the reference laptop, which is itself unmeasured.
They do not add; each assumes the ones above it are not yet built. Ranges are the consultants' and mine, reconciled.

| Rank | Lever | Class | Expected decode gain | What decides it | Sources |
|---|---|---|---|---|---|
| 1 | **Remove the per-layer drain** (P5a precise dependencies, or P5b sync-free GPU resolve) | E | +20% to +45% if the avoidable drain is 5-12 ms per token; near 0 if under 1.5 ms | M1.0 drain timing | FB, AS, CT |
| 2 | **Ranked residency**: warm start, ranked core, TinyLFU doorkeeper, transient slots, budgets by marginal gain (M1 itself) | E | +15% to +40% (hit rate from an estimated 70-80% to 85-90%) | M1.0 SLRU hit rate; replay | ST, MY, FB, AS, CT |
| 3 | **Speculation**: n-gram first, MTP if downloaded, window <= 7 rows, cost-aware length | D | +10% to +30% prose, +40% to +80% code and agent work | acceptance per window; Chris's preset decision | UP, ST, FB, AS |
| 4 | **Miss sourcing**: pinned staging, bounded ranked RAM tier, unbuffered NVMe tail | E | +10% to +30% steady state; removes the post-prefill collapse (stock fell 10.6 to 4.9 tok/s) | pinned vs pageable copy; endurance check | FB, AS, MY, ST, CT |
| 5 | **Look-ahead prefetch** (router l+1 on layer l's state, or an itinerary table) | E | +5% to +15% after lever 1, +10% to +25% before it | offline precision@10 from a trace | ST, LIT, FB, CT, AS |

Stacked, the consultants estimate the exact path at 35-50 tok/s [AS] or 45-60 tok/s [FB] after M2, against Strata's
35.5 tok/s answer-changing median; my own reading is that RAM sourcing (lever 4) decides whether those numbers survive
real sessions on a 32 GB machine.
