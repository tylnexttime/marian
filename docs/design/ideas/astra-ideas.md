# Recommendation

**Keep upstream’s expert banks and GPU matmuls. Replace the cache’s decision-making and byte supply, not the inference engine.**

Marian’s default should be:

> Profile-led residency, bounded RAM, explicit NVMe reads, and every selected expert computed on the GPU through the reference numerical path.

The most promising departure from Strata is **optimising exposed stall time across all three storage tiers**, rather than maximising VRAM lookup hit rate alone. A slightly lower VRAM hit rate can win if its misses are predictable, RAM-resident and cheap. Conversely, an 85% hit rate can still lose badly if the remaining requests fault randomly from NVMe.

Two important qualifications:

- There is **no measured upstream SLRU result at the pin** in this pack. The 7–10.6 tok/s stock result predates that cache. We cannot honestly call the difference between that result and 35.5 tok/s the opportunity remaining for Marian. [RESEARCH-PACK §1, §3]
- Strata’s 35.5 tok/s run used route-tail skipping and a separate MTP draft, with possible CPU/GPU rounding differences. It is a useful performance target, not an exact-output comparator. [RESEARCH-PACK §5]

My proposed target is **35–50 tok/s exact decode on the reference laptop**, subject to the measurements below. That is an engineering target, not a prediction supported by an existing Marian run.

## The byte budget explains the opportunity

Use the addendum’s corrected geometry: **34 IQ2_S layers, 11 IQ2_XXS layers and three IQ1_M layers**. The earlier count of 32 IQ2_S layers is superseded.

With ten experts selected in every layer, the total selected expert-weight volume is about **660.4 MiB per decode token before hits**. This does not require uniform popularity within a layer, since experts within each listed layout have the same size. [RESEARCH-PACK, addendum “Swift expert bytes by layout”]

| Byte-weighted VRAM hit rate | Expert bytes uploaded per token |
|---:|---:|
| 80% | 132.1 MiB |
| 85.4% | 96.4 MiB |
| 90% | 66.0 MiB |
| 93% | 46.2 MiB |

Strata’s reported 85.4% is a **lookup** hit rate. Treating it as a byte hit rate in this table is an approximation, not a measured Swift byte-transfer result. [RESEARCH-PACK §1]

For illustration, at an assumed, subsequently measured **12 GiB/s effective pinned upload rate**, 96.4 MiB takes 7.8 ms and 66 MiB takes 5.4 ms. More importantly, if just 1% of the selected bytes reach NVMe, that is another 6.6 MiB per token. At an assumed effective 2 GiB/s it costs 3.2 ms before considering overlap or request latency.

An illustrative serial budget is:

- GPU computation and ordinary graph work: **13 ms**, assumed.
- Residual host coordination: **2 ms**, assumed.
- Uploads at 90% byte hit rate: **5.4 ms**.
- NVMe serving 1% of selected bytes: **3.2 ms**.

That totals **23.6 ms, or 42 tok/s**. At 85% VRAM hits and 3% reaching NVMe, the same model gives roughly **30 tok/s**. These are sensitivity calculations, not benchmark results. They show why **RAM coverage and coordination matter as much as the last few points of VRAM hits**.

---

# Recommended architecture

## The thin llama.cpp patch

Keep upstream’s:

- Layout-grouped expert banks and guarded extra slot.
- Expert-ID to slot-ID mapping.
- Existing expert projection and aggregation operations.
- Large-batch selective-copy path.

Upstream already addresses expert slices using `w->nb[2]`; ordinary tensor overrides cannot place individual expert slices independently. Its cache also requires all collected expert tensors of an eligible layer to be host-backed. Marian should therefore start with eligible routed expert tensors host-backed, while leaving their computation assigned to the GPU cache path. [RESEARCH-PACK §2, §3]

Add three interfaces:

1. **Residency policy interface**  
   Receives actual selected IDs and returns protected residents, admissions, victims and slot assignments.

2. **Expert-source interface**  
   Supplies unchanged byte ranges from a RAM lease or asynchronous file read. Both small-cache fills and large-batch selective copies must use it.

3. **Telemetry interface**  
   Reports routed IDs, demand hits, bytes by tier, evictions, copy completion, exposed waits and batch geometry.

The control plane lives in Marian’s existing Python layer. The hot policy and I/O implementation should be a small native library loaded into the process, not Python callbacks on every layer.

Do **not** make the first patch a general-purpose replacement scheduler. Event-based overlap and layer-major prefill are later, separately gated changes.

## The exactness contract

“Same bytes” alone is insufficient. Changing bank dimensions, expert grouping, batch size or backend splits can change kernel selection or reduction order.

The default must preserve:

- Router-selected IDs and their original order.
- Router weights, without skipping or renormalising anything.
- GPU arithmetic for every expert.
- Projection and aggregation arithmetic.
- Reference prefill geometry and sampling/RNG behaviour.

For every supported configuration, compare against a repeatable all-VRAM reference with the same build, settings and compatible GPU execution path. Use a sufficiently large same-architecture GPU for end-to-end tests, plus local single-layer all-resident tests.

**Automatic adaptation may move bytes. It may not silently change numerical paths.** If a smaller bank changes dispatch or arithmetic, that configuration stays experimental until addressed. If memory becomes unsafe, pause or reject the request rather than silently switching misses to CPU.

This distinction is required by the pack’s documented numerical boundaries. [RESEARCH-PACK §4, §5, §10]

---

# Ranked ideas

The ranges below are **incremental throughput changes against pinned upstream SLRU on a matched workload**, once that baseline exists. They are hypotheses, not measured results or additive gains.

The ordering is my expected reference-laptop payoff. Rows 2 and 3 could readily swap after measurement. Costs are approximate engineer-days, including a basic test harness but excluding broad release qualification.

| Rank | Idea | Expected decode effect | Build cost |
|---:|---|---:|---:|
| 1 | Cost-weighted protected core, transient slots and admission control | **0 to +40%** | 6–10 days |
| 2 | Bounded, mostly exclusive RAM tier and expert-major NVMe source | **0 to +30%** steady state; potentially larger under paging | 10–16 days |
| 3 | Cache-aware prompt speculation, then compatible MTP | **0 to +25%** mixed tasks; **+20 to +60%** favourable repetitive tasks | 4–7 days initially |
| 4 | Remove unnecessarily broad synchronisation and pipeline byte delivery | **0 to +20%** | 7–12 days |
| 5 | Predict expert itineraries, not output tokens | **0 to +15%** | 5–9 days |
| 6 | Optional concurrent CPU miss lane | **−20 to +15%** before automatic disabling | 12–20 days |
| 7 | Coordinate PLE reads with expert I/O | **0 to +5%** | 3–5 days |
| 8 | Memory-pressure governor and phase-specific storage budgets | **−5 to +5%** when already safe; avoids catastrophic collapses | 4–7 days |
| 9 | Exact prompt-boundary checkpoints | **About 0%** decode | 3–6 days |
| 10 | Weight-stationary prefill without widening ubatches | **About 0%** decode | 20–35 days |

## 1. Cost-weighted protected core, transient slots and admission control

**What it is**

Retain upstream banks, but replace unconditional upload-and-admit SLRU with three classes:

- A profile-ranked protected core.
- A dynamic region using decayed frequency and recency.
- Transient slots that serve misses without necessarily retaining them.

Transient slots matter. **Rejecting admission must not mean computing on CPU or skipping the expert.** Upload it into a transient bank slot, execute the same bank matmul, then allow that slot to be reused.

For single-token decode, reserve enough space to serve one layer’s selected set without evicting hard-protected residents. Verification needs a larger reserve or the existing selective-copy fallback.

Rank candidates initially by:

\[
\text{value}(e)=
\frac{\text{expected requests}(e)\times
      \text{measured exposed miss cost}(e)}
     {\text{VRAM bytes}(e)}
\]

Then jointly evaluate RAM placement: keeping an expert on GPU can free its RAM allocation for another expert that would otherwise reach disk.

Use:

- Stored routing profiles for warm fill.
- Per-layer soft quotas to prevent early layers dominating.
- Group budgets based on marginal saved time, not just source bytes.
- Decayed online counts, with hysteresis before replacing residents.
- Separate prefill and decode statistics.

Upstream currently divides group budgets by source weight bytes and uses 50/50 SLRU rather than persistent frequency ranking. Swift has three layout groups. [RESEARCH-PACK §3; RESEARCH-PACK addendum]

**Expected effect**

- **Decode:** 0–40%. Example: raising byte hits from 80% to 88% cuts miss bytes by 40%. If exposed miss work is 40–70% of token time, Amdahl’s law gives approximately 19–39% throughput improvement.
- **Prefill:** 0–10%, mainly through retaining useful decode residents and supplying more large-batch bytes device-to-device.
- **Hit rate:** working target **85–90% held-out lookup hits at about Strata’s expert budget**, not a guarantee. If upstream already reaches that, the gain may be negligible. Do not project this target onto an 8 GB GPU.

**Exactness and location:** Exact candidate. Policy in Marian’s native layer; small bank-policy and budget hooks in llama.cpp. Preserve computation and aggregation order.

**Smallest falsifier:** Record 512 decode tokens from two distinct tasks. Train on the first task, replay both policies on the second at identical byte budgets. If the hybrid saves neither transfer bytes nor estimated exposed miss time, its profile-policy premise has failed for that workload.

**Source:** Strata’s ranked profiles and learned heat; upstream SLRU; TinyLFU-style admission as a conceptual borrowing. The joint GPU/RAM stall-cost objective is my proposal. [RESEARCH-PACK §3, §5]

## 2. Bounded RAM and an expert-major NVMe source

**What it is**

Use a three-tier hierarchy:

1. VRAM.
2. A bounded, mostly exclusive RAM cache of GPU misses.
3. Explicit NVMe reads.

At 9.16 GiB of expert VRAM, Swift’s routed-weight complement is still about **23.86 GiB**. That is before the OS, server, activations, PLE pages and other allocations. It is not a sensible mandatory pinned allocation on a 32 GB laptop, let alone a 16 GB one. [RESEARCH-PACK §1; RESEARCH-PACK addendum]

Do not pin the whole RAM tier. Start with ordinary bounded RAM plus a **small pinned staging ring**, for example a measured 64–256 MiB search range. Pin additional hot buffers only if the measured benefit justifies the physical-memory pressure.

Provide an optional derived, expert-major sidecar:

- Each record contains an expert’s native gate, up and down bytes, or its fused-role equivalent.
- No requantisation.
- Alignment and padding appropriate to the device’s direct-I/O requirements.
- A full model identity and per-record checksums.
- Original GGUF files remain the source of truth.

This turns up to three separated role reads into one expert-record read. Uploads may still need separate copies into upstream’s role banks.

Use `O_DIRECT` where supported on Linux and overlapped unbuffered reads on Windows, with aligned buffers and rounded reads. Keep a buffered fallback, but bound it and avoid simultaneously faulting the same payload through mmap.

**Both cache fills and large-batch copies must use this source**, otherwise prefill will reintroduce the uncontrolled page-cache path.

**Expected effect**

- **Decode:** 0–30% on an already usable reference configuration. Potentially much larger during paging, but the pack does not isolate a causal gain to claim.
- **Prefill:** 0–40% where file-cache churn and fragmented reads dominate.
- **Hit rate:** VRAM hits roughly unchanged. Report **combined VRAM-or-RAM byte coverage**, targeting 98–99% only if the measured reuse curve supports it.

For example, cutting disk-served bytes from 3% to 1% saves 13.2 MiB per token, or about 6.5 ms at an assumed effective 2 GiB/s.

**Exactness and location:** Byte-preserving. I/O library and sidecar tooling above llama.cpp; source hooks in both upstream copy paths.

**Smallest falsifier:** Replay 256 tokens’ actual miss reads under a constrained RAM budget, alternating mmap and explicit I/O. If explicit I/O does not improve p95 delivery time, physical read volume or memory headroom, do not enable it on that machine.

**Source:** Strata’s bounded complement and unbuffered source; Maya’s bounded RAM/NVMe hierarchy. Expert-major packing and coordinated exclusive-tier admission are the proposed refinements. Maya’s measured throughput is not a Swift comparison. [RESEARCH-PACK §5, §6]

## 3. Cache-aware speculation, starting without a draft model

**What it is**

Start with upstream prompt-lookup or n-gram drafting. These require no missing MTP tensors and no extra multi-gigabyte model.

Choose the verification length by **accepted tokens per predicted millisecond**, rather than acceptance rate alone:

\[
\text{value}(T)=
\frac{\mathbb{E}[\text{committed tokens in the window}]}
     {t_\text{draft}+t_\text{verify}
      +t_\text{exposed transfers}
      +t_\text{cache repair}}
\]

Estimate transfer cost using recent per-layer expert unions. For Swift, a window of \(T\) tokens can select up to \(\min(10T,512)\) distinct experts in each layer.

Separate residency accounting from routing correctness:

- All verification experts must execute.
- Rejected branches should not get the same long-term promotion weight as committed tokens.
- Do not reserve a large permanent speculation cache.
- Start with \(T=2\) or \(4\), and disable speculation when committed-token throughput falls.

Upstream’s slot-map path is limited to 32 tokens and also has a conservative selected-set capacity check. Larger or unsuitable batches take the selective-copy path and can reuse existing residents without filling the SLRU. [RESEARCH-PACK §3]

**MTP:** The pin supports qwen4exp MTP, but these Swift files do not contain its tensors. A separately supplied draft must be proven compatible with this checkpoint, architecture and tensor naming. Strata fetched a 5.2 GB draft; that is a substantial resource and distribution decision, not a free flag. [RESEARCH-PACK §4, §5]

**Expected effect**

- **Decode:** 0–25% across mixed tasks, 20–60% on repetitive code or tool text. Hypothesis: 2.2 committed tokens in 1.5 single-token execution costs gives 1.47× throughput.
- **Prefill:** approximately 0%; drafting does not fix prompt ingestion.
- **Hit rate:** resident lookup hit rate may fall. The relevant measure is **expert bytes and milliseconds per committed token**, which may improve by 10–35% on favourable windows.

**Exactness and location:** **Opt-in initially.** Target verification and distributional correctness do not establish seeded token identity when batching, kernels or RNG consumption change. Marian controls upstream speculation and records costs. Compatible MTP adds roughly 5–10 integration days after the tensors exist.

**Smallest falsifier:** On one repetitive task with good draft acceptance, compare \(T=1,2,4\). If bytes and wall time per committed token rise despite acceptance, cache-aware speculation has no benefit there.

**Source:** Upstream drafting, Strata prompt lookup and MTP. Cache-cost-aware window selection is my proposal. [RESEARCH-PACK §4, §5]

## 4. Replace broad synchronisation with precise dependencies

**What it is**

The current callback reads expert IDs to the host and calls `ggml_backend_synchronize`. That establishes a real dependency, but may wait for more work than necessary. Its Windows cost is unmeasured. [RESEARCH-PACK §3, §10]

Proceed in stages:

1. Read IDs into preallocated pinned buffers.
2. Signal a routing-complete event.
3. Let a worker plan and fetch bytes after that event.
4. Upload through a copy stream.
5. Make the consuming bank operation wait on the appropriate copy-complete event.

Overlap this with **already independent GPU work**, where the graph permits it. Also pipeline NVMe read, staging and upload across several records.

Do not initially split a layer into “hit experts now, miss experts later”. That changes the expert operation’s shape and potentially its numerical path. Likewise, moving the map to the GPU does not magically remove host involvement when a miss needs disk I/O.

**Expected effect**

- **Decode:** 0–20%. Saving an assumed 25–100 µs across 48 layers saves 1.2–4.8 ms. Against a 28 ms token, that is roughly 4–21% throughput.
- **Prefill:** 0–15% from better I/O/copy pipelining.
- **Hit rate:** unchanged.

**Exactness and location:** Exact candidate if only dependencies and transfer timing change. Requires backend/context patching plus native workers.

**Smallest falsifier:** Time 64 warm decode tokens with GPU events and host tracing. If avoidable synchronisation contributes less than 1 ms/token and no independent work can overlap, abandon the ambitious version.

**Source:** Upstream’s demonstrated callback barrier and Strata’s concurrent dispatch. Do not bank on DMA batching: Strata’s microbenchmark barely changed total transfer time and established no full-engine gain. [RESEARCH-PACK §3, §5, §10]

## 5. Predict expert itineraries, not output tokens

**What it is**

Learn a lightweight predictor from observed routing:

- Input: layer number, recent expert IDs and a compact signature of the current route.
- Output: likely misses in one or two upcoming layers.
- Action: fetch bytes only.

This can be a sparse transition table before anyone trains a neural predictor. It avoids copying later routers to CPU and applying them to approximate activations.

Use deadline-aware admission:

- Demand reads always take priority.
- Speculative bytes have a strict queue and bandwidth budget.
- Prefer RAM prefetch.
- Prefetch into VRAM only if a free or genuinely low-value slot exists.
- Charge unused reads and cache pollution against the predictor.

The actual GPU router remains authoritative.

**Expected effect**

- **Decode:** 0–15%. If exposed file waits occupy 30% of token time and prediction hides one third, throughput rises by about 11%.
- **Prefill:** 0–10%; known prompt work offers other, simpler read-ahead opportunities.
- **Hit rate:** true demand residency may improve by 0–3 percentage points, but **late-miss rate** is the more useful metric. A speculative upload is not free merely because the subsequent request is labelled a hit.

**Exactness and location:** Byte-only, therefore exact candidate. Predictor and I/O policy above llama.cpp; existing routing telemetry supplies inputs.

**Smallest falsifier:** Train on 256 tokens and evaluate the next 256 causally. If timely useful reads do not repay wasted reads and pollution, keep it disabled.

**Source:** Strata’s byte-only router look-ahead. The route-transition itinerary predictor is my proposal. Strata’s reserved foresight GPU slots found no consistent whole-engine gain, so do not repeat that reservation blindly. [RESEARCH-PACK §5]

## 6. Optional concurrent CPU miss lane

**What it is**

An explicit experimental mode computes a share of RAM-resident misses on CPU while the GPU handles residents and uploaded misses.

Choose the share from measured per-layout completion times, not Strata’s fitted formula. NVMe-backed misses still require storage service whichever processor eventually computes them.

Start with existing ggml CPU operations rather than importing a second suite of expert kernels. Preserve selected-output positions for aggregation, but do not call that sufficient for exactness.

**Expected effect**

- **Decode:** −20% to +15% before automatic disabling. If exposed miss work is only 25% of a token, even halving it gives about 14% throughput improvement.
- **Prefill:** −20% to +10%, especially sensitive to memory-bandwidth contention.
- **Hit rate:** unchanged; the lane changes miss execution, not residency.

On this reference laptop I expect less upside than on a machine with a weaker PCIe path or stronger CPU relative to GPU.

**Exactness and location:** **Answer-changing opt-in.** Native dispatcher plus a non-trivial graph/backend patch. Each CPU-share setting must receive blind quality testing before being offered as a recommended non-exact preset.

**Smallest falsifier:** Benchmark one representative expert of each layout, then ten-expert groups under simultaneous GPU load. If CPU completion is slower and steals useful host bandwidth, do not build the full dispatcher.

**Source:** Strata’s concurrent CPU pool and PCIe lane. Its documentation explicitly acknowledges split-dependent rounding. [RESEARCH-PACK §5]

## 7. Give PLE reads a place in the I/O scheduler

**What it is**

Treat PLE and expert traffic as competing users of the same NVMe.

For known prompt tokens, schedule required PLE rows ahead. During decode, schedule rows as soon as the token is known. Coalesce reads where the actual tensor layout permits, and protect a small bounded set of useful rows from expert-streaming pollution.

Do not preload the entire table. The **26.82 GiB PLE table is lazy**, and upstream already has row prefetch. The proposal is coordination, prioritisation and bounded retention, not inventing that prefetch again. [RESEARCH-PACK §1, §2]

**Expected effect**

- **Decode:** 0–5%.
- **Prefill:** 0–15% on cold prompts if PLE faults interrupt expert streaming.
- **Expert hit rate:** unchanged.

These ranges assume PLE accounts for no more than a modest fraction of exposed I/O stalls. Measure it separately before spending more effort.

**Exactness and location:** Byte-only. Mostly source scheduling above llama.cpp; possibly a small row-source coordination hook.

**Smallest falsifier:** Attribute file waits by tensor for one cold 4K prompt and 128 generated tokens. If PLE contributes under 1% of exposed time, stop.

**Source:** Upstream PLE row prefetch; coordinated admission with expert I/O is my proposal. [RESEARCH-PACK §2, §4]

## 8. Memory governor with phase-specific storage budgets

**What it is**

Budget from actual physical and driver limits:

\[
V_\text{experts}
\leq
V_\text{available budget}
-V_\text{dense}
-V_\text{KV}
-V_\text{workspace peak}
-V_\text{transfer buffers}
-V_\text{reserve}
\]

Likewise, bound the RAM tier after accounting for OS/application reserve, dense host state, activation storage and staging.

On Windows, include DXGI’s local-memory budget and usage where available, shared-memory growth, physical RAM and commit. `nvidia-smi` usage alone is not enough.

React with hysteresis at safe execution boundaries. Drain outstanding users before releasing banks or buffers. Resizing must remain within parity-qualified bank configurations. If that cannot be guaranteed mid-session, pause admission rather than changing computation.

**Expected effect**

- **Decode:** −5% to +5% when already below the cliff, because reserve costs cache capacity.
- **Prefill:** similar steady-state trade-off.
- **Hit rate:** potentially a few points lower when reserving headroom.
- **Tail behaviour:** potentially several-fold recovery relative to a collapsed configuration, but this is risk avoidance, not a general speedup.

The baseline’s adjacent configurations fell from 10.59 to 3.56 tok/s decode and from 70.72 to 11.27 tok/s prefill near full VRAM. WDDM fallback is the pack’s plausible explanation, not directly proven paging attribution. [RESEARCH-PACK §1]

**Exactness and location:** Storage-only adaptation above llama.cpp, with safe release/rebuild hooks where needed.

**Smallest falsifier:** Apply controlled competing GPU and RAM allocations during a short run. If Marian enters the same collapse before throttling or pausing, the governor has failed.

**Source:** M0 resource profiling, Strata’s physical/commit checks and the observed reference cliff. [RESEARCH-PACK §1, §5, §8]

## 9. Exact prompt-boundary checkpoints

**What it is**

Save a bounded checkpoint at an already completed reference prefill boundary. Reuse it for matching prefixes, including tool-loop branches.

The checkpoint must include **all model execution state needed to resume**, not an assumption that KV alone is sufficient for every architecture. Match model, adapters, token sequence, numerical settings and boundary geometry.

Do not claim exactness if restoring at a different chunk boundary changes the remaining computation.

**Expected effect**

- **Decode:** about 0%.
- **Prefill/TTFT:** 0–90% less repeated prompt work. At 70% reusable work, the simple proportional-work model gives 3.3× effective ingestion throughput.
- **Expert hit rate:** unchanged as a policy, but many expert reads disappear because corresponding prompt work is not repeated.

Strata reported 69.5% prompt-cache reuse in its agent session, illustrating the importance of this workload dimension. [RESEARCH-PACK §1]

**Exactness and location:** Exact candidate at qualified boundaries. Marian manages checkpoint selection and eviction; use existing server/state mechanisms where complete.

**Smallest falsifier:** Prefill a prefix, branch twice, and compare restored versus recomputed logits at the boundary and following tokens. One mismatch rejects that checkpoint path from exact mode.

**Source:** Strata’s retained conversation state and bounded prompt-cache-tail checkpoint. [RESEARCH-PACK §5]

## 10. Weight-stationary prefill without widening ubatches

**What it is**

This is the boldest idea and the least compatible with an immediate thin-fork deadline.

Keep the reference ubatch shape, but change execution order:

```text
Current:
    chunk 0: layers 0 ... 47
    chunk 1: layers 0 ... 47
    ...

Proposed:
    layer 0: chunks 0 ... N
    layer 1: chunks 0 ... N
    ...
```

Load a layer’s needed expert weights, process its prompt chunks, then release the layer weights. Store intermediate activations between layers in bounded RAM.

A plain FP32 `[tokens, 2560]` activation array is about 40 MiB at 4K tokens or 160 MiB at 16K, **before qwen4exp’s additional live state**. Inspect and budget that state rather than assuming one activation array is sufficient.

Preserve each chunk’s:

- Attention context extent and masking.
- KV update order.
- Tensor shapes and arithmetic.
- Cross-layer dependencies and auxiliary state.

This is not simply a loop interchange in today’s server.

**Expected effect**

- **Decode:** about 0%.
- **Prefill:** speculative **+20 to +100%** on long, storage-bound prompts.
- **Hit rate:** decode hit rate unchanged. Prefill should be measured by unique source bytes and reread amplification instead.

At 4K with 512-token chunks, the conventional traversal has eight opportunities to reread each layer’s host weights. Layer-major execution can approach one union read per layer, rather than one per chunk. That does **not** imply 8× total speed: GPU computation remains, selections overlap imperfectly, and activation handling costs time.

**Exactness and location:** Exact candidate in principle, **experimental until parity is demonstrated**. It needs a meaningful execution-scheduling patch and Marian activation-storage control. If it requires replacing the graph executor, defer it beyond M1.

Simply increasing ubatch size is a separate, potentially answer-changing option. The pack’s 512 versus 2048 measurements are confounded by VRAM pressure and do not establish that larger ubatches are intrinsically slower. [RESEARCH-PACK §1, §5]

**Smallest falsifier:** Implement a two-layer, two-chunk replay with unchanged operations. A logit/intermediate mismatch disproves the exact implementation. If weight bytes do not fall substantially or activation handling erases the saving, stop before building a full executor.

**Source:** My application of weight-stationary execution to fixed-geometry llama.cpp prefill. Strata’s larger chunks motivate the reuse opportunity but are not this mechanism. [RESEARCH-PACK §5, §9]

---

# Self-calibration and adaptation

## First start

Reuse M0’s process adapter, machine inspection and result storage, but fix the gaps documented in the pack. In particular, M0 currently inspects only GGUF headers, uses a partial-file model identity and does not restore `moe_cache_mib` in `plan.to_config`. [RESEARCH-PACK §8]

The first-start procedure should:

1. **Inspect every shard’s tensor directory.**  
   Record native type, dimensions, expert stride, role offsets, fused roles and alignment. Do not hard-code Swift’s three layouts.

2. **Establish model identity.**  
   Use verified full-shard hashes or a verified content manifest. A hash of one file’s first 64 MiB is insufficient for transferable expert profiles.

3. **Measure the actual pin.**  
   Run upstream SLRU before modifying policy. Capture per-token demand counts, uploaded bytes, file reads and exposed stalls.

4. **Measure the machine’s useful service rates.**  
   Test native expert-sized reads at several queue depths, pinned uploads, staging copies and concurrent I/O/upload behaviour. Measure latency distributions, not only peak bandwidth.

5. **Measure serving allocations.**  
   Include the requested context, KV representation, prefill workspace and copy buffers. Do not infer a safe serving budget from depth-zero decode.

6. **Replay candidate policies offline.**  
   A short route trace can cheaply eliminate bad budget/admission combinations before running expensive whole-engine sweeps.

7. **Confirm only the finalists live.**  
   Aim for a short initial calibration, followed by passive refinement. Maya’s 95-minute calibration is a warning against exhaustive noisy sweeps. [RESEARCH-PACK §6]

## Re-measure when it matters

Re-run relevant parts after:

- Driver or engine changes.
- Model, adapter, context or KV-setting changes.
- Sustained routing drift.
- Large changes in available VRAM/RAM.
- Power-mode or thermal changes.
- A model-file move to another storage device.

Keep **routing profiles separate from machine timing profiles**. A routing prior may transfer between machines using identical model content. PCIe rates and safe memory budgets should not.

For 8 GB VRAM and 16 GB RAM machines, use the same mechanism with smaller measured tiers. Do not promise 85% VRAM hits. Publish the measured cache-capacity curve and predicted disk demand before claiming useful throughput.

## Persistence and sharing

Persist:

- Full model and adapter identity.
- Tensor layout schema.
- Engine/build and numerical configuration.
- Frequency and transition statistics, with sample counts.
- Byte-weighted hit curves.
- Machine timing distributions.
- Profile age and workload category.

Shared profiles are **untrusted priors**, not executable configuration. Validate sizes and IDs, cap allocations, and allow local observations to override them.

Keep sharing explicit and offline by default. Routing statistics can reveal workload characteristics even without prompt text. Do not bundle automatic model or MTP downloads into calibration. The pack separates Strata’s engine from its launcher and download trust surfaces; Marian should maintain that separation. [RESEARCH-PACK §7]

---

# Two alternatives rejected

## 1. Replace upstream execution with a Strata-style custom engine

This could provide CPU/GPU overlap and specialised kernels sooner if importing substantial code.

I reject it for M1 because it replaces the very arithmetic and scheduling surface Marian needs to match. It also creates a large maintenance burden beside fast-moving upstream work. Upstream already supplies slice addressing, banks, slot maps and selective copies. [RESEARCH-PACK §2–§5]

Borrow mechanisms and data formats where appropriate, not the whole execution engine.

## 2. Pin the whole host complement, with mmap as the overflow plan

This is simple and can look excellent on a warm benchmark.

I reject it because Swift’s host complement at the reference expert-cache size is roughly 23.86 GiB. It leaves inadequate room on many 32 GB systems and cannot work on 16 GB systems. Letting mmap quietly absorb overflow recreates the uncontrolled paging behaviour Marian is meant to avoid. [RESEARCH-PACK §1, §5, §6]

Use a measured bounded tier and explicit file service instead.

---

# Principal failure modes

| Failure | Consequence | Required response |
|---|---|---|
| Profile overfits one task | Good demo, poor real requests | Held-out task evaluation, decay and dynamic capacity |
| Bank shape changes CUDA dispatch | Byte-correct storage, different answers | Dispatch audit and bitwise reference tests |
| Transient capacity is too small | Verify-window failure or eviction of live weights | Capacity checks and existing selective-copy fallback |
| Slot reused before all readers finish | Silent corruption | Generation-tagged leases and completion dependencies |
| Pinned memory consumes OS headroom | Paging, WDDM trouble, unstable latency | Small measured staging budget; bounded ordinary RAM |
| Prefetch competes with demand I/O | Higher p95 latency despite more “hits” | Strict demand priority and wasted-byte accounting |
| Sidecar and model disagree | Wrong weights served | Full identity binding and record verification |
| Large-batch path bypasses the source tier | Prefill returns to mmap thrashing | Test and instrument both copy paths |
| Speculation pollutes residency | Acceptance improves while decode slows | Optimise milliseconds per committed token |
| Layer-major execution changes attention geometry | Numerical divergence | Keep experimental until fixed-geometry parity holds |
| Telemetry adds substantial overhead | Optimiser measures its own cost | Buffered tracing and an instrumentation-off control |

---

# The smallest test that can prove the design wrong

**The first test should be an exactness test, not a throughput benchmark.**

Take one Swift layer of each layout and frozen input activations. Run the reference all-resident operation, then run the proposed bank path with:

1. Every selected expert resident.
2. One forced miss.
3. Permuted slot assignments.
4. An eviction followed by reuse.

Compare projection outputs and final layer output bit-for-bit.

**One mismatch is enough to reject the claim that this implementation is exact.** Correct copied bytes do not excuse it.

After that, the smallest useful performance screen is a held-out routing trace plus measured I/O costs. If upstream SLRU already reaches the hybrid policy’s hit curve, file waits are negligible and synchronisation is cheap, most of this design’s predicted decode upside disappears. That would be a valuable result, not a reason to manufacture another cache policy.

## Delivery order

1. **Days 1–3:** pin baseline, byte inventory, telemetry and exactness harness.
2. **Next 1–2 weeks:** profile-led admission and bounded byte source.
3. **Then:** precise synchronisation and byte-only prediction, guided by measured stalls.
4. **Separately opt-in:** cache-aware speculation and CPU misses, each with blind quality evaluation.
5. **Later research branch:** fixed-geometry layer-major prefill.

The best chance of beating Strata while respecting Chris’s constraints is not a more aggressive approximation. It is **less compulsory data movement, fewer uncontrolled faults and less exposed waiting, while leaving the model’s arithmetic alone**.