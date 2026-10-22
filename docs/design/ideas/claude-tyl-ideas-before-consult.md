# Claude Tyl's own ideas, written BEFORE reading Astra's or Fable's answers (22/10/2026)

Written while Sol was still building the research pack, from my own reading of `BASELINE.md`, Strata's
`RESULTS.md` and docs, and upstream llama.cpp at `a37cc461`. Recorded first so the design can credit each idea to
whoever had it first. Numbers are my estimates unless they cite a measurement.

## The finding that reframes M1

Upstream llama.cpp merged its own GPU cache for host-resident MoE experts on 7 Oct 2026 (`--moe-cache-mib`,
`src/llama-moe-cache.cpp`, #29887; multi-GPU #30112; SLRU instead of LRU #30637, 19 Oct). It caches only layers whose
experts all live in host memory, keeps a per-layer slot map, and on each decode step (up to 32 tokens) reads the
selected ids back with a blocking `ggml_backend_synchronize`, plans fills with an SLRU, uploads the missing experts and
runs `MUL_MAT_ID` on the device bank with slot ids (`llama-moe-cache.cpp:549-617`). **Every expert is computed on the
GPU**, so it is the natural exact path. It has no frequency ranking, no warm start, no prefetch, no overlap of
uploads with compute, no RAM tier and no CPU compute of misses. So Marian should build on it, not beside it.

## My ideas

1. **M1.0: measure upstream's cache before building anything.** `-ngl 99 -ncmoe 48 --moe-cache-mib ~9400` on the
   Swift files. SLRU over about 6,800 slots with a skewed routing distribution may already reach 70-80% hit rate.
   The gap between that and 85% is what M1 has to close. Smallest experiment, highest information.
2. **Warm start from a stored routing profile.** Collect per-(layer, expert) routing counts with llama.cpp's
   existing eval callback on `ffn_moe_topk-<il>` (no patch: the same mechanism the imatrix tool uses), store them in
   Marian's profile, and preload the top-ranked experts into the cache's protected segment at start. Exact.
3. **Frequency-aware admission and eviction**: a pluggable policy in `llama-moe-cache.cpp` (SLRU stays the
   default), with decayed counts and a pinned core that is never evicted. Exact (placement only).
4. **Per-layer budgets from the profile** instead of the byte-proportional split per layout group
   (`llama-moe-cache.cpp:344-363`): layers whose routing is flatter need more slots for the same hit rate. Exact.
5. **Next-layer prefetch by pre-gating**: apply layer l+1's router to layer l's output to predict its experts, and
   upload predicted misses on a copy stream while layer l computes. Misprediction costs only bandwidth. Exact.
   Source: the Pre-gated MoE idea (Hwang et al., 2024) and ProMoE-style prefetch; my adaptation to llama.cpp.
6. **Remove the per-layer blocking synchronize** (48 per token today): double-buffered slot maps and event-ordered
   uploads on a second stream. On Windows WDDM a sync can cost tens of microseconds or more; 48-96 per token could be
   several milliseconds. Exact. Needs measuring first (telemetry below).
7. **A bit-faithful CPU twin of the GPU expert kernel** (the most out-of-the-box one). Strata computes misses on the
   CPU, which rounds differently, so in Marian that would be opt-in. But the GPU's MMVQ path for these quant types
   quantises the activation to q8_1, does integer dot products per block (exact in any order) and then a float
   accumulation in a fixed tree (Strata's own `EXPERTDEC.md` documents such a tree lane by lane). A CPU kernel that
   reproduces the q8_1 rounding and the float tree in the same order, with the same FMA contraction, would give the
   GPU's bytes. Then CPU compute of misses becomes **exact**, and the Strata/Fiddler-style hybrid can be the default.
   Risk: the tree depends on kernel choice (MMVQ vs MMQ by batch size) and on GPU architecture; it needs a bitwise
   test per (quant type, batch size, arch), and may run slower than ggml's own CPU kernels.
8. **Treat the OS page cache as an unmanaged RAM tier and manage it**: on a machine whose RAM cannot hold all
   experts, pin (or at least prefetch and lock) a frequency-ranked RAM tier, and read the cold tail from NVMe with
   unbuffered reads so a prefill pass does not flush the hot pages (Strata notes Windows reads its file tier
   unbuffered). The baseline's 2x run-to-run spread is page-cache state; managing it should also cut the noise.
9. **Prefill: let the freed VRAM buy a bigger ubatch.** With all expert layers host-resident and a 9 GiB cache, the
   VRAM headroom that killed `-ub 2048` at `-ncmoe 34` is back. Fewer passes over 22-24 GiB of host experts per
   prompt; upstream already copies cached experts device-to-device for large batches (`copy_experts`). The calibration
   picks the ubatch from measured VRAM, not a constant.
10. **WDDM budget, not cudaMemGetInfo**: the calibration should size the cache from the DXGI video-memory budget
    (QueryVideoMemoryInfo) with a margin, because the baseline fell off the sysmem-fallback cliff at 16,018 of 16,376
    MiB without any error.
11. **MTP sidecar**: the Swift GGUFs have no MTP tensors, but upstream supports qwen4exp MTP (#29761) and Strata gets
    93/130 drafts accepted from the base model's 5.2 GB MTP layer. Graft the base model's MTP tensors into a sidecar
    GGUF (or a rewritten shard). Exact (speculation is verified). Interacts with the cache: a 3-4 token verify window
    touches up to 30-40 experts per layer, so the cache must be sized for the window, not for one token.
12. **Per-token telemetry as a first-class output**: hits, misses, bytes and wait time per layer and token, written
    by a callback, so every claim about hit rate is a measured per-token distribution, not a run total.

## Rough decode arithmetic for the reference laptop (my estimate, to be measured)

- Per token: 480 routed lookups, about 1.38 MiB each on average (33.02 GiB / 24,576), so about 660 MiB of expert
  reads, plus about 3.3 GiB of dense weights (2.98 GiB per-layer dense + 0.31 output). At an effective 400-450 GB/s
  that is about 8-9 ms of GPU time per token, a ceiling near 110-120 tok/s before speculation.
- Misses at 85% hit: about 72 experts, about 99 MiB per token. Over PCIe 4.0 x16 from pinned RAM (about 24 GB/s) that
  is about 4 ms; from pageable RAM perhaps 8-10 ms; from NVMe at 2-3 GB/s, 33-50 ms. **So where the misses come from
  matters more than the last few points of hit rate**, and the RAM tier is as important as the VRAM tier on 32 GB.
