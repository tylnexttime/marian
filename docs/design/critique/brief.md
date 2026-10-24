---
title: "Marian M1 design critique: per-expert VRAM residency on upstream llama.cpp"
problem: "Marian is a thin fork of the latest llama.cpp (pin a37cc461, 21 Oct 2026) that must lift the VRAM expert hit rate of a large MoE model on constrained laptops from about 33% (stock whole-layer placement) to about 85% (Strata's measured figure), exact by default, self-calibrating to any machine. DESIGN-M1.md proposes building on upstream's new --moe-cache-mib cache. Find its defects AND tell us how to make it faster."
solution_paths:
  - ../DESIGN-M1.md
  - ../RESEARCH-PACK.md
  - ../ideas/claude-tyl-ideas-before-consult.md
  - ../ideas/astra-ideas.md
  - ../ideas/fable-ideas.md
  - <research>/llama.cpp/src/llama-moe-cache.cpp
cite_roots:
  - ../../..
  - <research>/llama.cpp
in_scope: "The design in DESIGN-M1.md: architecture, the patch set on llama.cpp, the layer on top, the calibration loop, the exactness contract and its tests, the measurement protocol, milestones, risks, open decisions. Performance: every way the design leaves decode, prefill or hit rate on the table on 8-16 GB VRAM / 16-32 GB RAM laptops."
out_of_scope: "Whether Marian should exist; Chris's four decisions (thin fork on latest llama.cpp; exact by default with opt-in answer-changing switches; built on the reference laptop; self-calibrating, not tuned for one machine). Style and wording."
context: "Reference laptop: RTX 4090 Laptop 16 GB, i7-13700HX AVX2, 32 GB DDR5, Windows 11, NVMe. Model: Qwen3.8-Flash-Next Swift 1.5 IQ2_XS (qwen4exp, 48 layers, 512 experts, 10 used, 33.02 GiB routed experts in three layouts, 26.82 GiB lazily read PLE table). Strata: 35.5 tok/s decode median, 85.4% hit from 9.16 GiB, with MTP, route-tail skip and CPU/GPU split. Stock llama.cpp -ncmoe 32: 7-10.6 tok/s. The research pack has the citations. Nothing in the design has been run on the new pin."
known_decisions:
  - "Exact mode means bitwise identical to the same build with every expert in VRAM of the same GPU (section 3). Answer-changing mechanisms are opt-in; do not report their being off by default as a defect."
  - "Build on upstream llama-moe-cache.cpp rather than port Strata's engine (recommended to Chris, section 12.1). Arguments against it are welcome if they are concrete."
  - "No GPU benchmarks were run for this design; the numbers marked estimate are arithmetic."
  - "Marian launches upstream llama-server bound to 127.0.0.1; no server of its own."
access_rules: "Read-only design material and public MIT source. No secrets, no personal data in any file. Do not print credentials or e-mail addresses if you meet any."
severity_scale: "Critical = the design cannot meet M1 (85% hit, exact, decode >= 12 tok/s) or breaks exactness; High = a large performance loss or a wrong measurement conclusion likely; Medium = a real but bounded loss or gap; Low = minor."
brief_author: "Claude Tyl"
solution_author: "Claude Tyl"
---

# Brief

Critique `DESIGN-M1.md` (version 1). The research pack and the three idea files are context: the design should have
used the best of them, and you may say where it did not.

We want two kinds of item, both as findings with evidence:

1. **Defects**: anything that would stop M1 meeting its exit criterion, break the exactness contract, give a wrong
   measurement conclusion, or fail on a machine unlike the reference laptop (8 GB VRAM, 16 GB RAM, Linux, a slower
   NVMe, PCIe 3.0, another quant layout).
2. **Performance improvements**: concrete changes that would raise decode tok/s, prefill tok/s or the VRAM hit rate
   on this class of hardware, beyond what the design already plans, with your estimate of the gain, whether the
   change is exact or answer-changing, and where it would live (llama.cpp patch or the layer on top). Mark each such
   item's title with "PERF:".

Check the design's claims against `llama-moe-cache.cpp` where it cites line numbers. Be specific: a finding
without a line of evidence or a number is hard to act on.
