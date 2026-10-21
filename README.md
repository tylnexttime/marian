# Marian

**Run local LLMs by what your machine can actually do, measured, not guessed.**

Marian measures how a model behaves on *this* machine (speed and memory for each context size, KV-cache type and
GPU/CPU split), then picks the fastest configuration that fits a given request, and launches it. It is runtime-
agnostic: an engine is a small adapter, so new runtimes can be measured the same way when they appear.

Named after a popular Polish given name. Status: **M0, pre-release** (21/10/2026).

## Why

On consumer GPUs the right settings depend on the context you need. Measured on an 8 GB laptop GPU: a mixture-of-
experts model ran at 31 tok/s with one GPU/CPU split for short prompts, but the same split at an 8K context overflowed
VRAM and fell to 6 tok/s, because Windows pages GPU memory into system RAM instead of failing. The cure is not a
better guess; it is a measured table per machine, model and context size.

## Quick start (standard library only, Python 3.10+)

```
python -m marian machine                                   # GPU, VRAM, RAM, driver; a fingerprint, no hostname
python -m marian inspect model.gguf                        # architecture, layers, experts, MoE or dense
python -m marian profile model.gguf --engine path/to/llama-server --ctx 4096,8192,16384 --kv f16,q8_0
python -m marian plan model.gguf --need 6000               # the fastest measured configuration for 6,000 tokens
python -m marian serve model.gguf --engine path/to/llama-server --need 6000
```

`plan` refuses when nothing measured fits: it never extrapolates.

## How profiling works

For each context size and KV type, Marian walks the GPU/CPU split from the most conservative to the most aggressive
(for MoE models: experts of the first N layers kept in system RAM, `--n-cpu-moe N`; for dense models: layers on the
GPU, `-ngl N`). Each point starts a real server bound to 127.0.0.1, fills about 75 % of the context, generates a fixed
number of tokens, and records prompt and generation speed plus peak VRAM and the lowest free RAM. A point is an
**overflow** when VRAM is within a margin of full or speed collapses against the best point so far; the walk stops
there. Profiles are JSON files named by machine fingerprint, model id and engine.

## Roadmap

- **M0 (this):** machine fingerprint, GGUF inspection, profiler, planner and launcher for llama.cpp.
- **M1:** tiered expert storage (VRAM, pinned RAM, NVMe) and calibration, as a thin patch set on upstream llama.cpp
  (not a hard fork), with general parts offered upstream. Inspired by Project Maya (MIT) and Strata (MIT).
- **M2:** experts served from another machine on the LAN.
- Adapters for other runtimes (Strata, Project Maya) behind the same profile format.

## Licence

MIT (see LICENSE).
