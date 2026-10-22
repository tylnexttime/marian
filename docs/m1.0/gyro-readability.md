# Gyro-S readability by llama.cpp a37cc461 (22/10/2026, metadata only)

Repo agentionai/Qwen3.8-Flash-Next-Gyro-GGUF @ 273edc1d2ffae5241fe3cab69568d682dad4c718, file
Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf (58,487,015,808 B). Read: the first 32 MiB only (HTTP range), GGUF v3 header parsed.

- Architecture `qwen4exp`, 48 blocks, 512 experts, 10 used, 1,224 tensors; `general.file_type` 25.
- Routed expert tensors: `ffn_gate_exps` / `ffn_up_exps` type id **145**, `ffn_down_exps` type id **146** (48 each).
  Metadata names `agention.trellis.state_bits = 15`, `agention.encoder = Agention rotor encoder (APR)` and a Hadamard
  rotation (`prism.hadamard.*`).
- llama.cpp at the pin defines ggml types up to Q2_0 = 42 (`ggml/include/ggml.h:432`, `GGML_TYPE_COUNT = 43`).
- **Verdict: llama.cpp at a37cc461 cannot load Gyro-S (nor, by the same encoder, Gyro-M/L).** It is a Strata-only
  format. Not downloaded. Note for decision 3: the same repo publishes `mtp-Qwen3.8-Flash-Next-draft.gguf`
  (2,697,439,648 B), a GGUF MTP draft for Flash-Next.
