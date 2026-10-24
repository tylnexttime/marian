# Marian M1.2 groundwork: the routing-id trace (patch 0004) and `marian replay`

24/10/2026, a cloud session (Linux, no GPU) for Claude Tyl and Chris. **Nothing in this note ran on a GPU.** Everything
here was built and tested on a CPU: the trace writer by its own unit test (Linux, and Windows under wine), the replay on
SYNTHETIC traces and against the fork's own SLRU and P3 tier code compiled from the patched tree. Every number below says
whether it is measured here (CPU), taken from earlier notes, or synthetic. It does not depend on the P3 arms that are
being timed on the laptop now.

## 1. What was built

| What | Where | State |
|---|---|---|
| Patch 0004, `MARIAN_MOE_TRACE_IDS`: every routed expert id of every MoE layer step, hits included | `docs/m1.2/patches/0004-Marian-P1-routing-id-trace-of-every-MoE-layer-step-M.patch` | applies with `git am` on 0000 + 0001-0003; CPU build clean; unit test passes |
| `marian geometry`: expert geometry from a GGUF's tensor directory (headers only) | `marian/geometry.py`, `marian/gguf.py` (`tensor_infos`) | tested |
| Swift 1.5 IQ2_XS geometry | `docs/m1.2/swift-1.5-iq2_xs.geometry.json` | from the public shards' headers (section 4.1) |
| `marian replay`: SLRU exactly as the fork runs it, P2 candidates, RAM-tier policies for P3, a check against the server's own traces | `marian/replay.py`, `marian/idstrace.py`, `marian/cli.py` | 38 tests pass (`tests/test_replay.py`) |
| `marian synth`: SYNTHETIC routing traces shaped like a `run_arm` session | `marian/synth.py` | used by the tests only |
| Cross-checks against the fork's C++ | `docs/m1.2/tools/slru_xcheck.{sh,cpp}`, `tier_xcheck.{sh,cpp}`, `make_xcheck_inputs.py`; fixtures in `tests/data/` | Python equals C++ on every step (section 4.3) |
| Laptop build script for the patched fork | `docs/m1.2/tools/build_m12.bat` | not run (Windows only) |

## 2. CPU build log summary

Upstream `https://github.com/ggml-org/llama.cpp` at `a37cc4610` ("models : add Flint model", the design's pin), on
Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, `cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF` (Release, OpenMP, no BLAS).

| Step | Result |
|---|---|
| `git am` 0000-p0-trace, 0001 exactness floor, 0002 fill trace, 0003 P3 minimal | all four apply cleanly |
| Build of that stack (all 119 targets, including `llama-moe-source.cpp` on its Linux `O_DIRECT` path) | exit 0, **no compiler warning, no error**. Nothing CUDA-specific blocked the CPU build; no workaround was needed |
| 0004 on top, rebuild | exit 0, no warning (`llama-context.cpp`, `llama-moe-cache.cpp`, `llama-moe-trace-ids.cpp` rebuilt) |
| `test-moe-trace-ids` (new, CTest label `main`) | passes: format byte for byte, ids that do not fit in u16, the large-batch strides, the background flush, the destructor flush, an unopenable path |
| The writer's Windows branch (`CreateFileW` / `WriteFile`): mingw-w64 GCC 13 `-Wall -Wextra`, run under wine 9.0 | compiles clean, the test passes, and its sample file is **byte-identical** to the Linux one |
| `-Wall -Wextra -Wpedantic` (GCC) and clang on the writer | clean |
| Fresh worktree at `a37cc461` + the five patches with `git am` | applies; the tree equals the one built and tested (`git diff` empty) |
| llama.cpp's own CPU tests (`ctest -L main`) after 0004 | 51 of 54 passed together. The other 3 are not code failures (section 9). Run alone, all pass except the one that needs git-lfs |

Not done here: an MSVC build (the laptop's compiler) and an `nvcc` compile check. 0004 touches no `.cu` file, so CUDA
cannot be affected through it. MSVC is covered only indirectly, by mingw on the Windows branch of the writer.

## 3. Patch 0004: the routing-id trace (P1 in the design)

`MARIAN_MOE_TRACE_IDS=<file>` makes every MoE layer step whose ids reach the host write one record with all of its
routed ids, hits included:

- **S** (slot-map lookup, ubatches of up to 32 tokens): written in `llama_moe_cache::impl::copy()` right after the ids
  are read and before the plan. Its `t_us` is the same variable as the P0 S line's `t_us` and the F lines' `t_us`, so the
  three traces join exactly on (`t_us`, `layer`).
- **L** (large-batch selective copy in `sched_copy_experts`): one record per layer and ubatch, written when the first
  copy of the layer reads the ids. Its `t_us` and layer are those of that layer's first L line. Prefill routing
  therefore reaches the routing profile too (design C-05).

**Off by default.** When off, the only cost is one cached-pointer test per layer step: the same pattern as P0's
`llama_moe_trace_file()`. **When on, nothing that is computed changes.** The record copies ids the host already read
into `ids` / `st.ids_data`: no extra synchronize, no extra device read, no change to the plan, the slots or the uploads.
Records are appended to a memory buffer. A writer thread moves the buffer to the file every 100 ms (or at 1 MiB), so
the compute thread never waits on the disk. `record()` measured **70 ns per single-token step with 10 ids** on this
cloud x86 CPU (97 ns under wine), i.e. about 3.4 µs per Swift token of 48 layers. That is a CPU measurement, not a GPU
run.

Shutdown safety:

- The file is written with the OS API (`write` / `WriteFile`), not stdio, so the C runtime's exit-time stream cleanup
  never touches it while the thread runs.
- The writer is never destroyed (a leaked singleton), so no thread is joined during DLL unload.
- `~llama_context` flushes it.
- A killed server (`run_arm` uses `Stop-Process -Force`) loses at most the last 100 ms of records; a cut last record is
  dropped by the reader and counted.
- If the file cannot be opened, the trace is off and a WARN says so.

### 3.1 Format (documented in `src/llama-moe-trace-ids.h`)

Little-endian, packed, no padding.

```
header, 32 bytes:  char[8] "MRNIDS1\n" | u32 header_bytes = 32 | u32 version = 1 | u32 id_bytes = 2 | u32 0 | i64 t_open_us
record:            u8 kind ('S' 0x53, 'L' 0x4c) | u8 k | u16 layer | u32 n_tokens | i64 t_us | u16 ids[n_tokens*k]
```

Ids are written row by row (token by token), each row in the router's order (rank 0 first), as `ffn_moe_topk`
returns them. 0xffff marks an id that does not fit in u16 (never the case for 512 experts). Readers skip to
`header_bytes`, so fields can be added later.

**Size:** a single-token decode step of a 10-of-512 model is 16 + 20 = **36 bytes**. **2,400 decode tokens x 48 layers =
4,147,200 bytes = 3.96 MiB.** A prompt ubatch of n tokens adds 48 x (16 + 20 n) bytes. For the run in section 5 the
estimate is **about 3.9 MiB**. That uses `r2-c9400`'s phases warm + decode: 2,221 generated tokens, 272 prompt tokens in
11 ubatches, plus the server's 2-token warm-up. For scale, the P0 CSV of `r2-c9400` is 5.2 MiB.

## 4. `marian replay`

```
python -m marian replay TAG.ids.bin (--geometry G.json | --gguf SHARD1 SHARD2) --budget-mib 9400[,6500,...]
       [--requests TAG.requests.jsonl] [--trace TAG.trace.csv] [--fills TAG.fills.csv] [--test-prompts decode:1,decode:3]
       [--p2 core25,core50,door,guard,core50+door] [--tier-mib 1024,2048,...] [--tier-policies ...] [--md OUT] [--json OUT]
python -m marian geometry SHARD1 SHARD2 -o G.json
python -m marian synth OUTDIR            (SYNTHETIC; for trying the tool without a GPU)
```

Standard library only (Python 3.10+). Deterministic: the same inputs give the same report, byte for byte (tested).

### 4.1 Geometry and slot counts

The geometry is per MoE layer: the expert tensors in the cache's order (gate_up, gate, up, down), with ggml type, shape
and bytes per expert (`nb2`), plus `n_expert_used`. `nb2` comes from the pinned ggml's type table (printed from
`ggml_blck_size` / `ggml_type_size` of the built tree) and is checked against the gap to the next tensor's offset.

Slot counts follow the fork's constructor (0001 included):
- groups by identical layout, in layer order;
- `alloc_size = sum pad(nb2*(n+1), 128)`;
- the floor `min(min(n_ubatch,32)*n_expert_used, n_expert)`;
- the split by host bytes, iterated while floors bind;
- a refusal below the sum of the floors.

The Swift geometry comes from the public shards `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `99bb8f7f`.
Only the first 32 MiB of each shard was read (an HTTP range request); their sha256 are in the JSON. It gives 48 layers,
33.02 GiB of routed experts, and groups iq2_s 34 x 1,510,400 B, iq2_xxs 11 x 1,305,600 B, iq1_m 3 x 1,177,600 B. These
are the byte sizes M1.1's probe read from the laptop's files. The replay's slot counts reproduce **M1.1-NOTES section
2's table at all four budgets** (2,400: 1,105/356/320; 4,000: 1,973/637/320; 6,500: 3,329/1,076/320; 9,400:
4,838/1,564/426). They also reproduce the 1,223 MiB refusal and the floor binding below about 7,066 MiB (tests).

### 4.2 What it simulates

**Per-token hit rate.** The design's section 2 definition: per single-token decode step, the router selections
resident before that step's fills, over 48 x 10. It is reported as the median and 10th percentile over tokens, plus the
byte-weighted rate and MiB uploaded per token. Tokens are selected exactly as `analyse_m10.py` selects them, so the
replay and the server are measured over the same tokens.

**Split by prompt, never by token.** Each request maps to its prompt; `warm` is decode prompt 0 again and `post` k is
decode prompt k, so neither can leak across the split. By default the sorted prompts alternate train/test (decode:0
train, decode:1 test, ...); `--test-prompts` overrides this. The whole session is replayed in its real order (the cache
warms across prompts, as in a server). Metrics are taken on the test prompts; every profile is built from the train
prompts only.

**VRAM (P2 candidates against SLRU at the same budget).**

| Policy | What |
|---|---|
| `slru` | upstream's SLRU, line for line as in the fork (section 4.3) |
| `coreP` | a ranked core: P% of each group's slots, capped so the evictable rest still holds one 32-token ubatch (C-24), filled before the first token with the experts the train prompts' decode tokens selected most, never evicted; SLRU on the rest |
| `door` | a TinyLFU doorkeeper (4-bit count-min sketch, depth 4, halved every 10 x n_slots increments) in front of SLRU's victim. A miss is declined when the victim's estimate is higher than its own (ties admit). Declined misses use `n_expert_used` transient slots taken from the budget: uploaded and computed, not kept |
| `guard` | transient slots for prompt chunks: steps above 8 tokens admit nothing while transient slots last (up to one 32-token ubatch, within C-24) |
| `a+b` | combinations, e.g. `core50+door` |

These are my readings of DESIGN-M1 section 4 and P2: P2 itself does not exist yet. The parameters are choices, listed
in "Not verified".

**RAM tier for P3.** The tier is replayed on the SLRU's miss stream. The metric is the share of the test prompts'
decode-token miss bytes that the tier serves.

| Policy | What |
|---|---|
| `current` | the M1.1 tier exactly as `llama_moe_source` runs it (section 4.3): per-group split by host bytes, sector-rounded slots, decayed counts, CLOCK, count admission, busy stamps, ring wraps |
| `warm` | `current` starting from the state it reaches on the train prompts' misses (P4's warm start of learned counts and contents) |
| `static-sel` | static tier from the train prompts' decode selection counts |
| `static-miss` | static tier from the train prompts' VRAM misses at the same budget |
| `oracle-static` | the best static tier in hindsight (the test prompts' own misses) |
| `belady` | Belady MIN with bypass on the counted misses: an upper bound for every policy (exact for equal sizes; the three Swift slot sizes differ by up to 21%, so approximate there) |

Each row also shows the per-token fetch time read off the probe's linear model. That model is a given, not a replay
output: 46 ms at 0%, 15 ms at 75%, 8 ms at 90%, from the 24/10 probe write-up. The report ends with **the smallest tier
size in the sweep at which each policy reaches 75% and 90%**.

**Check against the server.** With `--trace` and `--fills` from the same run, every S line (row_hits, uniq, uniq_hits,
fills, bytes) and every step's uploads (expert and slot, in order) are compared with the replay. The server's own
median per-token hit rate is computed from its S lines, and the difference in points is printed. At the traced budget
an exact replay matches every step, so the M1.1 exit check ("within 3 points") should come out at 0.00, and any
mismatch points at the step.

### 4.3 Proof that the two "exactly as the fork" parts are exact

| Part | Against | Result |
|---|---|---|
| SLRU | `moe_cache_slru` cut **verbatim** from the patched `src/llama-moe-cache.cpp` by `slru_xcheck.sh` (139 lines, sha256 `8d4800325971dfd8`) and compiled with a driver; 403 steps (3 layers, 16 experts, 12 slots, one refused step) and 1,500 steps (4 layers, 512 experts, 10 per row, up to 32 rows, 420 slots) | identical on every step: row hits, hits, refusals, and every (expert, slot) |
| P3 tier `current` | the real `llama_moe_source` (patch 0003's `llama-moe-source.cpp`), CPU backend as the device, a random-bytes file as the GGUF (`tier_xcheck.sh`); three SYNTHETIC miss streams of 1,902 steps with tiers of 2/4, 13/33 and 61/149 slots, rings of 3, 2 and 8 slots (17 and 64 ring wraps) | identical on every fill ('t' / 'a' / 'r'), slot counts and ring sizes equal |
| SLRU worked by hand | 7 steps, 4 slots, segments checked after each step | `SlruHand` test |
| Slot counts | M1.1-NOTES section 2 table | equal at 2,400 / 4,000 / 6,500 / 9,400 MiB |

Known-answer tests on SYNTHETIC Zipf data:
- a cache that holds everything uploads each selected expert exactly once;
- large batches never change the cache;
- a static top-M tier on an IID Zipf(1) stream serves sum(top-M p) within 1 point;
- Belady >= the best static tier >= the current policy (minus noise) on a stationary stream;
- Belady by hand;
- a tier that holds everything admits each decode-missed expert exactly once, and prompt steps never admit;
- the ranked core respects C-24 and hits on the first token;
- the fetch model returns 46/15/8 at 0/75/90%;
- the split never shares a prompt.

### 4.4 Synthetic demonstration (SYNTHETIC: says nothing about Swift's routing)

This run used Swift's real geometry with ids from `marian synth` (48 layers, 512 experts, 10 per token, Zipf 1.0, 3
topics, stickiness 0.3; 10 prompts x 256 tokens, warm 64). The trace had 126,528 steps and an 8.0 MB ids file; the
generator took 2.0 s and the full report at 9,400 MiB took 41.5 s on this CPU. The synthetic routing is much flatter than
Swift's: SLRU 75.6% median, against the 90.8% measured on the laptop in M1.0. Its tier shares therefore say nothing
about the real answer.

They do show that the report works and that the bounds come out in order. At 8 GiB on the synthetic trace: current
27.7%, static-miss 33.4%, oracle-static 40.3%, Belady 53.0%. On these synthetic ids, `core50` raised the median hit
rate by 1.7 points, `door` lowered it by 4.8 and `guard` by 1.9. Real ids decide.

## 5. The one GPU run that produces a real ids trace

Arm: Swift 1.5 IQ2_XS, `-ncmoe 48 --moe-cache-mib 9400`, phases `warm,decode`. It is the E1 `e1-c9400` arm of
M1.1-NOTES section 7, run with the M1.2 build and `MARIAN_MOE_TRACE_IDS` set. It follows the measurement chain's rules:
- only when nothing else uses the GPU (not during the P3 arms);
- GPU-BUSY held, as for every GPU run;
- outputs under `m1.2\`, never `m1.0\runs` or `m1.1\runs`.

**5.1 Get the code (no GPU; touches no existing tree or build):**

```powershell
git -C <repo> fetch origin m1.2/replay
git -C <repo> worktree add <repo> origin/m1.2/replay
git -C <research>\llama.cpp worktree add -b marian/m1.2 <research>\llama.cpp-m1.2 marian/m1.1
git -C <research>\llama.cpp-m1.2 am <repo>\docs\m1.2\patches\0004-Marian-P1-routing-id-trace-of-every-MoE-layer-step-M.patch
git -C <research>\llama.cpp-m1.2 am <repo>\docs\m1.2\patches\0005-Marian-P1-review-fixes-S-record-head-and-writer-guards.patch
git -C <research>\llama.cpp-m1.2 log --oneline -5
```

The log must show the 0004 commit and the 0005 review fixes on top of `ebdf01e8`. **Use absolute paths for the
worktree:** with `git -C <research>\llama.cpp`, a relative worktree path is resolved inside the fork (this happened
once on 24/10). **0005 is required:** without it the S records are mis-headed (section 10).

**5.2 Build (CPU, about 15 minutes; GPU-BUSY absent, and not during CPU- or GPU-timed runs):**

```powershell
cmd /c <repo>\docs\m1.2\tools\build_m12.bat 8
```

It must print `BUILD OK`. The script builds into `build-m1.2` from `llama.cpp-m1.2` with build_m11's toolchain and
switches, so `build`, `build-m1.1` and `llama.cpp` stay untouched.

**5.3 The run (GPU, about 25 minutes, like E1's arms):**

```powershell
New-Item -ItemType Directory -Force <research>\m1.2\runs | Out-Null
$env:MARIAN_MOE_TRACE_IDS = '<research>\m1.2\runs\i1-c9400.ids.bin'
pwsh -NoProfile -File <research>\tools\run_arm_m11.ps1 -Tag i1-c9400 -ServerArgs "-ncmoe 48 --moe-cache-mib 9400" -Phases warm,decode -OutDir <research>\m1.2\runs -ServerExe <research>\build-m1.2\bin\llama-server.exe
Remove-Item Env:\MARIAN_MOE_TRACE_IDS
```

`run_arm_m11.ps1` needs no change: the server is started by `Start-Process`, which inherits the variable. The script
writes `i1-c9400.trace.csv` and `i1-c9400.fills.csv` itself, plus the requests, resource log, cmd and proc files.

**5.4 Analysis (CPU, about a minute):**

```powershell
$r = '<research>\m1.2\runs'
$m = '<models>\Swift-1.5-IQ2_XS\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000'
Set-Location <repo>
# (a) geometry of the laptop's own shards, compared with the one committed here
python -m marian geometry ($m + '1-of-00002.gguf') ($m + '2-of-00002.gguf') -o "$r\swift-local.geometry.json"
python -c "import json; a=json.load(open(r'$r\swift-local.geometry.json')); b=json.load(open(r'docs\m1.2\swift-1.5-iq2_xs.geometry.json')); print('geometry identical:', a['layers'] == b['layers'] and a['n_expert_used'] == b['n_expert_used'])"
# (b) exactness against E1's e1-c9400 (M1.1 build, ids trace off)
$x = '<research>\m1.2\exact-i1-vs-e1'
New-Item -ItemType Directory -Force $x | Out-Null
Copy-Item <research>\m1.1\runs\e1-c9400.requests.jsonl, "$r\i1-c9400.requests.jsonl" $x
python -I <research>\tools\analyse_m10.py $x "$x\summary.md"
# (c) the replay, checked step by step against the server's own traces, with the P2 and RAM-tier tables
python -m marian replay "$r\i1-c9400.ids.bin" --geometry "$r\swift-local.geometry.json" --budget-mib 9400 --requests "$r\i1-c9400.requests.jsonl" --trace "$r\i1-c9400.trace.csv" --fills "$r\i1-c9400.fills.csv" --label "i1-c9400 (real trace)" --md "$r\i1-c9400.replay.md" --json "$r\i1-c9400.replay.json"
```

If `e1-c9400` has not been run yet, (b) can use M1.0's `r2-c9400.requests.jsonl` instead. That is the cross-build
reference E1 check (4) uses; copy it into the scratch folder, never into `m1.0\runs`.

### 5.5 Acceptance checks (record each decision in `docs/TEST-REGISTER.md`)

1. **Output identity:** `$x\summary.md` shows `e1-c9400 | i1-c9400 | 10/10 | 10/10` (decode texts and top-3
   probabilities). This compares the M1.1 build with the trace off against the M1.2 build with the trace on: 0004 must
   not change a byte. If it fails, stop: run `i1` once more without the variable to separate the patch from the trace.
2. **Geometry:** (a) prints `geometry identical: True`. If False, the laptop's files differ from the public revision;
   use `swift-local.geometry.json` (the command above already does) and record it.
3. **Replay = server:** in `i1-c9400.replay.md`:
   - "S lines: N of N identical" with "0 not in the ids trace" (the last few steps before the kill may be missing from
     one of the files: say how many);
   - "fill lines: M of M steps with identical (expert, slot) uploads";
   - **replay minus server within 3 points** on the median per-token hit rate. This is M1.1's exit check; expected
     +0.00. Any non-identical step is a finding to explain before the replay's P2 or tier numbers are trusted.
4. **Trace size and health:** `i1-c9400.ids.bin` is about 3.9 MiB. The report says how many trailing bytes it dropped
   (at most one record). `i1-c9400.server.log` has no "routing-id trace is off" WARN.
5. **Speed (indicative, one run each):** decode tok/s of `i1-c9400` against `e1-c9400`, within M1.0's run-to-run floor
   (about +-5%). The CPU cost measured here is about 3.4 µs per token, so a larger difference would be the page cache,
   not the trace.

### 5.6 Reading the answer to the P3 question

The last table of `i1-c9400.replay.md` gives, per policy, the smallest tier in the sweep (1-16 GiB) that serves 75% and
90% of the test prompts' decode miss bytes.

- **A realistic policy (`current`, `warm`, `static-*`) reaching 75% at a size the 32 GB machine can pin** (M1.0 saw
  0.06-0.18 GiB free with the server up, so the tier competes with the page cache) is the path to ~15 ms of fetch per
  token, and so to the 12 tok/s exit.
- **Only `oracle-static` or `belady` reaching it** means the policy is the work.
- **Not even `belady`** means no RAM tier of that size gets there, and the lever is elsewhere (VRAM policy, prefetch,
  or the miss bytes themselves).

For a finer crossing, run `--tier-mib` with a finer list, e.g. `--tier-mib 5120,5632,6144,6656,7168`.

## 6. Branch and files

Branch `m1.2/replay` of `tylnexttime/marian`, from `m1.1` (`7fa8f50`). `master`, `m1.0-results` and `m1.1` are
untouched, and no PR was opened. The fork commit is `marian/m1.2` in the cloud clone only; the patch file is the
deliverable. `.gitattributes` keeps `*.patch` byte-exact for `git am` on Windows checkouts with autocrlf, and gives
`*.bat` CRLF.

## 7. Not verified

- **No GPU run of 0004:** the S and L hooks have never executed. The moe cache refuses a CPU-only backend and the copy
  callback never runs without a GPU, so only the writer was tested. Output identity with the trace on is argued from
  the code (it only copies host ids), not measured. That is check 1.
- **No MSVC build and no `nvcc` compile check of 0004.** mingw-w64 compiled and wine ran the Windows branch of the writer
  only; `llama-context.cpp` and `llama-moe-cache.cpp` with the hooks were compiled by GCC 13 on Linux only. The M1.1
  build has no `/EHsc`: a `std::thread` or allocation failure in the writer would terminate rather than unwind (the
  rest of the fork has the same property).
- **Tail loss at a kill on Windows** (at most 100 ms of records by design) is not measured; check 3 counts it.
- **That `run_arm_m11.ps1`'s server inherits `MARIAN_MOE_TRACE_IDS`** follows from `Start-Process` semantics; not run here.
- **That the laptop's shards are the public revision `99bb8f7f`:** not checked. The byte sizes and 33.02 GiB agree with
  M1.1's laptop-side dry run, and check 2 settles it.
- **Replay = server on a real trace:** shown only against the fork's own SLRU and tier code on synthetic steps. The
  "within 3 points" check awaits the run. The fork's split by free memory on one device is taken as exact
  (`size*free/free`); a 1-byte rounding there could move a slot only at an exact boundary.
- **The P2 candidates are my readings of the design, not P2's code**, with these parameters: sketch depth 4, width
  next power of 2 >= keys of the group, halving after 10 x n_slots increments, ties admit; `n_expert_used` transient
  slots for `door`; up to one 32-token ubatch for `guard`; the core ranked by train decode selections. They are
  candidates to rank, not predictions of P2.
- **Belady** is exact for equal-size experts only; Swift's three slot sizes differ (1.46 / 1.27 / 1.15 MiB with sector
  rounding).
- **The fetch-ms column** uses the probe's linear model as given (46 / 15 / 8 ms at 0 / 75 / 90%), not re-derived.
- **The P3 tier replay ignores read failures** (the source invalidates a slot on a failed read; none occurred in
  M1.1's self-test).
- **Without `--requests`,** prompts are inferred from prefill boundaries, and the server's warm-up joins the first
  request. Always pass the request log for real runs.

## 8. Commands used here (for the record)

```
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && git -C /tmp/llama.cpp checkout a37cc461
git -C /tmp/llama.cpp am docs/m1.0/patches/0000-p0-trace.patch docs/m1.1/patches/000[123]*.patch
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build -j
build/bin/test-moe-trace-ids tests/data/ids-sample-cpp.bin
docs/m1.2/tools/slru_xcheck.sh /tmp/llama.cpp tests/data/slru-xcheck-{small,swiftlike}.in ...out
python3 -I docs/m1.2/tools/make_xcheck_inputs.py tests/data && docs/m1.2/tools/tier_xcheck.sh /tmp/llama.cpp tests/data/tier-xcheck-{tiny,mid,big}.in ...out
python3 -m unittest discover -s tests
```

## 9. llama.cpp CPU test suite after 0004

`ctest -L main -j8 --timeout 300` on the patched build (4 vCPUs): 51 of 54 passed, including the new `test-moe-trace-ids`.

| Test | First run (`-j8`) | Run alone | Why |
|---|---|---|---|
| `test-tokenizers-ggml-vocabs` | failed in 1 s | not re-run | its vocab repo was cloned without git-lfs, so every `.gguf` is an LFS pointer ("invalid magic characters: 'vers'"); an environment gap, untouched by any Marian patch |
| `test-barrier` | timeout 300 s | **passed, 1 s** | a ggml thread-barrier test starved by 8 jobs on 4 vCPUs; 0004 changes no ggml file |
| `test-llama-context` | timeout 300 s | **passed, 20 s** | the same contention; it creates and frees many contexts, so it also runs 0004's flush in `~llama_context` |

## 10. The GPU run (24/10/2026, on the reference laptop)

**Review before the run.** An independent review of 0004 found one blocker for the data (not for exactness): the S
record took `k` and `n_tokens` from `sel`, which `build_moe_cache_slots` has already reshaped to 1-D, so every
multi-token S step was written as `n_tokens = 1, k = n_tokens * 10`. Single-token decode was unaffected, which is why
the size estimate and the unit test passed. A 28-token step (present in this arm's trace) would have made the reader
lose its place and cut decode prompts 5, 7 and 9 from the test set. Patch 0005 takes `k` from the reshape's source,
refuses any head that would misstate its payload, stops writing after a failed write (the file stays a clean prefix),
bounds the buffer if the writer stalls, and converts the path with the ANSI code page. On the Python side the reader
now refuses a record whose kind is not S/L or whose `k` differs from the model's `n_expert_used`; two tests feed it
the bad head. Lesson: the unit test exercised the writer, not the hook's arguments, so it could not have caught this.

**Acceptance checks (section 5.5):**

| Check | Result |
|---|---|
| 1. output identity, i1 (trace on) vs e1-c9400 (trace off) | **10/10 texts** (top-3: not recorded, see the note below the table) |
| 2. geometry of the laptop's shards | **identical** to `swift-1.5-iq2_xs.geometry.json` |
| 3. replay = server | S lines **107,008 / 107,008** identical; fill steps **58,110 / 58,110**; median hit replay minus server **+0.00** points (M1.1's exit check) |
| 4. trace size and health | 4.10 MB, 107,221 records, 0 truncated bytes, `k = 10` in every record, the 28-token step read correctly, no "trace is off" WARN |
| 5. speed vs e1 | **outside the band:** 6.6 vs 7.8 tok/s median (-15%); started with 4.5 GiB less free RAM (the page cache held other data) and 15 ms/token more host fetch; attributed to page-cache state, not isolated |

> **Correction 24/10/2026:** the top-3 probabilities in this check were never recorded (the client read `top_probs`/`prob`; this server sends `top_logprobs`/`logprob`), so the probability half compared nulls and could not fail; the token-identity half stands. Measured properly in JOURNEY A22: identical to full precision.

**The P3 question (section 5.6), answered on the real trace** (`i1-c9400.replay.md`, test prompts, budget 9,400 MiB):
no realistic tier policy reaches 75% of decode miss bytes within 16 GiB (current 56.9%, warm 59.4%, static-miss
61.7% at 16 GiB; 21-27% at 6 GiB); only `oracle-static` reaches 75%, at 16 GiB. On a 32 GB machine a bigger or
smarter tier is not the lever. The P2 candidates are all below SLRU at the same budget (-2.1 to -11.5 median points).

**Where the post-prefill time went** (per-phase split of the M1.1 P3 traces): after a 16K prompt the VRAM hit rate
barely changes (90.8% to 89.0%) and miss bytes rise 22%, but memory-mapped misses take twice the host time (page-cache
eviction by the prompt). The ring and the tier make miss cost nearly independent of the page cache (host +12-30%),
which is why P3 lifts post-prefill decode most. With the corrected analysis (requests that stop within 8 tokens no
longer count as 0 tok/s), the 6 GiB tier measured 14.0-14.7 tok/s warm and 14.4-14.8 post-prefill (median; 13.2-14.2
all-tokens) on two rounds; the design-9.3 five-round confirmation is the next entry in `docs/JOURNEY.md`.
