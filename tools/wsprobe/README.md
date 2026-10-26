# wsprobe: which tensors of a running model are in memory

`wsprobe` watches a running process (a `llama-server`, say) and reports, every 500 ms, how many MiB of each kind of
tensor in its memory-mapped GGUF shards are resident in that process's working set: the per-layer embedding table
(`ple`), the routed experts (`experts`, also per layer), everything else (`other`) and the GGUF header. Beside each
sample it records available RAM, the system cache size, and the process's page-fault count, working set and private
bytes. Windows only; no administrator rights needed. It was built for the register test E49 (does Swift's 26.8 GiB
per-layer embedding table crowd the experts out of memory during a long prompt?).

## How it works

1. `gguf_ranges.py` reads each shard's tensor directory with `marian.gguf.tensor_infos` and writes every tensor's
   absolute byte range: `data_offset + offset` to the next tensor's start (the file size closes the last one), so
   alignment padding counts with the tensor before it. Kind: `ple` if the name contains `per_layer`, `experts` if it
   contains `_exps`, else `other`; the bytes before the data section are one `header` range. Layer: N from `blk.N.`.
   The ranges must tile the file exactly; the script and the probe both refuse anything else.
2. `wsprobe` opens the process with `PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE` (a process of the same
   user grants this), walks its address space with `VirtualQueryEx`, names each `MEM_MAPPED` view with
   `GetMappedFileNameW`, and matches it to a shard by NT device path (the shard's drive letter translated with
   `QueryDosDeviceW`) or, if that fails, by a file name no other shard shares. A view at least as long as the file can
   only start at file offset 0, which is how llama.cpp maps a model on Windows (one whole-file view per shard); a
   shorter view is counted separately as `partial_view_mib`, unattributed.
3. Every sample asks `QueryWorkingSetEx` about the pages of those views, in batches (`--chunk`, default 65536 entries),
   and attributes each valid page to the tensor range holding its first byte.

It never writes to the target and never touches the target's pages.

## Limits, stated

- **Working set, not RAM.** A page in the standby list (in RAM, but not in this process's working set) counts as not
  resident. Telling standby pages apart needs administrator rights. `syscache_mib` (standby list plus the system
  working set, from `GetPerformanceInfo`) is printed beside each sample as system-wide context.
- **Cost.** `QueryWorkingSetEx` costs roughly 165-250 ns per page queried. On a 68 GB two-shard model (16.6 M pages)
  a full sample took 2.7-3.3 s. `--stride K` queries every K-th page and counts each valid one as K pages, a fixed-stride
  sample, recorded in every line. Against the exact count on the same frozen state (random single pages, random whole
  expert slices, contiguous blocks), stride 32 was within 0.7 MiB and took 94 ms; stride 64 was within 7.4 MiB
  (3% of 233 MiB of scattered single pages, the hardest case) and took 50 ms. Small kinds (the header) are coarse at a
  large stride: one sampled page stands for K.
- Fault read-ahead does not inflate the count: the self-test touches single pages and reads back exactly those, so
  clustered reads around a fault land outside the working set (in standby) and are, by the first limit, invisible.

## Build and run

```bat
build.bat                      :: from a Developer prompt, or set VCVARS64 to the full path of vcvars64.bat first
python gguf_ranges.py model-00001-of-00002.gguf model-00002-of-00002.gguf -o ranges.json
wsprobe --wait-image C:\path\to\llama-server.exe --wait-timeout 600 --ranges ranges.json --stride 64 --chunk 4096 --out run.wsprobe.jsonl
```

`--wait-image` waits for a process with that image (full path, or file name) created after the probe started, and
attaches within tens of milliseconds of its creation (10-31 ms in E49); `--pid N` attaches to a running process. The probe exits when the process
does. `--once [--dump-runs runs.json]` takes one sample (and, at stride 1, writes the resident runs as file offsets).

Output, one JSON object per line: a `start` line (pid, stride, attach delay, mapped MiB per kind, shards), a `views`
line whenever the set of matched views changes, one `sample` line per interval
(`t`, `qpc_us` on the same QueryPerformanceCounter clock as the Marian request logs, `q_ms` the sample's own cost,
`res_mib` and `res_pages` per kind, `exp_layer_mib` and `exp_layer_pages` per expert layer, `avail_mib`,
`syscache_mib`, `pf`, `ws_mib`, `priv_mib`), and an `end` line.

## Self-test

```bat
python selftest.py [--dir SCRATCH_DIR]
```

Writes a synthetic GGUF of about 1 GiB (seven tensors whose sizes are not page multiples, three kinds, two expert
layers), its ranges via `gguf_ranges.py`, and lets `wsprobe_target` map it the way llama.cpp does. A continuous
`wsprobe` watches throughout, and a one-shot `wsprobe --once --dump-runs` reads back each step: mapped and untouched
(nothing resident), a first touch set (resident runs, per-kind pages and per-layer pages must equal the touched pages
exactly, including ranges that cross tensor boundaries, a single page and the file's last partial page), a trimmed
working set (view still found, nothing resident), a different touch set (exact again), and unmapped (view gone,
residency 0). The continuous series must show the same rise and fall in order. Exit status = number of failures.
