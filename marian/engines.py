"""Engine adapters. An engine turns (model, configuration) into a running local server and back.

M0 ships the llama.cpp adapter (llama-server, OpenAI-compatible). Strata and Maya adapters follow the same small
interface: command(), start(), stop(). Every server binds 127.0.0.1 only.

A Marian build is told apart from a stock one by its `--help`, read once per engine: the fork lists the three P3
flags (`--moe-miss-source`, `--moe-tier-mib`, `--moe-ring-mib`, patch 0003); a stock build has none of them. Upstream's
own expert cache flag (`--moe-cache-mib`, PR #29887) is in both builds from the pin onwards, but not in older ones.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

TRANSIENT = ("shared object initialization failed",)   # a CUDA driver race on Windows when a process starts right
                                                       # after another exits; the next start works (seen 21/10/2026)
CACHE_FLAG = "--moe-cache-mib"                                        # upstream llama.cpp (PR #29887)
MARIAN_FLAGS = ("--moe-miss-source", "--moe-tier-mib", "--moe-ring-mib")  # Marian P3 (patch 0003), fork only
RING_DEFAULT_MIB = 64                                                 # the fork's --moe-ring-mib default
SOURCE_FIELDS = ("moe_cache_mib", "miss_source", "tier_mib", "ring_mib")  # recorded in a profile row when set


@dataclass
class Config:
    """One point in the search space. Dense models use n_gpu_layers; MoE models use n_cpu_moe with all layers on GPU.

    A cache arm keeps every expert layer in host memory (n_cpu_moe = all layers) and gives the GPU a per-expert cache
    of moe_cache_mib; the fork's miss source decides where a cache miss is read from."""
    ctx: int = 4096
    n_gpu_layers: int = 999
    n_cpu_moe: int | None = None
    cache_type: str = "f16"           # KV cache for K and V: f16 | q8_0 | q4_0
    threads: int | None = None
    load_mode: str = "none"           # read fully into RAM (mmap pays a page-fault tax on Windows); cache arms use mmap
    flash_attn: str = "on"
    moe_cache_mib: int | None = None  # upstream's per-expert VRAM cache for the experts kept in host memory
    miss_source: str | None = None    # Marian P3: mmap | ring | tier; None = the engine's default (mmap, upstream's path)
    tier_mib: int | None = None       # Marian P3: pinned RAM tier for miss_source tier
    ring_mib: int | None = None       # Marian P3: pinned staging ring; None = the fork's default (64 MiB)
    extra: list[str] = field(default_factory=list)

    def key(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)

    def marian_flags(self) -> list[str]:
        """The Marian-only flags this configuration passes; empty means a stock llama.cpp can run it."""
        return [f for f, v in zip(MARIAN_FLAGS, (self.miss_source, self.tier_mib, self.ring_mib)) if v is not None]

    def pinned_mib(self) -> int:
        """Host memory the fork pins for this configuration (the desktop cannot page it): ring, plus tier."""
        if self.miss_source not in ("ring", "tier"):
            return 0
        ring = self.ring_mib if self.ring_mib is not None else RING_DEFAULT_MIB
        return ring + ((self.tier_mib or 0) if self.miss_source == "tier" else 0)

    def exact_class(self, block_count: int | None = None) -> bool:
        """Every expert (every layer, for a dense model) computed on the GPU, so the answer is placement-independent.

        True for full offload and for cache arms (the fork's exactness floor, patch 0001, refuses a budget that would
        compute an expert on the CPU). Stock -ncmoe N > 0 computes N layers' experts on the CPU, which rounds
        differently: on 24/10/2026 it changed all 10 answers of Gemma 4 26B and moved log-probabilities by up to 0.59,
        while every cache arm matched the all-VRAM reference to the last bit (JOURNEY A22)."""
        all_layers = self.n_gpu_layers > block_count if block_count else self.n_gpu_layers >= 999
        if not all_layers:
            return False
        return bool(self.moe_cache_mib) or not self.n_cpu_moe

    def to_row(self) -> dict:
        """The configuration as a profile row records it (M0's keys first, then the cache and miss-source fields)."""
        r = {"ctx": self.ctx, "kv": self.cache_type}
        if self.n_cpu_moe is not None:
            r["n_cpu_moe"] = self.n_cpu_moe
        else:
            r["n_gpu_layers"] = self.n_gpu_layers
        r["threads"] = self.threads
        r["load_mode"] = self.load_mode
        r.update({k: getattr(self, k) for k in SOURCE_FIELDS if getattr(self, k) is not None})
        return r

    @classmethod
    def from_row(cls, row: dict) -> "Config":
        """The inverse of to_row, and it reads M0 rows too (no load_mode there: M0 always loaded with -lm none)."""
        c = cls(ctx=row["ctx"], cache_type=row["kv"], threads=row.get("threads"))
        if "n_cpu_moe" in row:
            c.n_gpu_layers, c.n_cpu_moe = 999, row["n_cpu_moe"]
        else:
            c.n_gpu_layers = row["n_gpu_layers"]
        if row.get("load_mode"):
            c.load_mode = row["load_mode"]
        for k in SOURCE_FIELDS:
            if row.get(k) is not None:
                setattr(c, k, row[k])
        return c


def parse_help(text: str) -> dict:
    """What a llama-server build offers, from its --help text.

    Only option lines count (they start with a dash at column 0; descriptions are indented and may mention other
    flags, as --moe-tier-mib's does). A Marian build lists all three P3 flags."""
    flags: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"(-[^\s,]+(?:,\s+-[^\s,]+)*)", line)
        if m:
            flags.update(t.strip() for t in m.group(1).split(","))
    if not flags:                                           # an unfamiliar layout: any long flag anywhere
        flags = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", text))
    return {"moe_cache": CACHE_FLAG in flags, "marian": all(f in flags for f in MARIAN_FLAGS),
            "moe_flags": sorted(f for f in flags if f.startswith("--moe-")), "flags": sorted(flags)}


def parse_version(text: str) -> str:
    """'0.7.0-dev (build 3203, commit 2c1cf30a)' from `llama-server --version` (M0 kept only the first four words,
    which cut the commit off: '0.7.0-dev (build 11760, commit')."""
    m = re.search(r"version:\s*([^\r\n]+)", text)
    return m.group(1).strip() if m else "?"


def version_key(v: str | None) -> tuple[str | None, str | None]:
    b = re.search(r"build\s+(\d+)", v or "")
    c = re.search(r"commit\s+([0-9a-f]{6,})", v or "")
    return (b.group(1) if b else None, c.group(1) if c else None)


def same_build(a: str | None, b: str | None) -> bool:
    """Same build number, and the same commit where both strings carry one (M0's truncated strings carry none)."""
    (ba, ca), (bb, cb) = version_key(a), version_key(b)
    if ba is None or bb is None:
        return (a or "").strip() == (b or "").strip()
    return ba == bb and (ca is None or cb is None or ca == cb)


class LlamaCpp:
    name = "llama.cpp"

    def __init__(self, server_exe: str | Path, port: int = 9951):
        self.exe = Path(server_exe)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self._caps: dict | None = None
        self._version: str | None = None

    def version(self) -> str:
        if self._version is None:
            out = subprocess.run([str(self.exe), "--version"], capture_output=True, text=True, errors="replace", timeout=60)
            self._version = parse_version(out.stdout + out.stderr)
        return self._version

    def capabilities(self) -> dict:
        """parse_help of this build's --help, read once."""
        if self._caps is None:
            out = subprocess.run([str(self.exe), "--help"], capture_output=True, text=True, errors="replace", timeout=60)
            self._caps = parse_help(out.stdout + out.stderr)
        return self._caps

    def command(self, model: Path, c: Config) -> list[str]:
        cmd = [str(self.exe), "-m", str(model), "-c", str(c.ctx), "-np", "1", "-ngl", str(c.n_gpu_layers),
               "-fa", c.flash_attn, "-lm", c.load_mode, "-ctk", c.cache_type, "-ctv", c.cache_type,
               "--host", "127.0.0.1", "--port", str(self.port)]
        if c.n_cpu_moe is not None:
            cmd += ["-ncmoe", str(c.n_cpu_moe)]
        if c.threads:
            cmd += ["-t", str(c.threads)]
        if c.moe_cache_mib:
            cmd += ["--moe-cache-mib", str(c.moe_cache_mib)]
        if c.miss_source is not None:
            cmd += ["--moe-miss-source", c.miss_source]
        if c.tier_mib is not None:
            cmd += ["--moe-tier-mib", str(c.tier_mib)]
        if c.ring_mib is not None:
            cmd += ["--moe-ring-mib", str(c.ring_mib)]
        return cmd + c.extra

    def start(self, model: Path, c: Config, log: Path, timeout: float = 900, attempts: int = 3) -> tuple[bool, str]:
        cmd = self.command(model, c)
        for attempt in range(1, attempts + 1):
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"\n# attempt {attempt}: {' '.join(cmd)}\n")
            lf = open(log, "a", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:
                    break
                if self._get("/health") is not None:
                    return True, f"ready in {time.time() - t0:.0f}s (attempt {attempt})"
                time.sleep(1.5)
            self.stop()
            text = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            if not any(t in text for t in TRANSIENT):
                return False, "failed to start (see log)"
            time.sleep(15)
        return False, "failed to start after retries"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=60)
        self.proc = None

    def _get(self, path: str):
        try:
            with urllib.request.urlopen(self.base + path, timeout=3) as r:
                return json.loads(r.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def post(self, path: str, body: dict, timeout: float = 1800) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tokenize(self, text: str) -> int:
        return len(self.post("/tokenize", {"content": text}).get("tokens", []))

    def complete(self, prompt: str, n_predict: int) -> dict:
        """Returns llama-server's timings: prompt_n, prompt_per_second, predicted_n, predicted_per_second."""
        r = self.post("/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                      "cache_prompt": False, "ignore_eos": True})
        return r.get("timings", {})
