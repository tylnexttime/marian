"""Engine adapters. An engine turns (model, configuration) into a running local server and back.

M0 ships the llama.cpp adapter (llama-server, OpenAI-compatible). Strata and Maya adapters follow the same small
interface: command(), start(), stop(). Every server binds 127.0.0.1 only.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

TRANSIENT = ("shared object initialization failed",)   # a CUDA driver race on Windows when a process starts right
                                                       # after another exits; the next start works (seen 21/10/2026)


@dataclass
class Config:
    """One point in the search space. Dense models use n_gpu_layers; MoE models use n_cpu_moe with all layers on GPU."""
    ctx: int = 4096
    n_gpu_layers: int = 999
    n_cpu_moe: int | None = None
    cache_type: str = "f16"           # KV cache for K and V: f16 | q8_0 | q4_0
    threads: int | None = None
    load_mode: str = "none"           # read fully into RAM (mmap pays a page-fault tax on Windows)
    flash_attn: str = "on"
    moe_cache_mib: int | None = None  # llama.cpp v0.7.0 expert cache for experts kept in host memory
    extra: list[str] = field(default_factory=list)

    def key(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


class LlamaCpp:
    name = "llama.cpp"

    def __init__(self, server_exe: str | Path, port: int = 9951):
        self.exe = Path(server_exe)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def version(self) -> str:
        out = subprocess.run([str(self.exe), "--version"], capture_output=True, text=True, timeout=60)
        return " ".join((out.stdout + out.stderr).split("version:")[-1].split()[:4]) if "version" in (out.stdout + out.stderr) else "?"

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
