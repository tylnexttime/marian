"""The machine Marian runs on: GPU, VRAM, RAM and driver, read without admin rights.

A profile is only valid for the machine it was measured on, so every profile carries this fingerprint.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class Gpu:
    index: int
    name: str
    vram_mib: int
    driver: str
    pcie_gen: str = ""
    pcie_width: str = ""


@dataclass
class Machine:
    hostname_hash: str          # a hash, so a published profile does not name the machine
    os: str
    cpu: str
    logical_cpus: int
    ram_total_mib: int
    gpus: list[Gpu] = field(default_factory=list)

    def fingerprint(self) -> str:
        key = {"cpu": self.cpu, "ram": self.ram_total_mib // 1024,
               "gpus": [(g.name, g.vram_mib, g.driver) for g in self.gpus]}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d


def nvidia_smi(query: str) -> list[list[str]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_memory_used_mib() -> list[int]:
    return [int(r[0]) for r in nvidia_smi("memory.used") if r and r[0].isdigit()]


def ram_mib() -> tuple[int, int]:
    """(total, available) MiB. Windows via GlobalMemoryStatusEx; elsewhere /proc/meminfo."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys // 2**20, m.ullAvailPhys // 2**20
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return info["MemTotal"] // 1024, info.get("MemAvailable", info["MemFree"]) // 1024


def detect() -> Machine:
    total, _ = ram_mib()
    gpus = [Gpu(int(r[0]), r[1], int(float(r[2])), r[3], r[4], r[5])
            for r in nvidia_smi("index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.current")
            if len(r) >= 6]
    cpu = platform.processor() or platform.machine()
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return Machine(hostname_hash=hashlib.sha256(platform.node().encode()).hexdigest()[:8],
                   os=f"{platform.system()} {platform.release()}", cpu=cpu, logical_cpus=os.cpu_count() or 0,
                   ram_total_mib=total, gpus=gpus)


if __name__ == "__main__":
    print(json.dumps(detect().to_dict(), indent=1))
