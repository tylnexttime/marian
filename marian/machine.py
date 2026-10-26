"""The machine Marian runs on: GPU, VRAM, RAM and driver, read without admin rights.

A profile is only valid for the machine it was measured on, so every profile carries this fingerprint. The
fingerprint is built from hardware only (CPU, RAM, GPU name, VRAM, driver). Nothing derived from the hostname is
recorded: a short hash of a hostname can be reversed by trying a list of likely names, and profiles get published.
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


def gpu_memory_free_mib() -> list[int]:
    return [int(r[0]) for r in nvidia_smi("memory.free") if r and r[0].isdigit()]


def _memstatus():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def _meminfo() -> dict:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return info


def ram_mib() -> tuple[int, int]:
    """(total, available) MiB. Windows via GlobalMemoryStatusEx; elsewhere /proc/meminfo."""
    if os.name == "nt":
        m = _memstatus()
        return m.ullTotalPhys // 2**20, m.ullAvailPhys // 2**20
    info = _meminfo()
    return info["MemTotal"] // 1024, info.get("MemAvailable", info["MemFree"]) // 1024


def commit_mib() -> tuple[int, int] | None:
    """(commit limit, commit available) MiB, or None. Windows: RAM plus page file, the limit pinned and private memory
    count against (the design's probe floor is 4 GiB of commit headroom, after Strata). Linux: CommitLimit and
    CommitLimit - Committed_AS, which only bind under strict overcommit."""
    try:
        if os.name == "nt":
            m = _memstatus()
            return m.ullTotalPageFile // 2**20, m.ullAvailPageFile // 2**20
        info = _meminfo()
        return info["CommitLimit"] // 1024, (info["CommitLimit"] - info["Committed_AS"]) // 1024
    except (OSError, KeyError, AttributeError, ValueError):
        return None


def process_memory_mib(pid: int) -> dict:
    """Peak working set and peak commit (private bytes) of a running process, in MiB; {} when unreadable.

    Read just before the process is stopped: both are peaks over its lifetime. Linux gives VmHWM (peak resident) and
    no per-process commit figure."""
    try:
        if os.name == "nt":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.windll.kernel32
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            h = k32.OpenProcess(0x1000, 0, pid)               # PROCESS_QUERY_LIMITED_INFORMATION: no admin needed
            if not h:
                return {}
            try:
                c = PMC()
                c.cb = ctypes.sizeof(PMC)
                if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                    return {}
                return {"working_set_peak_mib": c.PeakWorkingSetSize // 2**20, "commit_peak_mib": c.PeakPagefileUsage // 2**20}
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return {"working_set_peak_mib": int(line.split()[1]) // 1024}
    except (OSError, AttributeError, ValueError):
        pass
    return {}


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
    return Machine(os=f"{platform.system()} {platform.release()}", cpu=cpu, logical_cpus=os.cpu_count() or 0,
                   ram_total_mib=total, gpus=gpus)


if __name__ == "__main__":
    print(json.dumps(detect().to_dict(), indent=1))
