# -*- coding: utf-8 -*-
"""Marian pre-run check (dev-debt row 17, 28/10/2026): record the machine's state before a registered speed run.

On 27/10/2026 the reference laptop decoded the same configuration about 20% slower in the afternoon than that morning,
and one stock run fell from 16.49 to 7.77 tok/s back to back. Its resource logs showed two different things: the
afternoon drift came with a proportionally lower drive read rate (same bytes per token, same CPU performance, GPU
utilisation and free RAM), and the 2x slump with the CPU's "% Processor Performance" at 72 instead of 111-121. Neither
was visible before the run. This check makes both visible, and records the rest of the usual suspects.

What it records (Windows; no admin rights needed; about 10 s):
  - power: AC or battery, battery saver, the active power plan and power-mode overlay, uptime;
  - memory: available, standby, commit and commit limit;
  - CPU: % Processor Performance and Utility at rest, the ACPI thermal zone temperature, its passive limit and throttle
    reasons, and two short probes (one thread and all threads hashing a fixed buffer, MB/s);
  - drive: an unbuffered sequential read of 1 GiB from a file you name (the model, so the drive under test is the
    drive the run reads) and 4 KiB random unbuffered reads at queue depth 1 (the page-fault path);
  - GPU (nvidia-smi): temperature, SM and memory clocks against their maxima, power draw and limit, P-state, the active
    clock-event (throttle) reasons, memory used, and other processes on the GPU;
  - the processes using the most CPU over 2 s, and whether a llama-server is already running.

Flags are reported, never enforced (exit 0), unless --strict. With --baseline FILE (an earlier record on the same
machine), the probes are compared against it and a probe more than 10% slower is flagged.

Usage: python prerun_check.py --out RECORD.json [--drive-file MODEL.gguf] [--baseline BASE.json] [--label TEXT]
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time

K32 = ctypes.WinDLL("kernel32", use_last_error=True)


def ps_json(script):
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    try:
        return json.loads(r.stdout) if r.stdout.strip() else None
    except json.JSONDecodeError:
        return {"error": (r.stdout + r.stderr)[:300]}


class SPS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wt.DWORD), ("BatteryFullLifeTime", wt.DWORD)]


def power():
    s = SPS()
    K32.GetSystemPowerStatus(ctypes.byref(s))
    plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()
    overlay = ps_json(r"$k = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' "
                      r"-ErrorAction SilentlyContinue; @{ac=$k.ActiveOverlayAcPowerScheme; dc=$k.ActiveOverlayDcPowerScheme}"
                      r" | ConvertTo-Json -Compress")
    K32.GetTickCount64.restype = ctypes.c_ulonglong
    return {"ac": {0: "battery", 1: "ac"}.get(s.ACLineStatus, "unknown"), "battery_pct": s.BatteryLifePercent,
            "battery_saver": bool(s.SystemStatusFlag & 1), "plan": plan.split("(")[-1].rstrip(")") if "(" in plan else plan,
            "overlay": overlay, "uptime_h": round(K32.GetTickCount64() / 3.6e6, 2)}


COUNTERS = [r"\Memory\Available MBytes", r"\Memory\Committed Bytes", r"\Memory\Commit Limit",
            r"\Memory\Standby Cache Normal Priority Bytes", r"\Memory\Standby Cache Reserve Bytes",
            r"\Processor Information(_Total)\% Processor Performance", r"\Processor Information(_Total)\% Processor Utility",
            r"\Thermal Zone Information(*)\Temperature", r"\Thermal Zone Information(*)\% Passive Limit",
            r"\Thermal Zone Information(*)\Throttle Reasons", r"\Process(*)\% Processor Time"]


def counters():
    lst = ",".join("'" + c + "'" for c in COUNTERS)
    rows = ps_json(f"(Get-Counter -Counter @({lst}) -SampleInterval 2 -MaxSamples 1 -ErrorAction SilentlyContinue)"
                   ".CounterSamples | Select-Object Path, InstanceName, CookedValue | ConvertTo-Json -Compress")
    if not isinstance(rows, list):
        return {"error": rows}
    ncpu = os.cpu_count() or 1
    out, procs, tz = {}, [], {}
    for r in rows:
        p, v = r["Path"].lower(), float(r["CookedValue"])
        if "\\process(" in p:
            if r["InstanceName"] not in ("_total", "idle"):
                procs.append((r["InstanceName"], v / ncpu))
        elif "thermal zone" in p:
            key = p.rsplit("\\", 1)[-1]
            tz.setdefault(r["InstanceName"], {})[key] = v
        else:
            out[p.rsplit("\\", 1)[-1]] = v
    gib = 1024 ** 3
    agg = {}
    for n, v in procs:                       # several processes share a name (python#1, ...): sum by base name
        agg[n.split("#")[0]] = agg.get(n.split("#")[0], 0.0) + v
    top = sorted(agg.items(), key=lambda x: -x[1])[:8]
    return {
        "available_gib": round(out.get("available mbytes", -1024) / 1024, 2),
        "standby_gib": round((out.get("standby cache normal priority bytes", 0) + out.get("standby cache reserve bytes", 0)) / gib, 2),
        "commit_gib": round(out.get("committed bytes", 0) / gib, 2),
        "commit_limit_gib": round(out.get("commit limit", 0) / gib, 2),
        "cpu_perf_pct_idle": round(out.get("% processor performance", -1), 1),
        "cpu_util_pct_idle": round(out.get("% processor utility", -1), 1),
        "thermal_zones": {k: {"temp_c": round(v.get("temperature", 0) - 273.15, 1),
                              "passive_limit_pct": v.get("% passive limit"), "throttle_reasons": v.get("throttle reasons")}
                          for k, v in tz.items()},
        "top_cpu_pct_of_machine": [[n, round(v, 1)] for n, v in top],
    }


def cpu_probe(seconds=1.5):
    buf = os.urandom(8 << 20)

    def run(res, i):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            hashlib.sha256(buf).digest()     # releases the GIL on large buffers
            n += len(buf)
        res[i] = n / (time.perf_counter() - t0) / 1e6
    one = [0.0]
    run(one, 0)
    nthreads = os.cpu_count() or 1
    res = [0.0] * nthreads
    ts = [threading.Thread(target=run, args=(res, i)) for i in range(nthreads)]
    perf = {}

    def sample():                            # % Processor Performance while every core is busy: a clock cap shows here
        perf["v"] = ps_json(r"(Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval 1 "
                            r"-MaxSamples 1).CounterSamples[0].CookedValue")
    s = threading.Thread(target=sample)
    for t in ts:
        t.start()
    s.start()
    for t in ts:
        t.join()
    s.join()
    return {"one_thread_mb_s": round(one[0], 1), "all_threads_mb_s": round(sum(res), 1), "threads": nthreads,
            "cpu_perf_pct_loaded": round(perf["v"], 1) if isinstance(perf.get("v"), (int, float)) else None}


GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
FILE_SHARE_RW = 0x1 | 0x2
NO_BUFFERING, SEQUENTIAL = 0x20000000, 0x08000000


def drive_probe(path, seq_bytes=1 << 30, block=4 << 20, n_rand=2000):
    K32.CreateFileW.restype = wt.HANDLE
    K32.VirtualAlloc.restype = ctypes.c_void_p
    K32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
    K32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    size = os.path.getsize(path)
    h = K32.CreateFileW(path, GENERIC_READ, FILE_SHARE_RW, None, OPEN_EXISTING, NO_BUFFERING | SEQUENTIAL, None)
    if h in (None, wt.HANDLE(-1).value):
        return {"error": f"open failed ({ctypes.get_last_error()})"}
    buf = K32.VirtualAlloc(None, block, 0x3000, 0x04)        # page-aligned, as unbuffered reads need
    got = wt.DWORD()
    try:
        start = (size // 2) // block * block                 # the middle of the file: expert data, not the header
        seq_bytes = min(seq_bytes, (size - start) // block * block)
        K32.SetFilePointerEx(h, start, None, 0)
        t0, done = time.perf_counter(), 0
        while done < seq_bytes:
            if not K32.ReadFile(h, buf, block, ctypes.byref(got), None) or got.value == 0:
                break
            done += got.value
        seq = done / (time.perf_counter() - t0) / 1e6
        rng = random.Random(17)                              # the same offsets every run
        lat = []
        for _ in range(n_rand):
            off = rng.randrange(0, size // 4096) * 4096
            K32.SetFilePointerEx(h, off, None, 0)
            t = time.perf_counter()
            K32.ReadFile(h, buf, 4096, ctypes.byref(got), None)
            lat.append(time.perf_counter() - t)
        lat.sort()
        return {"file_gib": round(size / 1024 ** 3, 2), "seq_read_mb_s": round(seq, 1), "seq_bytes": done,
                "rand4k_iops_qd1": round(n_rand / sum(lat), 0), "rand4k_p50_us": round(lat[len(lat) // 2] * 1e6, 1),
                "rand4k_p99_us": round(lat[int(len(lat) * 0.99)] * 1e6, 1)}
    finally:
        K32.VirtualFree(ctypes.c_void_p(buf), 0, 0x8000)
        K32.CloseHandle(h)


GPU_Q = ("name,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,enforced.power.limit,power.default_limit,pstate,"
         "clocks_event_reasons.active,memory.used,memory.total,utilization.gpu")


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    if not r:
        return {"error": "no GPU"}
    vals = [v.strip() for v in r[0].split(",")]
    d = dict(zip(GPU_Q.split(","), vals))
    reasons = int(d.get("clocks_event_reasons.active", "0") or "0", 16)
    names = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal",
             0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}
    d["clocks_event_reasons"] = [n for b, n in names.items() if reasons & b]
    d["compute_apps"] = [ln.strip() for ln in apps.splitlines() if ln.strip()]
    return d


def llama_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True, text=True).stdout
    return "llama-server.exe" in out


def flags(rec, base):
    f = []
    p, c, g = rec["power"], rec["counters"], rec["gpu"]
    if p["ac"] != "ac":
        f.append(f"power: on {p['ac']}")
    if p["battery_saver"]:
        f.append("power: battery saver on")
    if "High performance" not in p["plan"] and "Ultimate" not in p["plan"]:
        f.append(f"power: plan is {p['plan']}")
    for z, v in c.get("thermal_zones", {}).items():
        if v["passive_limit_pct"] is not None and v["passive_limit_pct"] < 100:
            f.append(f"thermal: {z} passive limit {v['passive_limit_pct']:.0f}%")
        if v["throttle_reasons"]:
            f.append(f"thermal: {z} throttle reasons {v['throttle_reasons']:.0f}")
        if v["temp_c"] >= 90:
            f.append(f"thermal: {z} at {v['temp_c']} C")
    lp = rec["cpu"].get("cpu_perf_pct_loaded")
    if lp is not None and lp < 100:
        f.append(f"cpu: % Processor Performance {lp:.0f} under load (below base clock; 27/10's 2x slump ran at 72)")
    for n, v in c.get("top_cpu_pct_of_machine", []):
        if v >= 5 and n not in ("python", "powershell"):
            f.append(f"background: {n} uses {v:.0f}% of the machine")
    if isinstance(g, dict) and "error" not in g:
        bad = [r for r in g["clocks_event_reasons"] if r not in ("gpu_idle", "applications_clocks")]
        if bad:
            f.append(f"gpu: clock event reasons {bad}")
        lim, dflt = g.get("enforced.power.limit", ""), g.get("power.default_limit", "")
        try:
            if float(lim) < 0.95 * float(dflt):
                f.append(f"gpu: power limit {lim} W, below its default {dflt} W")
        except ValueError:
            pass
        if float(g.get("temperature.gpu", 0) or 0) >= 80:
            f.append(f"gpu: {g['temperature.gpu']} C at rest")
        if float(g.get("memory.used", 0) or 0) > 500 or g["compute_apps"]:
            f.append(f"gpu: {g['memory.used']} MiB in use, apps {g['compute_apps']}")
    if rec["llama_server_running"]:
        f.append("a llama-server is already running")
    if base:
        for sec, key in (("cpu", "one_thread_mb_s"), ("cpu", "all_threads_mb_s"), ("drive", "seq_read_mb_s"),
                         ("drive", "rand4k_iops_qd1")):
            a, b = rec.get(sec, {}).get(key), base.get(sec, {}).get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0 and a < 0.9 * b:
                f.append(f"{sec}: {key} {a} is {a / b:.2f} of the baseline's {b}")
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--drive-file", help="a large file on the drive the run reads (the model); omit to skip")
    ap.add_argument("--baseline", help="an earlier record on this machine to compare the probes against")
    ap.add_argument("--label", default="")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any flag is raised")
    a = ap.parse_args()
    rec = {"schema": 1, "label": a.label, "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "machine": os.environ.get("COMPUTERNAME", ""), "llama_server_running": llama_running()}
    rec["power"] = power()
    rec["gpu"] = gpu()
    rec["counters"] = counters()
    rec["cpu"] = cpu_probe()
    rec["drive"] = drive_probe(a.drive_file) if a.drive_file else {"skipped": True}
    base = json.load(open(a.baseline, encoding="utf-8")) if a.baseline else None
    rec["baseline"] = a.baseline and os.path.basename(a.baseline)
    rec["flags"] = flags(rec, base)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    c, d, g = rec["counters"], rec["drive"], rec["gpu"]
    print(f"prerun {rec['time']}: {rec['power']['ac']}, {rec['power']['plan']}; RAM avail {c.get('available_gib')} GiB; "
          f"cpu 1t {rec['cpu']['one_thread_mb_s']} / all {rec['cpu']['all_threads_mb_s']} MB/s, perf {rec['cpu']['cpu_perf_pct_loaded']}%; "
          f"drive seq {d.get('seq_read_mb_s')} MB/s, 4k {d.get('rand4k_iops_qd1')} IOPS; "
          f"gpu {g.get('temperature.gpu')} C {g.get('clocks.sm')}/{g.get('clocks.max.sm')} MHz; "
          f"flags: {'; '.join(rec['flags']) or 'none'}")
    return 1 if (a.strict and rec["flags"]) else 0


if __name__ == "__main__":
    sys.exit(main())
