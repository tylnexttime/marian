"""`marian serve` once the plan is chosen (M1.3): a front proxy, the one-sequence serving envelope, the memory
governor (`--governor watch|restart|off`), the passive live check and the prefix-reuse setting.

**Why a proxy.** llama-server listens on an internal port; a small standard-library proxy owns the port users call.
That gives Marian the two things llama-server cannot do for it:

- **The serving envelope (design 7.8, C-23).** Concurrent requests change rows per `MUL_MAT_ID`, the cache path and
  the peak workspace, so one sequence runs at a time (llama-server also runs with `-np 1`) and at most `queue` more
  wait (1 by default), in arrival order, for at most `queue_timeout_s`. Past that, or while the governor has paused
  admission, a request gets HTTP 503 with a `Retry-After` header and a JSON reason, in llama-server's own error shape:
  `{"error": {"code": 503, "type": "unavailable_error", "reason": "...", "message": "..."}}`. GET requests (health,
  props, models, metrics) and the cheap POSTs (`/tokenize`, `/detokenize`, `/apply-template`) are not queued.
  Responses pass through as they arrive, so streamed (server-sent event) answers stream; a client that hangs up
  closes the upstream connection, which makes llama-server stop generating.
- **The memory governor (design 7.7 as reshaped by critique C-02, then by register test E53).** Upstream allocates
  the expert cache's bank once, at construction, so a running server cannot give VRAM back. Every `interval_s` (2 s)
  the governor reads the VRAM free and the RAM available (and Windows commit headroom). Their floors are the margins
  the plan was checked with: free VRAM must stay above the VRAM margin (300 MiB by default), and available RAM above
  the RAM floor (6 GiB) for a row that pins memory, else above max(2 GiB, 10 % of RAM), the probe floor of design 6.1
  (C-21); commit headroom must stay above 4 GiB (Strata's floor, the same section).

  **What a crossing does depends on `--governor`.** E53 ran a VRAM hog (150 MiB left free) and then a RAM hog (under
  1 GiB available) against a plain llama-server: its decode never fell more than about 4%. The old default (pause at
  once, restart on the next smaller row after 10 s) moved Marian from 71-76 tok/s to a row running 9.5-11.7 tok/s,
  for about 240 s, and it never stepped back up once the hog left. So:

  - `watch` (the default): every crossing and every clearing is one line with the live decode rate. Only commit
    headroom pauses admission (503, reason `memory_pressure`; the running request finishes): under it, allocations
    fail, which is a crash risk, not a slowdown. VRAM and RAM crossings only warn. Nothing restarts on a reading.
  - `restart` (opt-in): the trigger is a measured slowdown. Each completed request's decode rate comes from the
    timings the proxy already reads. The first `baseline_n` (8) answers of at least 16 tokens on a row set its
    baseline, their median. The baseline is not the profile's figure: serve's request loop ran 10-16% under
    calibrate's figure on a clean machine (E53), so a bar set from the profile would sit that much closer to a false
    alarm. A slowdown is decode under `slowdown` (70%) of the baseline in at least two answers, the first and the last
    `persist_s` (10 s) or more apart, with no answer at or above the bar between them. It restarts only if a memory
    floor was crossed while it lasted; a slowdown with every floor held is logged, since a smaller row would not
    help. A memory crossing without a slowdown never restarts. The restart waits up to `drain_s` (30 s) for the
    running request, stops llama-server and starts it on the next smaller measured row that fits the memory free now
    (plan.next_smaller), saying why in one line: the KV and prompt cache are lost. An exact-class row is only
    replaced by an exact-class row unless non-exact rows were allowed. After `max_restarts` (3), or when nothing
    smaller fits, it gives up with a clear reason (on Windows the driver pages GPU memory into system RAM rather
    than failing; 42.7 -> 5.5 tok/s was measured on 21/10/2026, though E53 did not reproduce that cliff at its depth
    of pressure).
    **Step-up:** on a row smaller than the planned one, once memory has fitted the planned row (its share back,
    its floors kept) for `step_up_s` (60 s), it restarts on the planned row. Never while a request is queued (it
    tries again at the next reading); the running request drains first. A step-up counts against `max_restarts`; at
    the limit it stays on the smaller row and says so once, rather than giving up. A row met again keeps the
    baseline it set earlier in the session.
  - `off`: no governor thread at all.

  In `watch` and `restart`, a server that exits on its own is restarted on the same row (that is not a memory
  reading). Live shrinking of a bank is M2 work (design 7.7).

**Prefix reuse.** llama-server keeps the prompt of the last request and reuses the longest shared prefix by default
(`cache_prompt`). Whether a reused prefix gives the same answer as a fresh prefill is test E52, so until E52 passes
Marian serves with it OFF: llama-server starts with `--no-cache-prompt` where the build has it, and the proxy sets
`cache_prompt: false` in every generation request. `--cache-prompt on` opts in. `--gate-prefix FILE` (opt-in mode
only) sends a fixed instruction block once after every start (`n_predict` 0, so the slot holds its KV), for a gate
whose prompts all begin with it; the file must hold the exact text the requests begin with, after any chat template.
"""
from __future__ import annotations

import http.client
import json
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import confirm, machine, plan

CACHE_PROMPT_DEFAULT = False          # E52 decides; until it passes, prefix reuse is opt-in
UNGATED_POSTS = {"/tokenize", "/detokenize", "/apply-template"}
REWRITE_PATHS = {"/completion", "/completions", "/v1/completions", "/chat/completions", "/v1/chat/completions",
                 "/infill", "/v1/responses", "/responses", "/v1/messages"}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
       "transfer-encoding", "upgrade", "content-length", "host"}
RESP_SKIP = HOP | {"server", "date"}  # the proxy's own response line already sends these
SSE_DATA = re.compile(rb"^data:\s*(\{.*\})\s*$", re.M)
GOVERNOR_LINE = {
    "watch": "governor: watch (memory is read every interval; low VRAM or RAM warns, low commit headroom pauses "
             "admission; no restart on a reading)",
    "restart": "governor: restart (as watch, and a sustained decode slowdown under memory pressure restarts on a "
               "smaller row; steps back up when memory allows)",
    "off": "governor: off (no memory readings, no restarts)",
}


@dataclass
class ServeOptions:
    port: int = 9951                  # the port users call (the proxy)
    internal_port: int = 9952         # llama-server's port, 127.0.0.1 only
    need: int = 4096
    objective: str = "tg"
    margin_mib: int = 300             # the VRAM floor, as the launch check used it
    ram_floor_mib: int = 6144         # the RAM floor for rows that pin memory, as the launch check used it
    commit_floor_mib: int = 4096      # design 6.1 (C-21), after Strata
    queue: int = 1
    queue_timeout_s: float = 300
    retry_after_s: int = 5
    governor: str = "watch"           # watch | restart | off (register E53: restarting on a reading did harm)
    interval_s: float = 2.0
    persist_s: float = 10.0           # restart mode: how long a slowdown must last
    slowdown: float = 0.70            # restart mode: decode under this fraction of the row's baseline is a slowdown
    baseline_n: int = 8               # answers whose median is a row's baseline
    step_up_s: float = 60.0           # restart mode: memory fits the planned row this long -> step back up
    drain_s: float = 30.0
    max_restarts: int = 3
    upstream_timeout_s: float = 3600
    cache_prompt: bool = CACHE_PROMPT_DEFAULT
    gate_prefix: str | None = None
    live_n: int = 8
    slow: float = 0.15
    exact_only: bool = False
    stock: bool = False
    allow_non_exact: bool = False


# --- the serving envelope -----------------------------------------------------------------------------------------

class Gate:
    """One request at a time, `queue` more waiting, and a pause switch for the governor."""

    def __init__(self, queue: int = 1):
        self.cond = threading.Condition()
        self.queue = queue
        self.active = 0
        self.waiting = 0
        self.paused: tuple[str, str] | None = None

    def enter(self, timeout: float) -> tuple[bool, tuple[str, str] | None]:
        with self.cond:
            if self.paused:
                return False, self.paused
            if self.active + self.waiting >= 1 + self.queue:
                return False, ("queue_full", f"one request runs at a time and {self.queue} may wait (design 7.8); "
                                             "try again shortly")
            self.waiting += 1
            try:
                end = time.monotonic() + timeout
                while self.active or self.paused:
                    if self.paused:
                        return False, self.paused
                    left = end - time.monotonic()
                    if left <= 0:
                        return False, ("queue_timeout", f"waited {timeout:.0f} s for the running request")
                    self.cond.wait(left)
                self.active = 1
                return True, None
            finally:
                self.waiting -= 1

    def leave(self) -> None:
        with self.cond:
            self.active = 0
            self.cond.notify_all()

    def pause(self, reason: str, message: str) -> None:
        with self.cond:
            self.paused = (reason, message)
            self.cond.notify_all()

    def pause_unless_queued(self, reason: str, message: str) -> bool:
        """Pause only if nobody is waiting: a step up must not turn queued requests away."""
        with self.cond:
            if self.waiting:
                return False
            self.paused = (reason, message)
            self.cond.notify_all()
            return True

    def resume(self) -> None:
        with self.cond:
            self.paused = None
            self.cond.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        with self.cond:
            while self.active:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)
            return True


def timings_of(tail: bytes) -> dict | None:
    """llama-server's `timings` from the end of an answer: a JSON body, or the last server-sent event carrying them."""
    s = tail.strip()
    if s.startswith(b"{"):
        try:
            return (json.loads(s) or {}).get("timings")
        except ValueError:
            return None
    for m in reversed(SSE_DATA.findall(tail)):
        if b'"timings"' in m:
            try:
                return json.loads(m).get("timings")
            except ValueError:
                continue
    return None


class Front(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False       # never share the user-facing port with another server

    def __init__(self, addr, gate: Gate, upstream_port: int, opt: ServeOptions, live: confirm.LiveCheck | None = None,
                 say=print, on_timings=None):
        super().__init__(addr, Handler)
        self.gate, self.upstream_port, self.opt, self.live, self.say = gate, upstream_port, opt, live, say
        self.on_timings = on_timings                    # the supervisor's decode-rate record
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def count(self, k: str) -> None:
        with self.lock:
            self.counts[k] = self.counts.get(k, 0) + 1

    def rewrite(self, path: str, body: bytes) -> bytes:
        """Prefix reuse OFF (the exact default until E52 passes): every generation request gets cache_prompt false."""
        if self.opt.cache_prompt or path not in REWRITE_PATHS or not body:
            return body
        try:
            d = json.loads(body)
        except ValueError:
            return body                        # llama-server will refuse it with its own error
        if not isinstance(d, dict) or d.get("cache_prompt") is False:
            return body
        d["cache_prompt"] = False
        self.count("cache_prompt_forced_off")
        return json.dumps(d).encode("utf-8")

    def observe(self, tail: bytes) -> None:
        t = timings_of(tail)
        if self.live is not None:
            self.live.add(t)
        if self.on_timings is not None:
            self.on_timings(t)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Front

    def log_message(self, fmt, *args):                  # no access log: request lines can carry user text
        return

    def do_GET(self):
        self._handle()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_GET

    def _json(self, code: int, obj: dict, extra: dict | None = None) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _unavailable(self, why: tuple[str, str]) -> None:
        reason, message = why
        self.server.count(f"503_{reason}")
        self._json(503, {"error": {"code": 503, "type": "unavailable_error", "reason": reason, "message": message}},
                   {"Retry-After": str(self.server.opt.retry_after_s)})

    def _handle(self) -> None:
        front = self.server
        path = urlsplit(self.path).path
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True
            return self._json(411, {"error": {"code": 411, "type": "invalid_request_error",
                                              "message": "send a Content-Length; chunked request bodies are not supported"}})
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n > 0 else b""
        gated = self.command == "POST" and path not in UNGATED_POSTS
        if path == "/health" and front.gate.paused:
            return self._unavailable(front.gate.paused)
        if gated:
            ok, why = front.gate.enter(front.opt.queue_timeout_s)
            if not ok:
                return self._unavailable(why)
        try:
            if gated:
                body = front.rewrite(path, body)
            self._forward(body, measure=gated)
        finally:
            if gated:
                front.gate.leave()

    def _forward(self, body: bytes, measure: bool) -> None:
        front = self.server
        conn = http.client.HTTPConnection("127.0.0.1", front.upstream_port, timeout=front.opt.upstream_timeout_s)
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            if body or self.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))
            try:
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                return self._unavailable(("upstream_unavailable", f"the model server is not answering "
                                                                  f"({type(e).__name__}); it may be restarting"))
            chunked = resp.getheader("Transfer-Encoding", "").lower() == "chunked"
            length = resp.getheader("Content-Length")
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in RESP_SKIP:
                    self.send_header(k, v)
            if length is not None and not chunked:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
                tail = data[-65536:]
            else:
                http11 = self.request_version == "HTTP/1.1"
                if http11:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                buf = bytearray()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk) if http11 else chunk)
                    self.wfile.flush()
                    buf += chunk
                    if len(buf) > 131072:
                        del buf[:-65536]
                if http11:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                tail = bytes(buf)
            front.count("forwarded")
            if measure and resp.status == 200:
                front.observe(tail)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True               # the client left; closing upstream stops the generation
        except (OSError, http.client.HTTPException):
            self.close_connection = True               # upstream died mid-answer (a supervised restart cut it)
        finally:
            conn.close()


# --- the governor and the supervisor ------------------------------------------------------------------------------

def read_memory() -> dict:
    free = machine.gpu_memory_free_mib()
    total, avail = machine.ram_mib()
    commit = machine.commit_mib()
    return {"vram_free_mib": free[0] if free else None, "ram_total_mib": total, "ram_avail_mib": avail,
            "commit_avail_mib": commit[1] if commit else None}


def floors(row: dict, opt: ServeOptions, ram_total_mib: int) -> dict:
    ram = opt.ram_floor_mib if plan.pinned_mib(row) else max(2048, ram_total_mib // 10)
    return {"vram_free_mib": opt.margin_mib, "ram_avail_mib": ram, "commit_avail_mib": opt.commit_floor_mib}


def crossings(r: dict, f: dict) -> dict[str, str]:
    """Each floor the reading is under, as {kind: one-line detail}; kinds are vram, ram and commit."""
    out = {}
    if r.get("vram_free_mib") is not None and r["vram_free_mib"] < f["vram_free_mib"]:
        out["vram"] = f"VRAM free {r['vram_free_mib']} MiB < {f['vram_free_mib']} MiB floor"
    if r.get("ram_avail_mib") is not None and r["ram_avail_mib"] < f["ram_avail_mib"]:
        out["ram"] = f"RAM available {r['ram_avail_mib']} MiB < {f['ram_avail_mib']} MiB floor"
    if r.get("commit_avail_mib") is not None and r["commit_avail_mib"] < f["commit_avail_mib"]:
        out["commit"] = f"commit headroom {r['commit_avail_mib']} MiB < {f['commit_avail_mib']} MiB floor"
    return out


def pressure(r: dict, f: dict) -> tuple[set[str], str]:
    """The crossings as plan.next_smaller's pressure kinds (commit headroom is a RAM kind there) and one line."""
    c = crossings(r, f)
    return {"ram" if k == "commit" else k for k in c}, "; ".join(c.values())


def _tps(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f} tok/s"


class Supervisor:
    """Owns llama-server's life: start, prime, watch memory and decode speed (tick), restart, step up, give up."""

    def __init__(self, engine, model: Path, rows: list[dict], opt: ServeOptions, gate: Gate, log: Path,
                 read=read_memory, clock=time.monotonic, say=print):
        self.engine, self.model, self.rows, self.opt, self.gate = engine, model, rows, opt, gate
        self.log_path, self.read, self.clock, self.say = log, read, clock, say
        self.row: dict | None = None
        self.planned: dict | None = None                # the row the session started on: the step-up target
        self.state = "starting"
        self.restarts = 0
        self.reason: str | None = None
        self.done = threading.Event()
        self.events: list[str] = []
        self.lock = threading.RLock()
        self.prime_result: dict | None = None
        self.after_start = None                         # called with the row after every successful start
        # decode rates of completed requests: their own lock, because a handler thread records a rate while a
        # restart holds self.lock waiting for that same request to finish
        self.rate_lock = threading.Lock()
        self.baselines: dict[str, float] = {}           # row key -> median of its first baseline_n rates, this session
        self.baseline: float | None = None
        self.samples: list[float] = []                  # rates on this row before its baseline is set
        self.recent: list[tuple[float, float]] = []     # (clock, rate) on this row after its baseline is set
        self.crossed: dict[str, str] = {}               # the floors crossed at the last reading
        self.commit_paused = False
        self.slow_since: float | None = None
        self.slow_kinds: dict[str, str] = {}            # crossings seen while the current slow run lasted
        self.clear_since: float | None = None           # memory has fitted the planned row since this time
        self.limit_noted = False

    def event(self, msg: str) -> None:
        line = f"# marian serve {time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}"
        self.events.append(msg)
        self.say(msg)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def start(self, row: dict) -> tuple[bool, str]:
        ok, why = self.engine.start(self.model, plan.to_config(row), self.log_path)
        if ok:
            self.row = row
            if self.planned is None:
                self.planned = row
            self.state = "serving"
            self.new_row(row)
            if self.opt.gate_prefix is not None:
                self.prime()
            if self.after_start is not None:
                self.after_start(row)
        return ok, why

    def new_row(self, row: dict) -> None:
        """A (re)start: rates measured on another row say nothing about this one. A row already measured this
        session keeps its baseline (the planned row, after a step back up)."""
        with self.rate_lock:
            self.baseline = self.baselines.get(plan.key(row))
            self.samples, self.recent = [], []
        self.slow_since, self.slow_kinds, self.clear_since = None, {}, None
        self.commit_paused = False

    def prime(self) -> None:
        """Send the gate's fixed instruction block once; the slot keeps its KV for the next request (opt-in reuse)."""
        try:
            r = self.engine.post("/completion", {"prompt": self.opt.gate_prefix, "n_predict": 0, "cache_prompt": True,
                                                 "temperature": 0}, timeout=600)
            t = r.get("timings") or {}
            self.prime_result = {"prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms")}
            self.event(f"gate prefix primed: {t.get('prompt_n')} tokens in {t.get('prompt_ms') or 0:.0f} ms")
        except (OSError, ValueError, http.client.HTTPException) as e:
            self.event(f"gate prefix NOT primed ({type(e).__name__}); requests will prefill it themselves")

    # --- decode speed, from the timings the proxy already reads ---

    def record(self, timings: dict | None) -> None:
        """One completed request's decode rate (called by the proxy). The first baseline_n set the row's baseline:
        their median, not the profile's figure, because serve's loop runs 10-16% under calibrate's on a clean
        machine (register E53) and a profile-based 70% bar would sit that much closer to a false alarm."""
        if not timings or self.row is None:
            return
        n, ms = timings.get("predicted_n"), timings.get("predicted_ms")
        if not n or not ms or n < confirm.MIN_TOKENS:
            return
        rate, now, msg = n / (ms / 1000), self.clock(), None
        with self.rate_lock:
            if self.baseline is None:
                self.samples.append(rate)
                if len(self.samples) >= self.opt.baseline_n:
                    self.baseline = statistics.median(self.samples)
                    self.baselines[plan.key(self.row)] = self.baseline
                    msg = (f"baseline for this row: {_tps(self.baseline)}, the median of its first "
                           f"{len(self.samples)} answers (profile: {_tps(self.row.get('tg_tps'))})")
            else:
                self.recent.append((now, rate))
                del self.recent[:-64]
        if msg:
            self.event(msg)

    def live_rate(self) -> float | None:
        """The median of the last three decode rates on this row, for the log lines."""
        with self.rate_lock:
            xs = [x for _, x in self.recent[-3:]] or self.samples[-3:]
        return statistics.median(xs) if xs else None

    # --- one governor step ---

    def tick(self, r: dict | None = None) -> None:
        """One governor step: called every interval_s by the governor thread, and directly by the tests."""
        with self.lock:
            if self.state != "serving" or self.row is None:
                return
            r = r if r is not None else self.read()
            proc = getattr(self.engine, "proc", None)
            if proc is not None and proc.poll() is not None:
                return self.restart(set(), "the model server exited on its own", r, same_ok=True)
            cross = crossings(r, floors(self.row, self.opt, r.get("ram_total_mib") or 0))
            now = self.clock()
            self.warn(cross)
            self.admission(cross)
            if self.opt.governor != "restart":
                return
            if self.slowdown(cross, now, r):
                return
            self.step_up(r, now)

    def warn(self, cross: dict[str, str]) -> None:
        """One line when a floor is first crossed and one when it clears, with the live decode rate."""
        speed = f"live decode {_tps(self.live_rate())}, row baseline {_tps(self.baseline)}"
        for k, detail in cross.items():
            if k not in self.crossed:
                tail = ("admission pauses" if k == "commit" else
                        "admission continues; " + ("a slowdown would restart" if self.opt.governor == "restart"
                                                   else "a memory reading alone never restarts"))
                self.event(f"warning: {detail} ({speed}); {tail}")
        for k in self.crossed:
            if k not in cross:
                self.event(f"clear: {k} is back above its floor ({speed})")
        self.crossed = dict(cross)

    def admission(self, cross: dict[str, str]) -> None:
        """Only commit headroom pauses admission: under it, allocations fail (the crash risk). VRAM and RAM
        crossings only warn: register E53 found llama-server kept its speed with 150 MiB of VRAM free and under
        1 GiB of RAM available, while restarting cost more than the pressure did."""
        if "commit" in cross:
            if not self.commit_paused:
                self.commit_paused = True
                self.gate.pause("memory_pressure", f"{cross['commit']}; admission paused because allocations can "
                                                   "fail below it (the running request finishes)")
                self.event(f"pause: {cross['commit']}")
        elif self.commit_paused:
            self.commit_paused = False
            self.gate.resume()
            self.event("resume: commit headroom is back above its floor")

    def slowdown(self, cross: dict[str, str], now: float, r: dict) -> bool:
        """`restart` mode's trigger: the decode rate of completed requests stays under `slowdown` x the row's
        baseline across persist_s (at least two answers, the first and last that far apart, none in between at or
        above the bar), while a memory floor was crossed. True if it restarted."""
        with self.rate_lock:
            base, recent = self.baseline, list(self.recent)
        if base is None:
            return False
        bar = self.opt.slowdown * base
        run = []
        for t, x in reversed(recent):
            if x >= bar:
                break
            run.append((t, x))
        if not run:
            if self.slow_since is not None:
                self.event(f"decode recovered: {_tps(recent[-1][1])} against the bar of {_tps(bar)}")
            self.slow_since, self.slow_kinds = None, {}
            return False
        first, last = run[-1][0], run[0][0]
        if self.slow_since != first:
            self.slow_since, self.slow_kinds = first, {}
            self.event(f"slow: live decode {_tps(run[0][1])} is under {100 * self.opt.slowdown:.0f}% of this row's "
                       f"baseline {_tps(base)}")
        self.slow_kinds.update(cross)
        if len(run) < 2 or last - first < self.opt.persist_s:
            return False
        speed = (f"live decode {_tps(statistics.median(x for _, x in run))} stayed under "
                 f"{100 * self.opt.slowdown:.0f}% of this row's baseline {_tps(base)} for {last - first:.0f} s")
        if not self.slow_kinds:
            self.event(f"{speed}, but no memory floor was crossed: a smaller row would not help; serving on")
            with self.rate_lock:
                self.recent = []
            self.slow_since = None
            return False
        kinds = {"ram" if k == "commit" else k for k in self.slow_kinds}
        self.restart(kinds, f"{speed}, with {'; '.join(self.slow_kinds.values())}", r)
        return True

    def step_up(self, r: dict, now: float) -> None:
        """Back to the planned row once memory has fitted it, floors kept, for step_up_s. Never while requests are
        queued; the running one drains first, as for any restart. Counts against max_restarts."""
        if self.planned is None or plan.key(self.row) == plan.key(self.planned) or self.slow_since is not None:
            self.clear_since = None
            return
        p, cur = self.planned, self.row
        ram_floor = floors(p, self.opt, r.get("ram_total_mib") or 0)["ram_avail_mib"]
        ok = plan.fits_after_stop(p, cur, vram_free_mib=r.get("vram_free_mib"), ram_free_mib=r.get("ram_avail_mib"),
                                  margin_mib=self.opt.margin_mib, ram_floor_mib=ram_floor)
        c = r.get("commit_avail_mib")
        extra = max(0, plan.ram_need_mib(p) - plan.ram_need_mib(cur))
        ok = ok and (c is None or c - extra >= self.opt.commit_floor_mib)
        if not ok:
            self.clear_since = None
            return
        if self.clear_since is None:
            self.clear_since = now
            return
        if now - self.clear_since < self.opt.step_up_s:
            return
        if self.restarts >= self.opt.max_restarts:
            if not self.limit_noted:
                self.limit_noted = True
                self.event(f"memory fits the planned configuration again, but {self.restarts} supervised restarts "
                           f"were already made (limit {self.opt.max_restarts}); staying on {confirm.describe(cur)}")
            return
        if not self.gate.pause_unless_queued("restarting", "stepping back up to the planned configuration"):
            return                                      # requests are queued: try again at the next reading
        self.restart(set(), f"memory has fitted the planned configuration for {now - self.clear_since:.0f} s",
                     r, target=p)

    def restart(self, kinds: set[str], detail: str, r: dict, same_ok: bool = False,
                target: dict | None = None) -> None:
        self.state = "restarting"
        self.gate.pause("restarting", f"restarting the model server: {detail}")
        if self.restarts >= self.opt.max_restarts:
            return self.give_up(f"{detail}, and {self.restarts} supervised restarts were already made "
                                f"(limit {self.opt.max_restarts})")
        if not self.gate.wait_idle(self.opt.drain_s):
            self.event(f"the running request did not finish within {self.opt.drain_s:.0f} s; it is cut")
        cur = self.row
        if target is not None:
            nxt = target
        elif same_ok and not kinds:
            nxt = cur
        else:
            nxt = plan.next_smaller(
                self.rows, cur, self.opt.need, self.opt.objective, pressure=kinds, vram_free_mib=r.get("vram_free_mib"),
                ram_free_mib=r.get("ram_avail_mib"), margin_mib=self.opt.margin_mib,
                ram_floor_mib=floors(cur, self.opt, r.get("ram_total_mib") or 0)["ram_avail_mib"],
                exact_only=self.opt.exact_only, stock=self.opt.stock, allow_non_exact=self.opt.allow_non_exact)
        self.engine.stop()
        if nxt is None:
            extra = ("" if not plan.exact_class(cur) or self.opt.allow_non_exact else
                     " (non-exact rows were not considered; --allow-non-exact would)")
            return self.give_up(f"{detail}; no smaller measured configuration fits the memory free now{extra}")
        self.restarts += 1
        ok, why = self.start(nxt)
        if not ok and target is not None:               # the step up failed: go back to the row that was serving
            self.event(f"step up on {confirm.describe(nxt)} failed ({why}); back to {confirm.describe(cur)}")
            ok, why = self.start(cur)
            nxt = cur
        if not ok:
            return self.give_up(f"restart {self.restarts} on {confirm.describe(nxt)} failed: {why}")
        self.gate.resume()
        verb = "step up" if target is not None and nxt is target else "restart"
        self.event(f"{verb} {self.restarts} of {self.opt.max_restarts}: {detail}; now serving "
                   f"{confirm.describe(nxt)} (the KV and prompt cache were lost)")

    def give_up(self, reason: str) -> None:
        self.state = "gave_up"
        self.reason = reason
        self.gate.pause("gave_up", f"Marian stopped serving: {reason}")
        try:
            self.engine.stop()
        finally:
            self.event(f"GIVING UP: {reason}")
            self.done.set()


class Governor(threading.Thread):
    def __init__(self, sup: Supervisor, interval_s: float):
        super().__init__(daemon=True)
        self.sup, self.interval, self._halt = sup, interval_s, threading.Event()

    def run(self):
        while not self._halt.wait(self.interval):
            try:
                self.sup.tick()
            except Exception as e:                      # noqa: BLE001 - a broken reading must not kill the watch
                self.sup.event(f"governor reading failed: {type(e).__name__}")

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


# --- the whole serve -----------------------------------------------------------------------------------------------

@dataclass
class Session:
    gate: Gate
    sup: Supervisor
    front: Front
    gov: Governor
    live: confirm.LiveCheck | None
    threads: list = field(default_factory=list)

    def close(self) -> None:
        if self.gov.is_alive():
            self.gov.stop()
        self.front.shutdown()
        self.front.server_close()
        self.sup.engine.stop()


def open_session(engine, model: Path, rows: list[dict], row: dict, opt: ServeOptions, log: Path, *,
                 profile_dir: Path | None = None, read=read_memory, clock=time.monotonic, say=print,
                 governor: bool = True) -> tuple[Session | None, str]:
    """Start llama-server on the row, then the proxy and the governor. (None, reason) if the server did not start."""
    gate = Gate(opt.queue)
    sup = Supervisor(engine, model, rows, opt, gate, log, read=read, clock=clock, say=say)
    ok, why = sup.start(row)
    if not ok:
        return None, why

    def done(result: dict) -> None:
        sup.event(result["message"])
        if profile_dir is not None and sup.row is not None:
            try:
                confirm.write_back(profile_dir, [(sup.row, "live_check", result)])
            except (OSError, ValueError):
                pass

    live = confirm.LiveCheck(row.get("tg_tps"), opt.live_n, opt.slow, on_done=done) if opt.live_n > 0 else None
    if live is not None:                                 # after a supervised restart, judge the new row
        sup.after_start = lambda r: live.reset(r.get("tg_tps"))
    front = Front(("127.0.0.1", opt.port), gate, engine.port, opt, live, say=say, on_timings=sup.record)
    t = threading.Thread(target=front.serve_forever, daemon=True)
    t.start()
    gov = Governor(sup, opt.interval_s)
    sup.event(GOVERNOR_LINE[opt.governor])
    if governor and opt.governor != "off":
        gov.start()
    return Session(gate, sup, front, gov, live, [t]), "ok"
