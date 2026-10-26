"""Offline tests of M1.3's serving side: the CPU thread sweep, live confirmation (keep-if-faster), the front proxy
(one sequence, streaming pass-through, 503 with a reason, prefix reuse off), and the memory governor with supervised
restart. No GPU and no llama-server: a fake engine, a fake upstream HTTP server, fake memory readings and a fake clock.
python -m pytest -q tests"""
import contextlib
import http.client
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from marian import cli, confirm, machine, plan, profile, serve
from marian.engines import Config, parse_help
from marian.profile import ThreadOptions, thread_ladder

import test_m13
from test_m13 import HELP_MARIAN, FakeLlama, FakeMarian, Serve, fake_machine, row, toy_moe


# --- the CPU thread sweep -----------------------------------------------------------------------------------------

class ThreadSweep(unittest.TestCase):
    def test_ladder(self):
        self.assertEqual(thread_ladder(16, 24), [16, 14, 8, 4])
        self.assertEqual(thread_ladder(8, 16), [8, 6, 4])
        self.assertEqual(thread_ladder(4, 4), [4, 2])
        self.assertEqual(thread_ladder(None, 12), [12, 10, 8, 6, 4])         # no physical count: logical stands in
        self.assertEqual(thread_ladder(None, None), [])
        self.assertTrue(all(1 <= t <= 6 for t in thread_ladder(32, 6)))       # never above the logical processors

    def test_physical_cores_is_a_count_or_none(self):
        n = machine.physical_cores()
        self.assertTrue(n is None or (isinstance(n, int) and n > 0))

    def run_profile(self, eng, threads=None, sweep=ThreadOptions(ladder=(8, 6, 4))):
        with tempfile.TemporaryDirectory() as d, fake_machine(eng), \
                mock.patch.object(machine, "physical_cores", lambda: 8), \
                mock.patch.object(profile, "model_geometry", lambda m: object()), \
                mock.patch.object(profile.geometry, "floor_mib", lambda g, n=512: 700), \
                mock.patch.object(profile.geometry, "full_mib", lambda g, n=512: 2000):
            out = profile.run(toy_moe(Path(d)), eng, [4096], ["f16"], 2, threads, Path(d) / "p", 16, sweep=sweep)
            return json.loads(out.read_text(encoding="utf-8"))

    def test_cpu_rows_get_the_ladder_and_the_best_is_kept(self):
        class Threads(FakeMarian):
            def complete(self, text, n):
                t = super().complete(text, n)
                c = self.running
                if c.n_cpu_moe and not c.moe_cache_mib:                 # memory-bound: 6 threads is the sweet spot
                    t["predicted_per_second"] = {None: 20.0, 8: 18.0, 6: 31.0, 4: 26.0}[c.threads]
                return t
        eng = Threads()
        doc = self.run_profile(eng)
        sweep = [r for r in doc["rows"] if r["arm"] == "threads"]
        self.assertEqual([r["threads"] for r in sweep], [8, 6, 4])           # thread_ladder(8, logical)
        self.assertTrue(all(r["n_cpu_moe"] == 2 and not r["exact_class"] for r in sweep))
        cell = doc["thread_search"]["cells"][0]
        self.assertEqual((cell["best_threads"], cell["best_tg_tps"], cell["split"]), (6, 31.0, {"n_cpu_moe": 2}))
        self.assertEqual(doc["thread_search"]["ladder"], [8, 6, 4])
        # cache arms and the all-on-GPU point never ran with an explicit thread count
        self.assertFalse(any(c.threads for c in eng.started if c.moe_cache_mib or c.n_cpu_moe == 0))
        best = plan.choose(plan.rows_of([(Path("p.json"), doc)]), 3000, stock=True)
        self.assertEqual((best["n_cpu_moe"], best["threads"], best["tg_tps"]), (2, 6, 31.0))   # beats all-on-GPU's 30
        self.assertIn("-t 6", " ".join(test_m13.LlamaCpp("llama-server").command(Path("m"), plan.to_config(best))))

    def test_default_threads_can_win_and_fixed_threads_skip(self):
        doc = self.run_profile(FakeMarian())                                  # threads change nothing here
        cell = doc["thread_search"]["cells"][0]
        self.assertIsNone(cell["best_threads"])
        self.assertIn("default", cell["note"])
        doc = self.run_profile(FakeMarian(), threads=12)
        self.assertIn("--threads fixed", doc["thread_search"]["skipped"])
        self.assertFalse(any(r["arm"] == "threads" for r in doc["rows"]))
        doc = self.run_profile(FakeMarian(), sweep=ThreadOptions(enabled=False))
        self.assertNotIn("thread_search", doc)

    def test_no_cpu_row_no_sweep(self):
        s = {"block_count": 2, "moe": True}
        rows, summary = profile.thread_sweep(Path("m"), None, s, 4096, "f16",
                                             [dict(row(n_cpu_moe=0, exact_class=True))], Path("log"), 8000, 16, 300, [8])
        self.assertEqual(rows, [])
        self.assertIn("every expert", summary["skipped"])


# --- plan helpers: runner-up, live-confirmation preference, next smaller -----------------------------------------

def tag(r, i):
    return dict(r, _file="p.json", _index=i, _block_count=2)


ROWS = [tag(row(n_cpu_moe=2, moe_cache_mib=3000, load_mode="mmap", tg=40.0, share=6000, exact_class=True,
                ram_free_before_mib=20000, ram_avail_low_mib=18000), 0),
        tag(row(n_cpu_moe=2, moe_cache_mib=2000, load_mode="mmap", tg=35.0, share=5000, exact_class=True,
                ram_free_before_mib=20000, ram_avail_low_mib=17000), 1),
        tag(row(n_cpu_moe=2, moe_cache_mib=1000, load_mode="mmap", tg=30.0, share=4000, exact_class=True,
                ram_free_before_mib=20000, ram_avail_low_mib=16000), 2),
        tag(row(n_cpu_moe=1, tg=38.0, share=3000, exact_class=False, ram_free_before_mib=20000,
                ram_avail_low_mib=12000), 3)]


class PlanHelpers(unittest.TestCase):
    def test_runner_up_keeps_an_exact_plan_exact(self):
        best = plan.choose(ROWS, 3000)
        self.assertEqual(best["moe_cache_mib"], 3000)
        self.assertEqual(plan.runner_up(ROWS, best, 3000)["moe_cache_mib"], 2000)        # not the faster stock row
        self.assertEqual(plan.runner_up(ROWS, best, 3000, allow_non_exact=True)["n_cpu_moe"], 1)
        stock = plan.choose(ROWS, 3000, stock=True)
        self.assertIsNone(plan.runner_up(ROWS, stock, 3000, stock=True))                  # nothing else is stock

    def test_a_row_that_lost_gives_way(self):
        rows = [dict(r) for r in ROWS]
        rows[0]["confirmed"] = {"outcome": "replaced", "by": plan.key(rows[1])}
        self.assertEqual(plan.choose(rows, 3000)["moe_cache_mib"], 2000)
        rows[0]["confirmed"] = {"outcome": "replaced", "by": plan.key(rows[3])}
        self.assertEqual(plan.choose(rows, 3000, exact_only=True)["moe_cache_mib"], 3000)   # the winner must qualify

    def test_next_smaller(self):
        cur = ROWS[0]
        nxt = plan.next_smaller(ROWS, cur, 3000, pressure={"vram"}, vram_free_mib=200, ram_free_mib=20000)
        self.assertEqual(nxt["moe_cache_mib"], 2000)              # 5,000 + 300 fits in 200 + the 6,000 freed
        nxt = plan.next_smaller(ROWS, cur, 3000, pressure={"vram"}, vram_free_mib=-900, ram_free_mib=20000)
        self.assertEqual(nxt["moe_cache_mib"], 1000)              # a bigger competitor: one more step down
        self.assertIsNone(plan.next_smaller(ROWS, ROWS[2], 3000, pressure={"vram"}, vram_free_mib=100,
                                            ram_free_mib=20000))   # only the non-exact row is smaller
        self.assertEqual(plan.next_smaller(ROWS, ROWS[2], 3000, pressure={"vram"}, vram_free_mib=100, ram_free_mib=20000,
                                           allow_non_exact=True)["n_cpu_moe"], 1)
        # RAM pressure: rows needing less RAM than the current one (2,000 MiB), inside what is available + freed
        self.assertIsNone(plan.next_smaller(ROWS, cur, 3000, pressure={"ram"}, vram_free_mib=None, ram_free_mib=5000))
        self.assertEqual(plan.ram_need_mib(ROWS[3]), 8000)


# --- live confirmation --------------------------------------------------------------------------------------------

class Confirm(unittest.TestCase):
    def test_decide(self):
        o = confirm.ConfirmOptions(rounds=4)
        self.assertEqual(confirm.decide([10, 10, 10, 10], [11, 11, 11, 9], o)["outcome"], "replaced")   # 3 of 4, +10%
        self.assertEqual(confirm.decide([10, 10, 10, 10], [11, 11, 9, 9], o)["outcome"], "kept")        # 2 of 4
        self.assertEqual(confirm.decide([10, 10, 10, 10], [10.1, 10.1, 10.1, 10.1], o)["outcome"], "kept")  # +1% < 2%
        self.assertEqual(confirm.decide([None, 10, 10, 10], [9, 11, 11, 11], o)["outcome"], "replaced")  # plan failed
        self.assertEqual(confirm.decide([10, 10, 10, 10], [None, None, 30, 30], o)["outcome"], "kept")
        self.assertEqual(confirm.rounds_order(4), ["AB", "BA", "AB", "BA"])

    def test_live_check(self):
        got = []
        lc = confirm.LiveCheck(40.0, n=2, slow=0.15, on_done=got.append)
        self.assertIsNone(lc.add({"predicted_n": 4, "predicted_ms": 100}))           # too short to count
        self.assertIsNone(lc.add({"predicted_n": 64, "predicted_ms": 2000}))
        r = lc.add({"predicted_n": 64, "predicted_ms": 2000})                         # 32 tok/s: 20% below 40
        self.assertTrue(r["slow"] and got and "below the profiled" in r["message"])
        self.assertIsNone(lc.add({"predicted_n": 64, "predicted_ms": 1}))              # reports once
        self.assertFalse(confirm.live_verdict(36.0, 40.0, 0.15)["slow"])

    def test_ab_with_a_fake_engine_and_the_record(self):
        class Engine:
            name = "llama.cpp"

            def __init__(self, speeds):
                self.speeds, self.log, self.cur = speeds, [], None

            def start(self, model, c, log):
                self.cur = c
                self.log.append(c.moe_cache_mib)
                return True, "ok"

            def stop(self):
                self.cur = None

            def tokenize(self, text):
                return 30

            def complete(self, text, n):
                return {"predicted_per_second": self.speeds[self.cur.moe_cache_mib], "prompt_per_second": 500.0}

        o = confirm.ConfirmOptions(rounds=4)
        with mock.patch.object(confirm.time, "sleep", lambda s: None):
            eng = Engine({3000: 30.0, 2000: 36.0})               # the profile said 40 vs 35; live says otherwise
            d = confirm.ab(eng, Path("m.gguf"), ROWS[0], ROWS[1], Path("log"), o, say=lambda m: None)
        self.assertEqual(eng.log, [3000, 2000, 2000, 3000, 3000, 2000, 2000, 3000])   # ABBA ABBA, a fresh start each
        self.assertEqual((d["outcome"], d["runner_up_wins"]), ("replaced", 4))
        self.assertTrue(d["live_check"]["slow"])                  # 30 live against 40 profiled
        recs = confirm.records(ROWS[0], ROWS[1], d, o)
        self.assertEqual([r["outcome"] for _, r in recs], ["replaced", "won"])
        self.assertEqual(recs[0][1]["by"], plan.key(ROWS[1]))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p.json"
            p.write_text(json.dumps({"rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in ROWS]}),
                         encoding="utf-8")
            self.assertEqual(confirm.write_back(Path(td), [(r, "confirmed", rec) for r, rec in recs]), ["p.json"])
            back = plan.rows_of([(p, json.loads(p.read_text(encoding="utf-8")))])
            self.assertEqual(back[0]["confirmed"]["outcome"], "replaced")
            self.assertEqual(plan.choose(back, 3000)["moe_cache_mib"], 2000)       # the planner now prefers the winner
            self.assertFalse(list(Path(td).glob("*.tmp")))


# --- a fake llama-server ------------------------------------------------------------------------------------------

class FakeUpstream:
    """Speaks just enough of llama-server: /health, /completion and /v1/chat/completions (JSON or a server-sent event
    stream with timings in the last event), /tokenize. `hold` blocks answers until set; `up` False drops connections."""

    def __init__(self, tps=40.0):
        self.bodies, self.paths, self.tps = [], [], tps
        self.hold = threading.Event()
        self.hold.set()
        self.first_event_sent = threading.Event()
        self.up = True
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not up.up:
                    self.close_connection = True
                    return
                self._send(200, {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not up.up:
                    self.close_connection = True
                    return
                up.bodies.append(body)
                up.paths.append(self.path)
                timings = {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 64, "predicted_ms": 64000.0 / up.tps}
                if self.path == "/tokenize":
                    return self._send(200, {"tokens": [1, 2, 3]})
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    events = [b'data: {"content": "Hel"}\n\n', b'data: {"content": "lo"}\n\n',
                              b'data: ' + json.dumps({"content": "", "stop": True, "timings": timings}).encode() + b"\n\n"]
                    for i, e in enumerate(events):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(e), e))
                        self.wfile.flush()
                        if i == 0:
                            up.first_event_sent.set()
                            up.hold.wait(10)
                    self.wfile.write(b"0\r\n\r\n")
                    return
                up.hold.wait(10)
                self._send(200, {"content": "VERDICT: CLEAR", "timings": timings})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.hold.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeEngine:
    """The engine interface the supervisor uses, around one FakeUpstream on a fixed port (as llama-server keeps its
    internal port across restarts)."""
    name = "llama.cpp"

    def __init__(self, tps=40.0, fail_after=None):
        self.up = FakeUpstream(tps)
        self.port = self.up.port
        self.started, self.stops, self.fail_after, self.proc = [], 0, fail_after, None

    def start(self, model, c, log):
        self.started.append(c)
        if self.fail_after is not None and len(self.started) > self.fail_after:
            return False, "failed to start (see log)"
        self.up.up = True
        return True, "ready"

    def stop(self):
        self.stops += 1
        self.up.up = False

    def post(self, path, body, timeout=1800):
        req = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        req.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        return json.loads(req.getresponse().read())


def call(port, path, body=None, method="POST"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    data = json.dumps(body) if body is not None else None
    c.request(method, path, data, {"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@contextlib.contextmanager
def session(rows=ROWS, row=None, mem=None, **opt):
    eng = FakeEngine(opt.pop("tps", 40.0), opt.pop("fail_after", None))
    mem = mem if mem is not None else {}
    reading = {"vram_free_mib": 2000, "ram_total_mib": 32000, "ram_avail_mib": 20000, "commit_avail_mib": 30000}
    reading.update(mem)
    clock = Clock()
    o = serve.ServeOptions(port=0, internal_port=eng.port, need=3000, persist_s=10, drain_s=2, **opt)
    events = []
    with tempfile.TemporaryDirectory() as d:
        sess, why = serve.open_session(eng, Path("m.gguf"), rows, row or rows[0], o, Path(d) / "serve.log",
                                       read=lambda: dict(reading), clock=clock, say=events.append, governor=False)
        assert sess is not None, why
        try:
            yield sess, eng, reading, clock, events, sess.front.server_address[1]
        finally:
            sess.close()
            eng.up.close()


class Proxy(unittest.TestCase):
    def test_json_passes_and_prompt_reuse_is_forced_off(self):
        with session() as (sess, eng, *_ , port):
            code, _, body = call(port, "/completion", {"prompt": "x", "n_predict": 8})
            self.assertEqual((code, json.loads(body)["content"]), (200, "VERDICT: CLEAR"))
            call(port, "/v1/chat/completions", {"messages": [], "cache_prompt": True})
            call(port, "/tokenize", {"content": "x"})
            self.assertEqual([b.get("cache_prompt") for b in eng.up.bodies], [False, False, None])
            self.assertEqual(sess.front.counts["cache_prompt_forced_off"], 2)
            self.assertEqual(call(port, "/health", method="GET")[0], 200)

    def test_opt_in_leaves_requests_alone_and_primes_the_gate_prefix(self):
        with session(cache_prompt=True, gate_prefix="FIXED INSTRUCTIONS\n") as (sess, eng, *_ , port):
            self.assertEqual(eng.up.bodies[0], {"prompt": "FIXED INSTRUCTIONS\n", "n_predict": 0, "cache_prompt": True,
                                                "temperature": 0})
            call(port, "/completion", {"prompt": "FIXED INSTRUCTIONS\ndoc", "cache_prompt": True})
            self.assertTrue(eng.up.bodies[1]["cache_prompt"])
            self.assertEqual(sess.sup.prime_result["prompt_n"], 100)

    def test_streaming_passes_through_as_it_arrives(self):
        with session(live_n=1) as (sess, eng, *_ , port):
            eng.up.hold.clear()
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            c.request("POST", "/completion", json.dumps({"prompt": "x", "stream": True}), {"Content-Type": "application/json"})
            r = c.getresponse()
            self.assertEqual((r.status, r.getheader("Content-Type")), (200, "text/event-stream"))
            t0 = time.monotonic()
            first = r.read1(65536)                      # arrives while the upstream still holds the rest
            self.assertIn(b"Hel", first)
            self.assertNotIn(b'"timings"', first)       # not buffered: the last event has not been written yet
            self.assertLess(time.monotonic() - t0, 5)
            self.assertTrue(eng.up.first_event_sent.is_set())
            eng.up.hold.set()
            rest = r.read()
            self.assertIn(b'"timings"', rest)
            self.assertIsNotNone(sess.live.result)      # the live check read the last event's timings
            self.assertAlmostEqual(sess.live.result["live_tps"], 40.0, places=1)

    def test_one_at_a_time_then_queue_then_503(self):
        with session() as (sess, eng, *_ , port):
            eng.up.hold.clear()
            results = []
            ts = [threading.Thread(target=lambda: results.append(call(port, "/completion", {"prompt": "x"})[0]))
                  for _ in range(2)]
            for t in ts:
                t.start()
                time.sleep(0.2)
            deadline = time.time() + 5
            while (sess.gate.active, sess.gate.waiting) != (1, 1) and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual((sess.gate.active, sess.gate.waiting), (1, 1))
            self.assertEqual(len(eng.up.bodies), 1)                     # the second waits in the proxy, not upstream
            code, headers, body = call(port, "/completion", {"prompt": "x"})
            err = json.loads(body)["error"]
            self.assertEqual((code, err["code"], err["reason"]), (503, 503, "queue_full"))
            self.assertEqual(headers.get("Retry-After"), "5")
            eng.up.hold.set()
            for t in ts:
                t.join(10)
            self.assertEqual(results, [200, 200])

    def test_dead_upstream_is_a_503_not_a_hang(self):
        with session() as (sess, eng, *_ , port):
            eng.up.up = False
            code, _, body = call(port, "/completion", {"prompt": "x"})
            self.assertEqual((code, json.loads(body)["error"]["reason"]), (503, "upstream_unavailable"))


class Governor(unittest.TestCase):
    def test_pause_then_resume(self):
        with session() as (sess, eng, reading, clock, events, port):
            sess.sup.tick()
            self.assertIsNone(sess.gate.paused)
            reading["vram_free_mib"] = 150                       # a competing program took VRAM
            sess.sup.tick()
            code, _, body = call(port, "/completion", {"prompt": "x"})
            self.assertEqual((code, json.loads(body)["error"]["reason"]), (503, "memory_pressure"))
            self.assertIn("VRAM free 150 MiB < 300 MiB floor", json.loads(body)["error"]["message"])
            self.assertEqual(call(port, "/health", method="GET")[0], 503)
            clock.t = 5
            reading["vram_free_mib"] = 900
            sess.sup.tick()
            self.assertIsNone(sess.gate.paused)
            self.assertEqual(call(port, "/completion", {"prompt": "x"})[0], 200)
            self.assertEqual(len(eng.started), 1)                # no restart for a short dip
            self.assertTrue(any(e.startswith("pause:") for e in events) and any(e.startswith("resume") for e in events))

    def test_persisting_pressure_restarts_smaller_then_gives_up(self):
        with session(max_restarts=3) as (sess, eng, reading, clock, events, port):
            reading["vram_free_mib"] = 150
            sess.sup.tick()
            clock.t = 11                                         # past persist_s: restart on the next smaller row
            sess.sup.tick()
            self.assertEqual([c.moe_cache_mib for c in eng.started], [3000, 2000])
            self.assertEqual((sess.sup.restarts, sess.sup.state, sess.gate.paused), (1, "serving", None))
            self.assertTrue(any("restart 1 of 3" in e and "cache 2000 MiB" in e for e in events))
            self.assertEqual(call(port, "/completion", {"prompt": "x"})[0], 200)
            clock.t = 20
            sess.sup.tick()                                      # still short of VRAM: pause again...
            clock.t = 31
            sess.sup.tick()                                      # ...and one step further down
            self.assertEqual(eng.started[-1].moe_cache_mib, 1000)
            clock.t = 40
            sess.sup.tick()
            clock.t = 51
            sess.sup.tick()                                      # nothing exact is smaller: give up, clearly
            self.assertEqual(sess.sup.state, "gave_up")
            self.assertIn("no smaller measured configuration", sess.sup.reason)
            self.assertIn("--allow-non-exact", sess.sup.reason)
            self.assertTrue(sess.sup.done.is_set())
            code, _, body = call(port, "/completion", {"prompt": "x"})
            self.assertEqual((code, json.loads(body)["error"]["reason"]), (503, "gave_up"))

    def test_restart_limit(self):
        with session(max_restarts=1, allow_non_exact=True) as (sess, eng, reading, clock, events, port):
            reading["vram_free_mib"] = 150
            for t in (0, 11, 20, 31):
                clock.t = t
                sess.sup.tick()
            self.assertEqual(sess.sup.state, "gave_up")
            self.assertIn("1 supervised restarts were already made (limit 1)", sess.sup.reason)
            self.assertEqual(len(eng.started), 2)

    def test_ram_floor_depends_on_pinning(self):
        f = serve.floors(ROWS[0], serve.ServeOptions(), 32000)
        self.assertEqual(f["ram_avail_mib"], 3200)              # max(2 GiB, 10% of RAM) for a row that pins nothing
        tier = dict(ROWS[0], miss_source="tier", tier_mib=4096)
        self.assertEqual(serve.floors(tier, serve.ServeOptions(), 32000)["ram_avail_mib"], 6144)
        kinds, detail = serve.pressure({"vram_free_mib": 2000, "ram_avail_mib": 3000, "commit_avail_mib": 3000}, f)
        self.assertEqual(kinds, {"ram"})
        self.assertIn("commit headroom 3000 MiB", detail)

    def test_a_server_that_exits_is_restarted_on_the_same_row(self):
        with session() as (sess, eng, reading, clock, events, port):
            eng.proc = mock.Mock(poll=lambda: 1)                # the process is gone
            sess.sup.tick()
            eng.proc = None
            self.assertEqual([c.moe_cache_mib for c in eng.started], [3000, 3000])
            self.assertTrue(any("exited on its own" in e for e in events))


# --- the CLI ------------------------------------------------------------------------------------------------------

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ServeCli(unittest.TestCase):
    def setUp(self):
        FakeLlama.help_text, FakeLlama.version_text, FakeLlama.starts = HELP_MARIAN, "0.7.0-dev (build 3203, commit 2c1cf30a)", []

    def test_gate_prefix_needs_the_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "prefix.txt").write_text("FIXED", encoding="utf-8")
            rc, out, _ = Serve.run_cli(self, Path(d), "serve", "--engine", "x", "--need", "3000", "--json",
                                       "--gate-prefix", str(Path(d) / "prefix.txt"), doc=Serve.DOC)
        self.assertEqual(rc, 2)
        self.assertIn("E52", json.loads(out)["reason"])

    def test_dry_run_reports_the_proxy_and_no_prompt_reuse(self):
        FakeLlama.help_text = HELP_MARIAN + "--cache-prompt, --no-cache-prompt     whether to enable prompt caching\n"
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = Serve.run_cli(self, Path(d), "serve", "--engine", "x", "--need", "3000", "--dry-run", "--json",
                                       "--port", "9000", doc=Serve.DOC)
        doc = json.loads(out)
        self.assertEqual((rc, doc["port"], doc["internal_port"], doc["cache_prompt"]), (0, 9000, 9001, False))
        args = doc["server_args"]
        self.assertEqual(args[args.index("--port") + 1], "9001")
        self.assertIn("--no-cache-prompt", args)

    def test_confirm_records_the_ab_in_the_profile(self):
        class Live(FakeLlama):
            def start(self, model, c, log, timeout=900, attempts=3):
                FakeLlama.starts.append(c)
                self._c = c
                return True, "ready"

            def stop(self):
                pass

            def tokenize(self, text):
                return 30

            def complete(self, prompt, n_predict):
                fast = (self._c.miss_source == "tier")
                return {"predicted_per_second": 20.0 if fast else 33.0, "prompt_per_second": 500.0}

        doc = json.loads(json.dumps(Serve.DOC))                 # plan: the tier row; runner-up: all-on-GPU (exact)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(confirm.time, "sleep", lambda s: None), \
                mock.patch.object(cli.serve, "open_session", lambda *a, **k: (None, "stub: not serving")), \
                mock.patch.object(test_m13, "FakeLlama", Live):
            rc, out, err = Serve.run_cli(self, Path(d), "serve", "--engine", "x", "--need", "3000", "--confirm",
                                         "--confirm-rounds", "2", "--port", str(free_port()), doc=doc)
            prof = next((Path(d) / "profiles").glob("*.json"))
            rows = json.loads(prof.read_text(encoding="utf-8"))["rows"]
        self.assertEqual(rc, 2)                                  # the stub refused to serve
        self.assertEqual([c.miss_source for c in FakeLlama.starts], ["tier", None, None, "tier"])
        tier = next(r for r in rows if r.get("miss_source") == "tier")
        self.assertEqual((tier["confirmed"]["outcome"], tier["confirmed"]["role"]), ("replaced", "plan"))
        other = next(r for r in rows if r.get("n_cpu_moe") == 0)
        self.assertEqual(other["confirmed"]["outcome"], "won")
        self.assertNotIn("confirmed", next(r for r in rows if r.get("n_cpu_moe") == 2 and not r.get("moe_cache_mib")))
        self.assertIn("serving the runner-up", out + err)


if __name__ == "__main__":
    unittest.main()
