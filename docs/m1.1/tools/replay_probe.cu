// Marian M1.1 trace-replay probe: replays the slot-map fill sequence of a P0 trace (MARIAN_MOE_TRACE, S lines) against
// three miss sources and reports host time and GB/s per decode token for each:
//   mmap : cudaMemcpyAsync from the memory-mapped GGUF (pageable, may fault to the NVMe): what upstream does today
//   ring : unbuffered overlapped reads (FILE_FLAG_NO_BUFFERING, sector-aligned) into a pinned staging ring, then upload
//   tier : a bounded pinned RAM tier holding the experts with the highest expected misses per byte (within --tier-mib);
//          fills outside it fall back to the ring
// The P0 trace records per layer step the number of fills and their bytes, not which experts. Expert ids come from
// --fills (a MARIAN_MOE_TRACE_FILLS file of the same run, real ids) or else are SYNTHETIC: per layer a seeded random
// permutation of the experts ranked by a Zipf(--zipf) popularity; the --cache-mib share of each layer's top ranks is
// taken as VRAM-resident, and misses are drawn from the remaining ranks by popularity. The physical rates (fault, read,
// pinned upload) are real; the tier hit rate under synthetic ids is a model output and is labelled as such.
// Expert offsets come from the GGUF header of every shard (tensor directory + data offset; stride per expert from the
// gap to the next tensor, cross-checked against the trace's bytes per fill).
//
// Usage: replay_probe --trace T.trace.csv --gguf SHARD [--gguf SHARD ...] [options]
//   --requests R.requests.jsonl  restrict to requests of --phases (default decode,post); else every S step after load
//   --phases decode,post         --sources mmap,ring,tier       --tier-mib 6144   --ring-mib 64   --qd 8
//   --fills F.fills.csv          --train-frac 0.5 (with --fills: tier ranked on the first part, stats on the rest)
//   --zipf 1.2  --cache-mib 9400 --seed 42 (synthetic ids)     --max-tokens N    --timeout-s 1200
//   --leave-gib G   hold private memory until only G GiB are available (memory pressure like the server's; 0 = off)
//   --csv OUT.csv   one row per replayed layer step and source
//   --dry-run       parse the GGUF headers, the trace and the ids, pick the tier, print the plan; no CUDA call at all
// Build (same nvcc as copy_probe): tools\build_m11.bat 8 probe   (or the nvcc line in that file)
// Do not run while another process measures on the GPU.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <psapi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] static void die(const char * fmt, const char * a = "", long long b = 0) {
    std::fprintf(stderr, "replay_probe: ");
    std::fprintf(stderr, fmt, a, b);
    std::fprintf(stderr, "\n");
    std::exit(1);
}

// ---------------------------------------------------------------------------------------------------------------------
// GGUF header: tensor directory of every shard
// ---------------------------------------------------------------------------------------------------------------------

struct gguf_tensor {
    std::string name;
    int file = 0;
    uint64_t off = 0;   // absolute file offset of the tensor data
    uint64_t ne[4] = { 1, 1, 1, 1 };
    uint32_t type = 0;
};

struct gguf_reader {
    FILE * f = nullptr;
    explicit gguf_reader(const char * path) { f = std::fopen(path, "rb"); }
    ~gguf_reader() { if (f) std::fclose(f); }
    void raw(void * p, size_t n) { if (std::fread(p, 1, n, f) != n) die("short read in GGUF header%s", ""); }
    template <typename T> T get() { T v; raw(&v, sizeof(v)); return v; }
    std::string str() { const uint64_t n = get<uint64_t>(); std::string s(n, '\0'); if (n) raw(&s[0], n); return s; }
    void skip(uint32_t t) {
        static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
        if (t == 8) { str(); return; }
        if (t == 9) { const uint32_t et = get<uint32_t>(); const uint64_t n = get<uint64_t>(); for (uint64_t i = 0; i < n; ++i) skip(et); return; }
        if (t > 12) die("unknown GGUF value type%s %lld", "", t);
        _fseeki64(f, sz[t], SEEK_CUR);
    }
};

struct gguf_shard {
    std::string path;
    uint64_t size = 0;
    uint64_t data_off = 0;
    uint64_t alignment = 32;
};

static void read_gguf(const std::string & path, int file, std::vector<gguf_shard> & shards, std::vector<gguf_tensor> & tensors,
                      std::map<std::string, uint64_t> & kv_u) {
    gguf_reader r(path.c_str());
    if (!r.f) die("cannot open %s", path.c_str());
    char magic[4]; r.raw(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) die("%s is not a GGUF file", path.c_str());
    const uint32_t version = r.get<uint32_t>();
    if (version < 2) die("%s: GGUF version below 2", path.c_str());
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv      = r.get<uint64_t>();
    gguf_shard sh; sh.path = path;
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (t == 4 || t == 5)       { kv_u[key] = r.get<uint32_t>(); }
        else if (t == 10 || t == 11) { kv_u[key] = r.get<uint64_t>(); }
        else if (t == 2 || t == 3)  { kv_u[key] = r.get<uint16_t>(); }
        else { r.skip(t); }
        if (key == "general.alignment") sh.alignment = kv_u[key];
    }
    std::vector<gguf_tensor> mine;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor t; t.file = file;
        t.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (nd > 4) die("%s: tensor with more than 4 dims", path.c_str());
        for (uint32_t d = 0; d < nd; ++d) t.ne[d] = r.get<uint64_t>();
        t.type = r.get<uint32_t>();
        t.off  = r.get<uint64_t>();
        mine.push_back(t);
    }
    const uint64_t pos = (uint64_t) _ftelli64(r.f);
    sh.data_off = (pos + sh.alignment - 1)/sh.alignment*sh.alignment;
    _fseeki64(r.f, 0, SEEK_END);
    sh.size = (uint64_t) _ftelli64(r.f);
    for (auto & t : mine) { t.off += sh.data_off; tensors.push_back(t); }
    shards.push_back(sh);
}

// ---------------------------------------------------------------------------------------------------------------------
// expert layout per layer
// ---------------------------------------------------------------------------------------------------------------------

struct expert_tensor {
    int file;
    uint64_t off;    // file offset of expert 0
    uint64_t nb2;    // bytes per expert
};

struct layer_layout {
    std::vector<expert_tensor> t; // gate_up / gate / up / down, as present
    uint64_t expert_bytes = 0;
    int n_expert = 0;
};

static std::vector<layer_layout> build_layout(const std::vector<gguf_shard> & shards, const std::vector<gguf_tensor> & tensors) {
    // stride from the gap to the next tensor of the same file: gap = n_expert*nb2 + padding, padding < alignment <= n_expert
    std::vector<std::vector<const gguf_tensor *>> by_file(shards.size());
    for (const auto & t : tensors) by_file[t.file].push_back(&t);
    std::unordered_map<const gguf_tensor *, uint64_t> gap;
    for (size_t f = 0; f < by_file.size(); ++f) {
        auto & v = by_file[f];
        std::sort(v.begin(), v.end(), [](const gguf_tensor * a, const gguf_tensor * b) { return a->off < b->off; });
        for (size_t i = 0; i < v.size(); ++i) {
            const uint64_t end = i + 1 < v.size() ? v[i + 1]->off : shards[f].size;
            gap[v[i]] = end - v[i]->off;
        }
    }
    std::vector<layer_layout> layers;
    static const char * roles[] = { "ffn_gate_up_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (int role = 0; role < 4; ++role) {
        for (const auto & t : tensors) {
            int il = -1; char rest[128] = { 0 };
            if (std::sscanf(t.name.c_str(), "blk.%d.%127s", &il, rest) != 2) continue;
            if (std::strncmp(rest, roles[role], std::strlen(roles[role])) != 0 || std::strcmp(rest + std::strlen(roles[role]), ".weight") != 0) continue;
            if ((int) layers.size() <= il) layers.resize(il + 1);
            const uint64_t n_expert = t.ne[2];
            const uint64_t g = gap[&t];
            const uint64_t nb2 = g / n_expert;
            if (n_expert < 2 || g - nb2*n_expert >= shards[t.file].alignment) die("cannot derive the expert stride of %s", t.name.c_str());
            layers[il].t.push_back({ t.file, t.off, nb2 });
            layers[il].expert_bytes += nb2;
            layers[il].n_expert = (int) n_expert;
        }
    }
    return layers;
}

// ---------------------------------------------------------------------------------------------------------------------
// trace, requests, fills
// ---------------------------------------------------------------------------------------------------------------------

struct sline { long long t_us; int layer; int n_ids; int fills; long long bytes; };

static std::vector<sline> read_trace(const char * path) {
    FILE * f = std::fopen(path, "r");
    if (!f) die("cannot open trace %s", path);
    std::vector<sline> res;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), f)) {
        if (buf[0] != 'S' || buf[1] != ',') continue;
        // kind,t_us,layer,n_ids,row_hits,uniq,uniq_hits,fills,bytes,us_wait,us_host
        long long t, bytes; int layer, n_ids, row_hits, uniq, uniq_hits, fills;
        if (std::sscanf(buf + 2, "%lld,%d,%d,%d,%d,%d,%d,%lld", &t, &layer, &n_ids, &row_hits, &uniq, &uniq_hits, &fills, &bytes) == 8) {
            res.push_back({ t, layer, n_ids, fills, bytes });
        }
    }
    std::fclose(f);
    return res;
}

struct request { std::string phase; long long t0, t1; };

static std::vector<request> read_requests(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) die("cannot open requests %s", path);
    std::vector<request> res;
    std::string line; int c;
    auto flush = [&]() {
        const char * s = line.c_str();
        const char * p = std::strstr(s, "\"phase\":\"");
        const char * a = std::strstr(s, "\"t0_us\":");
        const char * b = std::strstr(s, "\"t1_us\":");
        if (p && a && b) {
            p += 9; const char * q = std::strchr(p, '"');
            if (q) res.push_back({ std::string(p, q - p), std::atoll(a + 8), std::atoll(b + 8) });
        }
        line.clear();
    };
    while ((c = std::fgetc(f)) != EOF) { if (c == '\n') flush(); else line.push_back((char) c); }
    flush();
    std::fclose(f);
    return res;
}

// F lines of MARIAN_MOE_TRACE_FILLS: kind,t_us,layer,n_ids,expert,slot,src,bytes ; keyed by (t_us, layer) of the S line
static std::unordered_map<std::string, std::vector<int>> read_fills(const char * path) {
    std::unordered_map<std::string, std::vector<int>> res;
    FILE * f = std::fopen(path, "r");
    if (!f) die("cannot open fills %s", path);
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f)) {
        if (buf[0] != 'F' || buf[1] != ',') continue;
        long long t; int layer, n_ids, expert;
        if (std::sscanf(buf + 2, "%lld,%d,%d,%d", &t, &layer, &n_ids, &expert) == 4) {
            res[std::to_string(t) + ":" + std::to_string(layer)].push_back(expert);
        }
    }
    std::fclose(f);
    return res;
}

// ---------------------------------------------------------------------------------------------------------------------
// unbuffered overlapped reader with a fixed queue depth
// ---------------------------------------------------------------------------------------------------------------------

struct reader {
    std::vector<HANDLE> h;
    std::vector<bool> unbuffered;
    uint64_t sector = 0;
    int qd = 8;
    struct req { OVERLAPPED ov; HANDLE ev; int file; uint8_t * buf; uint32_t alen; uint32_t skip; uint32_t len; uint8_t * dev; };
    std::deque<req> inflight;
    std::vector<HANDLE> free_ev;
    cudaStream_t stream = nullptr;   // where a completed read is uploaded, nullptr = no upload (tier fill)
    double bytes_read = 0, n_reads = 0, n_fallback = 0;

    void open(const std::vector<gguf_shard> & shards, int qd_) {
        qd = qd_;
        for (const auto & s : shards) {
            HANDLE f = CreateFileA(s.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
            bool ub = true;
            if (f == INVALID_HANDLE_VALUE) {
                std::fprintf(stderr, "replay_probe: unbuffered open of %s failed (error %lu), falling back to buffered overlapped reads\n", s.path.c_str(), GetLastError());
                f = CreateFileA(s.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
                ub = false;
                n_fallback++;
            }
            if (f == INVALID_HANDLE_VALUE) die("cannot open %s", s.path.c_str());
            // the sector size is queried, not assumed (design C-14); reads and buffers are aligned to the largest one seen
            FILE_STORAGE_INFO si = {};
            if (GetFileInformationByHandleEx(f, FileStorageInfo, &si, sizeof(si))) {
                const uint64_t sec = std::max<uint64_t>(si.LogicalBytesPerSector, si.PhysicalBytesPerSectorForPerformance);
                if (sec > 0 && (sec & (sec - 1)) == 0) sector = std::max<uint64_t>(sector, sec);
            } else {
                std::fprintf(stderr, "replay_probe: sector size query failed for %s (error %lu), assuming 4096\n", s.path.c_str(), GetLastError());
            }
            h.push_back(f);
            unbuffered.push_back(ub);
        }
        if (sector == 0) sector = 4096;
        if (sector > 4096) die("sector size %s%lld above the 4096-byte alignment of pinned allocations is not handled", "", (long long) sector);
        for (int i = 0; i < qd; ++i) free_ev.push_back(CreateEventA(nullptr, TRUE, FALSE, nullptr));
    }
    uint64_t region(uint64_t nb2) const { return (nb2 + sector - 1)/sector*sector + sector; } // aligned read of nb2 bytes at any offset fits
    // read nb2 bytes at file offset off into the region at buf (sector aligned); the payload starts at buf + (off % sector)
    void submit(int file, uint64_t off, uint64_t nb2, uint8_t * buf, uint8_t * dev) {
        if ((int) inflight.size() >= qd) complete_oldest();
        const uint64_t aoff = off/sector*sector;
        const uint32_t skip = (uint32_t) (off - aoff);
        const uint32_t alen = (uint32_t) ((skip + nb2 + sector - 1)/sector*sector);
        req r = {};
        r.ev = free_ev.back(); free_ev.pop_back();
        r.ov.Offset = (DWORD) (aoff & 0xffffffffull); r.ov.OffsetHigh = (DWORD) (aoff >> 32); r.ov.hEvent = r.ev;
        r.file = file; r.buf = buf; r.alen = alen; r.skip = skip; r.len = (uint32_t) nb2; r.dev = dev;
        inflight.push_back(r);
        req & q = inflight.back();
        if (!ReadFile(h[file], buf, alen, nullptr, &q.ov) && GetLastError() != ERROR_IO_PENDING) {
            die("ReadFile failed%s (error %lld)", "", (long long) GetLastError());
        }
    }
    void complete_oldest() {
        req & r = inflight.front();
        DWORD n = 0;
        if (!GetOverlappedResult(h[r.file], &r.ov, &n, TRUE)) die("read failed%s (error %lld)", "", (long long) GetLastError());
        if (n < r.skip + r.len) die("short unbuffered read%s (%lld bytes)", "", (long long) n);
        bytes_read += n; n_reads++;
        if (r.dev && stream) CK(cudaMemcpyAsync(r.dev, r.buf + r.skip, r.len, cudaMemcpyHostToDevice, stream));
        free_ev.push_back(r.ev);
        inflight.pop_front();
    }
    void drain() { while (!inflight.empty()) complete_oldest(); }
};

// pinned host memory in chunks (very large single pinned allocations can fail on Windows)
struct pinned_pool {
    std::vector<uint8_t *> chunks;
    size_t chunk = 256u << 20, used = 0, total = 0;
    uint64_t align = 4096;
    uint8_t * alloc(size_t n) {
        n = (n + align - 1)/align*align;
        if (chunks.empty() || used + n > chunk) {
            void * p = nullptr;
            const size_t sz = std::max(chunk, n);
            if (cudaHostAlloc(&p, sz, cudaHostAllocDefault) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
            chunks.push_back((uint8_t *) p); used = 0; total += sz; chunk = sz;
        }
        uint8_t * res = chunks.back() + used;
        used += n;
        return res;
    }
    ~pinned_pool() { for (auto * p : chunks) cudaFreeHost(p); }
};

// ---------------------------------------------------------------------------------------------------------------------

struct step { std::vector<int> lines; bool decode; std::string phase; bool test = true; };

struct pass_stats {
    std::vector<double> tok_us, tok_bytes;
    double prompt_us = 0, prompt_bytes = 0;
    double tier_fills = 0, tier_bytes = 0, all_fills = 0, all_bytes = 0;
    double tier_bytes_dec = 0, all_bytes_dec = 0; // decode tokens only
    double read_bytes = 0;
    bool truncated = false;
};

static double q(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p*(v.size() - 1) + 0.5))];
}

static void hold_memory(double leave_gib, std::vector<void *> & held) {
    MEMORYSTATUSEX ms = { sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    const double avail = ms.ullAvailPhys/1073741824.0;
    const double hold = avail - leave_gib;
    if (hold <= 0.25) { std::printf("memory: %.2f GiB available, nothing to hold for --leave-gib %.2f\n", avail, leave_gib); return; }
    const size_t step_b = 256u << 20;
    size_t n = (size_t) (hold*1073741824.0/step_b);
    for (size_t i = 0; i < n; ++i) {
        void * p = VirtualAlloc(nullptr, step_b, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!p) break;
        for (size_t o = 0; o < step_b; o += 4096) ((volatile uint8_t *) p)[o] = 1;
        held.push_back(p);
    }
    GlobalMemoryStatusEx(&ms);
    std::printf("memory: held %.2f GiB of private memory, %.2f GiB available now (asked to leave %.2f)\n",
        held.size()*(double) step_b/1073741824.0, ms.ullAvailPhys/1073741824.0, leave_gib);
}

static double avail_gib() { MEMORYSTATUSEX ms = { sizeof(ms) }; GlobalMemoryStatusEx(&ms); return ms.ullAvailPhys/1073741824.0; }

int main(int argc, char ** argv) {
    std::string trace_path, requests_path, fills_path, csv_path, phases = "decode,post", sources = "mmap,ring,tier";
    std::vector<std::string> gguf_paths;
    double tier_mib = 6144, ring_mib = 64, zipf = 1.2, cache_mib = 9400, train_frac = 0.5, leave_gib = 0, timeout_s = 1200;
    int qd = 8; long long max_tokens = -1; unsigned seed = 42; bool dry_run = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die("missing value for %s", a.c_str()); return argv[++i]; };
        if (a == "--trace") trace_path = next();
        else if (a == "--gguf") gguf_paths.push_back(next());
        else if (a == "--requests") requests_path = next();
        else if (a == "--phases") phases = next();
        else if (a == "--sources") sources = next();
        else if (a == "--fills") fills_path = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--tier-mib") tier_mib = std::atof(next());
        else if (a == "--ring-mib") ring_mib = std::atof(next());
        else if (a == "--qd") qd = std::max(1, std::atoi(next()));
        else if (a == "--zipf") zipf = std::atof(next());
        else if (a == "--cache-mib") cache_mib = std::atof(next());
        else if (a == "--train-frac") train_frac = std::atof(next());
        else if (a == "--leave-gib") leave_gib = std::atof(next());
        else if (a == "--timeout-s") timeout_s = std::atof(next());
        else if (a == "--max-tokens") max_tokens = std::atoll(next());
        else if (a == "--seed") seed = (unsigned) std::atoi(next());
        else if (a == "--dry-run") dry_run = true;
        else die("unknown option %s", a.c_str());
    }
    if (trace_path.empty() || gguf_paths.empty()) {
        std::printf("usage: replay_probe --trace T.trace.csv --gguf SHARD [--gguf SHARD ...] [--requests R.jsonl] [--fills F.csv]\n"
                    "       [--sources mmap,ring,tier] [--tier-mib 6144] [--ring-mib 64] [--qd 8] [--zipf 1.2] [--cache-mib 9400]\n"
                    "       [--leave-gib G] [--max-tokens N] [--csv OUT.csv]   (see the header of replay_probe.cu)\n");
        return 1;
    }

    // ---- model layout
    std::vector<gguf_shard> shards; std::vector<gguf_tensor> tensors; std::map<std::string, uint64_t> kv;
    for (size_t i = 0; i < gguf_paths.size(); ++i) read_gguf(gguf_paths[i], (int) i, shards, tensors, kv);
    uint64_t n_used = 0;
    const std::string used_key = ".expert_used_count";
    for (const auto & [k, v] : kv) {
        if (k.size() > used_key.size() && k.compare(k.size() - used_key.size(), used_key.size(), used_key) == 0) n_used = v;
    }
    std::vector<layer_layout> L = build_layout(shards, tensors);
    double total_expert_bytes = 0; int n_layers_moe = 0;
    for (const auto & l : L) if (!l.t.empty()) { total_expert_bytes += (double) l.expert_bytes*l.n_expert; n_layers_moe++; }
    if (n_layers_moe == 0) die("no expert tensors found%s", "");
    std::printf("model: %zu shard(s), %d MoE layers, %.2f GiB of routed experts, n_expert_used %llu\n",
        shards.size(), n_layers_moe, total_expert_bytes/1073741824.0, (unsigned long long) n_used);

    // ---- trace and steps
    std::vector<sline> S = read_trace(trace_path.c_str());
    if (S.empty()) die("no S lines in %s", trace_path.c_str());
    if (n_used == 0) { n_used = 1u << 30; for (const auto & s : S) n_used = std::min<uint64_t>(n_used, s.n_ids); }
    long long mismatched = 0;
    for (const auto & s : S) {
        if (s.layer < 0 || s.layer >= (int) L.size() || L[s.layer].t.empty()) die("trace layer %s%lld has no expert tensors", "", s.layer);
        if (s.fills > 0 && (unsigned long long) s.bytes != (unsigned long long) s.fills*L[s.layer].expert_bytes) mismatched++;
    }
    if (mismatched) die("%s%lld S lines disagree with the GGUF expert sizes: wrong model for this trace", "", mismatched);

    std::vector<request> reqs;
    if (!requests_path.empty()) reqs = read_requests(requests_path.c_str());
    auto phase_of = [&](long long t) -> std::string {
        for (const auto & r : reqs) if (t >= r.t0 && t <= r.t1) return r.phase;
        return "";
    };
    auto wanted = [&](const std::string & ph) { return ("," + phases + ",").find("," + ph + ",") != std::string::npos; };

    std::vector<step> steps;
    {
        int prev = 1 << 30;
        for (int i = 0; i < (int) S.size(); ++i) {
            if (S[i].layer <= prev) steps.emplace_back();
            steps.back().lines.push_back(i);
            prev = S[i].layer;
        }
        std::vector<step> kept;
        for (auto & st : steps) {
            st.decode = true;
            for (int i : st.lines) if ((uint64_t) S[i].n_ids > n_used) st.decode = false;
            if (!reqs.empty()) {
                st.phase = phase_of(S[st.lines[0]].t_us);
                if (st.phase.empty() || !wanted(st.phase)) continue;
            } else {
                st.phase = "all";
            }
            kept.push_back(st);
        }
        steps.swap(kept);
        if (max_tokens >= 0) {
            std::vector<step> lim; long long nd = 0;
            for (auto & st : steps) { if (st.decode && nd >= max_tokens) break; nd += st.decode; lim.push_back(st); }
            steps.swap(lim);
        }
    }
    long long n_dec = 0; for (const auto & st : steps) n_dec += st.decode;
    std::printf("trace: %zu S lines, %zu steps replayed (%lld decode tokens, %lld prompt steps), phases %s\n",
        S.size(), steps.size(), n_dec, (long long) steps.size() - n_dec, reqs.empty() ? "(all, no --requests)" : phases.c_str());

    // ---- expert ids per S line: real (--fills) or synthetic
    std::vector<std::vector<int>> ids(S.size());
    bool synthetic = fills_path.empty();
    std::vector<std::vector<int>> perm(L.size());
    std::vector<int> n_cached(L.size(), 0);
    const double cache_frac = std::min(1.0, cache_mib*1048576.0/total_expert_bytes);
    if (!synthetic) {
        auto F = read_fills(fills_path.c_str());
        long long miss = 0;
        for (const auto & st : steps) for (int i : st.lines) {
            auto it = F.find(std::to_string(S[i].t_us) + ":" + std::to_string(S[i].layer));
            if (it == F.end() || (int) it->second.size() != S[i].fills) { miss++; continue; }
            ids[i] = it->second;
        }
        if (miss) std::printf("fills: %lld S lines without matching F lines get synthetic ids\n", miss);
    }
    {
        std::mt19937_64 rng(seed);
        for (size_t l = 0; l < L.size(); ++l) {
            if (L[l].t.empty()) continue;
            perm[l].resize(L[l].n_expert);
            for (int e = 0; e < L[l].n_expert; ++e) perm[l][e] = e;
            std::shuffle(perm[l].begin(), perm[l].end(), rng);
            n_cached[l] = std::min(L[l].n_expert - 1, (int) (cache_frac*L[l].n_expert));
        }
        // misses from the ranks below the VRAM-resident top, by popularity r^-zipf
        std::vector<std::vector<double>> cdf(L.size());
        for (size_t l = 0; l < L.size(); ++l) {
            if (L[l].t.empty()) continue;
            double acc = 0;
            for (int r = n_cached[l] + 1; r <= L[l].n_expert; ++r) { acc += std::pow((double) r, -zipf); cdf[l].push_back(acc); }
        }
        std::uniform_real_distribution<double> U(0.0, 1.0);
        long long n_syn = 0;
        for (const auto & st : steps) for (int i : st.lines) {
            if (!ids[i].empty() || S[i].fills == 0) continue;
            const int l = S[i].layer;
            const int avail = (int) cdf[l].size();
            const int k = std::min(S[i].fills, avail);
            std::unordered_set<int> got;
            int guard = 0;
            while ((int) got.size() < k && guard++ < 100000) {
                const double x = U(rng)*cdf[l].back();
                const int j = (int) (std::lower_bound(cdf[l].begin(), cdf[l].end(), x) - cdf[l].begin());
                got.insert(perm[l][n_cached[l] + std::min(j, avail - 1)]);
            }
            ids[i].assign(got.begin(), got.end());
            std::sort(ids[i].begin(), ids[i].end());
            n_syn++;
        }
        if (n_syn) synthetic = true;
        if (n_syn) std::printf("ids: %lld S lines with SYNTHETIC expert ids (zipf %.2f, VRAM-resident share %.1f%% of each layer)\n", n_syn, zipf, 100*cache_frac);
    }

    // ---- train / test split (real ids only)
    size_t first_test = 0;
    if (!fills_path.empty() && train_frac > 0) {
        first_test = (size_t) (train_frac*steps.size());
        for (size_t s = 0; s < steps.size(); ++s) steps[s].test = s >= first_test;
        std::printf("split: tier ranked on the first %zu steps, statistics over the remaining %zu\n", first_test, steps.size() - first_test);
    }

    // ---- tier contents: experts by expected misses per byte, greedy within --tier-mib
    std::unordered_set<long long> in_tier;
    {
        std::vector<std::pair<double, long long>> val; // (misses per byte, key = layer*65536 + expert)
        if (fills_path.empty()) {
            std::vector<double> fills_per_tok(L.size(), 0.0);
            for (const auto & st : steps) if (st.decode) for (int i : st.lines) fills_per_tok[S[i].layer] += S[i].fills;
            for (size_t l = 0; l < L.size(); ++l) {
                if (L[l].t.empty()) continue;
                double z = 0; for (int r = n_cached[l] + 1; r <= L[l].n_expert; ++r) z += std::pow((double) r, -zipf);
                for (int r = n_cached[l] + 1; r <= L[l].n_expert; ++r) {
                    const double v = fills_per_tok[l]*std::pow((double) r, -zipf)/z/(double) L[l].expert_bytes;
                    val.push_back({ v, (long long) l*65536 + perm[l][r - 1] });
                }
            }
        } else {
            std::unordered_map<long long, double> cnt;
            for (size_t s = 0; s < first_test; ++s) for (int i : steps[s].lines) for (int e : ids[i]) cnt[(long long) S[i].layer*65536 + e] += 1;
            for (const auto & [k, c] : cnt) val.push_back({ c/(double) L[k/65536].expert_bytes, k });
        }
        std::sort(val.begin(), val.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        double used = 0;
        for (const auto & [v, k] : val) {
            const double b = (double) L[k/65536].expert_bytes;
            if (used + b > tier_mib*1048576.0) continue;
            used += b; in_tier.insert(k);
        }
        std::printf("tier: %zu experts, %.0f MiB of %.0f MiB (%s ranking)\n", in_tier.size(), used/1048576.0, tier_mib,
            fills_path.empty() ? "synthetic oracle" : "training-window counts");
    }

    if (dry_run) {
        // everything above touches only the GGUF headers and the trace: no CUDA call, no GPU context
        std::vector<double> tok_b;
        for (const auto & st : steps) if (st.decode && st.test) { double b = 0; for (int i : st.lines) b += (double) S[i].bytes; tok_b.push_back(b); }
        std::map<uint64_t, int> sizes; for (const auto & l : L) if (!l.t.empty()) sizes[l.expert_bytes]++;
        for (const auto & [b, n] : sizes) std::printf("layout: %d layers with %llu bytes per expert\n", n, (unsigned long long) b);
        std::printf("dry run: %zu decode tokens in the statistics, MiB uploaded per token median %.1f p10 %.1f p90 %.1f; no CUDA call made\n",
            tok_b.size(), q(tok_b, .5)/1048576.0, q(tok_b, .1)/1048576.0, q(tok_b, .9)/1048576.0);
        return 0;
    }

    // ---- device and streams
    CK(cudaSetDevice(0));
    cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint64_t max_step_bytes = 0;
    for (const auto & st : steps) for (int i : st.lines) max_step_bytes = std::max<uint64_t>(max_step_bytes, (uint64_t) S[i].bytes);
    const size_t dst_size = (size_t) std::min<uint64_t>(std::max<uint64_t>(max_step_bytes, 64ull << 20), 1ull << 30);
    uint8_t * dst; CK(cudaMalloc(&dst, dst_size + (4u << 20)));
    size_t dst_pos = 0;
    auto dst_at = [&](uint64_t n) { if (dst_pos + n > dst_size) dst_pos = 0; uint8_t * p = dst + dst_pos; dst_pos += n; return p; };

    // ---- sources
    std::vector<const uint8_t *> map_base(shards.size(), nullptr);
    reader rd;
    pinned_pool ring_pool, tier_pool;
    uint8_t * ring = nullptr; size_t ring_slot = 0, ring_slots = 0;
    std::unordered_map<long long, uint8_t *> tier_at;

    const bool want_mmap = sources.find("mmap") != std::string::npos;
    const bool want_ring = sources.find("ring") != std::string::npos;
    const bool want_tier = sources.find("tier") != std::string::npos;
    if (want_mmap) {
        for (size_t f = 0; f < shards.size(); ++f) {
            HANDLE h = CreateFileA(shards[f].path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE) die("cannot open %s", shards[f].path.c_str());
            HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
            map_base[f] = m ? (const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
            if (!map_base[f]) die("cannot map %s", shards[f].path.c_str());
        }
    }
    if (want_ring || want_tier) {
        rd.open(shards, qd);
        ring_pool.align = tier_pool.align = std::max<uint64_t>(4096, rd.sector);
        for (const auto & l : L) { size_t b = 0; for (const auto & t : l.t) b += rd.region(t.nb2); ring_slot = std::max(ring_slot, b); }
        ring_slots = std::max<size_t>(1, (size_t) (ring_mib*1048576.0/ring_slot));
        ring_pool.chunk = ring_slots*ring_slot;
        ring = ring_pool.alloc(ring_slots*ring_slot);
        if (!ring) die("cannot pin the %s staging ring", "");
        std::printf("ring: %zu slots of %zu bytes pinned, sector %llu, queue depth %d, %s reads\n", ring_slots, ring_slot,
            (unsigned long long) rd.sector, qd, rd.n_fallback ? "SOME BUFFERED (fallback)" : "unbuffered");
    }
    if (want_tier) {
        const double t0 = now_us();
        rd.stream = nullptr;
        size_t n_fail = 0;
        for (long long k : in_tier) {
            const int l = (int) (k/65536), e = (int) (k % 65536);
            size_t b = 0; for (const auto & t : L[l].t) b += rd.region(t.nb2);
            uint8_t * p = n_fail ? nullptr : tier_pool.alloc(b);
            if (!p) { n_fail++; continue; }
            tier_at[k] = p;
            size_t sub = 0;
            for (const auto & t : L[l].t) { rd.submit(t.file, t.off + (uint64_t) e*t.nb2, t.nb2, p + sub, nullptr); sub += rd.region(t.nb2); }
        }
        rd.drain();
        const double dt = now_us() - t0;
        std::printf("tier fill: %zu experts, %.0f MiB pinned, read %.2f GiB in %.1f s (%.2f GB/s incl. pinning)%s\n", tier_at.size(),
            tier_pool.total/1048576.0, rd.bytes_read/1073741824.0, dt/1e6, rd.bytes_read/dt/1e3, n_fail ? ", PINNING FAILED for some" : "");
        if (n_fail) std::printf("tier: %zu experts could not be pinned and are left out\n", n_fail);
        rd.bytes_read = 0; rd.n_reads = 0;
    }

    std::vector<void *> held;
    if (leave_gib > 0) hold_memory(leave_gib, held);

    FILE * csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
    if (csv) std::fprintf(csv, "source,step,phase,decode,test,t_us_trace,fills,bytes,us,tier_fills,tier_bytes,read_bytes\n");

    // ---- one pass per source
    std::map<std::string, pass_stats> results;
    std::vector<std::string> order;
    for (const char * s : { "mmap", "ring", "tier" }) if (sources.find(s) != std::string::npos) order.push_back(s);
    CK(cudaMemcpyAsync(dst, dst + 64, 64, cudaMemcpyDeviceToDevice, stream)); CK(cudaStreamSynchronize(stream)); // warm the driver

    for (const std::string & src : order) {
        pass_stats ps;
        PROCESS_MEMORY_COUNTERS pmc0 = { sizeof(pmc0) }; K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc0, sizeof(pmc0));
        std::printf("\npass %s: %.2f GiB available at start\n", src.c_str(), avail_gib());
        const double pass_t0 = now_us();
        rd.stream = stream; rd.bytes_read = 0;
        for (size_t si = 0; si < steps.size(); ++si) {
            const step & st = steps[si];
            double step_us = 0, step_bytes = 0;
            for (int i : st.lines) {
                const sline & s = S[i];
                if (s.fills == 0) continue;
                const layer_layout & ll = L[s.layer];
                const double t0 = now_us();
                const double rb0 = rd.bytes_read;
                double tf = 0, tb = 0;
                size_t ring_next = 0;
                const std::vector<int> & E = ids[i];
                if (src == "mmap") {
                    // consecutive experts are contiguous in each tensor and are uploaded together, as upstream does
                    for (size_t a = 0; a < E.size();) {
                        size_t n = 1; while (a + n < E.size() && E[a + n] == E[a] + (int) n) n++;
                        for (const auto & t : ll.t) {
                            CK(cudaMemcpyAsync(dst_at(n*t.nb2), map_base[t.file] + t.off + (uint64_t) E[a]*t.nb2, n*t.nb2, cudaMemcpyHostToDevice, stream));
                        }
                        a += n;
                    }
                } else {
                    for (int e : E) {
                        const long long k = (long long) s.layer*65536 + e;
                        auto it = src == "tier" ? tier_at.find(k) : tier_at.end();
                        if (it != tier_at.end()) {
                            size_t sub = 0;
                            for (const auto & t : ll.t) {
                                const uint64_t off = t.off + (uint64_t) e*t.nb2;
                                CK(cudaMemcpyAsync(dst_at(t.nb2), it->second + sub + (off % rd.sector), t.nb2, cudaMemcpyHostToDevice, stream));
                                sub += rd.region(t.nb2);
                            }
                            tf += 1; tb += (double) ll.expert_bytes;
                            continue;
                        }
                        if (ring_next == ring_slots) { rd.drain(); CK(cudaStreamSynchronize(stream)); ring_next = 0; }
                        uint8_t * slot = ring + ring_next++*ring_slot;
                        size_t sub = 0;
                        for (const auto & t : ll.t) {
                            rd.submit(t.file, t.off + (uint64_t) e*t.nb2, t.nb2, slot + sub, dst_at(t.nb2));
                            sub += rd.region(t.nb2);
                        }
                    }
                    rd.drain();
                }
                CK(cudaStreamSynchronize(stream));
                const double dt = now_us() - t0;
                step_us += dt; step_bytes += (double) s.bytes;
                if (st.test) {
                    ps.tier_fills += tf; ps.tier_bytes += tb; ps.all_fills += s.fills; ps.all_bytes += (double) s.bytes; ps.read_bytes += rd.bytes_read - rb0;
                    if (st.decode) { ps.tier_bytes_dec += tb; ps.all_bytes_dec += (double) s.bytes; }
                }
                if (csv) std::fprintf(csv, "%s,%zu,%s,%d,%d,%lld,%d,%lld,%.1f,%.0f,%.0f,%.0f\n", src.c_str(), si, st.phase.c_str(), (int) st.decode, (int) st.test,
                    s.t_us, s.fills, s.bytes, dt, tf, tb, rd.bytes_read - rb0);
            }
            if (st.test) {
                if (st.decode) { ps.tok_us.push_back(step_us); ps.tok_bytes.push_back(step_bytes); }
                else { ps.prompt_us += step_us; ps.prompt_bytes += step_bytes; }
            }
            if ((now_us() - pass_t0)/1e6 > timeout_s) { ps.truncated = true; std::printf("pass %s: timeout after step %zu of %zu\n", src.c_str(), si + 1, steps.size()); break; }
        }
        PROCESS_MEMORY_COUNTERS pmc1 = { sizeof(pmc1) }; K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc1, sizeof(pmc1));
        std::printf("pass %s: %.1f s, %lu page faults (soft and hard), %.2f GiB available at end\n", src.c_str(), (now_us() - pass_t0)/1e6,
            (unsigned long) (pmc1.PageFaultCount - pmc0.PageFaultCount), avail_gib());
        results[src] = ps;
    }
    if (csv) std::fclose(csv);

    // ---- summary
    std::printf("\n%s ids; per decode token = the sum over its layer steps of host time from the first upload to the stream sync\n",
        synthetic ? "SYNTHETIC" : "REAL");
    std::printf("| source | tokens | MiB/token | ms/token median | p10 | p90 | GB/s (median bytes / median time) | tier share fills / bytes | read GiB | prompt steps: GiB, s, GB/s |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
    for (const std::string & src : order) {
        const pass_stats & p = results[src];
        const double mb = q(p.tok_bytes, .5)/1048576.0, ms = q(p.tok_us, .5)/1e3;
        std::printf("| %s%s | %zu | %.1f | %.2f | %.2f | %.2f | %.2f | %s | %.2f | %.2f, %.1f, %.2f |\n", src.c_str(), p.truncated ? " (truncated)" : "",
            p.tok_us.size(), mb, ms, q(p.tok_us, .1)/1e3, q(p.tok_us, .9)/1e3, q(p.tok_bytes, .5)/(q(p.tok_us, .5)*1e3),
            src == "tier" ? (std::to_string((int) (100*p.tier_fills/std::max(1.0, p.all_fills))) + "% / " + std::to_string((int) (100*p.tier_bytes/std::max(1.0, p.all_bytes))) + "%").c_str() : "-",
            p.read_bytes/1073741824.0, p.prompt_bytes/1073741824.0, p.prompt_us/1e6, p.prompt_us > 0 ? p.prompt_bytes/p.prompt_us/1e3 : 0.0);
    }
    // what-if: a tier serving a share h of the miss bytes, from the measured per-byte times of the ring and tier passes
    if (results.count("ring") && results.count("tier")) {
        const pass_stats & r = results["ring"]; const pass_stats & t = results["tier"];
        double r_us = 0, r_b = 0, t_us = 0, t_b = 0;
        for (size_t i = 0; i < r.tok_us.size(); ++i) { r_us += r.tok_us[i]; r_b += r.tok_bytes[i]; }
        for (size_t i = 0; i < t.tok_us.size(); ++i) { t_us += t.tok_us[i]; t_b += t.tok_bytes[i]; }
        const double h = t.tier_bytes_dec/std::max(1.0, t.all_bytes_dec);
        const double us_per_b_ring = r_us/std::max(1.0, r_b);
        // the tier pass mixes tier and ring bytes: solve its per-byte time for the tier share
        const double us_per_b_tier = h > 0.01 ? std::max(0.0, (t_us/std::max(1.0, t_b) - (1 - h)*us_per_b_ring)/h) : NAN;
        const double mib_tok = q(r.tok_bytes, .5)/1048576.0;
        std::printf("\nlinear model from the ring and tier passes: ring %.2f GB/s, tier-resident %.2f GB/s per miss byte; at %.1f MiB/token:\n",
            1e-3/us_per_b_ring, std::isnan(us_per_b_tier) ? NAN : 1e-3/std::max(1e-9, us_per_b_tier), mib_tok);
        std::printf("| tier share of miss bytes | 0%% | 25%% | 50%% | 75%% | 90%% | 100%% |\n|---|---|---|---|---|---|---|\n| ms per token |");
        for (double hh : { 0.0, 0.25, 0.5, 0.75, 0.9, 1.0 }) {
            const double us = mib_tok*1048576.0*((1 - hh)*us_per_b_ring + hh*(std::isnan(us_per_b_tier) ? us_per_b_ring : us_per_b_tier));
            std::printf(" %.1f |", us/1e3);
        }
        std::printf("\n");
    }
    std::printf("DONE\n");
    for (void * p : held) VirtualFree(p, 0, MEM_RELEASE);
    cudaFree(dst);
    return 0;
}
