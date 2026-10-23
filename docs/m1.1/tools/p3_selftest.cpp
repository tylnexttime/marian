// Marian M1.1: CPU-only self-test of src/llama-moe-source.cpp (P3). No GPU: the "device" is the ggml CPU backend, so
// every upload is a memcpy into a CPU bank. Drives random decode and prefill-like steps through the ring and a small
// tier, then checks every filled bank slot byte for byte against the memory-mapped GGUF.
// Usage: p3_selftest <shard> [<shard> ...] [--steps N] [--tier-mib M] [--ring-mib R] [--slots S] [--seed X]
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "llama-moe-source.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

// the two libllama helpers the source uses
void llama_log_internal(ggml_log_level level, const char * fmt, ...) {
    (void) level;
    va_list ap; va_start(ap, fmt); std::vfprintf(stderr, fmt, ap); va_end(ap);
}
std::string format(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    char buf[4096]; std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

struct host_t { ggml_tensor * t; int file; uint64_t off; };

int main(int argc, char ** argv) {
    std::vector<std::string> shards;
    int steps = 300, n_slots = 64; double tier_mib = 64, ring_mib = 8; unsigned seed = 7;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--steps") steps = std::atoi(argv[++i]);
        else if (a == "--tier-mib") tier_mib = std::atof(argv[++i]);
        else if (a == "--ring-mib") ring_mib = std::atof(argv[++i]);
        else if (a == "--slots") n_slots = std::atoi(argv[++i]);
        else if (a == "--seed") seed = (unsigned) std::atoi(argv[++i]);
        else shards.push_back(a);
    }
    if (shards.empty()) { std::printf("usage: p3_selftest <shard> ...\n"); return 1; }

    // metadata and memory maps
    std::vector<ggml_context *> metas;
    std::vector<const uint8_t *> bases;
    std::map<int, std::vector<host_t>> layer_tensors; // layer -> gate_up / gate / up / down, in that order
    static const char * roles[] = { "ffn_gate_up_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (size_t f = 0; f < shards.size(); ++f) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { true, &meta };
        gguf_context * g = gguf_init_from_file(shards[f].c_str(), gp);
        if (!g) { std::printf("cannot read %s\n", shards[f].c_str()); return 1; }
        metas.push_back(meta);
        HANDLE h = CreateFileA(shards[f].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        const uint8_t * base = (const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base) { std::printf("cannot map %s\n", shards[f].c_str()); return 1; }
        bases.push_back(base);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            int il = -1; char rest[128] = { 0 };
            if (std::sscanf(name, "blk.%d.%127s", &il, rest) != 2) continue;
            for (int r = 0; r < 4; ++r) {
                if (std::string(rest) == std::string(roles[r]) + ".weight") {
                    ggml_tensor * t = ggml_get_tensor(meta, name);
                    const uint64_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
                    t->data = (void *) (base + off);
                    auto & v = layer_tensors[il];
                    v.push_back({ t, (int) f, off });
                }
            }
        }
    }
    // order each layer's tensors by role
    for (auto & [il, v] : layer_tensors) {
        std::sort(v.begin(), v.end(), [](const host_t & a, const host_t & b) {
            auto rank = [](const ggml_tensor * t) { for (int r = 0; r < 4; ++r) if (std::strstr(t->name, roles[r]) && std::strstr(t->name, roles[r])[std::strlen(roles[r])] == '.') return r; return 9; };
            return rank(a.t) < rank(b.t);
        });
    }

    // groups by layout (types and strides), banks in a CPU buffer
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const int n_layer = layer_tensors.rbegin()->first + 1;
    std::vector<int32_t> group_of(n_layer, -1);
    std::vector<std::string> keys;
    std::vector<size_t> group_bytes;
    std::vector<std::vector<ggml_tensor *>> group_banks;
    ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_banks = ggml_init(ip);
    int32_t n_expert = 0;
    for (auto & [il, v] : layer_tensors) {
        std::string key;
        for (auto & h : v) key += std::to_string(h.t->type) + ":" + std::to_string(h.t->nb[2]) + ";";
        auto it = std::find(keys.begin(), keys.end(), key);
        if (it == keys.end()) {
            keys.push_back(key); group_bytes.push_back(0); group_banks.emplace_back();
            for (auto & h : v) group_banks.back().push_back(ggml_new_tensor_3d(ctx_banks, h.t->type, h.t->ne[0], h.t->ne[1], n_slots + 1));
            it = keys.end() - 1;
        }
        const int g = (int) (it - keys.begin());
        group_of[il] = g;
        for (auto & h : v) group_bytes[g] += ggml_nbytes(h.t);
        n_expert = std::max<int32_t>(n_expert, (int32_t) v[0].t->ne[2]);
    }
    ggml_backend_buffer_t bank_buf = ggml_backend_alloc_ctx_tensors(ctx_banks, cpu);
    if (!bank_buf) { std::printf("cannot allocate the banks\n"); return 1; }
    std::printf("%d layers in %zu groups, %d experts, banks %.0f MiB (%d slots per group)\n", (int) layer_tensors.size(), keys.size(), n_expert,
        ggml_backend_buffer_get_size(bank_buf)/1048576.0, n_slots);

    std::vector<std::vector<llama_moe_source_tensor>> layers(n_layer);
    for (auto & [il, v] : layer_tensors) {
        for (size_t k = 0; k < v.size(); ++k) {
            layers[il].push_back({ v[k].file, v[k].off, (uint64_t) v[k].t->nb[2], group_banks[group_of[il]][k], v[k].t });
        }
    }

    llama_moe_source_params sp;
    sp.type = LLAMA_MOE_MISS_SOURCE_TIER;
    sp.tier_size = (size_t) (tier_mib*1048576);
    sp.ring_size = (size_t) (ring_mib*1048576);
    llama_moe_source src(sp, ggml_backend_cpu_buffer_type(), shards, layers, group_of, group_bytes, n_expert);
    if (!src.error().empty()) { std::printf("source error: %s\n", src.error().c_str()); return 1; }

    // a few layers from every group, a hot set per layer so the tier gets hits
    std::vector<int> test_layers;
    for (size_t g = 0; g < keys.size(); ++g) { int n = 0; for (int il = 0; il < n_layer && n < 2; ++il) if (group_of[il] == (int) g) { test_layers.push_back(il); n++; } }
    std::mt19937 rng(seed);
    std::map<char, long> by_src; long mismatched = 0, checked = 0;
    for (int step = 0; step < steps; ++step) {
        const int il = test_layers[rng() % test_layers.size()];
        const bool decode = rng() % 4 != 0;
        const int k = decode ? 1 + (int) (rng() % 10) : 16 + (int) (rng() % std::max(1, n_slots - 16));
        std::set<int> es;
        while ((int) es.size() < std::min(k, n_expert)) es.insert(rng() % 10 < 7 ? (int) (rng() % 24) : (int) (rng() % n_expert));
        std::vector<int> E(es.begin(), es.end());
        std::vector<int> S(n_slots); for (int s = 0; s < n_slots; ++s) S[s] = s;
        std::shuffle(S.begin(), S.end(), rng);
        src.begin(il, decode, E.data(), E.size());
        for (size_t i = 0; i < E.size(); ++i) by_src[src.fill(cpu, E[i], S[i])]++;
        src.end(cpu);
        ggml_backend_synchronize(cpu);
        for (size_t i = 0; i < E.size(); ++i) {
            for (const auto & t : layers[il]) {
                checked++;
                if (std::memcmp((const uint8_t *) t.bank->data + (size_t) S[i]*t.nb2, (const uint8_t *) t.host->data + (size_t) E[i]*t.nb2, t.nb2) != 0) mismatched++;
            }
        }
    }
    std::printf("fills: tier %ld, admitted %ld, ring %ld; %ld expert tensors checked against the memory map, %ld mismatched\n",
        by_src['t'], by_src['a'], by_src['r'], checked, mismatched);
    src.log_stats();
    std::printf(mismatched == 0 ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
    return mismatched == 0 ? 0 : 1;
}
