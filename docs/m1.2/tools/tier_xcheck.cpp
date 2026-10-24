// Marian M1.2: cross-check of `marian replay`'s "current" RAM-tier policy against the fork's own llama_moe_source
// (src/llama-moe-source.cpp, patch 0003), CPU only. The ggml CPU backend plays the device, a temporary file of random
// bytes plays the GGUF, and every step's fills go through the real begin()/fill()/end(), so the tier decisions printed
// here ('t' tier hit, 'a' read and admitted, 'r' read through the ring) are the server's. Build: tier_xcheck.sh.
//
// input (stdin):
//   n_layer n_expert ring_bytes tier_bytes
//   n_layer lines: group nb2 nb2 ...          (group -1: not cached; nb2 of each expert tensor)
//   one step per line: layer decode n_uniq uniq... n_fills expert slot expert slot ...
// output: one line per step with its fill sources ('-' for none), then "stats ..." from the source

#include "llama-moe-source.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "/tmp/marian-tier-xcheck.bin";
    int n_layer = 0, n_expert = 0;
    long long ring_bytes = 0, tier_bytes = 0;
    if (std::scanf("%d %d %lld %lld", &n_layer, &n_expert, &ring_bytes, &tier_bytes) != 4) {
        return 1;
    }
    std::vector<int32_t> group_of(n_layer, -1);
    std::vector<std::vector<long long>> nb2(n_layer);
    int n_group = 0;
    for (int il = 0; il < n_layer; ++il) {
        int g = -1, n_t = 0;
        if (std::scanf("%d %d", &g, &n_t) != 2) {
            return 1;
        }
        group_of[il] = g;
        nb2[il].resize(n_t);
        for (int k = 0; k < n_t; ++k) {
            if (std::scanf("%lld", &nb2[il][k]) != 1) {
                return 1;
            }
        }
        n_group = std::max(n_group, g + 1);
    }

    // the "GGUF": every expert tensor of every layer, 4 KiB aligned, random bytes
    std::mt19937_64 rng(42);
    std::vector<uint8_t> file;
    std::vector<std::vector<uint64_t>> off(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        for (long long b : nb2[il]) {
            file.resize((file.size() + 4095)/4096*4096);
            off[il].push_back(file.size());
            const size_t n = (size_t) b*n_expert;
            const size_t p = file.size();
            file.resize(p + n);
            for (size_t i = 0; i < n; ++i) {
                file[p + i] = (uint8_t) rng();
            }
        }
    }
    file.resize(file.size() + 4096);
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(file.data(), 1, file.size(), f) != file.size()) {
        return 1;
    }
    std::fclose(f);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_init_params ip = { (size_t) 4*n_layer*ggml_tensor_overhead() + (1 << 16), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<std::vector<llama_moe_source_tensor>> layers(n_layer);
    std::vector<size_t> group_bytes(n_group, 0);
    std::vector<ggml_tensor *> banks;
    for (int il = 0; il < n_layer; ++il) {
        if (group_of[il] < 0) {
            continue;
        }
        for (size_t k = 0; k < nb2[il].size(); ++k) {
            ggml_tensor * host = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, nb2[il][k], 1, n_expert);
            host->data = file.data() + off[il][k];
            ggml_tensor * bank = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, nb2[il][k], 1, n_layer*n_expert + 1); // any VRAM slot
            banks.push_back(bank);
            layers[il].push_back({ 0, off[il][k], (uint64_t) nb2[il][k], bank, host });
            group_bytes[group_of[il]] += (size_t) nb2[il][k]*n_expert;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu); // the banks; host tensors keep their data

    llama_moe_source_params sp;
    sp.type      = LLAMA_MOE_MISS_SOURCE_TIER;
    sp.ring_size = (size_t) ring_bytes;
    sp.tier_size = (size_t) tier_bytes;
    llama_moe_source src(sp, ggml_backend_cpu_buffer_type(), { path }, layers, group_of, group_bytes, n_expert);
    if (!src.error().empty()) {
        std::fprintf(stderr, "source: %s\n", src.error().c_str());
        return 1;
    }

    int il = 0, dec = 0, n_uniq = 0;
    while (std::scanf("%d %d %d", &il, &dec, &n_uniq) == 3) {
        std::vector<int32_t> uniq(n_uniq);
        for (int i = 0; i < n_uniq; ++i) {
            if (std::scanf("%d", &uniq[i]) != 1) {
                return 1;
            }
        }
        int n_fill = 0;
        if (std::scanf("%d", &n_fill) != 1) {
            return 1;
        }
        std::string out;
        src.begin(il, dec != 0, uniq.data(), uniq.size());
        for (int i = 0; i < n_fill; ++i) {
            int e = 0, s = 0;
            if (std::scanf("%d %d", &e, &s) != 2) {
                return 1;
            }
            out += src.fill(cpu, e, s);
        }
        src.end(cpu);
        ggml_backend_synchronize(cpu); // the caller's next copy synchronizes before the next begin
        std::printf("%s\n", out.empty() ? "-" : out.c_str());
    }
    std::fflush(stdout);
    src.log_stats();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    std::remove(path.c_str());
    return 0;
}
