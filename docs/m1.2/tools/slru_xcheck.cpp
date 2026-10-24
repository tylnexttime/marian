// Marian M1.2: cross-check of `marian replay`'s SLRU against the fork's own moe_cache_slru (CPU only, no ggml).
// slru_xcheck.sh extracts `struct moe_cache_slru { ... };` verbatim from src/llama-moe-cache.cpp into slru_extracted.inc
// and compiles this driver with it, so the code under test is the fork's, not a copy typed by hand.
//
// input (stdin):  n_layer n_expert n_slots, then one step per line: layer n_ids id id ...
// output:         one line per step: row_hits n_hit n_fills expert:slot ... (row_hits counted before the plan, like
//                 the P0 trace), or REFUSED when the step selects more distinct experts than there are slots

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GGML_ASSERT(x) do { if (!(x)) { std::fprintf(stderr, "assert: %s\n", #x); std::exit(2); } } while (0)
#define GGML_ABORT(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(3); } while (0)

#include "slru_extracted.inc"

int main() {
    int n_layer = 0, n_expert = 0, n_slots = 0;
    if (std::scanf("%d %d %d", &n_layer, &n_expert, &n_slots) != 3) {
        return 1;
    }
    moe_cache_slru slru;
    slru.init(n_layer, n_expert, n_slots);
    std::vector<std::vector<int32_t>> maps(n_layer, std::vector<int32_t>(n_expert, -1));
    for (int il = 0; il < n_layer; ++il) {
        slru.slot_map[il] = maps[il].data();
    }
    std::vector<moe_cache_slru::fill> fills;
    int il = 0, n = 0;
    while (std::scanf("%d %d", &il, &n) == 2) {
        std::vector<int32_t> ids(n);
        for (int i = 0; i < n; ++i) {
            if (std::scanf("%d", &ids[i]) != 1) {
                return 1;
            }
        }
        size_t row_hits = 0;
        for (int32_t e : ids) {
            row_hits += maps[il][e] >= 0;
        }
        size_t n_hit = 0;
        if (!slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            std::printf("REFUSED\n");
            continue;
        }
        std::printf("%zu %zu %zu", row_hits, n_hit, fills.size());
        for (const auto & f : fills) {
            std::printf(" %d:%d", f.expert, f.slot);
        }
        std::printf("\n");
    }
    return 0;
}
