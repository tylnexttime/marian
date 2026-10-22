// Marian M1.0 copy probe: host-to-device upload rate for expert-sized blocks from (1) the mmap'd GGUF, first touch,
// (2) the same mapped blocks again (resident pageable), (3) a malloc'd pageable copy, (4) a cudaMallocHost pinned copy;
// plus the round trip of a stream synchronise. Usage: copy_probe <gguf shard> [n_blocks=400] [block_bytes=1510400]
#include <cuda_runtime.h>
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static void report(const char * name, std::vector<double> us, size_t bytes) {
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2], p90 = us[us.size() * 9 / 10], p10 = us[us.size() / 10];
    double sum = 0; for (double u : us) sum += u;
    std::printf("%-28s median %8.1f us (%6.2f GB/s)  p10 %8.1f  p90 %8.1f us  mean %6.2f GB/s\n", name, med,
        bytes / med / 1e3, p10, p90, bytes * us.size() / sum / 1e3);
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: copy_probe <file> [n] [bytes]\n"); return 1; }
    const int    n     = argc > 2 ? std::atoi(argv[2]) : 400;
    const size_t bytes = argc > 3 ? (size_t) std::atoll(argv[3]) : 1510400;

    HANDLE f = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { std::printf("cannot open %s\n", argv[1]); return 1; }
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const uint8_t * base = (const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) { std::printf("map failed\n"); return 1; }

    std::mt19937_64 rng(42);
    const size_t n_blocks_file = (size_t) (sz.QuadPart / bytes) - 1;
    std::vector<size_t> off(n);
    for (auto & o : off) o = (rng() % n_blocks_file) * bytes;

    void * dev; CK(cudaMalloc(&dev, bytes));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    // warm the driver
    std::vector<uint8_t> tmp(bytes); CK(cudaMemcpyAsync(dev, tmp.data(), bytes, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s));

    std::printf("file %s (%.1f GiB), %d blocks of %zu bytes\n", argv[1], sz.QuadPart / 1073741824.0, n, bytes);
    std::vector<double> t(n);
    auto run = [&](const char * name, auto src_of) {
        for (int i = 0; i < n; ++i) {
            const double a = now_us();
            CK(cudaMemcpyAsync(dev, src_of(i), bytes, cudaMemcpyHostToDevice, s));
            CK(cudaStreamSynchronize(s));
            t[i] = now_us() - a;
        }
        report(name, t, bytes);
    };
    run("mmap first touch", [&](int i) { return base + off[i]; });
    run("mmap again (resident)", [&](int i) { return base + off[i]; });

    std::vector<uint8_t *> pageable(n);
    for (int i = 0; i < n; ++i) { pageable[i] = (uint8_t *) std::malloc(bytes); std::memcpy(pageable[i], base + off[i], bytes); }
    run("malloc pageable", [&](int i) { return pageable[i]; });

    uint8_t * pinned; CK(cudaMallocHost(&pinned, bytes * (size_t) n));
    for (int i = 0; i < n; ++i) std::memcpy(pinned + (size_t) i * bytes, base + off[i], bytes);
    run("cudaMallocHost pinned", [&](int i) { return pinned + (size_t) i * bytes; });

    // host blocking time of an async pinned upload (submission only), and of a pageable one
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < n; ++i) {
            const double a = now_us();
            CK(cudaMemcpyAsync(dev, pass == 0 ? (void *) (pinned + (size_t) i * bytes) : (void *) pageable[i], bytes, cudaMemcpyHostToDevice, s));
            t[i] = now_us() - a;
            CK(cudaStreamSynchronize(s));
        }
        report(pass == 0 ? "submit only, pinned" : "submit only, pageable", t, bytes);
    }

    // round trip of a synchronise on an idle stream, and after a tiny copy
    for (int i = 0; i < n; ++i) { const double a = now_us(); CK(cudaStreamSynchronize(s)); t[i] = now_us() - a; }
    report("sync, idle stream", t, 1);
    int32_t ids[16] = {0}; int32_t * dids; CK(cudaMalloc(&dids, sizeof(ids)));
    for (int i = 0; i < n; ++i) {
        const double a = now_us();
        CK(cudaMemcpyAsync(ids, dids, sizeof(ids), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        t[i] = now_us() - a;
    }
    report("64 B D2H + sync (ids read)", t, 1);
    std::printf("DONE\n");
    return 0;
}
