// Marian M1.0 VRAM ballast: holds <mib> MiB of device memory (touched, so it is really resident) until <stop file>
// exists, to simulate a smaller GPU. Usage: ballast <mib> <stop file>
#include <cuda_runtime.h>
#include <windows.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 3) { std::printf("usage: ballast <mib> <stop file>\n"); return 1; }
    const size_t mib = (size_t) std::atoll(argv[1]);
    void * p = nullptr;
    if (cudaMalloc(&p, mib << 20) != cudaSuccess) { std::printf("cudaMalloc %zu MiB failed\n", mib); return 2; }
    cudaMemset(p, 1, mib << 20);
    cudaDeviceSynchronize();
    size_t free_b, total_b; cudaMemGetInfo(&free_b, &total_b);
    std::printf("ballast %zu MiB held; device free %zu MiB of %zu\n", mib, free_b >> 20, total_b >> 20);
    std::fflush(stdout);
    while (GetFileAttributesA(argv[2]) == INVALID_FILE_ATTRIBUTES) {
        Sleep(500);
    }
    cudaFree(p);
    std::printf("ballast released\n");
    return 0;
}
