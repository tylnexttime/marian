// Copyright 2026 tylnexttime (Chris Tyl and Claude Tyl) and contributors
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (see LICENSE at the repository root).
//
// wsprobe_target: the process wsprobe's self-test measures. It maps each file named on the command line the way
// llama.cpp does on Windows (CreateFileMapping of the whole file, one read-only view at offset 0), prints
// "ready <pid>", then obeys one command per line on stdin and answers each with one "ok ..." line:
//   touch I A B   read one byte of every page of file I's bytes [A, B)   -> ok touch <pages> <sum of bytes read>
//   touchrand I A B N SEED   read one byte of N pseudo-random pages of [A, B) -> ok touchrand N <sum>
//   trim          empty this process's working set                       -> ok trim
//   unmap I       unmap file I's view                                     -> ok unmap
//   quit
// It touches nothing else in the views, so after a "touch" exactly the touched pages should be resident.

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 8

int main(int argc, char ** argv) {
    if (argc < 2 || argc - 1 > MAX_FILES) {
        fprintf(stderr, "usage: wsprobe_target FILE [FILE ...] (at most %d)\n", MAX_FILES);
        return 2;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t page = si.dwPageSize;
    int nf = argc - 1;
    volatile const uint8_t * base[MAX_FILES];
    uint64_t size[MAX_FILES];
    for (int i = 0; i < nf; i++) {
        HANDLE f = CreateFileA(argv[i + 1], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (f == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s: %lu\n", argv[i + 1], GetLastError()); return 3; }
        LARGE_INTEGER sz;
        GetFileSizeEx(f, &sz);
        size[i] = (uint64_t) sz.QuadPart;
        HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!m) { fprintf(stderr, "CreateFileMapping %s: %lu\n", argv[i + 1], GetLastError()); return 4; }
        base[i] = (volatile const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base[i]) { fprintf(stderr, "MapViewOfFile %s: %lu\n", argv[i + 1], GetLastError()); return 5; }
        CloseHandle(m);   // the view keeps the section alive
        CloseHandle(f);
    }
    printf("ready %lu\n", (unsigned long) GetCurrentProcessId());
    fflush(stdout);

    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        int i;
        unsigned long long a, b, cnt, seed;
        if (sscanf(line, "touch %d %llu %llu", &i, &a, &b) == 3) {
            if (i < 0 || i >= nf || !base[i] || b > size[i] || a > b) { printf("err touch\n"); fflush(stdout); continue; }
            uint64_t sum = 0, n = 0;
            for (uint64_t p = a / page * page; p < b; p += page) { sum += base[i][p]; n++; }
            printf("ok touch %llu %llu\n", (unsigned long long) n, (unsigned long long) sum);
        } else if (sscanf(line, "touchrand %d %llu %llu %llu %llu", &i, &a, &b, &cnt, &seed) == 5) {
            // cnt pseudo-random pages of [a, b) (repeats allowed), a fixed LCG so a run is reproducible
            if (i < 0 || i >= nf || !base[i] || b > size[i] || a >= b) { printf("err touchrand\n"); fflush(stdout); continue; }
            uint64_t first = a / page, n_pages = (b - 1) / page - first + 1, x = seed * 2862933555777941757ULL + 3037000493ULL, sum = 0;
            for (unsigned long long k = 0; k < cnt; k++) {
                x = x * 6364136223846793005ULL + 1442695040888963407ULL;
                sum += base[i][(first + (x >> 17) % n_pages) * page];
            }
            printf("ok touchrand %llu %llu\n", cnt, (unsigned long long) sum);
        } else if (!strncmp(line, "trim", 4)) {
            SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);
            printf("ok trim\n");
        } else if (sscanf(line, "unmap %d", &i) == 1) {
            if (i < 0 || i >= nf || !base[i]) { printf("err unmap\n"); fflush(stdout); continue; }
            UnmapViewOfFile((LPCVOID) base[i]);
            base[i] = NULL;
            printf("ok unmap\n");
        } else if (!strncmp(line, "quit", 4)) {
            break;
        } else {
            printf("err unknown command\n");
        }
        fflush(stdout);
    }
    return 0;
}
