// Copyright 2026 tylnexttime (Chris Tyl and Claude Tyl) and contributors
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (see LICENSE at the repository root).
//
// wsprobe: which bytes of a process's memory-mapped GGUF shards are resident in its working set, by tensor kind,
// sampled over time, without administrator rights (Windows).
//
// Given a process (by PID, or by waiting for a new process with a given image name) and the tensor ranges of each
// shard (JSON from gguf_ranges.py), it finds the process's mapped views of the shards (VirtualQueryEx for MEM_MAPPED
// regions, GetMappedFileNameW to name them, matched by NT device path or, failing that, by a unique file name), and
// every --interval-ms asks QueryWorkingSetEx which pages of those views are valid in the process's working set.
// Each page is attributed to the tensor range holding its first byte. One JSON line per sample: resident MiB per kind
// (ple, experts, other, header) and per expert layer, available RAM, system cache size, and the process's page-fault
// count, working set and private bytes.
//
// What it cannot see: a page in the standby list (in RAM, but not in this process's working set) counts as not
// resident. Telling standby pages apart needs admin rights (the PFN database); the system-wide cache size
// (GetPerformanceInfo SystemCache = standby list + system working set) is reported beside each sample as context.
//
// Access needed: PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE on the target, which a process of the same
// user grants. Nothing is written to the target and none of its pages are touched.
//
// usage:
//   wsprobe (--pid N | --wait-image NAME_OR_PATH [--wait-timeout S]) --ranges R.json [--out S.jsonl]
//           [--interval-ms 500] [--stride 1] [--chunk 65536] [--rescan-ms 2000] [--max-seconds S]
//           [--once [--dump-runs RUNS.json]]
// --stride K queries every K-th page and counts each valid one as K pages (a fixed-stride sample, said in every line).
// --once takes one sample and exits; --dump-runs then also writes the resident runs [start, end) in file offsets.

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

enum { K_PLE, K_EXP, K_OTHER, K_HEADER, K_N };
static const char * KIND_NAME[K_N] = { "ple", "experts", "other", "header" };
#define MAX_LAYERS 256
#define MAX_VIEWS 64

static __declspec(noreturn) void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "wsprobe: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

// ---------------------------------------------------------------- minimal JSON reader (enough for the ranges file)

typedef struct JV JV;
struct JV {
    int t;              // 0 null, 1 bool, 2 number, 3 string, 4 array, 5 object
    int isint;
    int64_t i;
    double d;
    char * s;
    int n, cap;
    JV ** v;
    char ** k;
};
static const char * jp;

static void jws(void) { while (*jp == ' ' || *jp == '\n' || *jp == '\r' || *jp == '\t') jp++; }

static void jputc(char ** o, size_t * n, size_t * cap, unsigned c) {
    if (*n + 5 >= *cap) { *cap *= 2; *o = (char *) realloc(*o, *cap); if (!*o) die("out of memory"); }
    (*o)[(*n)++] = (char) c;
}

static void jput_utf8(char ** o, size_t * n, size_t * cap, unsigned cp) {
    if (cp < 0x80) { jputc(o, n, cap, cp); }
    else if (cp < 0x800) { jputc(o, n, cap, 0xC0 | (cp >> 6)); jputc(o, n, cap, 0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { jputc(o, n, cap, 0xE0 | (cp >> 12)); jputc(o, n, cap, 0x80 | ((cp >> 6) & 0x3F)); jputc(o, n, cap, 0x80 | (cp & 0x3F)); }
    else { jputc(o, n, cap, 0xF0 | (cp >> 18)); jputc(o, n, cap, 0x80 | ((cp >> 12) & 0x3F)); jputc(o, n, cap, 0x80 | ((cp >> 6) & 0x3F)); jputc(o, n, cap, 0x80 | (cp & 0x3F)); }
}

static unsigned jhex4(void) {
    unsigned cp = 0;
    for (int i = 0; i < 4; i++) {
        char h = *jp++;
        if (h >= '0' && h <= '9') cp = cp * 16 + (unsigned) (h - '0');
        else if ((h | 32) >= 'a' && (h | 32) <= 'f') cp = cp * 16 + (unsigned) ((h | 32) - 'a' + 10);
        else die("bad \\u escape in JSON");
    }
    return cp;
}

static char * jstring(void) {
    if (*jp != '"') die("JSON: expected a string");
    jp++;
    size_t cap = 32, n = 0;
    char * o = (char *) xmalloc(cap);
    while (*jp && *jp != '"') {
        char c = *jp++;
        if (c != '\\') { jputc(&o, &n, &cap, (unsigned char) c); continue; }
        c = *jp++;
        switch (c) {
            case 'n': jputc(&o, &n, &cap, '\n'); break;
            case 't': jputc(&o, &n, &cap, '\t'); break;
            case 'r': jputc(&o, &n, &cap, '\r'); break;
            case 'b': jputc(&o, &n, &cap, '\b'); break;
            case 'f': jputc(&o, &n, &cap, '\f'); break;
            case 'u': {
                unsigned cp = jhex4();
                if (cp >= 0xD800 && cp < 0xDC00 && jp[0] == '\\' && jp[1] == 'u') {
                    jp += 2;
                    unsigned lo = jhex4();
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                jput_utf8(&o, &n, &cap, cp);
                break;
            }
            case 0: die("JSON: unterminated string");
            default: jputc(&o, &n, &cap, (unsigned char) c);
        }
    }
    if (*jp != '"') die("JSON: unterminated string");
    jp++;
    o[n] = 0;
    return o;
}

static JV * jnew(int t) { JV * v = (JV *) calloc(1, sizeof(JV)); if (!v) die("out of memory"); v->t = t; return v; }

static void jpush(JV * a, char * key, JV * item) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 8;
        a->v = (JV **) realloc(a->v, sizeof(JV *) * a->cap);
        a->k = (char **) realloc(a->k, sizeof(char *) * a->cap);
        if (!a->v || !a->k) die("out of memory");
    }
    a->k[a->n] = key;
    a->v[a->n++] = item;
}

static JV * jvalue(void) {
    jws();
    if (*jp == '{') {
        JV * o = jnew(5);
        jp++; jws();
        if (*jp == '}') { jp++; return o; }
        for (;;) {
            jws();
            char * key = jstring();
            jws();
            if (*jp++ != ':') die("JSON: expected ':'");
            jpush(o, key, jvalue());
            jws();
            if (*jp == ',') { jp++; continue; }
            if (*jp == '}') { jp++; return o; }
            die("JSON: expected ',' or '}'");
        }
    }
    if (*jp == '[') {
        JV * a = jnew(4);
        jp++; jws();
        if (*jp == ']') { jp++; return a; }
        for (;;) {
            jpush(a, NULL, jvalue());
            jws();
            if (*jp == ',') { jp++; continue; }
            if (*jp == ']') { jp++; return a; }
            die("JSON: expected ',' or ']'");
        }
    }
    if (*jp == '"') { JV * s = jnew(3); s->s = jstring(); return s; }
    if (!strncmp(jp, "true", 4)) { jp += 4; JV * b = jnew(1); b->i = 1; return b; }
    if (!strncmp(jp, "false", 5)) { jp += 5; return jnew(1); }
    if (!strncmp(jp, "null", 4)) { jp += 4; return jnew(0); }
    if (*jp == '-' || (*jp >= '0' && *jp <= '9')) {
        JV * num = jnew(2);
        const char * s = jp;
        char * e1;
        num->d = strtod(s, &e1);
        int isint = 1;
        for (const char * q = s; q < e1; q++) if (*q == '.' || *q == 'e' || *q == 'E') isint = 0;
        if (isint) { num->i = _strtoi64(s, NULL, 10); num->isint = 1; } else { num->i = (int64_t) num->d; }
        jp = e1;
        return num;
    }
    die("JSON: unexpected character '%c'", *jp ? *jp : '0');
}

static JV * jget(JV * o, const char * key) {
    if (!o || o->t != 5) return NULL;
    for (int i = 0; i < o->n; i++) if (!strcmp(o->k[i], key)) return o->v[i];
    return NULL;
}

static JV * jneed(JV * o, const char * key, int t) {
    JV * v = jget(o, key);
    if (!v || v->t != t) die("ranges JSON: missing or mistyped field '%s'", key);
    return v;
}

// ---------------------------------------------------------------- shards and views

typedef struct {
    uint64_t start, end;
    int kind, layer;
} Range;

typedef struct {
    wchar_t path[1024];      // full DOS path
    wchar_t dev[1024];       // the same path in NT device form (\Device\HarddiskVolumeN\...), or empty
    const wchar_t * base;    // file name part of path
    uint64_t size;
    Range * r;
    int nr;
    uint64_t bytes_kind[K_N];
} Shard;

typedef struct {
    uint8_t * base;
    uint64_t size;
    int shard;               // index into shards
    int full;                // the view covers the whole file, so its base is file offset 0
    int match;               // 1 device path, 2 unique file name
} View;

static Shard * g_sh;
static int g_nsh;
static DWORD g_page;

static const wchar_t * wbase(const wchar_t * p) {
    const wchar_t * b = p;
    for (const wchar_t * q = p; *q; q++) if (*q == L'\\' || *q == L'/') b = q + 1;
    return b;
}

static void utf8_to_w(const char * s, wchar_t * out, int cap) {
    if (!MultiByteToWideChar(CP_UTF8, 0, s, -1, out, cap)) die("path too long or not UTF-8: %s", s);
}

// JSON-escaped UTF-8 of a wide string, into a static buffer
static const char * jesc_w(const wchar_t * w) {
    static char out[4096];
    char u[2048];
    if (!WideCharToMultiByte(CP_UTF8, 0, w, -1, u, sizeof u, NULL, NULL)) strcpy(u, "?");
    size_t n = 0;
    for (const char * p = u; *p && n + 8 < sizeof out; p++) {
        unsigned char c = (unsigned char) *p;
        if (c == '"' || c == '\\') { out[n++] = '\\'; out[n++] = (char) c; }
        else if (c < 0x20) { n += (size_t) sprintf(out + n, "\\u%04x", c); }
        else out[n++] = (char) c;
    }
    out[n] = 0;
    return out;
}

static void device_path(const wchar_t * full, wchar_t * dev, int cap) {
    dev[0] = 0;
    const wchar_t * p = full;
    if (!wcsncmp(p, L"\\\\?\\", 4)) p += 4;
    if (!(p[0] && p[1] == L':')) return;
    wchar_t drive[3] = { p[0], L':', 0 };
    wchar_t target[1024];
    if (!QueryDosDeviceW(drive, target, 1024)) return;
    if ((int) (wcslen(target) + wcslen(p + 2)) + 1 > cap) return;
    wcscpy(dev, target);
    wcscat(dev, p + 2);
}

static void load_ranges(const char * file) {
    FILE * f = fopen(file, "rb");
    if (!f) die("cannot open ranges file %s", file);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * buf = (char *) xmalloc((size_t) n + 1);
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) die("cannot read %s", file);
    buf[n] = 0;
    fclose(f);
    jp = buf;
    JV * root = jvalue();
    JV * shards = jneed(root, "shards", 4);
    g_nsh = shards->n;
    if (g_nsh < 1) die("ranges JSON has no shards");
    g_sh = (Shard *) calloc((size_t) g_nsh, sizeof(Shard));
    for (int i = 0; i < g_nsh; i++) {
        JV * s = shards->v[i];
        Shard * sh = &g_sh[i];
        wchar_t given[1024];
        utf8_to_w(jneed(s, "path", 3)->s, given, 1024);
        if (!GetFullPathNameW(given, 1024, sh->path, NULL)) wcscpy(sh->path, given);
        device_path(sh->path, sh->dev, 1024);
        sh->base = wbase(sh->path);
        sh->size = (uint64_t) jneed(s, "size", 2)->i;
        JV * ts = jneed(s, "tensors", 4);
        sh->nr = ts->n;
        sh->r = (Range *) xmalloc(sizeof(Range) * (size_t) sh->nr);
        for (int j = 0; j < ts->n; j++) {
            JV * t = ts->v[j];
            Range * r = &sh->r[j];
            r->start = (uint64_t) jneed(t, "start", 2)->i;
            r->end = (uint64_t) jneed(t, "end", 2)->i;
            JV * ly = jget(t, "layer");
            r->layer = ly && ly->t == 2 ? (int) ly->i : -1;
            const char * k = jneed(t, "kind", 3)->s;
            r->kind = -1;
            for (int q = 0; q < K_N; q++) if (!strcmp(k, KIND_NAME[q])) r->kind = q;
            if (r->kind < 0) die("unknown kind '%s'", k);
            if (r->layer >= MAX_LAYERS) die("layer %d >= %d", r->layer, MAX_LAYERS);
            // the probe attributes pages by walking the ranges in order: they must tile the file
            if (j == 0 ? r->start != 0 : r->start != sh->r[j - 1].end) die("shard %d: ranges are not contiguous at %d", i, j);
            if (r->end < r->start) die("shard %d: range %d ends before it starts", i, j);
            sh->bytes_kind[r->kind] += r->end - r->start;
        }
        if (sh->nr == 0 || sh->r[sh->nr - 1].end != sh->size) die("shard %d: ranges do not end at the file size", i);
    }
    // the shard file names must be unique for the file-name fallback to be safe; otherwise that fallback is disabled
}

static int name_unique(int idx) {
    for (int i = 0; i < g_nsh; i++) if (i != idx && !_wcsicmp(g_sh[i].base, g_sh[idx].base)) return 0;
    return 1;
}

// Walk the target's address space; collect the views that are mappings of a shard. Returns the count.
static int scan_views(HANDLE h, View * out, int * n_other_gguf, wchar_t (*other_names)[512], int max_other) {
    int nv = 0;
    *n_other_gguf = 0;
    uint8_t * p = NULL;
    MEMORY_BASIC_INFORMATION mbi;
    wchar_t name[2048];
    while (VirtualQueryEx(h, p, &mbi, sizeof mbi) == sizeof mbi) {
        uint8_t * next = (uint8_t *) mbi.BaseAddress + mbi.RegionSize;
        if (mbi.State != MEM_FREE && mbi.Type == MEM_MAPPED && mbi.AllocationBase == mbi.BaseAddress) {
            uint8_t * vb = (uint8_t *) mbi.BaseAddress;
            uint64_t vsize = mbi.RegionSize;
            MEMORY_BASIC_INFORMATION m2;
            uint8_t * q = next;
            while (VirtualQueryEx(h, q, &m2, sizeof m2) == sizeof m2 && m2.AllocationBase == (PVOID) vb && m2.State != MEM_FREE) {
                vsize += m2.RegionSize;
                q = (uint8_t *) m2.BaseAddress + m2.RegionSize;
            }
            next = q;
            DWORD len = GetMappedFileNameW(h, vb, name, 2048);
            if (len) {
                name[len < 2048 ? len : 2047] = 0;
                int hit = -1, how = 0;
                for (int i = 0; i < g_nsh && hit < 0; i++) if (g_sh[i].dev[0] && !_wcsicmp(name, g_sh[i].dev)) { hit = i; how = 1; }
                for (int i = 0; i < g_nsh && hit < 0; i++) if (name_unique(i) && !_wcsicmp(wbase(name), g_sh[i].base)) { hit = i; how = 2; }
                if (hit >= 0 && nv < MAX_VIEWS) {
                    View * v = &out[nv++];
                    v->base = vb;
                    v->size = vsize;
                    v->shard = hit;
                    v->match = how;
                    // a view at least as long as the file can only start at file offset 0 (the section is the file)
                    v->full = vsize >= g_sh[hit].size;
                } else if (hit < 0) {
                    size_t L = wcslen(name);
                    if (L > 5 && !_wcsicmp(name + L - 5, L".gguf") && *n_other_gguf < max_other) {
                        wcsncpy(other_names[*n_other_gguf], name, 511);
                        other_names[*n_other_gguf][511] = 0;
                        (*n_other_gguf)++;
                    }
                }
            }
        }
        if (next <= p) break;
        p = next;
    }
    return nv;
}

// ---------------------------------------------------------------- sampling

typedef struct {
    uint64_t pages_kind[K_N];      // valid pages (x stride)
    uint64_t pages_layer[MAX_LAYERS];
    uint64_t pages_partial;        // valid pages in views that do not cover the whole file (offset unknown)
    uint64_t queried;              // pages actually queried
    int ok;
    DWORD err;
} Sample;

static PSAPI_WORKING_SET_EX_INFORMATION * g_buf;
static int g_chunk = 65536;
static int g_stride = 1;

typedef struct { uint64_t a, b; } Run;
typedef struct { Run * r; size_t n, cap; } Runs;

static void runs_add(Runs * rs, uint64_t a, uint64_t b) {
    if (rs->n && rs->r[rs->n - 1].b == a) { rs->r[rs->n - 1].b = b; return; }
    if (rs->n == rs->cap) { rs->cap = rs->cap ? rs->cap * 2 : 64; rs->r = (Run *) realloc(rs->r, sizeof(Run) * rs->cap); if (!rs->r) die("out of memory"); }
    rs->r[rs->n].a = a;
    rs->r[rs->n].b = b;
    rs->n++;
}

static void sample_views(HANDLE h, const View * views, int nv, Sample * s, Runs * runs /* per view, may be NULL */) {
    memset(s, 0, sizeof *s);
    s->ok = 1;
    for (int vi = 0; vi < nv; vi++) {
        const View * v = &views[vi];
        const Shard * sh = &g_sh[v->shard];
        uint64_t limit = v->full ? sh->size : v->size;
        uint64_t npages = (limit + g_page - 1) / g_page;
        int ri = 0;
        for (uint64_t p0 = 0; p0 < npages; p0 += (uint64_t) g_chunk * (uint64_t) g_stride) {
            int n = 0;
            for (uint64_t p = p0; p < npages && n < g_chunk; p += (uint64_t) g_stride) {
                g_buf[n].VirtualAddress = v->base + p * g_page;
                g_buf[n].VirtualAttributes.Flags = 0;
                n++;
            }
            if (!QueryWorkingSetEx(h, g_buf, (DWORD) (sizeof(g_buf[0]) * (size_t) n))) {
                s->ok = 0;
                s->err = GetLastError();
                return;
            }
            s->queried += (uint64_t) n;
            for (int j = 0; j < n; j++) {
                uint64_t page = p0 + (uint64_t) j * (uint64_t) g_stride;
                if (!g_buf[j].VirtualAttributes.Valid) continue;
                if (runs) runs_add(&runs[vi], page * g_page, (page + 1) * g_page);
                if (!v->full) { s->pages_partial += (uint64_t) g_stride; continue; }
                uint64_t byte = page * g_page;
                while (ri < sh->nr - 1 && sh->r[ri].end <= byte) ri++;
                const Range * r = &sh->r[ri];
                s->pages_kind[r->kind] += (uint64_t) g_stride;
                if (r->kind == K_EXP && r->layer >= 0) s->pages_layer[r->layer] += (uint64_t) g_stride;
            }
        }
    }
}

static double g_qpf;
static double now_us(void) { LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double) c.QuadPart * 1e6 / g_qpf; }
static int64_t unix_ms(void) {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ((uint64_t) ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (int64_t) ((t - 116444736000000000ULL) / 10000ULL);
}
static int64_t ft_unix_ms(FILETIME ft) {
    uint64_t t = ((uint64_t) ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (int64_t) ((t - 116444736000000000ULL) / 10000ULL);
}

static double mib(uint64_t pages) { return (double) pages * (double) g_page / 1048576.0; }

// ---------------------------------------------------------------- waiting for a process by image name

static DWORD wait_for_image(const char * want8, double timeout_s, FILETIME after) {
    wchar_t want[1024], wantfull[1024];
    utf8_to_w(want8, want, 1024);
    int by_path = wcschr(want, L'\\') || wcschr(want, L'/');
    if (by_path && GetFullPathNameW(want, 1024, wantfull, NULL)) wcscpy(want, wantfull);
    uint64_t after64 = ((uint64_t) after.dwHighDateTime << 32) | after.dwLowDateTime;
    double t0 = now_us();
    static DWORD pids[65536];
    for (;;) {
        DWORD got = 0;
        if (EnumProcesses(pids, sizeof pids, &got)) {
            DWORD best = 0;
            uint64_t best_ct = 0;
            for (DWORD i = 0; i < got / sizeof(DWORD); i++) {
                HANDLE hp = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pids[i]);
                if (!hp) continue;
                wchar_t img[1024];
                DWORD len = 1024;
                FILETIME ct, et, kt, ut;
                if (QueryFullProcessImageNameW(hp, 0, img, &len) && GetProcessTimes(hp, &ct, &et, &kt, &ut)) {
                    uint64_t ct64 = ((uint64_t) ct.dwHighDateTime << 32) | ct.dwLowDateTime;
                    int same = by_path ? !_wcsicmp(img, want) : !_wcsicmp(wbase(img), want);
                    if (same && ct64 >= after64 && ct64 > best_ct) { best = pids[i]; best_ct = ct64; }
                }
                CloseHandle(hp);
            }
            if (best) return best;
        }
        if (timeout_s > 0 && (now_us() - t0) / 1e6 > timeout_s) die("no new process '%s' within %.0f s", want8, timeout_s);
        Sleep(20);
    }
}

// ---------------------------------------------------------------- main

static const char * next_arg(int * i, int argc, char ** argv) {
    if (*i + 1 >= argc) die("%s needs a value", argv[*i]);
    return argv[++*i];
}

int main(int argc, char ** argv) {
    DWORD pid = 0;
    const char * wait_image = NULL, * ranges = NULL, * outp = NULL, * dump = NULL;
    double wait_timeout = 0, max_seconds = 0;
    int interval_ms = 500, rescan_ms = 2000, once = 0;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
#define NEXT() next_arg(&i, argc, argv)
        if (!strcmp(a, "--pid")) pid = (DWORD) strtoul(NEXT(), NULL, 10);
        else if (!strcmp(a, "--wait-image")) wait_image = NEXT();
        else if (!strcmp(a, "--wait-timeout")) wait_timeout = atof(NEXT());
        else if (!strcmp(a, "--ranges")) ranges = NEXT();
        else if (!strcmp(a, "--out")) outp = NEXT();
        else if (!strcmp(a, "--interval-ms")) interval_ms = atoi(NEXT());
        else if (!strcmp(a, "--stride")) g_stride = atoi(NEXT());
        else if (!strcmp(a, "--chunk")) g_chunk = atoi(NEXT());
        else if (!strcmp(a, "--rescan-ms")) rescan_ms = atoi(NEXT());
        else if (!strcmp(a, "--max-seconds")) max_seconds = atof(NEXT());
        else if (!strcmp(a, "--once")) once = 1;
        else if (!strcmp(a, "--dump-runs")) dump = NEXT();
        else die("unknown argument %s (see the comment at the top of wsprobe.c)", a);
#undef NEXT
    }
    if (!ranges) die("--ranges is required");
    if (!pid && !wait_image) die("--pid or --wait-image is required");
    if (pid && wait_image) die("--pid and --wait-image are exclusive");
    if (g_stride < 1 || g_chunk < 1 || interval_ms < 1) die("--stride, --chunk and --interval-ms must be >= 1");
    if (dump && (!once || g_stride != 1)) die("--dump-runs needs --once and --stride 1");

    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_qpf = (double) f.QuadPart;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    g_page = si.dwPageSize;
    load_ranges(ranges);
    g_buf = (PSAPI_WORKING_SET_EX_INFORMATION *) xmalloc(sizeof(PSAPI_WORKING_SET_EX_INFORMATION) * (size_t) g_chunk);

    FILE * out = stdout;
    if (outp && !(out = fopen(outp, "ab"))) die("cannot open %s", outp);

    FILETIME started;
    GetSystemTimeAsFileTime(&started);
    if (wait_image) pid = wait_for_image(wait_image, wait_timeout, started);
    double t0 = now_us();
    HANDLE h = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE, FALSE, pid);
    if (!h) die("OpenProcess(%lu) failed: error %lu", (unsigned long) pid, (unsigned long) GetLastError());
    FILETIME ct, et, kt, ut;
    int64_t created_ms = GetProcessTimes(h, &ct, &et, &kt, &ut) ? ft_unix_ms(ct) : 0;
    int64_t attach_ms = unix_ms();

    // header line: what is being measured, and the denominators
    {
        uint64_t tot[K_N] = { 0 };
        for (int i = 0; i < g_nsh; i++) for (int k = 0; k < K_N; k++) tot[k] += g_sh[i].bytes_kind[k];
        fprintf(out, "{\"type\":\"start\",\"pid\":%lu,\"page\":%lu,\"stride\":%d,\"chunk\":%d,\"interval_ms\":%d,"
                     "\"created_unix_ms\":%lld,\"attach_unix_ms\":%lld,\"attach_after_create_ms\":%lld,\"qpc_us\":%.0f,\"mapped_mib\":{",
                (unsigned long) pid, (unsigned long) g_page, g_stride, g_chunk, interval_ms, (long long) created_ms,
                (long long) attach_ms, (long long) (created_ms ? attach_ms - created_ms : -1), t0);
        for (int k = 0; k < K_N; k++) fprintf(out, "%s\"%s\":%.2f", k ? "," : "", KIND_NAME[k], (double) tot[k] / 1048576.0);
        fprintf(out, "},\"shards\":[");
        for (int i = 0; i < g_nsh; i++) fprintf(out, "%s{\"path\":\"%s\",\"size\":%llu,\"dev\":\"%s\"}", i ? "," : "",
                                                jesc_w(g_sh[i].path), (unsigned long long) g_sh[i].size, g_sh[i].dev[0] ? "yes" : "no");
        fprintf(out, "]}\n");
        fflush(out);
    }

    View views[MAX_VIEWS], prev[MAX_VIEWS];
    int nv = 0, nprev = -1;
    wchar_t other[8][512];
    int nother = 0;
    double last_scan = -1e18;
    long long nsamples = 0;
    double qmax = 0, qsum = 0, scan_ms = 0;
    const char * reason = "max-seconds";
    for (long long k = 0;; k++) {
        double t_target = t0 + (double) k * interval_ms * 1000.0;
        // rescan the address space every rescan_ms, and on every sample until every shard has a full view
        int all_found = 1;
        for (int i = 0; i < g_nsh; i++) {
            int have = 0;
            for (int j = 0; j < nv; j++) if (views[j].shard == i && views[j].full) have = 1;
            all_found &= have;
        }
        double ts = now_us();
        if (!all_found || (ts - last_scan) / 1000.0 >= rescan_ms || once) {
            nv = scan_views(h, views, &nother, other, 8);
            last_scan = now_us();
            scan_ms = (last_scan - ts) / 1000.0;
            int changed = nv != nprev;
            for (int j = 0; !changed && j < nv; j++) changed = views[j].base != prev[j].base || views[j].size != prev[j].size;
            if (changed) {
                fprintf(out, "{\"type\":\"views\",\"t\":%.3f,\"qpc_us\":%.0f,\"scan_ms\":%.1f,\"views\":[", (ts - t0) / 1e6, ts, scan_ms);
                for (int j = 0; j < nv; j++)
                    fprintf(out, "%s{\"shard\":%d,\"size\":%llu,\"full\":%d,\"match\":\"%s\"}", j ? "," : "", views[j].shard,
                            (unsigned long long) views[j].size, views[j].full, views[j].match == 1 ? "device-path" : "file-name");
                fprintf(out, "],\"unmatched_gguf_views\":[");
                for (int j = 0; j < nother; j++) fprintf(out, "%s\"%s\"", j ? "," : "", jesc_w(other[j]));
                fprintf(out, "]}\n");
                memcpy(prev, views, sizeof(View) * (size_t) nv);
                nprev = nv;
            }
        }

        Sample s;
        Runs * runs = dump ? (Runs *) calloc((size_t) (nv ? nv : 1), sizeof(Runs)) : NULL;
        double q0 = now_us();
        sample_views(h, views, nv, &s, runs);
        double q1 = now_us();
        double qms = (q1 - q0) / 1000.0;

        PROCESS_MEMORY_COUNTERS_EX pmc;
        memset(&pmc, 0, sizeof pmc);
        pmc.cb = sizeof pmc;
        BOOL pm_ok = GetProcessMemoryInfo(h, (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof pmc);
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof ms;
        GlobalMemoryStatusEx(&ms);
        PERFORMANCE_INFORMATION pi;
        memset(&pi, 0, sizeof pi);
        pi.cb = sizeof pi;
        GetPerformanceInfo(&pi, sizeof pi);
        int exited = WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
        if (exited && !s.ok) { reason = "process exited"; break; }

        fprintf(out, "{\"type\":\"sample\",\"t\":%.3f,\"qpc_us\":%.0f,\"unix_ms\":%lld,\"q_ms\":%.1f,\"ok\":%d,", (q0 - t0) / 1e6, q0,
                (long long) unix_ms(), qms, s.ok);
        if (!s.ok) fprintf(out, "\"err\":%lu,", (unsigned long) s.err);
        fprintf(out, "\"stride\":%d,\"views\":%d,\"queried\":%llu,\"res_mib\":{", g_stride, nv, (unsigned long long) s.queried);
        for (int q = 0; q < K_N; q++) fprintf(out, "%s\"%s\":%.2f", q ? "," : "", KIND_NAME[q], mib(s.pages_kind[q]));
        fprintf(out, "},\"res_pages\":{");
        for (int q = 0; q < K_N; q++) fprintf(out, "%s\"%s\":%llu", q ? "," : "", KIND_NAME[q], (unsigned long long) s.pages_kind[q]);
        fprintf(out, "},\"partial_view_mib\":%.2f,\"exp_layer_mib\":[", mib(s.pages_partial));
        int maxl = -1;
        for (int i = 0; i < g_nsh; i++) for (int j = 0; j < g_sh[i].nr; j++) if (g_sh[i].r[j].kind == K_EXP && g_sh[i].r[j].layer > maxl) maxl = g_sh[i].r[j].layer;
        for (int l = 0; l <= maxl; l++) fprintf(out, "%s%.2f", l ? "," : "", mib(s.pages_layer[l]));
        fprintf(out, "],\"exp_layer_pages\":[");
        for (int l = 0; l <= maxl; l++) fprintf(out, "%s%llu", l ? "," : "", (unsigned long long) s.pages_layer[l]);
        fprintf(out, "],\"avail_mib\":%.0f,\"syscache_mib\":%.0f,", (double) ms.ullAvailPhys / 1048576.0,
                (double) pi.SystemCache * (double) pi.PageSize / 1048576.0);
        if (pm_ok)
            fprintf(out, "\"pf\":%lu,\"ws_mib\":%.1f,\"priv_mib\":%.1f}\n", (unsigned long) pmc.PageFaultCount,
                    (double) pmc.WorkingSetSize / 1048576.0, (double) pmc.PrivateUsage / 1048576.0);
        else
            fprintf(out, "\"pf\":null,\"ws_mib\":null,\"priv_mib\":null}\n");
        fflush(out);
        nsamples++;
        qsum += qms;
        if (qms > qmax) qmax = qms;

        if (dump) {
            FILE * d = fopen(dump, "wb");
            if (!d) die("cannot open %s", dump);
            fprintf(d, "{\"page\":%lu,\"views\":[", (unsigned long) g_page);
            for (int j = 0; j < nv; j++) {
                fprintf(d, "%s{\"shard\":%d,\"path\":\"%s\",\"full\":%d,\"runs\":[", j ? "," : "", views[j].shard, jesc_w(g_sh[views[j].shard].path), views[j].full);
                for (size_t r = 0; r < runs[j].n; r++) fprintf(d, "%s[%llu,%llu]", r ? "," : "", (unsigned long long) runs[j].r[r].a, (unsigned long long) runs[j].r[r].b);
                fprintf(d, "]}");
            }
            fprintf(d, "]}\n");
            fclose(d);
        }
        if (runs) { for (int j = 0; j < nv; j++) free(runs[j].r); free(runs); }

        if (once) { reason = "once"; break; }
        if (exited) { reason = "process exited"; break; }
        if (max_seconds > 0 && (now_us() - t0) / 1e6 >= max_seconds) break;
        double wait_ms = (t_target + interval_ms * 1000.0 - now_us()) / 1000.0;
        if (wait_ms > 0 && WaitForSingleObject(h, (DWORD) wait_ms) == WAIT_OBJECT_0) {
            // the process ended while we slept: its address space is gone, so there is nothing left to sample
            reason = "process exited";
            break;
        }
    }
    fprintf(out, "{\"type\":\"end\",\"t\":%.3f,\"reason\":\"%s\",\"samples\":%lld,\"q_ms_mean\":%.1f,\"q_ms_max\":%.1f}\n",
            (now_us() - t0) / 1e6, reason, nsamples, nsamples ? qsum / (double) nsamples : 0.0, qmax);
    fflush(out);
    if (out != stdout) fclose(out);
    CloseHandle(h);
    return 0;
}
