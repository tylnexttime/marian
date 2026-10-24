#!/usr/bin/env bash
# tier_xcheck.sh LLAMA_CPP_DIR IN.txt OUT.txt : build tier_xcheck against the fork's src/llama-moe-source.cpp and a
# finished CPU build of that tree (LLAMA_CPP_DIR/build/bin: libggml-base, libggml-cpu, libggml), then run it.
set -euo pipefail
L="$1"; IN="$2"; OUT="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
c++ -std=c++17 -O2 -Wall -I"$L/src" -I"$L/include" -I"$L/ggml/include" -o "$WORK/tier_xcheck" \
    "$HERE/tier_xcheck.cpp" "$L/src/llama-moe-source.cpp" -L"$L/build/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"$L/build/bin"
"$WORK/tier_xcheck" "$WORK/data.bin" < "$IN" > "$OUT" 2> "$WORK/log.txt"
grep -E 'llama_moe_source' "$WORK/log.txt" || true
echo "steps: $(wc -l < "$OUT") -> $OUT"
rm -rf "$WORK"
