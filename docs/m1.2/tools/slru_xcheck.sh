#!/usr/bin/env bash
# slru_xcheck.sh LLAMA_CPP_DIR IN.txt OUT.txt : build the SLRU cross-check against the fork's own moe_cache_slru and run it.
# The struct is cut verbatim from LLAMA_CPP_DIR/src/llama-moe-cache.cpp (from "struct moe_cache_slru {" to the first
# line that is exactly "};" after it), so the replay is compared with the code the server runs.
set -euo pipefail
SRC="$1/src/llama-moe-cache.cpp"; IN="$2"; OUT="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
awk '/^struct moe_cache_slru \{/ {on=1} on {print} on && /^};$/ {exit}' "$SRC" > "$WORK/slru_extracted.inc"
[ -s "$WORK/slru_extracted.inc" ] || { echo "moe_cache_slru not found in $SRC"; exit 2; }
c++ -std=c++17 -O2 -Wall -I"$WORK" -o "$WORK/slru_xcheck" "$HERE/slru_xcheck.cpp"
"$WORK/slru_xcheck" < "$IN" > "$OUT"
echo "struct: $(wc -l < "$WORK/slru_extracted.inc") lines from $SRC, sha256 $(sha256sum "$WORK/slru_extracted.inc" | cut -c1-16)"
echo "steps: $(wc -l < "$OUT") -> $OUT"
rm -rf "$WORK"
