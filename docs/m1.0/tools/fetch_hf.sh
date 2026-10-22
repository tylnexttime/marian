#!/usr/bin/env bash
# fetch_hf.sh REPO SHA PATH SIZE OID DEST_DIR : one Hugging Face LFS file pinned to a commit, capped at 50 MB/s,
# resumable, verified against the tree API's size and sha256 oid; writes <file>.ok on success.
set -u
REPO="$1"; SHA="$2"; P="$3"; SIZE="$4"; OID="$5"; DEST="$6"
F="$(basename "$P")"
mkdir -p "$DEST"; cd "$DEST" || exit 2
if [ -f "$F.ok" ] && [ -f "$F" ]; then echo "ALREADY $F"; exit 0; fi
for i in 1 2 3 4 5 6; do
  curl -sSL --fail --limit-rate 50M -C - -o "$F.part" "https://huggingface.co/$REPO/resolve/$SHA/$P" && break
  echo "retry $i"; sleep 10
done
S=$(stat -c %s "$F.part" 2>/dev/null || echo 0)
H=$(sha256sum "$F.part" | cut -d' ' -f1)
if [ "$S" = "$SIZE" ] && [ "$H" = "$OID" ]; then
  mv "$F.part" "$F" && echo "$H  $F  repo $REPO@$SHA path $P" > "$F.ok" && echo "VERIFIED $F"
else
  echo "MISMATCH $F size $S hash $H"; exit 3
fi
