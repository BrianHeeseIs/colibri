#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-METAL=0}
[[ $# -le 1 ]] || { printf 'usage: %s [METAL=0|1]\n' "$0" >&2; exit 2; }
[[ $MODE =~ ^METAL=([01])$ ]] || { printf 'usage: %s [METAL=0|1]\n' "$0" >&2; exit 2; }
METAL=${BASH_REMATCH[1]}

if pgrep -f '[d]eepseek_v4' >/dev/null 2>&1; then
    printf 'refusing to rebuild while deepseek_v4 is running\n' >&2
    exit 2
fi

rm -f "$ROOT"/c/*.o "$ROOT/c/deepseek_v4"
make -C "$ROOT/c" -f Makefile.deepseek-v4 METAL="$METAL" deepseek-v4

if [[ $METAL == 1 ]]; then
    DEST="$ROOT/c/deepseek_v4.metal"
else
    DEST="$ROOT/c/deepseek_v4.cpu"
fi
rm -f "$DEST"
cp "$ROOT/c/deepseek_v4" "$DEST"

object_count=$(find "$ROOT/c" -maxdepth 1 -type f -name '*.o' | wc -l | tr -d ' ')
metal_symbol_count=$(nm "$DEST" 2>/dev/null | awk '/coli_v4_metal_/ { count++ } END { print count + 0 }')
printf 'BUILD METAL=%s objects=%s metal_symbols=%s binary=%s\n' \
    "$METAL" "$object_count" "$metal_symbol_count" "$DEST"
