#!/usr/bin/env bash
# B5 — p512 decode generalisation of E126/E127. Third data point after p064 (+13.20%) and
# p256 (+15.27%). N=5, which is the mandated minimum for resolving sub-10% decode deltas.
#
# Run on the CURRENT binary, deliberately BEFORE the hot-pack default flip, so all three
# prompt lengths in the series were measured on one engine. Comparability across length is
# the entire point of this experiment, and it costs nothing to preserve.
#
# TMPDIR is pointed at .backlog/lab so tokps.sh's per-run logs land somewhere durable:
# it uses mktemp -d "${TMPDIR:-/tmp}/tokps.XXXXXX", and /tmp can vanish between commands.
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri
TS=$(date +%Y%m%d-%H%M%S)
MARK=.backlog/lab/E135_B5_DONE
rm -f "$MARK"

WORKROOT=".backlog/lab/b5_work_$TS"
mkdir -p "$WORKROOT"

echo "seed md5: $(md5 -q .backlog/lab/coli_usage.snapshot)  (expect 599f3d12e9347ef30541bd6f9ba18bde)"
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot

OFF="COLI_V4_HEAD_ILP=0 COLI_V4_HC_OMP=0 COLI_V4_FP8_DUAL_ROWS16=0 COLI_V4_SPARSE_OMP=0 COLI_V4_INDEXER_OMP=0"
TMPDIR="$(pwd)/$WORKROOT" TOKENS=40 N=5 \
  PROMPT_FILE=.backlog/prefill_prompts/p512.txt \
  bash .backlog/lab/tokps.sh "off=@=$OFF" 'on=@='
echo "tokps_rc=$?"

echo "work dir retained: $WORKROOT"
ls -1 "$WORKROOT" 2>/dev/null | head
echo "E135_B5_DONE" | tee "$MARK"
