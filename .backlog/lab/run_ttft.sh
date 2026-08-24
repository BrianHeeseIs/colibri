#!/usr/bin/env bash
# Absolute TTFT capture, invoked exactly as bench/ab.sh invokes the binary.
# Cross-binary instrument: feed its value lists to `bench/ab.sh --math`.
# usage: run_ttft.sh LABEL N "ENV KEY=VAL ..."
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# /tmp is cleared on reboot and the snapshot vanished once mid-run, which would have
# silently invalidated every measurement. Prefer the durable in-repo copy.
SEED_SNAP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/coli_usage.snapshot"
# NO /tmp FALLBACK. /tmp is cleared on reboot; relying on it is what let a gate run start
# against a missing seed. The in-repo copy is the only source of truth.
if [ ! -f "$SEED_SNAP" ]; then echo "ABORT: no usage snapshot at $SEED_SNAP"; exit 1; fi
if [ "$(md5 -q "$SEED_SNAP")" != "599f3d12e9347ef30541bd6f9ba18bde" ]; then
  echo "ABORT: snapshot md5 mismatch - refusing to measure against a drifted seed"; exit 1
fi
LABEL=${1:?label}; N=${2:?n}; ENVSPEC=${3:-}
LOG=.backlog/lab/ttft_${LABEL}_$(date +%Y%m%d-%H%M%S).log
unset COLI_V4_MOE_GROUPED COLI_V4_METAL_ATTN COLI_V4_MOE_BATCHED COLI_V4_METAL COLI_NO_OMP_TUNE
SNAP=$SEED_SNAP; USAGE=models/deepseek-v4-flash/.coli_usage
if pgrep -f '[d]eepseek_v4' >/dev/null; then echo "ABORT: engine already running"; exit 1; fi
printf '\033[1;36m=== TTFT %s  N=%s  env: %s ===\033[0m\n' "$LABEL" "$N" "${ENVSPEC:-<bare>}" | tee -a "$LOG"
echo "binary md5 $(md5 -q c/deepseek_v4)" | tee -a "$LOG"
for p in p064 p256; do
  vals=""
  PROMPT=$(<".backlog/prefill_prompts/$p.txt")
  for i in $(seq 1 "$N"); do
    cp "$SNAP" "$USAGE"
    t=$(env $ENVSPEC COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 models/deepseek-v4-flash "$PROMPT" \
         --max-tokens 1 --memory-gb 96 2>&1 | grep -oE 'time_to_first_token=[0-9.]+' | cut -d= -f2)
    cp "$SNAP" "$USAGE"
    echo "RUN $p i=$i ttft=${t}s" | tee -a "$LOG"
    vals="$vals $t"
  done
  echo "VALUES $p:$vals" | tee -a "$LOG"
done
echo "log: $LOG"
