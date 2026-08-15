#!/usr/bin/env bash
# Isolated single-flag A/B. Exports the shipped baseline in the PARENT so ab.sh's
# OFF arm (env -u on each ON_ENV var) inherits it. usage: run_gate.sh "ON_ENV" N
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ON=${1:?on-env}; N=${2:-5}
LOG=.backlog/lab/gate_$(date +%Y%m%d-%H%M%S).log
if pgrep -f '[d]eepseek_v4' >/dev/null; then echo "ABORT: engine already running"; exit 1; fi
printf '\033[1;36m=== GATE  ON=%s  N=%s ===\033[0m\n' "$ON" "$N" | tee -a "$LOG"
printf '\033[1;33mSIGN: delta=100*(on-off)/off -> FASTER IS NEGATIVE\033[0m\n' | tee -a "$LOG"
echo "binary md5 $(md5 -q c/deepseek_v4)" | tee -a "$LOG"
export COLI_V4_MOE_GROUPED=1 COLI_V4_METAL_ATTN=1 COLI_V4_MOE_BATCHED=1
cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
N=$N ./bench/ab.sh "$ON" ./c/deepseek_v4 2>&1 | tee -a "$LOG" | grep -E '^(RUN|AB) '
cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
echo "log: $LOG"
