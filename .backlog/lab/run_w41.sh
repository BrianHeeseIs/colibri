#!/usr/bin/env bash
# W4.1 adjacent-surface regression: three A/Bs, N=5, sequential.
# Clears ab.sh's lock dir between runs - an aborted ab.sh leaves /tmp/colibri-prefill-bench.lock
# behind and the next run reports "another deepseek_v4 process is already running", which is
# misleading: pgrep is clean, it is a stale lock.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
clear
echo "=== W4.1 ADJACENT REGRESSION  (3 A/Bs, N=5) ==="
echo "SIGN: delta = 100*(on-off)/off  ->  FASTER IS NEGATIVE"
echo "binary $(md5 -q c/deepseek_v4)"
echo
run_one() {
  local label=$1 flag=$2 expect=$3
  rm -rf /tmp/colibri-prefill-bench.lock
  pkill -f '[d]eepseek_v4' 2>/dev/null; sleep 2
  echo ">>> $label   (expect $expect)"
  N=5 ./bench/ab.sh "$flag" ./c/deepseek_v4 2>&1 | tee ".backlog/lab/w41_${label}.log" | grep -E '^(RUN|AB) '
  echo
}
run_one grouped "COLI_V4_MOE_GROUPED=1" "~ -4.4% / -3.6%"
run_one attn    "COLI_V4_METAL_ATTN=1"  "~ -11.5% / -13.0%"
export COLI_V4_MOE_GROUPED=1 COLI_V4_METAL_ATTN=1
run_one batched "COLI_V4_MOE_BATCHED=1" "~ -10.8% / -11.7%  (incremental)"
unset COLI_V4_MOE_GROUPED COLI_V4_METAL_ATTN
echo "=== W4.1 SUMMARY ==="
grep -hE '^AB ' .backlog/lab/w41_*.log
echo W41_ALL_DONE
