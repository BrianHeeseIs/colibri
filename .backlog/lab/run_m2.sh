#!/usr/bin/env bash
# M2 adjacent-surface regression: prove the batched-MoE compile-in did not disturb
# the already-shipped lanes. Parent env MUST be clean so ab.sh's OFF arm is bare.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
LOG=.backlog/lab/m2_$(date +%Y%m%d-%H%M%S).log
unset COLI_V4_MOE_GROUPED COLI_V4_METAL_ATTN COLI_V4_MOE_BATCHED COLI_V4_METAL

banner() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
say()    { printf '\033[1;33m%s\033[0m\n' "$*"; }

banner "COLIBRI LAB — M2 adjacent-surface regression"
say "binary md5 : $(md5 -q c/deepseek_v4)"
say "SIGN TRAP  : delta = 100*(on-off)/off  ->  FASTER IS NEGATIVE"
say "log        : $LOG"
say "expectation: GROUPED alone ~ -4.1% / -2.1% ; METAL_ATTN alone ~ -11.7% / -12.4%"
say "pass band  : within +/-1.5pp of those, else the compile-in disturbed a neighbour"

if pgrep -f '[d]eepseek_v4' >/dev/null; then
  say "ABORT: another deepseek_v4 is running"; exit 1
fi

run_ab() {
  local flag=$1 want=$2
  banner "AB  $flag     (expect $want)"
  date '+  start %H:%M:%S'
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
  N=5 ./bench/ab.sh "$flag" ./c/deepseek_v4 2>&1 | tee -a "$LOG" | grep -E "^(RUN|AB) " 
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
  date '+  done  %H:%M:%S'
}

run_ab "COLI_V4_MOE_GROUPED=1" "-4.1% / -2.1%"
run_ab "COLI_V4_METAL_ATTN=1"  "-11.7% / -12.4%"

banner "M2 SUMMARY"
grep -E "^AB " "$LOG" | sed 's/^/  /'
say "seed restored: $(md5 -q models/deepseek-v4-flash/.coli_usage | cut -c1-8) (want 599f3d12)"
banner "M2 COMPLETE — session stays open for inspection"
