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

# PRE-FLIGHT. bench/ab.sh is `set -euo pipefail`; its run_model sends the engine's own
# output to a per-run log inside a mktemp $WORK dir, and `trap cleanup EXIT` deletes that
# dir on the way out. So when the engine exits non-zero, ab.sh dies instantly, prints
# NOTHING, and destroys the only copy of the error. That is exactly how the
# 2026-08-15 15:47 gate run vanished after its two WARMUP lines.
# Exercise the engine once here, with ab.sh's exact invocation, so a broken engine is
# reported with its real output instead of a silent exit.
echo "--- pre-flight: one engine run with ab.sh's invocation ---" | tee -a "$LOG"
PREFLIGHT=$(mktemp "${TMPDIR:-/tmp}/gate-preflight.XXXXXX")
if env $ON COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 models/deepseek-v4-flash \
      "$(<.backlog/prefill_prompts/p064.txt)" --max-tokens 1 --memory-gb 96 \
      > "$PREFLIGHT" 2>&1; then
  grep -oE 'time_to_first_token=[0-9.]+' "$PREFLIGHT" | head -1 \
    | sed 's/^/  pre-flight OK: /' | tee -a "$LOG"
else
  rc=$?
  printf '\033[1;31mPRE-FLIGHT FAILED (exit %d). Engine output:\033[0m\n' "$rc" | tee -a "$LOG"
  tail -25 "$PREFLIGHT" | tee -a "$LOG"
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
  rm -f "$PREFLIGHT"
  echo "aborting before ab.sh; fix the engine first" | tee -a "$LOG"
  exit 1
fi
rm -f "$PREFLIGHT"
cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
# Do NOT filter this pipeline. An earlier version piped through
#   grep -E '^(RUN|AB) '
# which silently swallowed ab.sh's own diagnostics: a run exited straight after both
# WARMUP lines with no RUN/AB output and no visible reason, and the cause was invisible
# because grep had eaten it. Show everything; the log is the record.
N=$N ./bench/ab.sh "$ON" ./c/deepseek_v4 2>&1 | tee -a "$LOG"
ab_rc=${PIPESTATUS[0]}
if [ "$ab_rc" -ne 0 ]; then
  printf '\033[1;31mab.sh exited %d — see %s\033[0m\n' "$ab_rc" "$LOG"
fi
if ! grep -qE '^AB ' "$LOG"; then
  printf '\033[1;31mNO AB LINES: the gate did not produce a result. Do not interpret this run.\033[0m\n'
fi
cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
echo "log: $LOG"
