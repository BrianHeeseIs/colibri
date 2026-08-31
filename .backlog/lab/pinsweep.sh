#!/usr/bin/env bash
# E131 Stage 1 — mechanism probe: does raising COLI_V4_PIN_SLOTS cut the decode miss rate?
#
# Four arms, ~62 s each. A3 repeats A0 and runs LAST, so instrument drift across the block is
# bounded by |A3-A0| and can be checked against the pre-registered <=3% validity precondition.
# Every arm carries the decode trace, because the COUNTERS are the evidence here: the whole
# expert-wait avenue's ceiling is +9.04% tok/s, which is inside the 5-13% decode noise floor,
# so tok/s cannot be the primary measurement.
#
# Guards below are cloned from decodetrace.sh deliberately and must not be relaxed:
# the seed is verified by HASH not existence, one engine at a time is enforced, and each arm
# re-seeds .coli_usage so pin history cannot leak from the previous arm into the next.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BIN=${BIN:-./c/deepseek_v4}
TOKENS=${TOKENS:-40}
PROMPT_FILE=${PROMPT_FILE:-.backlog/prefill_prompts/p256.txt}
MEMGB=${MEMGB:-96}
EXTRACT="$ROOT/.backlog/lab/e131_extract.sh"
COMPLETION_MARKER=${COMPLETION_MARKER:-$ROOT/.backlog/lab/E131_PINSWEEP_DONE}

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -x "$(basename "$BIN")" >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }
rm -f "$COMPLETION_MARKER" || { echo "FATAL: cannot clear marker" >&2; exit 2; }

BIN_MD5_BEFORE=$(md5 -q "$BIN")
PROMPT=$(<"$PROMPT_FILE")

# name =@= expected_pins =@= extra env
set -- \
  'A0=@=16=@=' \
  'A1=@=48=@=COLI_V4_PIN_SLOTS=48' \
  'A2=@=96=@=COLI_V4_PIN_SLOTS=96' \
  'A3=@=16=@='
declare -a NAMES=() PINS=() ENVS=()
while [[ $# -gt 0 ]]; do
  rest=${1#*=@=}
  NAMES+=("${1%%=@=*}"); PINS+=("${rest%%=@=*}"); ENVS+=("${rest#*=@=}"); shift
done

ext() { awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal|v4_prefill_trace) /{print} /^timing /{f=0}' "$1"; }

printf 'E131 pin sweep  tokens=%s prompt=%s memory_gb=%s binary_md5=%s\n' \
       "$TOKENS" "$(basename "$PROMPT_FILE")" "$MEMGB" "$BIN_MD5_BEFORE"
declare -a LOGS=() MD5S=()
failed=0
clamped=0
for i in "${!NAMES[@]}"; do
  cp "$DURABLE" /tmp/coli_usage.snapshot || { echo "FATAL: cannot refresh /tmp seed" >&2; exit 2; }
  cp "$DURABLE" "$MODEL/.coli_usage"     || { echo "FATAL: cannot refresh model seed" >&2; exit 2; }
  timestamp=$(date '+%Y%m%d-%H%M%S')
  log="$ROOT/.backlog/lab/decode_trace_${NAMES[$i]}_${timestamp}.log"
  LOGS[$i]=$log
  # shellcheck disable=SC2086
  env ${ENVS[$i]} COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1 COLI_V4_SAVE_USAGE=0 \
      "$BIN" "$MODEL" "$PROMPT" --max-tokens "$TOKENS" --memory-gb "$MEMGB" \
      >"$log" 2>&1
  rc=$?
  md5v=$(ext "$log" | md5)
  MD5S[$i]=$md5v
  printf '  %-3s rc=%-3s pins_req=%-4s text_md5=%s log=%s\n' \
         "${NAMES[$i]}" "$rc" "${PINS[$i]}" "$md5v" "$(basename "$log")"
  if (( rc != 0 )); then
    echo "  ENGINE FAILED rc=$rc arm=${NAMES[$i]}" >&2; tail -5 "$log" >&2; failed=1
  fi
  # A silently clamped arm is identical to baseline and would manufacture a false null.
  if ! bash "$EXTRACT" verify_applied "$log" "${PINS[$i]}"; then
    echo "  CLAMPED/UNAPPLIED arm=${NAMES[$i]} -- this arm is VOID" >&2; clamped=1
  fi
done
cp "$DURABLE" "$MODEL/.coli_usage" || { echo "FATAL: cannot restore model seed" >&2; exit 2; }

BIN_MD5_AFTER=$(md5 -q "$BIN")
echo
printf 'binary md5 before=%s after=%s %s\n' "$BIN_MD5_BEFORE" "$BIN_MD5_AFTER" \
       "$([[ $BIN_MD5_BEFORE == "$BIN_MD5_AFTER" ]] && echo '(unchanged: both golden verdicts still hold)' || echo '(CHANGED -- goldens must be re-run)')"
echo
bash "$EXTRACT" table "${LOGS[@]}"
echo
printf 'text md5 per arm: %s %s %s %s\n' "${MD5S[0]}" "${MD5S[1]}" "${MD5S[2]}" "${MD5S[3]}"
if (( failed == 0 && clamped == 0 )); then
  echo "E131_PINSWEEP: COMPLETE (all arms ran and applied their requested pins)"
else
  echo "E131_PINSWEEP: DEGRADED (failed=$failed clamped=$clamped) -- read the log bodies"
fi
touch "$COMPLETION_MARKER"
