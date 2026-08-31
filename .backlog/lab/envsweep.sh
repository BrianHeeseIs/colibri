#!/usr/bin/env bash
# Generic decode-trace arm sweep. Generalises pinsweep.sh: an arm is
#   name =@= memory_gb =@= verify_specs =@= env
# where verify_specs is a ';'-separated list of field<op>value (or '-' for none), checked
# against the arm's own log after it runs.
#
# Everything load-bearing is inherited from pinsweep.sh unchanged: the seed is verified by
# HASH not existence, one engine at a time, each arm re-seeds .coli_usage so pin history
# cannot leak between arms, COLI_V4_SAVE_USAGE=0, the exact 7-prefix ext() strip list, and
# the completion marker is touched as the final action.
#
# The one-engine guard uses `pgrep -x` rather than `pgrep -f`. `-f` matches the full command
# line of ANY process, so a caller whose own pre-flight merely mentions the binary path trips
# it and the harness refuses a run while no engine exists. That cost two dispatches on
# 2026-08-30. `-x` matches the process NAME only and closes the trap class.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BIN=${BIN:-./c/deepseek_v4}
TOKENS=${TOKENS:-40}
PROMPT_FILE=${PROMPT_FILE:-.backlog/prefill_prompts/p256.txt}
EXTRACT="$ROOT/.backlog/lab/e131_extract.sh"
TAG=${TAG:-batchA}
COMPLETION_MARKER=${COMPLETION_MARKER:-$ROOT/.backlog/lab/E132_E133_DONE}

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -x "$(basename "$BIN")" >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }
rm -f "$COMPLETION_MARKER" || { echo "FATAL: cannot clear marker" >&2; exit 2; }

BIN_MD5_BEFORE=$(md5 -q "$BIN")
PROMPT=$(<"$PROMPT_FILE")

COMMON_SPEC='pin_slots_applied=16;target_slots=164;head_tier=resident-bf16'
MEM_SPEC='target_slots!=164;head_tier=resident-bf16'
# Batch A ordering is load-bearing. Batch 1 voided E132 because a 48 GiB arm sat between the
# anchors and left 9.4 GB of extra disk traffic in the page cache, so the later anchor drifted
# 20.2%. Here E132's four arms run first and CONTIGUOUSLY, and every memory arm runs after both
# of its anchors, so no heavy arm can contaminate the comparison it is measured against.
set -- \
  "R0=@=96=@=$COMMON_SPEC=@=" \
  "R1=@=96=@=$COMMON_SPEC=@=COLI_V4_HOT_PACK_UNLOCKED=1" \
  "R2=@=96=@=$COMMON_SPEC=@=COLI_V4_HOT_PACK_UNLOCKED=1" \
  "R3=@=96=@=$COMMON_SPEC=@=" \
  "M1=@=108=@=$MEM_SPEC=@=" \
  "M2=@=120=@=$MEM_SPEC=@=" \
  "R4=@=96=@=$COMMON_SPEC=@="
declare -a NAMES=() MEMS=() SPECS=() ENVS=()
while [[ $# -gt 0 ]]; do
  r1=${1#*=@=}; r2=${r1#*=@=}
  NAMES+=("${1%%=@=*}"); MEMS+=("${r1%%=@=*}"); SPECS+=("${r2%%=@=*}"); ENVS+=("${r2#*=@=}"); shift
done

ext() { awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal|v4_prefill_trace) /{print} /^timing /{f=0}' "$1"; }

printf '%s  tokens=%s prompt=%s binary_md5=%s\n' \
       "$TAG" "$TOKENS" "$(basename "$PROMPT_FILE")" "$BIN_MD5_BEFORE"
declare -a LOGS=() MD5S=()
failed=0; voided=0
for i in "${!NAMES[@]}"; do
  cp "$DURABLE" /tmp/coli_usage.snapshot || { echo "FATAL: cannot refresh /tmp seed" >&2; exit 2; }
  cp "$DURABLE" "$MODEL/.coli_usage"     || { echo "FATAL: cannot refresh model seed" >&2; exit 2; }
  timestamp=$(date '+%Y%m%d-%H%M%S')
  log="$ROOT/.backlog/lab/decode_trace_${NAMES[$i]}_${timestamp}.log"
  LOGS[$i]=$log
  # shellcheck disable=SC2086
  env ${ENVS[$i]} COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1 COLI_V4_SAVE_USAGE=0 \
      "$BIN" "$MODEL" "$PROMPT" --max-tokens "$TOKENS" --memory-gb "${MEMS[$i]}" \
      >"$log" 2>&1
  rc=$?
  md5v=$(ext "$log" | md5)
  MD5S[$i]=$md5v
  printf '  %-3s rc=%-3s mem=%-4s env=%-32s text_md5=%s log=%s\n' \
         "${NAMES[$i]}" "$rc" "${MEMS[$i]}" "${ENVS[$i]:-<defaults>}" "$md5v" "$(basename "$log")"
  if (( rc != 0 )); then
    echo "  ENGINE FAILED rc=$rc arm=${NAMES[$i]}" >&2; tail -5 "$log" >&2; failed=1
  fi
  if ! bash "$EXTRACT" verify_specs "$log" "${SPECS[$i]}"; then
    echo "  ARM VOID: ${NAMES[$i]} did not apply its requested settings" >&2; voided=1
  fi
done
cp "$DURABLE" "$MODEL/.coli_usage" || { echo "FATAL: cannot restore model seed" >&2; exit 2; }

BIN_MD5_AFTER=$(md5 -q "$BIN")
echo
printf 'binary md5 before=%s after=%s %s\n' "$BIN_MD5_BEFORE" "$BIN_MD5_AFTER" \
       "$([[ $BIN_MD5_BEFORE == "$BIN_MD5_AFTER" ]] && echo '(unchanged: both golden verdicts still hold)' || echo '(CHANGED -- goldens must be re-run)')"
echo
printf '%-4s %10s %10s %10s %10s %10s %9s %10s %8s %9s\n' \
  arm lock_ms pack_ms hitscan_ms waitblk_ms disk_ms misses expfwd_ms rows16 wall_ms
for i in "${!NAMES[@]}"; do
  L=${LOGS[$i]}
  printf '%-4s %10s %10s %10s %10s %10s %9s %10s %8s %9s\n' "${NAMES[$i]}" \
    "$(bash "$EXTRACT" field "$L" store_lock_ms)" \
    "$(bash "$EXTRACT" field "$L" store_pack_ms)" \
    "$(bash "$EXTRACT" field "$L" store_hit_scan_ms)" \
    "$(bash "$EXTRACT" field "$L" wait_finish_complete_ms)" \
    "$(bash "$EXTRACT" field "$L" store_disk_read_ms)" \
    "$(bash "$EXTRACT" field "$L" store_disk_read_calls)" \
    "$(bash "$EXTRACT" field "$L" expert_forward_ms)" \
    "$(bash "$EXTRACT" field "$L" expert_calls_rows16)" \
    "$(bash "$EXTRACT" field "$L" decode_wall_ms)"
done
echo
printf '%-4s %10s %10s %8s %14s %s\n' arm ttft_s decode_s tok_s packed_slots pack_calls
for i in "${!NAMES[@]}"; do
  L=${LOGS[$i]}; ds=$(bash "$EXTRACT" field "$L" decode_sec)
  printf '%-4s %10s %10s %8.4f %14s %s\n' "${NAMES[$i]}" \
    "$(bash "$EXTRACT" field "$L" ttft_sec)" "$ds" \
    "$(awk -v d="$ds" 'BEGIN{if(d>0) print (39/d); else print 0}')" \
    "$(bash "$EXTRACT" field "$L" packed_slots)" \
    "$(bash "$EXTRACT" field "$L" store_pack_calls)"
done
echo
for i in "${!NAMES[@]}"; do printf 'md5 %-3s %s\n' "${NAMES[$i]}" "${MD5S[$i]}"; done
if (( failed == 0 && voided == 0 )); then
  echo "${TAG}: COMPLETE (all arms ran and applied their requested settings)"
else
  echo "${TAG}: DEGRADED (failed=$failed voided=$voided) -- read the log bodies"
fi
touch "$COMPLETION_MARKER"
