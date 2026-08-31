#!/usr/bin/env bash
# Decode-trace prefix contract harness. Runs one trace-off control and two
# trace-on replicates while retaining every raw engine log under .backlog/lab.
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
COMPLETION_MARKER=${COMPLETION_MARKER:-$ROOT/.backlog/lab/decodetrace.complete}

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -f '[d]eepseek_v4' >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }
rm -f "$COMPLETION_MARKER" || { echo "FATAL: cannot clear completion marker: $COMPLETION_MARKER" >&2; exit 2; }

PROMPT=$(<"$PROMPT_FILE")
set -- \
  'off=@=' \
  'on_1=@=COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1' \
  'on_2=@=COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1'
declare -a NAMES=() ENVS=()
while [[ $# -gt 0 ]]; do NAMES+=("${1%%=@=*}"); ENVS+=("${1#*=@=}"); shift; done

ext() { awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal|v4_prefill_trace) /{print} /^timing /{f=0}' "$1"; }

printf 'decode trace prefix contract  tokens=%s  prompt=%s  memory_gb=%s\n' \
       "$TOKENS" "$(basename "$PROMPT_FILE")" "$MEMGB"
declare -a MD5S=() LOGS=()
failed=0
for i in "${!NAMES[@]}"; do
  cp "$DURABLE" /tmp/coli_usage.snapshot || { echo "FATAL: cannot refresh /tmp seed" >&2; exit 2; }
  cp "$DURABLE" "$MODEL/.coli_usage" || { echo "FATAL: cannot refresh model seed" >&2; exit 2; }
  timestamp=$(date '+%Y%m%d-%H%M%S')
  log="$ROOT/.backlog/lab/decode_trace_${NAMES[$i]}_${timestamp}.log"
  LOGS[$i]=$log
  arm_env="${ENVS[$i]}"
  if [[ -z $arm_env ]]; then
    env -u COLI_V4_PROFILE -u COLI_V4_DECODE_TRACE COLI_V4_SAVE_USAGE=0 \
        "$BIN" "$MODEL" "$PROMPT" --max-tokens "$TOKENS" --memory-gb "$MEMGB" \
        >"$log" 2>&1
  else
    # shellcheck disable=SC2086
    env $arm_env COLI_V4_SAVE_USAGE=0 \
        "$BIN" "$MODEL" "$PROMPT" --max-tokens "$TOKENS" --memory-gb "$MEMGB" \
        >"$log" 2>&1
  fi
  rc=$?
  md5v=$(ext "$log" | md5)
  MD5S[$i]=$md5v
  printf '  %-5s rc=%-3s generated_text_md5=%s log=%s\n' \
         "${NAMES[$i]}" "$rc" "$md5v" "$log"
  if (( rc != 0 )); then
    echo "  ENGINE FAILED rc=$rc arm=${NAMES[$i]} log=$log" >&2
    tail -5 "$log" >&2
    failed=1
  fi
done
cp "$DURABLE" "$MODEL/.coli_usage" || { echo "FATAL: cannot restore model seed" >&2; exit 2; }

printf 'logs: %s %s %s\n' "${LOGS[0]}" "${LOGS[1]}" "${LOGS[2]}"
if (( failed == 0 )) && [[ ${MD5S[0]} == "${MD5S[1]}" && ${MD5S[0]} == "${MD5S[2]}" ]]; then
  echo "S6_PREFIX_CONTRACT: PASS"
else
  echo "S6_PREFIX_CONTRACT: FAIL"
fi
touch "$COMPLETION_MARKER"
