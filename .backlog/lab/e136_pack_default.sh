#!/usr/bin/env bash
# E136 known-answer gate for the hot-pack policy default.
#
# A default flip is behaviour-changing config with no natural unit-test seam:
# hot_pack_unlocked() is static inside a 15k-line amalgamation unit. So the test surface is
# the engine's own reported state on the v4_hot_policy line, and each arm fails for a
# DISTINCT real reason:
#
#   1  (no env)                                   -> unlocked   fails if the flip did not happen
#   2  COLI_V4_BASELINE=1                         -> locked     fails if the baseline guard is
#                                                               missing -- this is the assertion
#                                                               protecting the SACRED md5
#   3  COLI_V4_HOT_PACK_UNLOCKED=0                -> locked     fails if explicit override stopped
#                                                               beating the default
#   4  BASELINE=1 + HOT_PACK_UNLOCKED=1           -> unlocked   fails if the explicit flag stopped
#                                                               beating baseline
#
# --max-tokens 1 because only the store-open line is needed; this is an engine run but NOT a
# timing run, so the no-background-agents rule does not bind. The one-engine rule still does.
#
# Guard is `pgrep -x`, matching the process NAME. `pgrep -f` matches the full command line of
# ANY process and therefore also matches a caller whose own pre-flight mentions the binary path,
# which refuses a run while no engine exists.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BIN=${BIN:-./c/deepseek_v4}
OUT=${OUT:-$ROOT/.backlog/lab/e136_pack_default_$(date +%Y%m%d-%H%M%S).log}

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -x "$(basename "$BIN")" >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }

PROMPT="hi"
bad=0
echo "e136 pack-policy gate   binary_md5=$(md5 -q "$BIN")"

run_arm() { # run_arm <n> <expected> <env...>
  local n=$1 want=$2; shift 2
  cp "$DURABLE" /tmp/coli_usage.snapshot 2>/dev/null
  cp "$DURABLE" "$MODEL/.coli_usage" 2>/dev/null
  local log="$OUT.arm$n"
  # shellcheck disable=SC2086
  env "$@" COLI_V4_SAVE_USAGE=0 "$BIN" "$MODEL" "$PROMPT" --max-tokens 1 >"$log" 2>&1
  local rc=$? got
  got=$(sed -n 's/.*v4_hot_policy .*pack=\([a-z]*\).*/\1/p' "$log" | tail -1)
  if [[ -z $got ]]; then got="<no-pack-field>"; fi
  if [[ $got == "$want" ]]; then
    printf '  ok    arm%-2s %-46s pack=%s\n' "$n" "$*" "$got"
  else
    printf '  FAIL  arm%-2s %-46s expected=%s actual=%s (rc=%s)\n' "$n" "$*" "$want" "$got" "$rc"
    bad=$((bad+1))
  fi
}

run_arm 1 unlocked COLI_V4_PACK_GATE_NOOP=1
run_arm 2 locked   COLI_V4_BASELINE=1
run_arm 3 locked   COLI_V4_HOT_PACK_UNLOCKED=0
run_arm 4 unlocked COLI_V4_BASELINE=1 COLI_V4_HOT_PACK_UNLOCKED=1

cp "$DURABLE" "$MODEL/.coli_usage" 2>/dev/null
if (( bad )); then
  echo "e136_pack_default: $bad of 4 arms failed"
  exit 1
fi
echo "e136_pack_default: all 4 arms passed"
