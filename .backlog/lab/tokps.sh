#!/usr/bin/env bash
# tok/s A/B harness. The repo's bench/ab.sh runs --max-tokens 1 and parses only
# time_to_first_token, so DECODE IS EXCLUDED BY CONSTRUCTION there. This measures the
# decode axis instead: tok/s = (max_tokens-1) / after_first, where after_first is
# gen_stats.decode_sec (c/deepseek_v4.c:11708).
#
# Also captures the generated-text md5 per arm so correctness/meaning can be judged
# in the same run rather than needing a second pass.
#
# Unlike ab.sh/golden.sh this VERIFIES THE SEED HASH, not merely its existence -
# a present-but-wrong seed silently corrupts results (AGENTS.md).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BIN=${BIN:-./c/deepseek_v4}
TOKENS=${TOKENS:-60}
N=${N:-2}
PROMPT_FILE=${PROMPT_FILE:-.backlog/prefill_prompts/p064.txt}
MEMGB=${MEMGB:-96}      # default residency; per-arm MEMGB=<n> overrides

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -f '[d]eepseek_v4' >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }

PROMPT=$(<"$PROMPT_FILE")
declare -a NAMES=() ENVS=()
while [[ $# -gt 0 ]]; do NAMES+=("${1%%=@=*}"); ENVS+=("${1#*=@=}"); shift; done
(( ${#NAMES[@]} >= 2 )) || { echo "usage: $0 'name=@=ENV1=1 ENV2=1' 'name2=@=...'" >&2; exit 2; }

ext() { awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal) /{print} /^timing /{f=0}' "$1"; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/tokps.XXXXXX")

printf 'tok/s A/B  tokens=%s  N=%s  prompt=%s\n' "$TOKENS" "$N" "$(basename "$PROMPT_FILE")"
for r in $(seq 1 "$N"); do
  for i in "${!NAMES[@]}"; do
    cp "$DURABLE" /tmp/coli_usage.snapshot; cp "$DURABLE" "$MODEL/.coli_usage"
    log="$WORK/${NAMES[$i]}_$r.log"
    # --memory-gb was hardcoded to 96, so an arm could not vary residency and a sweep would
    # silently compare three identical configurations. An arm may now carry MEMGB=<n> in its env
    # string; it is stripped from the child env and turned into the CLI flag it actually needs.
    arm_env="${ENVS[$i]}"; arm_mem=$MEMGB
    if [[ $arm_env =~ (^|[[:space:]])MEMGB=([0-9]+) ]]; then
      arm_mem=${BASH_REMATCH[2]}
      arm_env=$(sed -E 's/(^|[[:space:]])MEMGB=[0-9]+//' <<<"$arm_env")
    fi
    # shellcheck disable=SC2086
    env $arm_env COLI_V4_SAVE_USAGE=0 "$BIN" "$MODEL" "$PROMPT" \
        --max-tokens "$TOKENS" --memory-gb "$arm_mem" >"$log" 2>&1
    rc=$?
    if (( rc != 0 )); then echo "  ENGINE FAILED rc=$rc arm=${NAMES[$i]} log=$log" >&2; tail -5 "$log" >&2; continue; fi
    line=$(grep -m1 '^timing ' "$log")
    ttft=$(sed -E 's/.*time_to_first_token=([0-9.]+)s.*/\1/' <<<"$line")
    dec=$(sed -E 's/.*after_first=([0-9.]+)s.*/\1/' <<<"$line")
    tps=$(awk -v t="$TOKENS" -v d="$dec" 'BEGIN{ if(d>0) printf "%.4f", (t-1)/d; else print "n/a" }')
    md5v=$(ext "$log" | md5)
    printf '  run%s %-22s ttft=%8ss decode=%8ss tok/s=%-8s md5=%s\n' \
           "$r" "${NAMES[$i]}" "$ttft" "$dec" "$tps" "$md5v"
    echo "$tps" >>"$WORK/${NAMES[$i]}.tps"; echo "$md5v" >>"$WORK/${NAMES[$i]}.md5"
  done
done
cp "$DURABLE" "$MODEL/.coli_usage"
echo
echo "SUMMARY (median tok/s)"
base=""
for i in "${!NAMES[@]}"; do
  f="$WORK/${NAMES[$i]}.tps"; [[ -f $f ]] || continue
  med=$(sort -n "$f" | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  uniq_md5=$(sort -u "$WORK/${NAMES[$i]}.md5" | tr '\n' ' ')
  det=$( [[ $(sort -u "$WORK/${NAMES[$i]}.md5" | wc -l) -eq 1 ]] && echo deterministic || echo NONDETERMINISTIC )
  if [[ -z $base ]]; then base=$med; delta="baseline"; else
    delta=$(awk -v m="$med" -v b="$base" 'BEGIN{printf "%+.2f%% tok/s", 100*(m-b)/b}'); fi
  printf '  %-22s %8s tok/s  %-18s  %s  md5=%s\n' "${NAMES[$i]}" "$med" "$delta" "$det" "$uniq_md5"
done
echo "logs: $WORK"
