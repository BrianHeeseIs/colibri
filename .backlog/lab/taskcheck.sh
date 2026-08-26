#!/usr/bin/env bash
# Task-level correctness gate for NON-BIT-EXACT changes (user-chosen acceptance bar).
#
# A non-bit-exact change is acceptable only if CAPABILITY is unchanged: the generated text may
# differ token-for-token, but answers to verifiable questions must stay correct. This is the
# gate for things like COLI_V4_KERNELS=all (reassociated FP) where golden's md5 cannot apply.
#
# Cost note: model load dominates each run (~35 s), so all questions are packed into ONE prompt
# and graded together - 1 run per arm per repeat instead of 1 run per question.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BIN=${BIN:-./c/deepseek_v4}; TOKENS=${TOKENS:-120}; N=${N:-2}

[[ -f $DURABLE ]] || { echo "FATAL: seed missing" >&2; exit 2; }
[[ $(md5 -q "$DURABLE") == "$SEED_MD5" ]] || { echo "FATAL: seed hash wrong" >&2; exit 2; }
pgrep -f '[d]eepseek_v4' >/dev/null && { echo "FATAL: engine running" >&2; exit 2; }

PROMPT='Answer these questions. Put each answer on its own line, numbered.
1. What is 17 multiplied by 23?
2. What is the capital city of Australia?
3. How many sides does a hexagon have?
4. What is the chemical symbol for gold?
5. Is 97 a prime number, yes or no?'

# grader: label|extended-regex (case-insensitive)
CHECKS=(
  "arithmetic|391"
  "geography|canberra"
  "geometry|(^|[^0-9])6([^0-9]|$)|six"
  "chemistry|\bAu\b|gold is au"
  "primality|yes"
)
ext(){ awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal) /{print} /^timing /{f=0}' "$1"; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/taskcheck.XXXXXX")
declare -a NAMES=() ENVS=()
while [[ $# -gt 0 ]]; do NAMES+=("${1%%=@=*}"); ENVS+=("${1#*=@=}"); shift; done

printf 'task-level correctness gate  tokens=%s  N=%s  checks=%s\n' "$TOKENS" "$N" "${#CHECKS[@]}"
for r in $(seq 1 "$N"); do
  for i in "${!NAMES[@]}"; do
    cp "$DURABLE" /tmp/coli_usage.snapshot; cp "$DURABLE" "$MODEL/.coli_usage"
    log="$WORK/${NAMES[$i]}_$r.log"
    # shellcheck disable=SC2086
    env ${ENVS[$i]} COLI_V4_SAVE_USAGE=0 "$BIN" "$MODEL" "$PROMPT" \
        --max-tokens "$TOKENS" --memory-gb 96 >"$log" 2>&1
    rc=$?; (( rc != 0 )) && { echo "  ENGINE FAILED rc=$rc ${NAMES[$i]}"; tail -3 "$log"; continue; }
    txt=$(ext "$log"); vec=""
    for c in "${CHECKS[@]}"; do
      lab=${c%%|*}; pat=${c#*|}
      if grep -qiE "$pat" <<<"$txt"; then vec+="1"; else vec+="0"; echo "    MISS ${NAMES[$i]} r$r: $lab"; fi
    done
    printf '  run%s %-14s correctness=%s (%s/%s)\n' "$r" "${NAMES[$i]}" "$vec" \
           "$(tr -cd 1 <<<"$vec" | wc -c | tr -d ' ')" "${#CHECKS[@]}"
    echo "$vec" >>"$WORK/${NAMES[$i]}.vec"; cp "$log" "$WORK/keep_${NAMES[$i]}_$r.log"
  done
done
cp "$DURABLE" "$MODEL/.coli_usage"
echo; echo "VERDICT"
ref=""; ok=1
for i in "${!NAMES[@]}"; do
  f="$WORK/${NAMES[$i]}.vec"; [[ -f $f ]] || continue
  u=$(sort -u "$f" | tr '\n' ' ')
  [[ -z $ref ]] && ref=$(head -1 "$f")
  same=$(grep -c "^$ref$" "$f"); tot=$(wc -l <"$f" | tr -d ' ')
  printf '  %-14s vectors=%-14s stable=%s/%s\n' "${NAMES[$i]}" "$u" "$same" "$tot"
  grep -qv "^$ref$" "$f" && ok=0
done
if (( ok )); then echo "  PASS - capability identical across arms (vector $ref)"
else echo "  FAIL - capability DIFFERS between arms; non-bit-exact change is NOT acceptable"; fi
echo "  transcripts: $WORK"
