#!/usr/bin/env bash
# Prompt-length sweep, interleaved OFF/ON per length. Appends INCREMENTALLY to a durable log so a
# long run that gets interrupted still leaves usable data (AGENTS.md: never keep results in /tmp).
# Metric is TTFT (--max-tokens 1) unless TOKENS>1, in which case tok/s is also recorded.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"; DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
SEED=599f3d12e9347ef30541bd6f9ba18bde
[[ $(md5 -q "$DURABLE") == "$SEED" ]] || { echo "FATAL seed hash" >&2; exit 2; }
pgrep -f '[d]eepseek_v4' >/dev/null && { echo "FATAL engine running" >&2; exit 2; }
N=${N:-2}; TOKENS=${TOKENS:-1}; PROMPTS=${PROMPTS:-"p064 p256 p512"}
OFF_ENV=${OFF_ENV:-"COLI_V4_METAL=1 COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1"}
ON_EXTRA=${ON_EXTRA:-"COLI_V4_KERNELS=all"}
LOG=${LOG:-"$ROOT/.backlog/lab/lensweep_$(date +%Y%m%d-%H%M%S).log"}
echo "lensweep N=$N TOKENS=$TOKENS prompts='$PROMPTS' on_extra='$ON_EXTRA'" | tee "$LOG"
for p in $PROMPTS; do
  f="$ROOT/.backlog/prefill_prompts/$p.txt"; [[ -f $f ]] || { echo "  skip $p (missing)" | tee -a "$LOG"; continue; }
  PROMPT=$(<"$f"); wc=$(wc -w <"$f" | tr -d ' ')
  : >/tmp/ls_off.txt; : >/tmp/ls_on.txt
  for r in $(seq 1 "$N"); do
    for st in off on; do
      extra=""; [[ $st == on ]] && extra="$ON_EXTRA"
      cp "$DURABLE" "$MODEL/.coli_usage"
      # shellcheck disable=SC2086
      env $OFF_ENV $extra COLI_V4_SAVE_USAGE=0 "$ROOT/c/deepseek_v4" "$MODEL" "$PROMPT" \
          --max-tokens "$TOKENS" --memory-gb 96 >/tmp/ls.log 2>&1
      rc=$?; (( rc )) && { echo "  ENGINE FAIL $p/$st rc=$rc" | tee -a "$LOG"; tail -3 /tmp/ls.log | tee -a "$LOG"; continue; }
      line=$(grep -m1 '^timing' /tmp/ls.log)
      t=$(sed -E 's/.*token=([0-9.]+)s.*/\1/' <<<"$line")
      echo "$t" >>/tmp/ls_$st.txt
      printf '  %-6s(%sw) r%s %-3s ttft=%ss\n' "$p" "$wc" "$r" "$st" "$t" | tee -a "$LOG"
    done
  done
  mo=$(sort -n /tmp/ls_off.txt | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  mn=$(sort -n /tmp/ls_on.txt  | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  [[ -n $mo && -n $mn ]] && awk -v p="$p" -v w="$wc" -v a="$mo" -v b="$mn" \
    'BEGIN{printf "  ==> %s (%sw) off=%ss on=%ss delta=%+.2f%%\n",p,w,a,b,100*(b-a)/a}' | tee -a "$LOG"
done
cp "$DURABLE" "$MODEL/.coli_usage"; rm -f /tmp/ls_*.txt /tmp/ls.log
echo "log: $LOG" | tee -a "$LOG"
