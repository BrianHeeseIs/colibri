#!/bin/bash
# E130 T13: extract the decode-trace tables from the three decodetrace.sh arms and
# compute the quantities the T14 decision gate is stated against.
#
# Every number this prints is traceable to a log line, because the gate rule is that a
# figure with no log line behind it is not evidence. Pass the three logs explicitly.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OFF=${1:-$(ls -t .backlog/lab/decode_trace_off_*.log 2>/dev/null | head -1)}
ON1=${2:-$(ls -t .backlog/lab/decode_trace_on_1_*.log 2>/dev/null | head -1)}
ON2=${3:-$(ls -t .backlog/lab/decode_trace_on_2_*.log 2>/dev/null | head -1)}
for f in "$OFF" "$ON1" "$ON2"; do
  [[ -n $f && -f $f ]] || { echo "FATAL: missing arm log (off='$OFF' on1='$ON1' on2='$ON2')" >&2; exit 2; }
done
echo "arms: off=$OFF on_1=$ON1 on_2=$ON2"

# TC1 execution proof: the line must be ABSENT when off and present when on.
echo
echo "== TC1 execution proof =="
for name in off on_1 on_2; do
  case $name in off) f=$OFF;; on_1) f=$ON1;; on_2) f=$ON2;; esac
  n=$(grep -c 'decode_trace_total_calls' "$f")
  line=$(grep 'decode_trace_total_calls' "$f" | tail -1)
  printf '  %-5s lines=%s  %s\n' "$name" "$n" "${line:-<absent>}"
done

# TC2 named-subset liveness: each of these must have calls>0 in both on arms.
echo
echo "== TC2 named-subset liveness (calls must be > 0) =="
for stage in wait_start_lock finish_calls store_lock tensor_lookup decode_alloc; do
  for name in on_1 on_2; do
    case $name in on_1) f=$ON1;; on_2) f=$ON2;; esac
    ln=$(grep -n "stage=$stage " "$f" | tail -1)
    printf '  %-16s %-5s %s\n' "$stage" "$name" "${ln:-<MISSING>}"
  done
done

# S: wait share of decode wall, the quantity the 3% criterion is stated against.
echo
echo "== S = total main-thread wait as pct of decode wall =="
for name in on_1 on_2; do
  case $name in on_1) f=$ON1;; on_2) f=$ON2;; esac
  printf '  %-5s %s\n' "$name" "$(grep 'table=wait stage=total' "$f" | tail -1)"
done

# TC3 reconciliation against the pre-existing, untouched expert_wait bucket.
echo
echo "== TC3 reconciliation vs the existing expert_wait bucket =="
for name in on_1 on_2; do
  case $name in on_1) f=$ON1;; on_2) f=$ON2;; esac
  wait_total=$(grep 'table=wait stage=total' "$f" | tail -1 | sed -n 's/.*total_ms=\([0-9.]*\).*/\1/p')
  bucket=$(grep 'phase=expert_wait' "$f" | tail -1 | sed -n 's/.*total_ms=\([0-9.]*\).*/\1/p')
  awk -v a="$wait_total" -v b="$bucket" -v n="$name" 'BEGIN{
    if (b+0 == 0) { printf "  %-5s trace=%s bucket=%s  R=undefined (bucket zero)\n", n, a, b; }
    else { r=a/b; printf "  %-5s trace_ms=%s bucket_ms=%s  R=%.4f  %s\n", n, a, b, r,
           (r>=0.80 && r<=1.20) ? "IN BAND" : "OUT OF BAND -> trace under suspicion"; } }'
done

# TC5 zero cost: tok/s = (tokens-1)/decode_sec. Both axes are mandatory (AGENTS.md).
echo
echo "== TC5 both axes: TTFT and tok/s per arm =="
for name in off on_1 on_2; do
  case $name in off) f=$OFF;; on_1) f=$ON1;; on_2) f=$ON2;; esac
  printf '  %-5s %s\n' "$name" "$(grep -E 'time_to_first_token|decode_sec' "$f" | tail -2 | tr '\n' ' ')"
done

echo
echo "== full trace tables (on_1) =="
grep 'decode_trace table=' "$ON1" || echo "  <none: the trace produced no table lines>"
