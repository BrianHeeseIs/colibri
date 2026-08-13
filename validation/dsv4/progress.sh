#!/usr/bin/env bash
DUR=${1:-900}; START=$(date +%s)
while :; do
  N=$(date +%s); EL=$((N-START)); [ "$EL" -gt "$DUR" ] && EL=$DUR
  PCT=$((EL*100/DUR)); FILL=$((PCT*50/100))
  BAR=$(printf '%*s' "$FILL" '' | tr ' ' '#'); GAP=$(printf '%*s' $((50-FILL)) '')
  clear; echo "=== EXPERIMENT PROGRESS ==="; echo
  echo "  [${BAR}${GAP}] ${PCT}%"; echo
  printf "  elapsed   %02d:%02d\n" $((EL/60)) $((EL%60))
  printf "  remaining %02d:%02d\n" $(((DUR-EL)/60)) $(((DUR-EL)%60))
  echo; echo "  --ram 48  (prev run: --ram 96 -> swap thrash)"
  echo "  started $(date -r $START +%H:%M:%S)  now $(date +%H:%M:%S)"
  [ "$EL" -ge "$DUR" ] && { echo; echo "  *** TIME COMPLETE ***"; }
  sleep 1
done
