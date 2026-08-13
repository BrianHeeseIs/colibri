#!/usr/bin/env bash
# Progress bar keyed off the ACTUALLY RUNNING experiment (not file existence).
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
valid_rows(){ [ -f "$1" ] || { echo 0; return; }
  awk -F, 'NR>1 && $4+0>0' "$1" 2>/dev/null | wc -l | tr -d ' '; }
# macOS ps has no `etimes`; parse `etime` (DD-HH:MM:SS | HH:MM:SS | MM:SS) into seconds.
etime2s(){ awk -F'[-:]' 'NF==4{print $1*86400+$2*3600+$3*60+$4;next}
                          NF==3{print $1*3600+$2*60+$3;next}
                          NF==2{print $1*60+$2;next}{print 0}'; }
while :; do
  # Detect ANY harness. The [0-9]* covers coldwarm.sh AND coldwarm2.sh (the prefetch variant),
  # which a bare 'coldwarm\.sh' missed - the pane read "idle" during a live prefetch arm.
  # Do NOT loosen to bare 'coldwarm': that also matches `tee coldwarm96_prefetch.log` and
  # `tail -f coldwarm96_prefetch.log`, so pgrep would return a pipe process, not the harness.
  HPID=$(pgrep -f '(run_experiment|coding_agent|coldwarm[0-9]*|[a-z0-9_]+_ab|t[0-9]_[a-z_]+)\.sh' 2>/dev/null | head -1)
  # HARD GUARD: `ps -p ""` does not reliably return nothing - with a bare -p some ps
  # implementations fall back to listing the tty's processes, which made this pane report
  # "running: progress2.sh  4/4  100%" while nothing was running. Empty pid => idle, full stop.
  if [ -n "$HPID" ]; then
    CMD=$(ps -o command= -p "$HPID" 2>/dev/null)
    # env vars are exported into the shell, so ps -o command= cannot see them; read the real env
    # TAG and PREFETCH MUST be here: coldwarm2.sh writes coldwarm_*_ram<RAM><TAG>.csv, so
    # dropping TAG resolved the CSVs to the PREVIOUS untagged arm and the pane reported that
    # arm's 8/8 completion as the new arm's progress.
    ENV=$(ps -Ewww -p "$HPID" 2>/dev/null | tr ' ' '\n' \
          | grep -E '^(V4_[A-Z_]+|V4RAM|NTOK|CSV|TAG|PREFETCH)=' | tr '\n' ' ')
  else
    CMD=""; ENV=""
  fi
  # CSV: argv first, then CSV= in env, then V4RAM convention, then newest csv touched in last 30min
  CSV=$(echo "$CMD" | grep -oE '[A-Za-z0-9_]+\.csv' | head -1)
  [ -z "$CSV" ] && CSV=$(echo "$ENV" | grep -oE 'CSV=[A-Za-z0-9_]+\.csv' | cut -d= -f2)
  if [ -z "$CSV" ] && echo "$CMD" | grep -q 'coding_agent\.sh'; then
    R=$(echo "$ENV" | grep -oE 'V4RAM=[0-9]+' | cut -d= -f2)
    [ -n "$R" ] && CSV="coding_ram$R.csv"
  fi
  [ -n "$CMD" ] && [ -z "$CSV" ] && \
    CSV=$(find . -maxdepth 1 -name '*.csv' -mmin -30 -not -name 'rss_*' 2>/dev/null \
          | xargs -r ls -t 2>/dev/null | head -1 | sed 's|^\./||')
  if [ -n "$CSV" ]; then
    LABEL=$(echo "$CMD" | grep -oE '[a-z_0-9]+\.sh' | head -1)
    [ -n "$ENV" ] && LABEL="$LABEL [$(echo $ENV)]"
  else
    LABEL="(idle - no experiment running)"
  fi
  # coldwarm runs TWO passes of 4. Showing only one CSV read as "0/4 after 40 minutes",
  # hiding a completed cold pass. Derive both passes so the scope matches the elapsed time.
  DUAL=""
  case "$CMD" in *coldwarm*)
    R=$(echo "$ENV" | grep -oE 'V4RAM=[0-9]+' | cut -d= -f2)
    T=$(echo "$ENV" | grep -oE 'TAG=[A-Za-z0-9_]+' | cut -d= -f2)
    CC="coldwarm_cold_ram${R}${T}.csv"; WC="coldwarm_warm_ram${R}${T}.csv"
    if [ -n "$R" ]; then
      DUAL=$(printf "cold %s/4 · warm %s/4" "$(valid_rows "$CC")" "$(valid_rows "$WC")")
      # progress across BOTH passes so the bar and the elapsed clock agree
      ROWS_ALL=$(( $(valid_rows "$CC") + $(valid_rows "$WC") ))
    fi
  ;; esac
  if [ -n "$CSV" ] && [ -f "$CSV" ]; then
    # elapsed MUST come from the harness process, not the CSV birth time: on a re-run the
    # CSV pre-exists and birth-time reported 130:25 for a 6-second-old decoy run.
    EL=$(ps -o etime= -p "$HPID" 2>/dev/null | tr -d ' ' | etime2s); [ -z "$EL" ] && EL=0
    ROWS=$(valid_rows "$CSV")
    # total prompts differ per harness: run_experiment.sh=10, coding_agent.sh=4
    case "$CMD" in *coding_agent*|*coldwarm*|*_ab.sh*) TOT=4;; *) TOT=10;; esac
    # coldwarm: count BOTH passes (8 prompts total) so bar/ETA match the elapsed clock
    if [ -n "$DUAL" ]; then ROWS=$ROWS_ALL; TOT=8; fi
    # CLAMP EVERYTHING: a stale/mismatched CSV gave 9 rows against TOT=4 -> 225% and ETA -72:-25
    [ "$ROWS" -gt "$TOT" ] && ROWS=$TOT
    # progress by COMPLETED ROWS, not a hardcoded wall-clock cap; ETA from observed pace
    if [ "$ROWS" -gt 0 ]; then
      PER=$((EL/ROWS)); REM=$(( (TOT-ROWS)*PER )); [ "$REM" -lt 0 ] && REM=0
    else PER=0; REM=0; fi
    PCT=$((ROWS*100/TOT)); [ "$PCT" -gt 100 ] && PCT=100; [ "$PCT" -lt 0 ] && PCT=0
    FILL=$((PCT*50/100)); [ "$FILL" -gt 50 ] && FILL=50; [ "$FILL" -lt 0 ] && FILL=0
    BAR=$(printf "%*s" "$FILL" "" | tr " " "#"); GAP=$(printf "%*s" $((50-FILL)) "")
  else EL=0; REM=0; PER=0; TOT=10; PCT=0; BAR=""; GAP=$(printf "%*s" 50 ""); ROWS=0; fi
  clear
  echo "========== COLIBRI EXPERIMENT PROGRESS =========="; echo
  echo "  running : $LABEL"
  if [ -z "$CMD" ]; then
    # idle: never imply a run is pending or that a prompt is in flight
    echo "  prompts : --"; echo
    echo "  [${GAP}] --%"; echo
    echo "  no experiment in flight; last results in the queue below"
  else
    echo "  prompts : $ROWS/$TOT completed"; echo
    [ -n "$DUAL" ] && echo "            ($DUAL)" && echo
    echo "  [${BAR}${GAP}] ${PCT}%"; echo
    printf "  elapsed   %02d:%02d\n" $((EL/60)) $((EL%60))
    if [ "$ROWS" -gt 0 ]; then
      printf "  per-prompt %02d:%02d   ETA %02d:%02d\n" $((PER/60)) $((PER%60)) $((REM/60)) $((REM%60))
    else
      printf "  per-prompt --:--   ETA --:--  (first prompt still running)\n"
    fi
  fi
  echo; echo "  --- queue (valid rows only) ---"
  printf "   %-22s %s/10\n" "baseline --ram 48" "$(valid_rows warmup48.csv)"
  for f in spec_*.csv; do
    [ -e "$f" ] || continue
    case "$f" in *INVALID*) continue;; esac
    printf "   %-22s %s/10\n" "${f%.csv}" "$(valid_rows "$f")"
  done
  echo; echo "  $(date +%H:%M:%S)   chime + speech on each completion"
  sleep 2
done
