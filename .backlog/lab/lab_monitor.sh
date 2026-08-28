#!/usr/bin/env bash
# Live monitor for engine benchmark runs. READ-ONLY: never starts an engine, never touches the
# seed. Shows progress and an ETA derived from completed runs.
#   usage: .backlog/lab/lab_monitor.sh <logfile> [expected_runs]
LOG=${1:-}; EXP=${2:-0}; START=$(date +%s)
BAR="--------------------------------------------------------------------"
while true; do
  clear
  echo " colibri-lab   $(date '+%H:%M:%S')   elapsed $(( ($(date +%s)-START)/60 ))m$(( ($(date +%s)-START)%60 ))s"
  echo "$BAR"
  pid=$(pgrep -f '[d]eepseek_v4' | head -1)
  if [ -n "$pid" ]; then
    ps -o rss=,%cpu=,etime= -p "$pid" | awk -v p="$pid" \
      '{printf " engine  RUNNING pid=%s rss=%.1fGB cpu=%s%% up=%s", p, $1/1048576, $2, $3;
        if ($2+0 < 5) printf "   <-- 0%% CPU: possible DEADLOCK"; print ""}'
  else
    echo " engine  idle (between runs, or finished)"
  fi
  vm_stat | awk '/Pages free/{gsub(/\./,"",$3);f=$3} /Pages inactive/{gsub(/\./,"",$3);i=$3}
                 END{printf " memory  free+inactive %.1f GB of 128\n",(f+i)*16384/1073741824}'
  printf " disk    %s free\n" "$(df -h / | tail -1 | awk '{print $4}')"
  echo "$BAR"
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    n=$(grep -c '^  run' "$LOG" 2>/dev/null); n=${n:-0}
    el=$(( $(date +%s)-START ))
    if [ "$EXP" -gt 0 ] && [ "$n" -gt 0 ]; then
      per=$(( el / n )); left=$(( (EXP-n)*per ))
      printf " progress %s/%s runs   ~%ds/run   ETA %dm%02ds\n" "$n" "$EXP" "$per" "$((left/60))" "$((left%60))"
    else
      printf " progress %s/%s runs   (ETA after first run completes)\n" "$n" "${EXP:-?}"
    fi
    echo "$BAR"; tail -n 12 "$LOG"
  else
    echo " waiting for log: ${LOG:-<none>}"
  fi
  sleep 3
done
