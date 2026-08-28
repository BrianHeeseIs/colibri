#!/usr/bin/env bash
# Live monitor pane for engine benchmark runs. READ-ONLY: never starts an engine, never touches
# the seed, safe to run alongside a benchmark. Refreshes every 3 s.
#   usage: .backlog/lab/lab_monitor.sh [logfile]
LOG=${1:-}
BAR="----------------------------------------------------------------"
while true; do
  clear
  echo " colibri-lab monitor    $(date '+%H:%M:%S')"
  echo "$BAR"
  pid=$(pgrep -f '[d]eepseek_v4' | head -1)
  if [ -n "$pid" ]; then
    ps -o rss=,%cpu=,etime= -p "$pid" | awk -v p="$pid" \
      '{printf " engine   RUNNING pid=%s  rss=%.1f GB  cpu=%s%%  up=%s\n", p, $1/1048576, $2, $3}'
  else
    echo " engine   idle (between runs, or finished)"
  fi
  # AGENTS.md caps engine + non-engine at ~100 GB here; swap > 0 invalidates a run
  vm_stat | awk '/Pages free/{gsub(/\./,"",$3);f=$3} /Pages inactive/{gsub(/\./,"",$3);i=$3}
                 END{printf " memory   free+inactive %.1f GB of 128 GB\n", (f+i)*16384/1073741824}'
  echo " swap    $(sysctl -n vm.swapusage 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="used") print $(i+2)}')"
  echo "$BAR"
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    n=$(grep -c '^  run' "$LOG" 2>/dev/null)
    echo " $(basename "$LOG")   [runs complete: ${n:-0} / 15]"
    echo "$BAR"
    tail -n 15 "$LOG"
  else
    echo " waiting for log: ${LOG:-<none>}"
  fi
  sleep 3
done
