#!/usr/bin/env bash
while :; do
  clear; echo "=== SYSTEM MONITOR ===  $(date +%H:%M:%S)"
  P=$(pgrep -f 'libexec/colibri/deepseek_v4' | head -1)
  if [ -n "$P" ]; then
    ps -o rss=,%cpu=,%mem= -p "$P" | awk '{printf "engine RSS %.1f GB | CPU %s%% | MEM %s%%\n",$1/1048576,$2,$3}'
  else echo "engine: not running"; fi
  sysctl -n vm.swapusage | sed 's/^/swap: /'
  echo "free RAM: $(memory_pressure 2>/dev/null | grep -i 'free percentage' | awk '{print $NF}')"
  echo "disk free: $(df -H / | tail -1 | awk '{print $4}')"
  echo; echo "--- .coli_usage (learning cache) ---"
  ls -l /Users/cptn/workbench/ai/colibri/models/deepseek-v4-flash/.coli_usage 2>/dev/null | awk '{print "  "$5" bytes  "$9}' || echo "  absent"
  sleep 5
done
