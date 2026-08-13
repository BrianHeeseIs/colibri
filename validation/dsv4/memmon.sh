#!/usr/bin/env bash
# Memory-pressure monitor for the DeepSeek-V4 engine.
#
# WHY THIS EXISTS: `ps` RSS is BLIND to compressed pages on macOS. During the
# 2026-08-12 --ram 96 run, RSS read 17-21 GB while the engine's real
# phys_footprint was 87 GB - the cache was fully retained but macOS had
# compressed ~69 GB of it. Reading RSS produced two wrong conclusions in a row.
# phys_footprint is the authoritative number; compressor+swap explain throughput.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
P=16384
while :; do
  PID=$(pgrep -f '[d]eepseek_v4' | head -1)
  clear
  printf '\033[1m  DeepSeek-V4 memory monitor\033[0m            %s\n' "$(date '+%H:%M:%S')"
  printf '  ─────────────────────────────────────────────────────\n'
  if [ -z "$PID" ]; then
    printf '  engine: \033[31mnot running\033[0m\n'
  else
    RSS=$(ps -o rss= -p "$PID" | tr -d ' ')
    ET=$(ps -o etime= -p "$PID" | tr -d ' ')
    FP=$(footprint -p "$PID" 2>/dev/null | grep -m1 'phys_footprint:' | grep -oE '[0-9.]+ [KMG]B')
    printf '  engine pid %-8s elapsed %s\n' "$PID" "$ET"
    printf '  phys_footprint  \033[1;32m%-12s\033[0m  <- AUTHORITATIVE\n' "${FP:-?}"
    printf '  ps RSS          %-12s  <- excludes compressed, do not trust\n' \
      "$(awk -v r="$RSS" 'BEGIN{printf "%.1f GB", r/1048576}')"
  fi
  printf '  ─────────────────────────────────────────────────────\n'
  vm_stat | awk -v p=$P '
    /Pages free/{gsub(/\./,"",$3); f=$3}
    /Pages active/{gsub(/\./,"",$3); a=$3}
    /Pages inactive/{gsub(/\./,"",$3); i=$3}
    /Pages wired/{gsub(/\./,"",$4); w=$4}
    /occupied by compressor/{gsub(/\./,"",$5); c=$5}
    /stored in compressor/{gsub(/\./,"",$5); s=$5}
    END{
      printf "  free %.1f  active %.1f  inactive %.1f  wired %.1f GB\n",f*p/1e9,a*p/1e9,i*p/1e9,w*p/1e9
      printf "  compressor: holds \033[1;33m%.1f GB\033[0m compressed into %.1f GB physical\n",s*p/1e9,c*p/1e9
      if (c*p/1e9 > 20) printf "  \033[31m*** COMPRESSION ACTIVE - throughput is being taxed ***\033[0m\n"
    }'
  sysctl vm.swapusage 2>/dev/null | sed -E 's/vm.swapusage: /  swap: /'
  printf '  ─────────────────────────────────────────────────────\n'
  # glob, never a hardcoded list - future runs appear without rewiring this pane
  for f in $(ls -t coding_*.csv coldwarm_*.csv prewarm_*.csv 2>/dev/null | grep -v INVALID | head -3); do
    printf '  \033[1m%s\033[0m\n' "$f"
    # coding_*.csv col8=task ; coldwarm_*.csv col12=pass - print whichever exists
    awk -F, 'NR>1{lbl=(NF>=12?$12:$8); printf "     #%s  %6.1fs  %s tok/s  %s\n",$1,$3,$5,lbl}' "$f"
    awk -F, 'NR>1&&$5+0>0{n++;s+=$5} END{if(n>1) printf "     mean %.4f tok/s over %d rows\n",s/n,n}' "$f"
  done
  sleep 15
done
