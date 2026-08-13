#!/usr/bin/env bash
# Session status board: what has run, what is running, what is queued.
# Everything is derived at refresh time - no hardcoded counts or captions.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
rows(){ [ -f "$1" ] || { echo 0; return; }; awk -F, 'NR>1 && $5+0>0' "$1" 2>/dev/null | wc -l | tr -d ' '; }
mean(){ [ -f "$1" ] || { echo "-"; return; }
  awk -F, 'NR>1&&$5+0>0{n++;s+=$5} END{if(n)printf "%.4f",s/n; else printf "-"}' "$1"; }
while :; do
  clear
  printf '\033[1m  COLIBRI SESSION BOARD\033[0m                    %s\n' "$(date '+%H:%M:%S')"
  printf '  ═══════════════════════════════════════════════════════\n'
  # [0-9]* covers coldwarm.sh AND coldwarm2.sh; a bare 'coldwarm' would match the tee/tail
  # processes on coldwarm*.log instead of the harness itself.
  HPID=$(pgrep -f '(run_experiment|coding_agent|coldwarm[0-9]*|[a-z0-9_]+_ab|t[0-9]_[a-z_]+)\.sh' | head -1)
  if [ -n "$HPID" ]; then
    E=$(ps -Ewww -p "$HPID" 2>/dev/null | tr ' ' '\n' | grep -E '^(V4_[A-Z_]+|V4RAM|NTOK)=' | tr '\n' ' ')
    printf '  \033[32mRUNNING\033[0m  %s\n' "$(ps -o command= -p "$HPID" | sed 's|.*/||') $E"
  else
    printf '  \033[33mIDLE\033[0m     no harness running\n'
  fi
  ENG=$(pgrep -f '[d]eepseek_v4' | head -1)
  printf '  engine   %s   port8090 %s\n' \
    "$([ -n "$ENG" ] && echo "up pid $ENG" || echo down)" \
    "$(lsof -nP -iTCP:8090 -sTCP:LISTEN 2>/dev/null | tail -n +2 | wc -l | tr -d ' ') listener(s)"
  printf '  ═══════════════════════════════════════════════════════\n'
  printf '  \033[1m%-24s %5s  %s\033[0m\n' "experiment" "rows" "mean tok/s"
  for f in $(ls -t *.csv 2>/dev/null | grep -viE 'invalid|^rss_'); do
    printf '  %-24s %5s  %s\n' "${f%.csv}" "$(rows "$f")" "$(mean "$f")"
  done
  printf '  ═══════════════════════════════════════════════════════\n'
  printf '  quarantined: %s\n' "$(ls *INVALID*.csv 2>/dev/null | tr '\n' ' ')"
  sleep 20
done
