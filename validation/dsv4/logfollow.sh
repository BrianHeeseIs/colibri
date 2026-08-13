#!/usr/bin/env bash
# Follow the ACTIVE experiment log, and switch automatically when a new run starts.
#
# WHY: a plain `tail -f coding96.log` goes stale the instant the next experiment starts.
# But "newest file" is ALSO wrong: a queued waiter (queue_*.log) is created after the
# harness log and then sits static, so this pane followed a file whose only content was
# "waiting for the current arm to finish" while the real run streamed elsewhere.
#
# RULE: prefer the log of the harness that is actually running; ignore queue_*.log
# (the queue panes already display that); fall back to newest only when nothing runs.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
CUR=""
TPID=""

active_log(){
  local hpid env ram tag
  hpid=$(pgrep -f 'coldwarm2?\.sh|coding_agent\.sh|run_experiment\.sh' 2>/dev/null | head -1)
  if [ -n "$hpid" ]; then
    env=$(ps -Ewww -p "$hpid" 2>/dev/null | tr ' ' '\n')
    ram=$(echo "$env" | grep -oE '^V4RAM=[0-9]+' | cut -d= -f2)
    tag=$(echo "$env" | grep -oE '^TAG=[A-Za-z0-9_]+' | cut -d= -f2)
    # coldwarm.sh tees to coldwarm<RAM>.log ; the prefetch variant to coldwarm<RAM>_prefetch.log
    for c in "coldwarm${ram}${tag:+_prefetch}.log" "coldwarm${ram}.log" "coding${ram}.log"; do
      [ -n "$ram" ] && [ -f "$c" ] && { echo "$c"; return; }
    done
  fi
  # nothing running (or log not matched): newest non-queue log
  ls -t *.log 2>/dev/null | grep -v '^queue' | head -1
}

while :; do
  NEW=$(active_log)
  if [ -n "$NEW" ] && [ "$NEW" != "$CUR" ]; then
    [ -n "$TPID" ] && kill "$TPID" 2>/dev/null
    CUR="$NEW"
    clear
    printf '\033[1m>>> following %s\033[0m  (tracks the RUNNING harness; queue_*.log ignored)\n\n' "$CUR"
    tail -f "$CUR" &
    TPID=$!
  fi
  sleep 10
done
