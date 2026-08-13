#!/usr/bin/env bash
# T4 (primary evidence only): does the learning loop CLOSE across a restart?
#
# The design's primary signal is the startup line: run 2 prints
#   v4_autopin history=... selections=N
# and run 1 does not. That requires policy->history_seeded, i.e. history_total >= 5000
# (deepseek_v4.c:6054). The model already carries 2064 selections from the T2 QA, so a
# couple more turns crosses it.
#
# This deliberately does NOT attempt the full T4 hit-rate protocol (2x 10 prompts plus the
# COLI_V4_SAVE_USAGE=0/=1 confound control) -- that is multiple hours. What this DOES settle
# is the mechanism: history accumulates per turn, survives a NON-graceful kill, and is
# re-loaded and seeded on the next start.
#
# Teardown is SIGTERM on purpose: destroy_hot() must NOT be what saves the history, otherwise
# this proves nothing that the pre-existing behaviour didn't already do.
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
HIST=$MODEL/.coli_usage
RAM=${V4RAM:-48}
cd "$BASE"

cleanup(){ pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
           pkill -f '[d]eepseek_v4' 2>/dev/null
           for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done; }

start(){   # $1 = log file
  COLI_MODEL=$MODEL python3 c/coli serve --ram $RAM --port 8090 >"$1" 2>&1 &
  for i in $(seq 1 120); do
    sleep 5
    MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null \
          | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$MID" ] && return 0
  done
  return 1
}

ask(){     # $1 = prompt text
  curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":16,\"temperature\":0}" \
    --max-time 900 -o /tmp/t4_reply.json
  python3 -c "import json;u=json.load(open('/tmp/t4_reply.json')).get('usage',{});print('    tokens:',u)" 2>/dev/null
}

echo "############ T4 primary evidence: does the loop close? ############"
echo "  starting history: $([ -f "$HIST" ] && stat -f '%z bytes' "$HIST" || echo ABSENT)"
cleanup

echo; echo "===== RUN 1: accumulate past the 5000 seed threshold ====="
start /tmp/t4_run1.log || { echo "  RUN1 FAILED TO START"; tail -20 /tmp/t4_run1.log; cleanup; exit 1; }
echo "  ready ($MID)"
echo "  run 1 startup history line: $(grep -c 'v4_autopin history=' /tmp/t4_run1.log) occurrence(s)"
for p in "explain a hash map" "write a for loop"; do
  echo "  asking: $p"; ask "$p"
done
echo "  saves during run 1:"
grep -o 'selections=[0-9]* distinct=[0-9]*' /tmp/t4_run1.log | tail -3 | sed 's/^/    /'

echo; echo "===== NON-GRACEFUL kill (SIGTERM) -- destroy_hot must not be the saver ====="
cleanup
echo "  history after kill: $([ -f "$HIST" ] && stat -f '%z bytes' "$HIST" || echo ABSENT)"
SEL=$(grep -o 'selections=[0-9]*' /tmp/t4_run1.log | tail -1 | cut -d= -f2)
echo "  last recorded selections: ${SEL:-?}  (seed threshold 5000)"

echo; echo "===== RUN 2: does startup now SEED from history? ====="
start /tmp/t4_run2.log || { echo "  RUN2 FAILED TO START"; tail -20 /tmp/t4_run2.log; cleanup; exit 1; }
echo "  ready ($MID)"
HIST_LINE=$(grep -m1 'v4_autopin history=' /tmp/t4_run2.log)
cleanup

echo
echo "############ VERDICT ############"
if [ -n "$HIST_LINE" ]; then
  echo "  GREEN -- run 2 seeded from persisted history:"
  echo "    $HIST_LINE"
  echo "  run 1 had $(grep -c 'v4_autopin history=' /tmp/t4_run1.log) such line(s); run 2 has 1."
  exit 0
fi
echo "  NOT SEEDED -- no 'v4_autopin history=' line in run 2."
echo "  If selections (${SEL:-?}) < 5000 this is EXPECTED, not a defect: the seed"
echo "  threshold simply was not crossed. Run more turns and repeat."
exit 1
