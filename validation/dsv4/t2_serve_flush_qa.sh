#!/usr/bin/env bash
# T2 QA from the cache-flush design: prove .coli_usage appears after ONE served request
# while the server is STILL RUNNING.
#
# The "still running" part is the entire point. destroy_hot() has always written the file
# on graceful teardown, so checking after shutdown would pass even with the feature absent.
# The new per-turn epilogue is the only thing that can produce the file mid-session.
#
# Uses the in-repo launcher c/coli, not bin/coli (bin/ is a local `make install` artifact
# and is absent from a fresh checkout).
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
HIST=$MODEL/.coli_usage
RAM=${V4RAM:-48}
cd "$BASE"

cleanup(){ pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
           pkill -f 'libexec/colibri/deepseek_v4' 2>/dev/null; }

echo "=== T2 serve-level flush QA @ --ram $RAM ==="
cleanup
for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done

rm -f "$HIST" "$HIST.tmp"
echo "  pre-condition: .coli_usage $([ -f "$HIST" ] && echo PRESENT-BAD || echo absent-ok)"

COLI_MODEL=$MODEL python3 c/coli serve --ram $RAM --port 8090 >/tmp/t2_serve.log 2>&1 &
SRV=$!
echo "  server pid=$SRV, waiting for readiness..."
MID=""
for i in $(seq 1 120); do
  sleep 5
  MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MID" ] && break
done
[ -n "$MID" ] || { echo "  SERVER FAILED TO START"; tail -20 /tmp/t2_serve.log; cleanup; exit 1; }
echo "  server ready ($MID)"

# The file must still be absent after load -- only a served turn may create it.
echo "  after load, before any request: .coli_usage $([ -f "$HIST" ] && echo PRESENT || echo absent)"

echo "  sending ONE request..."
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}" \
  --max-time 900 -o /tmp/t2_reply.json
echo "  reply tokens: $(python3 -c "import json;print(json.load(open('/tmp/t2_reply.json')).get('usage',{}))" 2>/dev/null)"

# THE ASSERTION -- server deliberately still alive.
# NOTE: the pattern must NOT be 'libexec/colibri/deepseek_v4'. c/coli's engine_for() prefers
# c/deepseek_v4 when it exists (see coli:249), so a libexec-only pattern matches nothing and
# reports a live engine as 0 procs -- which it did on the first run of this script.
ALIVE=$(pgrep -f '[d]eepseek_v4' | wc -l | tr -d ' ')
if [ -f "$HIST" ]; then
  echo "  RESULT: GREEN -- .coli_usage EXISTS after one request, engine still running ($ALIVE proc)"
  ls -l "$HIST" | sed 's/^/    /'
  grep -o 'v4_autopin saved=[^ ]* selections=[0-9]* distinct=[0-9]*' /tmp/t2_serve.log | tail -3 | sed 's/^/    /'
  RC=0
else
  echo "  RESULT: RED -- .coli_usage ABSENT after one request (engine still running: $ALIVE)"
  RC=1
fi

cleanup
exit $RC
