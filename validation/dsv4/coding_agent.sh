#!/usr/bin/env bash
# Emulates a coding-agent workload: long code context in, long code out.
# Primary signal = does engine RSS grow (cache actually filling) at high --ram?
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
RAM=${V4RAM:-96}; CSV=${CSV:-coding_ram${RAM}.csv}; NTOK=${NTOK:-128}
cd "$BASE/validation/dsv4"


# ENGINE PROVENANCE GUARD (added 2026-08-12, RESULTS.md S10).
# bin/coli and c/coli resolve to DIFFERENT binaries: engine_for() (c/coli:249) prefers
# HERE/deepseek_v4 over libexec/. Running a feature test against a stale libexec build
# produces a false negative that looks like a real result. Print what we are about to run.
_ENG="$BASE/libexec/colibri/deepseek_v4"
[ -x "$BASE/bin/deepseek_v4" ] && _ENG="$BASE/bin/deepseek_v4"
echo "  ENGINE: $_ENG  (built $(stat -f '%Sm' -t '%m-%d %H:%M' "$_ENG" 2>/dev/null || echo '?'))"
echo "  NOTE  : c/deepseek_v4 is $(stat -f '%Sm' -t '%m-%d %H:%M' "$BASE/c/deepseek_v4" 2>/dev/null || echo absent) -- if newer, this harness is testing OLD code"

pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
pkill -f 'libexec/colibri/deepseek_v4' 2>/dev/null
tmux send-keys -t colibri-lab:0.0 C-c 2>/dev/null
for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done
sleep 4
echo "=== coding-agent workload: --ram $RAM, $NTOK tokens/prompt ==="
tmux send-keys -t colibri-lab:0.0 "clear && env V4_DRAFT=4 V4_NGRAM=1 COLI_MODEL=$MODEL $BASE/bin/coli serve --ram $RAM --port 8090" Enter
MID=""
for i in $(seq 1 90); do
  sleep 5
  MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MID" ] && break
done
[ -n "$MID" ] || { echo "SERVER FAILED"; ./notify.sh "Coding test failed to start"; exit 1; }
echo "server ready ($MID)"

# RSS sampler in background -> proves whether the cache fills
( while :; do
    P=$(pgrep -f 'libexec/colibri/deepseek_v4' | head -1)
    [ -n "$P" ] && echo "$(date +%H:%M:%S),$(ps -o rss= -p $P | awk '{printf "%.1f",$1/1048576}')" >> rss_ram${RAM}.csv
    sleep 20
  done ) & RSSPID=$!

echo "n,ts,wall_s,tokens,tok_s,prompt_chars,rss_gb,task" > "$CSV"
P1='Refactor this Python function to be async and add proper error handling. Return only the code.

def fetch_user_orders(db, user_id):
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE user_id = " + str(user_id))
    rows = cur.fetchall()
    results = []
    for r in rows:
        item = db.get_item(r[2])
        results.append({"id": r[0], "item": item.name, "qty": r[3]})
    conn.close()
    return results'
P2='Find and fix the bug in this Go concurrency code, and explain the race.

func processJobs(jobs []Job) map[string]int {
    results := make(map[string]int)
    var wg sync.WaitGroup
    for _, j := range jobs {
        wg.Add(1)
        go func() {
            defer wg.Done()
            results[j.Name] = j.Compute()
        }()
    }
    wg.Wait()
    return results
}'
P3='Write a C function that parses a safetensors header: read 8 bytes little-endian as header length N, then N bytes of JSON. Validate N is sane, handle short reads, return an error code. Include the struct definition.'
P4='Review this SQL migration for correctness and performance problems on a 50M row table.

ALTER TABLE orders ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT (pending);
CREATE INDEX idx_orders_status ON orders(status);
UPDATE orders SET status = shipped WHERE shipped_at IS NOT NULL;'

n=0
for P in "$P1" "$P2" "$P3" "$P4"; do
  n=$((n+1)); S=$(date +%s)
  BODY=$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.stdin.read()}],'max_tokens':$NTOK,'temperature':0}))" <<< "$P")
  curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$BODY" --max-time 3000 -o /tmp/code_$n.json
  E=$(date +%s); W=$((E-S))
  T=$(python3 -c "import json;print(json.load(open('/tmp/code_$n.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  PT=$(python3 -c "import json;print(json.load(open('/tmp/code_$n.json')).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null||echo 0)
  R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
  PID=$(pgrep -f 'libexec/colibri/deepseek_v4'|head -1)
  RSS=$(ps -o rss= -p $PID 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
  echo "$n,$(date +%H:%M:%S),$W,$T,$R,$PT,$RSS,task$n" >> "$CSV"
  printf "%2d  %ss  %s tok  %s tok/s  prompt=%s tok  RSS=%sGB\n" "$n" "$W" "$T" "$R" "$PT" "$RSS"
done
kill $RSSPID 2>/dev/null
echo "=== DONE --ram $RAM ==="
./notify.sh "Coding agent test at RAM $RAM complete"
