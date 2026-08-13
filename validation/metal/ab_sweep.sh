#!/usr/bin/env bash
# Metal ON vs OFF A/B for the DeepSeek-V4 engine, INTERLEAVED and PAIRED.
#
# WHY INTERLEAVED. RESULTS.md:723-726 already warns that absolute tok/s on this host drifts with
# thermal and memory pressure. That warning was re-confirmed today: an identical PREWARM config
# measured 946s on prompt 1 in the afternoon vs 712s overnight -- a 25% swing from machine state
# alone. Running all-OFF then all-ON would fold that drift straight into the delta. So every
# prompt is measured OFF then ON back-to-back, and only the PAIRED per-prompt delta is reported.
#
# WHY A DISPATCH COUNTER GATE. A silent CPU fallback (device unavailable, unsupported format,
# kernel entry missing) would look exactly like "Metal is running but slow". Every ON run must
# prove Metal actually executed, or the row is marked INVALID rather than reported as a result.
#
# WHY TOKEN-EXACTNESS IS CHECKED. A speedup that changes the output is not a speedup. The
# generated text of the ON run is compared against the OFF run for the same prompt.
set -u
BASE=/Users/cptn/workbench/ai/colibri
MODEL=${MODEL:-$BASE/models/deepseek-v4-flash}
RAM=${V4RAM:-96}
NTOK=${NTOK:-128}
VARIANT=${1:-ordered_cold}
GATE=${GATE_COMPRESSOR_GB:-20}
PORT=${PORT:-8090}
PANE=${PANE:-colibri-lab:0.0}
HIST=$MODEL/.coli_usage
SNAP=/tmp/coli_usage.snapshot
cd "$BASE"

comp(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }
cleanup(){ pkill -f 'coli serve' 2>/dev/null; pkill -f '[d]eepseek_v4' 2>/dev/null
           for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:$PORT 2>/dev/null)" ] && break; done; }

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

# start a server with Metal either off or on; echoes the resolved model id
start_server(){ # $1 = 0|1 metal
  local M=$1 MID=""
  cleanup
  tmux send-keys -t "$PANE" "clear && env V4_DRAFT=4 V4_NGRAM=1 COLI_V4_SAVE_USAGE=0 \
COLI_V4_METAL=$M COLI_V4_METAL_VARIANT=$VARIANT COLI_MODEL=$MODEL \
python3 $BASE/c/coli serve --ram $RAM --port $PORT" Enter
  for i in $(seq 1 150); do
    sleep 5
    MID=$(curl -s --max-time 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null \
          | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$MID" ] && break
  done
  echo "$MID"
}

# ask one prompt; writes tok/s + text to globals
ask(){ # $1 = model id  $2 = prompt  $3 = tag
  local MID=$1 P=$2 TAG=$3 S E W T R BODY
  S=$(date +%s)
  BODY=$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.stdin.read()}],'max_tokens':$NTOK,'temperature':0}))" <<< "$P")
  curl -s -X POST http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
       -d "$BODY" --max-time 3000 -o "/tmp/ab_${TAG}.json"
  E=$(date +%s); W=$((E-S))
  T=$(python3 -c "import json;print(json.load(open('/tmp/ab_${TAG}.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
  ASK_WALL=$W; ASK_TOK=$T; ASK_RATE=$R
  ASK_TEXT=$(python3 -c "
import json,hashlib
try:
    d=json.load(open('/tmp/ab_${TAG}.json'))
    t=d['choices'][0]['message']['content']
    print(hashlib.sha256(t.encode()).hexdigest()[:16])
except Exception: print('ERR')" 2>/dev/null)
}

echo "########## METAL A/B  variant=$VARIANT  ram=$RAM  ntok=$NTOK ##########"
[ -f "$SNAP" ] && cp "$SNAP" "$HIST" && echo "  history restored from snapshot ($(stat -f%z "$HIST") B)"
echo "n,prompt,metal,wall_s,tokens,tok_s,text_sha,comp_gb,gate,dispatches"

n=0
for P in "$P1" "$P2" "$P3" "$P4"; do
  n=$((n+1))
  for M in 0 1; do                      # OFF then ON, back to back, same prompt
    MID=$(start_server $M)
    if [ -z "$MID" ]; then echo "$n,p$n,$M,NA,NA,NA,NA,NA,START_FAIL,NA"; continue; fi
    C0=$(comp); ask "$MID" "$P" "p${n}_m${M}"; C1=$(comp)
    CMAX=$(awk -v a="$C0" -v b="$C1" 'BEGIN{print (a>b)?a:b}')
    if awk -v c="$CMAX" -v l="$GATE" 'BEGIN{exit !(c<l)}'; then G=ok; else G=GATE_FAIL; fi
    # dispatch counter: engine prints it at shutdown; scrape the pane
    D=$(tmux capture-pane -t "$PANE" -p 2>/dev/null | grep -oE 'metal_dispatches=[0-9]+' | tail -1 | cut -d= -f2)
    [ -z "$D" ] && D=0
    # a metal-ON run that never dispatched is NOT a valid measurement
    if [ "$M" = "1" ] && [ "$D" = "0" ]; then G="INVALID_NO_DISPATCH"; fi
    echo "$n,p$n,$M,$ASK_WALL,$ASK_TOK,$ASK_RATE,$ASK_TEXT,$CMAX,$G,$D"
    cleanup
  done
done
[ -f "$SNAP" ] && cp "$SNAP" "$HIST"
echo "########## paired deltas ##########"
