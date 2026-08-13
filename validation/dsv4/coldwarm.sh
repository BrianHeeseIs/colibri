#!/usr/bin/env bash
# COLD vs WARM at fixed --ram, under a held-low memory-pressure gate.
#
# PURPOSE. Today's --ram 96 run climbed 0.1557 -> 0.3478 tok/s, but TWO variables moved
# together: system memory pressure fell AND the expert cache warmed. That run cannot
# attribute the gain. This script holds pressure low throughout and varies ONLY cache
# warmth: one cold pass on a fresh engine, then an immediate second pass on the SAME
# process with the cache already populated.
#
# READ THIS BEFORE TRUSTING ANY NUMBER: `ps` RSS excludes compressed pages on macOS and
# misled this investigation twice. Every row records phys_footprint (retention) AND rss
# (residency); the gap between them is the compression penalty. A pass whose gate fails
# is a pressure test, not a cache test - Oracle's protocol says discard it.
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
RAM=${V4RAM:-96}; NTOK=${NTOK:-128}
GATE_COMPRESSOR_GB=${GATE_COMPRESSOR_GB:-20}
cd "$BASE/validation/dsv4"


# ENGINE PROVENANCE GUARD (added 2026-08-12, RESULTS.md S10).
# bin/coli and c/coli resolve to DIFFERENT binaries: engine_for() (c/coli:249) prefers
# HERE/deepseek_v4 over libexec/. Running a feature test against a stale libexec build
# produces a false negative that looks like a real result. Print what we are about to run.
_ENG="$BASE/libexec/colibri/deepseek_v4"
[ -x "$BASE/bin/deepseek_v4" ] && _ENG="$BASE/bin/deepseek_v4"
echo "  ENGINE: $_ENG  (built $(stat -f '%Sm' -t '%m-%d %H:%M' "$_ENG" 2>/dev/null || echo '?'))"
echo "  NOTE  : c/deepseek_v4 is $(stat -f '%Sm' -t '%m-%d %H:%M' "$BASE/c/deepseek_v4" 2>/dev/null || echo absent) -- if newer, this harness is testing OLD code"

fp_gb(){ local p=$1 v
  v=$(footprint -p "$p" 2>/dev/null | grep -m1 'phys_footprint:' | grep -oE '[0-9.]+ [KMG]B')
  case "$v" in *GB) echo "${v% GB}";; *MB) awk -v m="${v% MB}" 'BEGIN{printf "%.1f",m/1024}';; *) echo 0;; esac; }
rss_gb(){ ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}'; }
comp_gb(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }

# Returns 0 when the machine is quiet enough for the measurement to mean anything.
gate(){ local c; c=$(comp_gb)
  awk -v c="$c" -v lim="$GATE_COMPRESSOR_GB" 'BEGIN{exit !(c<lim)}'; }

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

run_pass(){                      # $1=label  $2=csv
  local label=$1 csv=$2 n=0 P S E W T PT R PID FP RSS GAP C
  echo "n,ts,wall_s,tokens,tok_s,prompt_tok,footprint_gb,rss_gb,gap_gb,compressor_gb,gate,pass" > "$csv"
  for P in "$P1" "$P2" "$P3" "$P4"; do
    n=$((n+1))
    C=$(comp_gb); if gate; then G=ok; else G=FAIL; fi
    [ "$G" = FAIL ] && echo "  !! gate FAIL before $label #$n (compressor ${C}GB) - row will be marked"
    S=$(date +%s)
    BODY=$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.stdin.read()}],'max_tokens':$NTOK,'temperature':0}))" <<< "$P")
    curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
      -d "$BODY" --max-time 3000 -o /tmp/cw_${label}_$n.json
    E=$(date +%s); W=$((E-S))
    T=$(python3 -c "import json;print(json.load(open('/tmp/cw_${label}_$n.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
    PT=$(python3 -c "import json;print(json.load(open('/tmp/cw_${label}_$n.json')).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null||echo 0)
    R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
    PID=$(pgrep -f 'libexec/colibri/deepseek_v4'|head -1)
    FP=$(fp_gb "$PID"); RSS=$(rss_gb "$PID")
    GAP=$(awk -v a="$FP" -v b="$RSS" 'BEGIN{printf "%.1f",a-b}')
    echo "$n,$(date +%H:%M:%S),$W,$T,$R,$PT,$FP,$RSS,$GAP,$C,$G,$label" >> "$csv"
    printf "  %-4s #%d  %ss  %s tok  %s tok/s  fp=%sGB rss=%sGB gap=%sGB comp=%sGB %s\n" \
           "$label" "$n" "$W" "$T" "$R" "$FP" "$RSS" "$GAP" "$C" "$G"
  done
}

echo "=== COLD vs WARM @ --ram $RAM, $NTOK tok/prompt ==="
echo "--- tearing down for a genuinely cold start ---"
pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
pkill -f 'libexec/colibri/deepseek_v4' 2>/dev/null
tmux send-keys -t colibri-lab:0.0 C-c 2>/dev/null
for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done
sleep 4
echo "  pre-start compressor: $(comp_gb) GB (gate limit ${GATE_COMPRESSOR_GB})"
tmux send-keys -t colibri-lab:0.0 "clear && env V4_DRAFT=4 V4_NGRAM=1 COLI_MODEL=$MODEL $BASE/bin/coli serve --ram $RAM --port 8090" Enter
MID=""
for i in $(seq 1 90); do
  sleep 5
  MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MID" ] && break
done
[ -n "$MID" ] || { echo "SERVER FAILED"; ./notify.sh "Cold/warm test failed to start"; exit 1; }
echo "server ready ($MID)"

# CSV names MUST carry the RAM budget: they were hardcoded, so a second arm at a different
# --ram silently overwrote the first arm's results. Never again.
COLD_CSV="coldwarm_cold_ram${RAM}.csv"
WARM_CSV="coldwarm_warm_ram${RAM}.csv"
for f in "$COLD_CSV" "$WARM_CSV"; do
  [ -f "$f" ] && { mv "$f" "$f.$(date +%H%M%S).bak"; echo "  preserved existing $f as a .bak"; }
done
run_pass cold "$COLD_CSV"
echo "--- second pass on the SAME process: cache now warm, engine NOT restarted ---"
run_pass warm "$WARM_CSV"

echo "=== DONE cold vs warm @ --ram $RAM ==="
python3 - <<'PY'
import csv,os
import glob
def rows(f):
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []
ram=os.environ.get('V4RAM','96')
c,w=rows(f'coldwarm_cold_ram{ram}.csv'),rows(f'coldwarm_warm_ram{ram}.csv')
def mean(r,k='tok_s'):
    v=[float(x[k]) for x in r if float(x[k])>0]
    return sum(v)/len(v) if v else 0
if c and w:
    print(f"\n  --ram {ram}:  cold mean {mean(c):.4f} tok/s   warm mean {mean(w):.4f} tok/s")
    if mean(c): print(f"  warm vs cold: {(mean(w)-mean(c))/mean(c)*100:+.1f}%")
    print("  paired per-prompt:")
    for a,b in zip(c,w):
        d=(float(b['tok_s'])-float(a['tok_s']))/max(1e-9,float(a['tok_s']))*100
        print(f"    #{a['n']}  cold {float(a['tok_s']):.4f} -> warm {float(b['tok_s']):.4f}  {d:+.1f}%")
    bad=[x for x in c+w if x['gate']!='ok']
    print(f"  gate failures: {len(bad)} (any >0 => that pass is a pressure test, discard per protocol)")
# cross-arm comparison whenever two budgets exist
arms={}
for f in glob.glob('coldwarm_warm_ram*.csv'):
    r=rows(f)
    if r: arms[f.split('ram')[-1].replace('.csv','')]=mean(r)
if len(arms)>1:
    print("\n  === CROSS-ARM (warm steady state) ===")
    for k in sorted(arms,key=lambda x:int(x)): print(f"    --ram {k}: {arms[k]:.4f} tok/s")
    lo,hi=sorted(arms,key=lambda x:int(x))[0],sorted(arms,key=lambda x:int(x))[-1]
    if arms[lo]: print(f"    --ram {hi} vs {lo}: {(arms[hi]-arms[lo])/arms[lo]*100:+.1f}%")
PY
./notify.sh "Cold vs warm test at RAM $RAM complete"
