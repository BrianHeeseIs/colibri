#!/usr/bin/env bash
# COLI_V4_PREWARM 0 vs 1, back-to-back, cold start, at a budget that FITS.
#
# WHAT THIS TESTS. COLI_V4_PREWARM=1 eagerly loads the pinned experts from .coli_usage at
# startup (deepseek_v4.c:6092, gated on policy->history_seeded). It targets exactly the
# cold-start penalty measured in RESULTS.md S4b: +33.2% on prompt 1, +17.6% over a 4-prompt
# pass. Until the per-turn flush landed (commit 8c17016) this branch was unreachable, because
# history_total never crossed 5000 across restarts. It is reachable now: 14190 selections.
#
# WHY --ram 80 AND NOT 96. The approved budget was 96, but desktop load grew to ~31 GB
# (a 12 GB colima/QEMU VM, two opencode processes, Chrome), so 96 computes to
# 100 + 31 + 8 = 139 GB against 137 physical and WOULD compress -- the same contamination
# that forced the prefetch arm to be aborted. 80 gives 84 + 31 + 8 = 124 GB, a 13 GB margin,
# and stays above the 64-96 retention threshold where the warm effect is real.
#
# THE CONFOUND THIS AVOIDS. Both arms must seed from the SAME history. If arm A were allowed
# to write, arm B would start from a larger .coli_usage than arm A did, and any difference
# would be partly "more history" rather than "prewarm". So: snapshot once, restore before
# each arm, and run both with COLI_V4_SAVE_USAGE=0 so neither mutates it.
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
HIST=$MODEL/.coli_usage
SNAP=/tmp/coli_usage.snapshot
RAM=${V4RAM:-80}
NTOK=${NTOK:-128}
GATE=${GATE_COMPRESSOR_GB:-20}
cd "$BASE"

comp(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }
freeg(){ vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%.1f",$3*16384/1e9}'; }
fp_gb(){ local v i; for i in 1 2 3; do
    v=$(footprint -p "$1" 2>/dev/null | grep -m1 'phys_footprint:' | grep -oE '[0-9.]+ [KMG]B')
    case "$v" in *GB) echo "${v% GB}"; return;; *MB) awk -v m="${v% MB}" 'BEGIN{printf "%.1f",m/1024}'; return;; esac
    sleep 1; done; echo NA; }
rss_gb(){ local v i; for i in 1 2 3; do
    v=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' ')
    [ -n "$v" ] && [ "$v" -gt 0 ] 2>/dev/null && { awk -v r="$v" 'BEGIN{printf "%.1f",r/1048576}'; return; }
    sleep 1; done; echo NA; }

cleanup(){ pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
           pkill -f '[d]eepseek_v4' 2>/dev/null
           for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done; }

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

arm(){                      # $1 = PREWARM value  $2 = csv
  local PW=$1 csv=$2 n=0 P S E W T PT R PID FP RSS GAP C C_END C_MAX G
  echo; echo "===== ARM: COLI_V4_PREWARM=$PW ====="
  cleanup
  cp "$SNAP" "$HIST"                      # identical seed history for every arm
  echo "  restored history: $(stat -f '%z bytes' "$HIST")"
  echo "  pre-start compressor: $(comp) GB  free: $(freeg) GB"
  # env MUST be inline in send-keys: the server runs in another tmux pane and would not
  # inherit exported vars. COLI_V4_SAVE_USAGE=0 freezes the history for the whole arm.
  tmux send-keys -t colibri-lab:0.0 "clear && env V4_DRAFT=4 V4_NGRAM=1 COLI_V4_PREWARM=$PW COLI_V4_SAVE_USAGE=0 COLI_MODEL=$MODEL python3 $BASE/c/coli serve --ram $RAM --port 8090" Enter
  MID=""
  for i in $(seq 1 120); do
    sleep 5
    MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null \
          | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$MID" ] && break
  done
  [ -n "$MID" ] || { echo "  ARM FAILED TO START"; return 1; }
  echo "  server ready ($MID)"
  echo "n,ts,wall_s,tokens,tok_s,prompt_tok,footprint_gb,rss_gb,gap_gb,comp_start_gb,comp_end_gb,gate,prewarm" > "$csv"
  for P in "$P1" "$P2" "$P3" "$P4"; do
    n=$((n+1)); C=$(comp); S=$(date +%s)
    BODY=$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.stdin.read()}],'max_tokens':$NTOK,'temperature':0}))" <<< "$P")
    curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
      -d "$BODY" --max-time 3000 -o /tmp/pw_${PW}_$n.json
    E=$(date +%s); W=$((E-S)); C_END=$(comp)
    C_MAX=$(awk -v a="$C" -v b="$C_END" 'BEGIN{print (a>b)?a:b}')
    if awk -v c="$C_MAX" -v l="$GATE" 'BEGIN{exit !(c<l)}'; then G=ok; else G=FAIL; fi
    [ "$G" = FAIL ] && echo "  !! gate FAIL on #$n (compressor ${C}->${C_END}GB) - DISCARD"
    T=$(python3 -c "import json;print(json.load(open('/tmp/pw_${PW}_$n.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
    PT=$(python3 -c "import json;print(json.load(open('/tmp/pw_${PW}_$n.json')).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null||echo 0)
    R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
    PID=$(pgrep -f '[d]eepseek_v4'|head -1); FP=$(fp_gb "$PID"); RSS=$(rss_gb "$PID")
    if [ "$FP" = NA ] || [ "$RSS" = NA ]; then GAP=NA
    else GAP=$(awk -v a="$FP" -v b="$RSS" 'BEGIN{printf "%.1f",a-b}'); fi
    echo "$n,$(date +%H:%M:%S),$W,$T,$R,$PT,$FP,$RSS,$GAP,$C,$C_END,$G,$PW" >> "$csv"
    printf "  pw=%s #%d  %ss  %s tok  %s tok/s  fp=%sGB gap=%sGB comp=%s->%sGB %s\n" \
           "$PW" "$n" "$W" "$T" "$R" "$FP" "$GAP" "$C" "$C_END" "$G"
  done
  cleanup
}

echo "############ COLI_V4_PREWARM A/B @ --ram $RAM ############"
[ -f "$HIST" ] || { echo "  NO .coli_usage -- prewarm cannot fire, aborting"; exit 1; }
cp "$HIST" "$SNAP"
echo "  seed history snapshot: $(stat -f '%z bytes' "$SNAP")  (must be >= 5000 selections to seed)"
echo "  engine: $BASE/c/deepseek_v4 (built $(stat -f '%Sm' -t '%m-%d\ %H:%M' $BASE/c/deepseek_v4))"

arm 0 validation/dsv4/prewarm_off_ram${RAM}.csv
arm 1 validation/dsv4/prewarm_on_ram${RAM}.csv
cp "$SNAP" "$HIST"    # leave the history as we found it

echo; echo "############ RESULT ############"
RAM=$RAM python3 - <<'PY'
import csv,os
ram=os.environ.get('RAM','80')
def rows(f): return list(csv.DictReader(open(f))) if os.path.exists(f) else []
def m(r):
    v=[float(x['tok_s']) for x in r if float(x['tok_s'])>0]
    return sum(v)/len(v) if v else 0
a,b=rows(f'validation/dsv4/prewarm_off_ram{ram}.csv'),rows(f'validation/dsv4/prewarm_on_ram{ram}.csv')
if a and b:
    print("   #     OFF       ON       delta")
    for x,y in zip(a,b):
        p,q=float(x['tok_s']),float(y['tok_s'])
        flag='' if x['gate']=='ok' and y['gate']=='ok' else '  <-- GATE FAIL'
        print(f"   {x['n']}    {p:.4f}    {q:.4f}   {(q-p)/p*100:+6.1f}%{flag}")
    print(f"  mean  {m(a):.4f}    {m(b):.4f}   {(m(b)-m(a))/m(a)*100:+6.1f}%")
    bad=sum(1 for x in a+b if x['gate']!='ok')
    print(f"  gate failures: {bad}")
    print(f"  prompt #1 is the cold-start row -- S4b measured a +33.2% warm advantage there,")
    print(f"  which is what prewarm is supposed to recover.")
PY
./notify.sh "PREWARM A/B complete"
