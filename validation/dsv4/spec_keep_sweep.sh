#!/usr/bin/env bash
# M1 spec_keep: does V4_NGRAM_PARTIAL_KEEP=1 (+ draft-depth tuning) beat the self-disabling
# default? Paired, frozen history, gate on compressor. Metal OFF throughout.
set -u
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
SNAP=/tmp/coli_usage.snapshot
NTOK=${NTOK:-64}
MEM=${MEM:-96}
OUT=${OUT:-$BASE/.backlog/spec_keep_sweep.csv}
cd "$BASE"
comp(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }

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

echo "arm,draft,keep,prompt,wall_s,tokens,tok_s,attempts,drafted,accepted,acc_pct,disabled,text_sha,comp_gb,gate" > "$OUT"
run_arm(){ # $1=draft $2=keep $3=label
  local D=$1 K=$2 L=$3 n=0
  for P in "$P1" "$P2" "$P3" "$P4"; do
    n=$((n+1))
    cp "$SNAP" "$MODEL/.coli_usage"
    local C0=$(comp) S=$(date +%s)
    local OUTTXT
    OUTTXT=$(timeout 2400 env V4_DRAFT=$D V4_NGRAM=1 V4_NGRAM_PARTIAL_KEEP=$K \
      COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 "$MODEL" "$P" --max-tokens $NTOK --memory-gb $MEM 2>&1)
    local E=$(date +%s) W=$((E-S)) C1=$(comp)
    local T=$(echo "$OUTTXT" | grep -oE 'generated=[0-9]+' | head -1 | cut -d= -f2); T=${T:-0}
    local SP=$(echo "$OUTTXT" | grep -oE 'attempts=[0-9]+ drafted=[0-9]+ accepted=[0-9]+ acceptance=[0-9.]+%? adaptive_disabled=[0-9]+' | tail -1)
    local AT=$(echo "$SP"|grep -oE 'attempts=[0-9]+'|cut -d= -f2); AT=${AT:-0}
    local DR=$(echo "$SP"|grep -oE 'drafted=[0-9]+'|cut -d= -f2); DR=${DR:-0}
    local AC=$(echo "$SP"|grep -oE 'accepted=[0-9]+'|cut -d= -f2); AC=${AC:-0}
    local AP=$(echo "$SP"|grep -oE 'acceptance=[0-9.]+'|cut -d= -f2); AP=${AP:-0}
    local DIS=$(echo "$SP"|grep -oE 'adaptive_disabled=[0-9]+'|cut -d= -f2); DIS=${DIS:-NA}
    local SHA=$(echo "$OUTTXT" | grep -m1 'generated_text=' | shasum -a256 | cut -c1-12)
    local R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
    local CMAX=$(awk -v a="$C0" -v b="$C1" 'BEGIN{print (a>b)?a:b}')
    local G=ok; awk -v c="$CMAX" 'BEGIN{exit !(c<20)}' || G=GATE_FAIL
    echo "$L,$D,$K,p$n,$W,$T,$R,$AT,$DR,$AC,$AP,$DIS,$SHA,$CMAX,$G" | tee -a "$OUT"
  done
}
echo "### baseline: no speculation"
run_arm 0 0 base
echo "### default behavior (self-disabling)"
run_arm 4 0 d4k0
echo "### the M1 hypothesis"
run_arm 4 1 d4k1
echo "### depth sweep with KEEP"
run_arm 2 1 d2k1
run_arm 8 1 d8k1
cp "$SNAP" "$MODEL/.coli_usage"
echo "### done -> $OUT"
