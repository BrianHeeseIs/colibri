#!/usr/bin/env bash
# Wait for the in-flight coldwarm run to finish, then run the next arm.
#
# WHY A QUEUE: each coldwarm arm is ~55 min (cold pass + warm pass). Chaining avoids dead
# machine time between arms and keeps both arms close together in wall-clock, which limits
# how much the ambient memory state can drift between them.
#
# GATE DISCIPLINE: this does NOT bypass the pressure gate. coldwarm.sh still checks the
# compressor before every prompt and stamps ok/FAIL per row. If the machine gets busy the
# rows are marked, not silently accepted - discard them per the protocol in RESULTS.md S4.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
NEXT_RAM=${NEXT_RAM:-96}

echo "=== queue: waiting for the current coldwarm arm to finish ==="
while pgrep -f coldwarm.sh >/dev/null 2>&1; do sleep 30; done
echo "=== queue: previous arm done at $(date '+%H:%M:%S') ==="
sleep 10

C=$(vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}')
echo "=== queue: compressor before next arm = ${C} GB (coldwarm.sh tears down first, which reclaims the engine's own pages) ==="
echo "=== queue: starting --ram ${NEXT_RAM} cold/warm at $(date '+%H:%M:%S') ==="
V4RAM=$NEXT_RAM NTOK=128 ./coldwarm.sh 2>&1 | tee "coldwarm${NEXT_RAM}_rerun.log"

echo "=== queue: ALL ARMS COMPLETE $(date '+%H:%M:%S') ==="
python3 - <<'PY'
import csv,os,glob
def rows(f): return list(csv.DictReader(open(f))) if os.path.exists(f) else []
def mean(r):
    v=[float(x['tok_s']) for x in r if float(x['tok_s'])>0]
    return sum(v)/len(v) if v else 0
print("\n  === CROSS-ARM SUMMARY (all completed arms) ===")
for kind in ('cold','warm'):
    print(f"  {kind}:")
    for f in sorted(glob.glob(f'coldwarm_{kind}_ram*.csv')):
        r=rows(f); bad=[x for x in r if x['gate']!='ok']
        print(f"    {f:34s} n={len(r)}  mean {mean(r):.4f} tok/s  gate-fail={len(bad)}")
w={f.split('ram')[-1].replace('.csv',''):mean(rows(f)) for f in glob.glob('coldwarm_warm_ram*.csv')}
w={k:v for k,v in w.items() if v>0}
if len(w)>1:
    ks=sorted(w,key=lambda x:int(x)); lo,hi=ks[0],ks[-1]
    print(f"\n  VERDICT (warm steady state): --ram {hi} vs --ram {lo}: {(w[hi]-w[lo])/w[lo]*100:+.1f}%")
    print("  NOTE: n=4 per arm. Treat as directional, not definitive.")
PY
./notify.sh "All coldwarm arms complete"
