#!/usr/bin/env bash
# Wait for the in-flight arm, then test COLI_V4_EXPERT_PREFETCH=1 at --ram 96.
#
# WHY --ram 96: we already have a gate-clean --ram 96 baseline with prefetch OFF
# (cold 0.3375 / warm 0.3968 tok/s, RESULTS.md S4b). Running the prefetch arm at the same
# budget changes exactly ONE variable.
#
# CAVEAT TO CARRY INTO THE WRITE-UP: the baseline came from an earlier session (15:56-16:45),
# not interleaved with this arm. Ambient state can drift between sessions - that is precisely
# what invalidated the original -16.8% result. Treat the comparison as directional and, if
# prefetch looks like a win, re-run both arms back-to-back before claiming it.
#
# COLI_V4_EXPERT_PREFETCH is V4's analogue of GLM's PILOT (router-lookahead prefetch). It is
# OFF by default (deepseek_v4.c:3270) and appears in NO documentation - found only by grepping
# getenv() out of the engine. It has never been measured on this host.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4

echo "=== queue: waiting for the current arm to finish ==="
while pgrep -f 'coldwarm\.sh|coldwarm2\.sh' >/dev/null 2>&1; do sleep 30; done
echo "=== queue: previous arm done at $(date '+%H:%M:%S') ==="
sleep 10

C=$(vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}')
echo "=== queue: compressor before prefetch arm = ${C} GB (teardown reclaims the engine's pages) ==="
echo "=== queue: starting --ram 96 with COLI_V4_EXPERT_PREFETCH=1 at $(date '+%H:%M:%S') ==="
PREFETCH=1 TAG=_pf1 V4RAM=96 NTOK=128 ./coldwarm2.sh 2>&1 | tee coldwarm96_prefetch.log

echo "=== queue: PREFETCH ARM COMPLETE $(date '+%H:%M:%S') ==="
python3 - <<'PY'
import csv,os,glob
def rows(f): return list(csv.DictReader(open(f))) if os.path.exists(f) else []
def mean(r):
    v=[float(x['tok_s']) for x in r if float(x['tok_s'])>0]
    return sum(v)/len(v) if v else 0
print("\n  === ALL ARMS ===")
for kind in ('cold','warm'):
    print(f"  {kind}:")
    for f in sorted(glob.glob(f'coldwarm_{kind}_ram*.csv')):
        r=rows(f); bad=[x for x in r if x['gate']!='ok']
        print(f"    {f:38s} n={len(r)}  mean {mean(r):.4f} tok/s  gate-fail={len(bad)}")
base_c,base_w=mean(rows('coldwarm_cold_ram96.csv')),mean(rows('coldwarm_warm_ram96.csv'))
pf_c,pf_w   =mean(rows('coldwarm_cold_ram96_pf1.csv')),mean(rows('coldwarm_warm_ram96_pf1.csv'))
if base_w and pf_w:
    print("\n  === PREFETCH EFFECT (--ram 96, one variable changed) ===")
    print(f"    cold: {base_c:.4f} -> {pf_c:.4f}  {(pf_c-base_c)/base_c*100:+.1f}%")
    print(f"    warm: {base_w:.4f} -> {pf_w:.4f}  {(pf_w-base_w)/base_w*100:+.1f}%")
    print("    NOTE: n=4/arm, baseline from an earlier session. Directional only.")
PY
./notify.sh "Prefetch arm complete"
