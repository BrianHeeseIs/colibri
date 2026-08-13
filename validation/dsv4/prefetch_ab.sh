#!/usr/bin/env bash
# Prefetch A/B: COLI_V4_EXPERT_PREFETCH 0 vs 1, back-to-back, at a budget that FITS.
#
# WHY --ram 64 AND NOT 96. The first attempt ran prefetch-ON at --ram 96 against the earlier
# --ram 96 baseline and had to be aborted: a ~94 GB engine plus ~18 GB of desktop load exceeds
# what 137 GB holds uncompressed, so macOS compressed 41 GB of the engine mid-run (RESULTS.md
# S4d). At --ram 64 the engine needs ~64 GB, leaving ~55 GB for everything else - it should stay
# resident even with Chrome/QEMU running.
#
# WHY BOTH ARMS HERE instead of reusing an old baseline: S4c's --ram 48 vs 96 comparison is
# caveated because the arms ran ~90 min apart (protocol step 5: never all-A-then-all-B). Running
# OFF then ON back-to-back on the same machine state removes that objection for this question.
#
# Each arm is a full coldwarm2.sh run (cold pass + warm pass, 4 prompts each) with the FIXED
# gate that samples the compressor before AND after every prompt.
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
RAM=${V4RAM:-64}

comp(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }

echo "############ PREFETCH A/B @ --ram ${RAM} ############"
echo "  start $(date '+%H:%M:%S')  compressor=$(comp) GB  free=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%.1f",$3*16384/1e9}') GB"

echo; echo "===== ARM A: prefetch OFF ====="
PREFETCH=0 TAG=_pf0 V4RAM=$RAM NTOK=128 ./coldwarm2.sh 2>&1 | tee "prefetch_ab_off_ram${RAM}.log"

echo; echo "===== ARM B: prefetch ON ====="
PREFETCH=1 TAG=_pf1 V4RAM=$RAM NTOK=128 ./coldwarm2.sh 2>&1 | tee "prefetch_ab_on_ram${RAM}.log"

echo; echo "############ PREFETCH A/B COMPLETE $(date '+%H:%M:%S') ############"
RAM=$RAM python3 - <<'PY'
import csv,os
ram=os.environ.get('RAM','64')
def rows(f): return list(csv.DictReader(open(f))) if os.path.exists(f) else []
def m(r):
    v=[float(x['tok_s']) for x in r if float(x['tok_s'])>0]
    return sum(v)/len(v) if v else 0
def bad(r): return sum(1 for x in r if x['gate']!='ok')
print(f"\n  === PREFETCH EFFECT @ --ram {ram} (back-to-back, one variable) ===")
ok=True
for kind in ('cold','warm'):
    a=rows(f'coldwarm_{kind}_ram{ram}_pf0.csv'); b=rows(f'coldwarm_{kind}_ram{ram}_pf1.csv')
    if not a or not b: continue
    print(f"  --- {kind} ---")
    print("   #    OFF       ON        delta")
    for x,y in zip(a,b):
        p,q=float(x['tok_s']),float(y['tok_s'])
        flag='' if x['gate']=='ok' and y['gate']=='ok' else '  <-- GATE FAIL, DISCARD'
        print(f"   {x['n']}   {p:.4f}    {q:.4f}    {(q-p)/p*100:+6.1f}%{flag}")
    print(f"  mean  {m(a):.4f}    {m(b):.4f}    {(m(b)-m(a))/m(a)*100:+6.1f}%")
    nb=bad(a)+bad(b)
    print(f"  gate failures: OFF={bad(a)} ON={bad(b)}")
    if nb: ok=False
print("\n  VERDICT VALID" if ok else "\n  VERDICT INVALID - discard failed rows and re-run")
print("  NOTE: n=4 per pass. Directional.")
PY
./notify.sh "Prefetch A/B complete"
