#!/usr/bin/env bash
# Live paired view of the PREWARM A/B. Shows each arm's rows and, once both have a row at
# the same index, the paired delta -- which is the only number that matters here, because
# absolute tok/s is not comparable across sessions (ambient load moved 34% tonight).
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
RAM=${V4RAM:-64}
while :; do
  OFF="prewarm_off_ram${RAM}.csv"; ON="prewarm_on_ram${RAM}.csv"
  clear
  printf '\033[1m  PREWARM A/B  --ram %s\033[0m                 %s\n' "$RAM" "$(date '+%H:%M:%S')"
  printf '  ═══════════════════════════════════════════════════════\n'
  H=$(pgrep -f prewarm_ab.sh | head -1)
  if [ -n "$H" ]; then
    printf '  \033[32mRUNNING\033[0m  arm: %s\n' \
      "$(grep -oE '===== ARM: COLI_V4_PREWARM=[01] =====' prewarm_ab${RAM}.log 2>/dev/null | tail -1 | grep -oE '[01] ' | tr -d ' ')"
  else
    printf '  \033[33mIDLE\033[0m\n'
  fi
  printf '  ───────────────────────────────────────────────────────\n'
  printf '   #   PREWARM=0    PREWARM=1     delta\n'
  python3 - "$OFF" "$ON" <<'PY'
import csv,os,sys
def rows(f): return list(csv.DictReader(open(f))) if os.path.exists(f) else []
a,b=rows(sys.argv[1]),rows(sys.argv[2])
for i in range(4):
    x=a[i] if i<len(a) else None
    y=b[i] if i<len(b) else None
    p=f"{float(x['tok_s']):.4f}" if x else "  --  "
    q=f"{float(y['tok_s']):.4f}" if y else "  --  "
    d=""
    if x and y and float(x['tok_s'])>0:
        pct=(float(y['tok_s'])-float(x['tok_s']))/float(x['tok_s'])*100
        d=f"{pct:+7.1f}%"
        if x['gate']!='ok' or y['gate']!='ok': d+="  GATE FAIL"
    print(f"   {i+1}    {p}      {q}     {d}")
def m(r):
    v=[float(z['tok_s']) for z in r if float(z['tok_s'])>0]
    return sum(v)/len(v) if v else 0
if a and b and m(a):
    print(f"  mean  {m(a):.4f}      {m(b):.4f}     {(m(b)-m(a))/m(a)*100:+7.1f}%")
bad=sum(1 for z in a+b if z.get('gate')!='ok')
print(f"\n  rows: OFF {len(a)}/4  ON {len(b)}/4   gate failures: {bad}")
print("  prompt #1 is the cold-start row -- the +33.2% penalty prewarm should recover")
PY
  printf '  ───────────────────────────────────────────────────────\n'
  vm_stat | awk -v p=16384 '/Pages free/{gsub(/\./,"",$3);f=$3} /occupied by compressor/{gsub(/\./,"",$5);c=$5} END{
    printf "  free %.1f GB   compressor %.1f GB   %s\n",f*p/1e9,c*p/1e9,(c*p/1e9<20?"GATE OK":"GATE FAIL")}'
  sleep 15
done
