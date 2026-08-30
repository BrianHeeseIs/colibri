#!/bin/bash
# merged-tree golden gates (E129 merge verification, V2). Generated 20260830-171749
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri
TS=20260830-174206
MARK=.backlog/lab/GOLDEN_MERGED_DONE_$TS
rm -f "$MARK"
echo "seed md5: $(md5 -q .backlog/lab/coli_usage.snapshot)"
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
echo "== golden.sh (expect 5d04890413ff539e802985ce8c727814) =="
bash bench/golden.sh ./c/deepseek_v4 > .backlog/lab/golden_merged_$TS.log 2>&1
echo "golden_rc=$?"
grep -E 'PASS|FAIL' .backlog/lab/golden_merged_$TS.log | tail -2
echo "== golden_default.sh (expect cc09015d089d9a25d10d75753f9e849a) =="
bash bench/golden_default.sh ./c/deepseek_v4 > .backlog/lab/golden_default_merged_$TS.log 2>&1
echo "golden_default_rc=$?"
grep -E 'PASS|FAIL' .backlog/lab/golden_default_merged_$TS.log | tail -2
echo "ALL_GOLDENS_DONE" | tee "$MARK"
