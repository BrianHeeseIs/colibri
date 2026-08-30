#!/bin/bash
# E130 Wave 4 / T10: both golden gates against the decode-trace build.
# Run from tmux colibri-lab:0.0. Writes durable logs and touches a completion marker
# as its final action, so the caller polls the marker in a LATER turn rather than
# blocking a tool call that would time out and silently abandon the work.
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri
TS=$(date +%Y%m%d-%H%M%S)
MARK=.backlog/lab/E130_GOLDENS_DONE
rm -f "$MARK"

echo "seed md5: $(md5 -q .backlog/lab/coli_usage.snapshot)  (expect 599f3d12e9347ef30541bd6f9ba18bde)"
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot || { echo "FATAL: seed sync failed"; }
pgrep -f '[d]eepseek_v4' >/dev/null && echo "WARNING: an engine is already running"

echo "== golden.sh  (sacred, expect 5d04890413ff539e802985ce8c727814) =="
bash bench/golden.sh ./c/deepseek_v4 > ".backlog/lab/e130_golden_$TS.log" 2>&1
echo "golden_rc=$?"
tail -6 ".backlog/lab/e130_golden_$TS.log"

echo "== golden_default.sh  (shipping, expect cc09015d089d9a25d10d75753f9e849a) =="
bash bench/golden_default.sh ./c/deepseek_v4 > ".backlog/lab/e130_golden_default_$TS.log" 2>&1
echo "golden_default_rc=$?"
tail -6 ".backlog/lab/e130_golden_default_$TS.log"

echo "logs: .backlog/lab/e130_golden_$TS.log .backlog/lab/e130_golden_default_$TS.log"
echo "E130_GOLDENS_DONE" | tee "$MARK"
