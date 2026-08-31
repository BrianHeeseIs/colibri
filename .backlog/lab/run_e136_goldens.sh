#!/usr/bin/env bash
# E136 golden gates for the hot-pack default flip.
#
# golden.sh sets COLI_V4_BASELINE=1, which must still force the pack policy to locked, so the
# SACRED md5 5d04890413ff539e802985ce8c727814 must be unchanged. If it is not, the baseline guard
# is broken and the gate's arm 2 should have caught it first -- stop and debug, never re-record.
#
# golden_default.sh exercises the SHIPPING path, which now runs with the flag ON for the first
# time. E132 measured all seven p256 arms byte-identical with the flag on and off, so no change is
# expected. GOLDEN_DEFAULT_MD5 is not sacred and may be re-recorded, but only deliberately and
# only after the full protocol: diff the text, run taskcheck, and repeat the arm against itself.
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri
TS=$(date +%Y%m%d-%H%M%S)
MARK=.backlog/lab/E136_GOLDEN_DONE
rm -f "$MARK"

echo "binary md5: $(md5 -q c/deepseek_v4)"
echo "expected sacred:   5d04890413ff539e802985ce8c727814"
echo "expected shipping: $(cat bench/GOLDEN_DEFAULT_MD5 2>/dev/null | tr -d '[:space:]')"
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot

echo "== golden.sh (BASELINE=1; pack must be locked) =="
bash bench/golden.sh ./c/deepseek_v4 > ".backlog/lab/e136_golden_$TS.log" 2>&1
echo "golden_rc=$?"
grep -E 'PASS|FAIL|already running' ".backlog/lab/e136_golden_$TS.log" | tail -3
grep -m1 'v4_hot_policy' ".backlog/lab/e136_golden_$TS.log" || true

echo "== golden_default.sh (shipping; pack now unlocked by default) =="
bash bench/golden_default.sh ./c/deepseek_v4 > ".backlog/lab/e136_golden_default_$TS.log" 2>&1
echo "golden_default_rc=$?"
grep -E 'PASS|FAIL|already running' ".backlog/lab/e136_golden_default_$TS.log" | tail -3
grep -m1 'v4_hot_policy' ".backlog/lab/e136_golden_default_$TS.log" || true

echo "logs: .backlog/lab/e136_golden_$TS.log .backlog/lab/e136_golden_default_$TS.log"
echo "E136_GOLDEN_DONE" | tee "$MARK"
