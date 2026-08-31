#!/usr/bin/env bash
# Post-change verification for the three approved fixes:
#   (a) COLI_V4_PREWARM now reports what it did, always
#   (b) v4_prefill_trace added to every strip list  (script-only, no rebuild)
#   (c) omp_head_wall wired to the RESIDENT head path, closing the E130 gap
#
# Both goldens must pass because the binary changed. Then two short trace arms prove the new
# code paths actually execute -- a golden PASS says nothing about whether a new counter fired.
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri
TS=$(date +%Y%m%d-%H%M%S)
MARK=.backlog/lab/E134_VERIFY_DONE
rm -f "$MARK"
MODEL=models/deepseek-v4-flash
DURABLE=.backlog/lab/coli_usage.snapshot

echo "binary md5: $(md5 -q c/deepseek_v4)"
cp "$DURABLE" /tmp/coli_usage.snapshot

echo "== golden.sh (sacred, expect 5d04890413ff539e802985ce8c727814) =="
bash bench/golden.sh ./c/deepseek_v4 > ".backlog/lab/e134_golden_$TS.log" 2>&1
echo "golden_rc=$?"; tail -4 ".backlog/lab/e134_golden_$TS.log"

echo "== golden_default.sh (shipping, expect cc09015d089d9a25d10d75753f9e849a) =="
bash bench/golden_default.sh ./c/deepseek_v4 > ".backlog/lab/e134_golden_default_$TS.log" 2>&1
echo "golden_default_rc=$?"; tail -4 ".backlog/lab/e134_golden_default_$TS.log"

PROMPT=$(<.backlog/prefill_prompts/p256.txt)
for arm in headwall prewarm; do
  cp "$DURABLE" /tmp/coli_usage.snapshot
  cp "$DURABLE" "$MODEL/.coli_usage"
  log=".backlog/lab/e134_${arm}_$TS.log"
  if [[ $arm == prewarm ]]; then EXTRA="COLI_V4_PREWARM=1"; else EXTRA=""; fi
  echo "== arm=$arm env='${EXTRA:-<defaults>}' =="
  # shellcheck disable=SC2086
  env $EXTRA COLI_V4_PROFILE=1 COLI_V4_DECODE_TRACE=1 COLI_V4_SAVE_USAGE=0 \
      ./c/deepseek_v4 "$MODEL" "$PROMPT" --max-tokens 40 --memory-gb 96 > "$log" 2>&1
  echo "  rc=$?  log=$(basename "$log")"
  echo "  (c) omp_head_wall: $(grep -o 'stage=omp_head_wall total_ms=[0-9.]* calls=[0-9]*' "$log" | tail -1)"
  echo "  (a) prewarm line : $(grep -o 'v4_hot_policy prewarm=.*' "$log" | tail -1 || echo '<none>')"
  echo "  misses/wall      : $(grep -o 'stage=store_disk_read .*calls=[0-9]*' "$log" | tail -1)"
done
cp "$DURABLE" "$MODEL/.coli_usage"
echo "E134_VERIFY_DONE" | tee "$MARK"
