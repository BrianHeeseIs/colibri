#!/usr/bin/env bash
# ============================================================================
# STOCK vs RECOMMENDED - the comparison nobody had ever run in one session.
#
#   ARM A "stock"  : --ram 48, V4_DRAFT=0  (speculation OFF; V4_NGRAM is inert
#                    unless V4_DRAFT>0, so this is genuinely unaccelerated)
#   ARM B "reco"   : --ram 96, V4_DRAFT=4 V4_NGRAM=1
#
# WHY THIS EXISTS. RESULTS.md quotes n-gram at +18.5% and --ram 48->96 at
# +33.8% cold / +52.5% warm, and those compose to a MODELLED +80.7% warm.
# But the two deltas came from DIFFERENT workloads and different sessions, and
# S11 documents absolute tok/s swinging 22% on ambient load alone. The
# composition has never been measured end to end. This measures it.
#
# ORDER = A,B,B,A (protocol step 5: never all-A-then-all-B). Each arm is a
# fresh engine (the plan is fixed at open) running coldwarm2.sh, which gates
# the compressor BEFORE AND AFTER every prompt.
#
# KNOWN RISK, STATED UP FRONT: S11's working rule is engine+non-engine <= ~100 GB
# on this 137 GB host. --ram 96 needs ~94 GB, so arm B is only clean if
# non-engine stays under ~10-15 GB. If B gate-fails, that is not a wasted run -
# it is a measurement that the recommended config is unreachable under this
# host's realistic load, which is exactly what S4d/S11 predict.
# ============================================================================
set -uo pipefail
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
NTOK=${NTOK:-64}
OUT=/Users/cptn/workbench/ai/colibri/validation/dsv4/ab_stock_reco.log

comp(){ vm_stat | awk '/occupied by compressor/{gsub(/\./,"",$5); printf "%.1f",$5*16384/1e9}'; }
freeg(){ vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%.1f",$3*16384/1e9}'; }
nonengine(){ ps -Ao rss=,comm= | grep -v deepseek_v4 | awk '{s+=$1} END{printf "%.1f", s*1024/1e9}'; }

banner(){ echo; echo "############################################################"; echo "# $*"; echo "#   $(date '+%H:%M:%S')  compressor=$(comp)GB free=$(freeg)GB non-engine=$(nonengine)GB"; echo "############################################################"; }

banner "STOCK vs RECOMMENDED  ABBA  NTOK=$NTOK  START"
echo "engine under test: $(md5 -q /Users/cptn/workbench/ai/colibri/libexec/colibri/deepseek_v4 | cut -c1-8)"

MODELDIR=/Users/cptn/workbench/ai/colibri/models/deepseek-v4-flash
HIST=$MODELDIR/.coli_usage
SNAP=/tmp/coli_usage.snapshot
SNAP_MD5=$(md5 -q "$SNAP")

run_arm(){ # tag ram draft ngram label
  local tag=$1 ram=$2 draft=$3 ngram=$4 label=$5
  # Restore the frozen seed history before EVERY arm, and verify it. coldwarm2.sh runs with
  # COLI_V4_SAVE_USAGE=0 so nothing should mutate it - this asserts that rather than trusting it.
  cp "$SNAP" "$HIST"
  local now; now=$(md5 -q "$HIST")
  if [ "$now" != "$SNAP_MD5" ]; then echo "  !! HISTORY RESTORE FAILED ($now != $SNAP_MD5)"; exit 1; fi
  echo "  seed history restored + verified: $SNAP_MD5 ($(stat -f %z "$HIST") bytes)"
  banner "ARM $label : --ram $ram  V4_DRAFT=$draft V4_NGRAM=$ngram"
  V4RAM=$ram NTOK=$NTOK V4DRAFT=$draft V4NGRAM=$ngram TAG=$tag PREFETCH=0 ./coldwarm2.sh 2>&1
  local after; after=$(md5 -q "$HIST")
  if [ "$after" != "$SNAP_MD5" ]; then
    echo "  !! WARNING: history MUTATED during arm $label ($after) - later arms are confounded"
  else echo "  history unchanged after arm ($SNAP_MD5) - freeze held"; fi
  echo "### arm $label finished $(date '+%H:%M:%S')"
}

run_arm _stockA1 48 0 0 "A1 stock"
run_arm _recoB1  96 4 1 "B1 reco"
run_arm _recoB2  96 4 1 "B2 reco"
run_arm _stockA2 48 0 0 "A2 stock"

banner "ALL ARMS COMPLETE - analysing"
python3 /Users/cptn/workbench/ai/colibri/validation/dsv4/ab_stock_reco_report.py
