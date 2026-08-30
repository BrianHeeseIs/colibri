#!/usr/bin/env bash
# Known-answer tests for the E131 extractor. Costs ZERO engine time: every assertion runs
# against the E130 logs already committed, whose values were read from disk and are fixed.
#
# Two things are under test.
#   1. The extractor recovers the ten fields E131's gate is computed from. If it silently
#      mis-parses one, the gate would compare a wrong number against a pre-registered
#      threshold and produce a confident wrong verdict.
#   2. verify_applied catches a SILENT CLAMP. COLI_V4_PIN_SLOTS is capped at maximum_pins;
#      if a requested 48 were clamped back to 16, two arms would be identical and the probe
#      would manufacture a false null. The guard must FAIL when the log disagrees with the
#      request, not merely when the log is missing.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

EXTRACT=.backlog/lab/e131_extract.sh
REF=.backlog/lab/decode_trace_on_1_20260830-192200.log

bad=0
check() { # check <label> <expected> <actual>
  if [[ "$2" == "$3" ]]; then
    printf '  ok    %-34s %s\n' "$1" "$3"
  else
    printf '  FAIL  %-34s expected=%s actual=%s\n' "$1" "$2" "$3"
    bad=$((bad+1))
  fi
}

[[ -f $REF ]] || { echo "FATAL: reference log missing: $REF" >&2; exit 2; }
if [[ ! -x $EXTRACT ]]; then
  echo "pinsweep_extract: FAIL extractor not present or not executable: $EXTRACT" >&2
  echo "pinsweep_extract: 1 checks failed" >&2
  exit 1
fi

echo "== KAT: extractor against the committed E130 arm =="
check store_disk_read_calls        765        "$(bash "$EXTRACT" field "$REF" store_disk_read_calls)"
check store_disk_read_ms           2987.867   "$(bash "$EXTRACT" field "$REF" store_disk_read_ms)"
check wait_finish_complete_ms      1488.243   "$(bash "$EXTRACT" field "$REF" wait_finish_complete_ms)"
check finish_slept_calls           723        "$(bash "$EXTRACT" field "$REF" finish_slept_calls)"
check finish_completed_at_entry    9339       "$(bash "$EXTRACT" field "$REF" finish_completed_at_entry)"
check start_slept_calls            0          "$(bash "$EXTRACT" field "$REF" start_slept_calls)"
check expert_forward_ms            7485.917   "$(bash "$EXTRACT" field "$REF" expert_forward_ms)"
check decode_wall_ms               17885.966  "$(bash "$EXTRACT" field "$REF" decode_wall_ms)"
check expert_calls_rows16          4484       "$(bash "$EXTRACT" field "$REF" expert_calls_rows16)"
check expert_calls_scalar          10775      "$(bash "$EXTRACT" field "$REF" expert_calls_scalar)"
check pin_slots_applied            16         "$(bash "$EXTRACT" field "$REF" pin_slots_applied)"

echo "== GUARD: verify_applied must reject a clamped arm =="
bash "$EXTRACT" verify_applied "$REF" 16 >/dev/null 2>&1
check "verify_applied 16 (true)"   0 "$?"
bash "$EXTRACT" verify_applied "$REF" 48 >/dev/null 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  printf '  ok    %-34s rejected as required (rc=%s)\n' "verify_applied 48 (clamped)" "$rc"
else
  printf '  FAIL  %-34s accepted a clamped arm\n' "verify_applied 48 (clamped)"
  bad=$((bad+1))
fi

if (( bad )); then
  echo "pinsweep_extract: $bad checks failed" >&2
  exit 1
fi
echo "pinsweep_extract: all checks passed"
