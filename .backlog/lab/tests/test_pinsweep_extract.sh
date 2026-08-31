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

REF_ON1=.backlog/lab/decode_trace_on_1_20260830-192200.log
REF_A0=.backlog/lab/decode_trace_A0_20260830-223845.log
REF_A2=.backlog/lab/decode_trace_A2_20260830-224050.log

# These are E131 A0's values, so they must be asserted against A0's log, not against $REF
# (which is the E130 on_1 arm). The two arms are the same configuration but different runs,
# so their stage totals differ by run-to-run spread -- which is exactly the 1.84% figure the
# E132 gate is sized against.
echo "== KAT: E132/E133 fields (store_lock is the E132 gate quantity) =="
check store_pack_ms         697.363        "$(bash "$EXTRACT" field "$REF_A0" store_pack_ms)"
check store_pack_calls      3910           "$(bash "$EXTRACT" field "$REF_A0" store_pack_calls)"
check store_lock_ms         471.319        "$(bash "$EXTRACT" field "$REF_A0" store_lock_ms)"
check store_hit_scan_ms     44.877         "$(bash "$EXTRACT" field "$REF_A0" store_hit_scan_ms)"
check packed_slots          902            "$(bash "$EXTRACT" field "$REF_A0" packed_slots)"
check target_slots          164            "$(bash "$EXTRACT" field "$REF_A0" target_slots)"
check head_tier             resident-bf16  "$(bash "$EXTRACT" field "$REF_A0" head_tier)"
check ram_available_gib     96.00          "$(bash "$EXTRACT" field "$REF_A0" ram_available_gib)"

# Cross-log: proves the parsers track a genuinely different arm rather than returning a
# constant. A2 is the high-contention 96-pin arm, so every stage differs sharply from A0.
echo "== KAT: cross-log, parsers must track the arm not a constant =="
check on1_store_lock_ms     466.225        "$(bash "$EXTRACT" field "$REF_ON1" store_lock_ms)"
check on1_store_pack_calls  3908           "$(bash "$EXTRACT" field "$REF_ON1" store_pack_calls)"
check on1_packed_slots      904            "$(bash "$EXTRACT" field "$REF_ON1" packed_slots)"
check a2_store_lock_ms      981.905        "$(bash "$EXTRACT" field "$REF_A2" store_lock_ms)"
check a2_store_pack_ms      1760.313       "$(bash "$EXTRACT" field "$REF_A2" store_pack_ms)"
check a2_store_pack_calls   8370           "$(bash "$EXTRACT" field "$REF_A2" store_pack_calls)"
check a2_store_hit_scan_ms  757.773        "$(bash "$EXTRACT" field "$REF_A2" store_hit_scan_ms)"
check a2_packed_slots       4568           "$(bash "$EXTRACT" field "$REF_A2" packed_slots)"

# verify_field is the generalised guard. The != form is what catches a silently clamped
# --memory-gb arm, which would otherwise be identical to baseline and manufacture a false
# null -- the same trap class as the E131 pin clamp.
echo "== GUARD: verify_field, equality and inequality =="
bash "$EXTRACT" verify_field "$REF" target_slots = 164 >/dev/null 2>&1
check "verify_field slots = 164"      0 "$?"
bash "$EXTRACT" verify_field "$REF" target_slots != 164 >/dev/null 2>&1
rc=$?; if [[ $rc -ne 0 ]]; then printf '  ok    %-34s rejected as required\n' "verify_field slots != 164"; else printf '  FAIL  %-34s accepted an unmoved memory arm\n' "verify_field slots != 164"; bad=$((bad+1)); fi
bash "$EXTRACT" verify_field "$REF" head_tier = streamed-bf16 >/dev/null 2>&1
rc=$?; if [[ $rc -ne 0 ]]; then printf '  ok    %-34s rejected as required\n' "verify_field head demoted"; else printf '  FAIL  %-34s missed a demoted head tier\n' "verify_field head demoted"; bad=$((bad+1)); fi

# Regression test for the guard false positive that cost two dispatches: pgrep -f matches
# ANY process whose command line contains the binary name, including the caller's own
# pre-flight. pgrep -x matches the process NAME only, so this cmdline -- which contains
# the literal c/deepseek_v4 -- must not match.
echo "== GUARD: pgrep -x must not match a caller that merely names c/deepseek_v4 =="
pgrep -x deepseek_v4 >/dev/null 2>&1
rc=$?; if [[ $rc -ne 0 ]]; then printf '  ok    %-34s no false positive\n' "pgrep -x vs own cmdline"; else printf '  FAIL  %-34s matched something (engine running?)\n' "pgrep -x vs own cmdline"; bad=$((bad+1)); fi

if (( bad )); then
  echo "pinsweep_extract: $bad checks failed" >&2
  exit 1
fi
echo "pinsweep_extract: all checks passed"
