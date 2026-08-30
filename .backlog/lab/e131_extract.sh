#!/usr/bin/env bash
# E131 extractor. Pulls the ten fields the pre-registered gate is computed from out of a
# decode-trace arm log, and asserts that a requested COLI_V4_PIN_SLOTS was actually applied.
#
# Modes:
#   field <log> <name>            print one value
#   verify_applied <log> <n>      rc 0 if the log records pin_slots_per_layer=<n>, else rc 1
#   table <log> [<log> ...]       one row per arm, for the gate comparison
set -uo pipefail

field() { # field <log> <name>
  local log=$1 name=$2
  [[ -f $log ]] || { echo "MISSING_LOG" ; return 2; }
  case $name in
    store_disk_read_calls)
      sed -n 's/.*stage=store_disk_read .*calls=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    store_disk_read_ms)
      sed -n 's/.*stage=store_disk_read total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    wait_finish_complete_ms)
      sed -n 's/.*stage=wait_finish_complete_block total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    wait_total_ms)
      sed -n 's/.*table=wait stage=total total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    wait_total_pct)
      sed -n 's/.*table=wait stage=total .*pct_decode=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    finish_slept_calls)
      sed -n 's/.*stage=finish_slept_calls count=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    finish_completed_at_entry)
      sed -n 's/.*stage=finish_completed_at_entry count=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    start_slept_calls)
      sed -n 's/.*stage=start_slept_calls count=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    expert_forward_ms)
      sed -n 's/.*phase=expert_forward total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    decode_wall_ms)
      sed -n 's/.*decode_wall_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    expert_calls_rows16)
      sed -n 's/.*expert_calls_rows16=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    expert_calls_scalar)
      sed -n 's/.*expert_calls_scalar=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    pin_slots_applied)
      sed -n 's/.*v4_hot_policy pin_slots_per_layer=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    decode_sec)
      sed -n 's/.*after_first=\([0-9.]*\)s.*/\1/p' "$log" | tail -1 ;;
    ttft_sec)
      sed -n 's/.*time_to_first_token=\([0-9.]*\)s.*/\1/p' "$log" | tail -1 ;;
    *) echo "UNKNOWN_FIELD"; return 2 ;;
  esac
}

# A silent clamp to maximum_pins would make two arms identical and manufacture a false null,
# so a requested value that did not land must be a hard failure, not a warning.
verify_applied() { # verify_applied <log> <expected>
  local log=$1 want=$2 got
  got=$(field "$log" pin_slots_applied)
  if [[ -z $got ]]; then
    echo "verify_applied: no v4_hot_policy line in $log" >&2
    return 1
  fi
  if [[ $got != "$want" ]]; then
    echo "verify_applied: requested $want but engine applied $got in $log" >&2
    return 1
  fi
  echo "verify_applied: $log applied pin_slots_per_layer=$got"
  return 0
}

table() {
  printf '%-34s %8s %8s %10s %10s %9s %9s %10s %8s %8s\n' \
    log pins misses disk_ms waitblk_ms slept at_entry expfwd_ms rows16 scalar
  local f
  for f in "$@"; do
    printf '%-34s %8s %8s %10s %10s %9s %9s %10s %8s %8s\n' \
      "$(basename "$f" | cut -c1-34)" \
      "$(field "$f" pin_slots_applied)" \
      "$(field "$f" store_disk_read_calls)" \
      "$(field "$f" store_disk_read_ms)" \
      "$(field "$f" wait_finish_complete_ms)" \
      "$(field "$f" finish_slept_calls)" \
      "$(field "$f" finish_completed_at_entry)" \
      "$(field "$f" expert_forward_ms)" \
      "$(field "$f" expert_calls_rows16)" \
      "$(field "$f" expert_calls_scalar)"
  done
}

mode=${1:-}
case $mode in
  field)          shift; field "$@" ;;
  verify_applied) shift; verify_applied "$@" ;;
  table)          shift; table "$@" ;;
  *) echo "usage: $0 {field <log> <name>|verify_applied <log> <n>|table <log>...}" >&2; exit 2 ;;
esac
