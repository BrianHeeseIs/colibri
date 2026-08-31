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
    store_pack_ms)
      sed -n 's/.*stage=store_pack total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    store_pack_calls)
      sed -n 's/.*stage=store_pack total_ms=[0-9.]* calls=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    store_lock_ms)
      sed -n 's/.*stage=store_lock total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    store_hit_scan_ms)
      sed -n 's/.*stage=store_hit_scan total_ms=\([0-9.]*\).*/\1/p' "$log" | tail -1 ;;
    packed_slots)
      sed -n 's/.*packed_slots=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    target_slots)
      sed -n 's/.*target_slots=\([0-9]*\).*/\1/p' "$log" | tail -1 ;;
    head_tier)
      sed -n 's/.*ram_tiers .*head=\([a-z0-9-]*\).*/\1/p' "$log" | tail -1 ;;
    ram_available_gib)
      sed -n 's/.*ram_tiers available=\([0-9.]*\)GiB.*/\1/p' "$log" | tail -1 ;;
    decode_sec)
      sed -n 's/.*after_first=\([0-9.]*\)s.*/\1/p' "$log" | tail -1 ;;
    ttft_sec)
      sed -n 's/.*time_to_first_token=\([0-9.]*\)s.*/\1/p' "$log" | tail -1 ;;
    *) echo "UNKNOWN_FIELD"; return 2 ;;
  esac
}

# Generalised arm guard. An arm whose requested setting silently did not land is identical to
# baseline and would manufacture a false null -- the E131 pin clamp, and the same risk for a
# --memory-gb arm that gets clamped or a head tier that gets demoted. Supports = and !=.
verify_field() { # verify_field <log> <field> <op> <value>
  local log=$1 name=$2 op=$3 want=$4 got
  got=$(field "$log" "$name")
  if [[ -z $got || $got == UNKNOWN_FIELD || $got == MISSING_LOG ]]; then
    echo "verify_field: cannot read $name from $log (got '${got:-empty}')" >&2
    return 1
  fi
  case $op in
    =)  if [[ $got != "$want" ]]; then
          echo "verify_field: $name expected $want but engine reported $got in $log" >&2
          return 1
        fi ;;
    !=) if [[ $got == "$want" ]]; then
          echo "verify_field: $name is still $got in $log -- the setting did not take effect" >&2
          return 1
        fi ;;
    *)  echo "verify_field: unsupported op '$op' (use = or !=)" >&2; return 2 ;;
  esac
  echo "verify_field: $(basename "$log") $name $op $want (actual $got)"
  return 0
}

# Kept as a thin wrapper so the committed E131 tests and pinsweep.sh keep working unchanged.
verify_applied() { # verify_applied <log> <expected>
  verify_field "$1" pin_slots_applied = "$2"
}

# Arm spec is a ';'-separated list of field<op>value, or '-' for no checks.
verify_specs() { # verify_specs <log> <spec>
  local log=$1 spec=$2 one name op want rc=0
  [[ -z $spec || $spec == "-" ]] && return 0
  local IFS=';'
  for one in $spec; do
    [[ -z $one ]] && continue
    if [[ $one == *"!="* ]]; then name=${one%%!=*}; op="!="; want=${one#*!=}
    else                          name=${one%%=*};  op="=";  want=${one#*=}
    fi
    verify_field "$log" "$name" "$op" "$want" || rc=1
  done
  return $rc
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
  verify_field)   shift; verify_field "$@" ;;
  verify_specs)   shift; verify_specs "$@" ;;
  table)          shift; table "$@" ;;
  *) echo "usage: $0 {field <log> <name>|verify_applied <log> <n>|verify_field <log> <f> <op> <v>|verify_specs <log> <spec>|table <log>...}" >&2; exit 2 ;;
esac
