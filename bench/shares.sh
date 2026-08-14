#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { printf 'usage: %s LOGFILE\n' "$0" >&2; exit 2; }
[[ -f $1 ]] || { printf 'log file not found: %s\n' "$1" >&2; exit 2; }

awk '
    /^v4_profile phase=/ {
        phase=""
        total=""
        for (field=1; field<=NF; field++) {
            if ($field ~ /^phase=/) { phase=$field; sub(/^phase=/, "", phase) }
            if ($field ~ /^total_ms=/) { total=$field; sub(/^total_ms=/, "", total) }
        }
        if (phase != "" && total != "") {
            if (!(phase in seen)) order[++phase_count]=phase
            seen[phase]=1
            phase_ms[phase]=total
        }
    }
    /^v4_profile / && /decode_wall_ms=/ {
        for (field=1; field<=NF; field++) {
            if ($field ~ /^decode_wall_ms=/) {
                decode_wall=$field
                sub(/^decode_wall_ms=/, "", decode_wall)
            }
        }
    }
    /^timing / && /time_to_first_token=/ {
        for (field=1; field<=NF; field++) {
            if ($field ~ /^time_to_first_token=/) {
                ttft=$field
                sub(/^time_to_first_token=/, "", ttft)
                sub(/s$/, "", ttft)
            }
        }
    }
    END {
        if (ttft == "") {
            print "missing timing time_to_first_token" > "/dev/stderr"
            exit 1
        }
        if (decode_wall == "" || decode_wall == 0) {
            print "missing or zero v4_profile decode_wall_ms" > "/dev/stderr"
            exit 1
        }
        number="^[0-9]+([.][0-9]+)?$"
        if (ttft !~ number || decode_wall !~ number) {
            print "invalid timing numeric value" > "/dev/stderr"
            exit 1
        }
        printf "TIMING ttft=%.3fs decode_wall_ms=%.3f\n", ttft, decode_wall
        for (phase_index=1; phase_index<=phase_count; phase_index++) {
            phase=order[phase_index]
            if (phase_ms[phase] !~ number) {
                print "invalid phase numeric value: " phase > "/dev/stderr"
                exit 1
            }
            printf "SHARE phase=%s total_ms=%.3f pct=%.2f\n", \
                phase, phase_ms[phase], 100 * phase_ms[phase] / decode_wall
        }
    }
' "$1"
