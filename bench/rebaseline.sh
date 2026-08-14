#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -le 1 ]] || { printf 'usage: %s [BINARY]\n' "$0" >&2; exit 2; }
BINARY=${1:-"$ROOT/c/deepseek_v4"}
N=${N:-3}
MODEL="$ROOT/models/deepseek-v4-flash"
SNAPSHOT=/tmp/coli_usage.snapshot
USAGE="$MODEL/.coli_usage"
LOCK=/tmp/colibri-prefill-bench.lock
WORK=$(mktemp -d "${TMPDIR:-/tmp}/colibri-rebaseline.XXXXXX")
ARTIFACT="$ROOT/artifacts/baseline.md"
LOCK_HELD=0
RUNS_STARTED=0

cleanup() {
    local status=$?
    local cleanup_status=0
    set +e
    if (( RUNS_STARTED )); then
        pkill -f '[d]eepseek_v4' >/dev/null 2>&1 || true
    fi
    if (( RUNS_STARTED )) && [[ -f $SNAPSHOT && -d $MODEL ]]; then
        cp "$SNAPSHOT" "$USAGE" || cleanup_status=$?
    fi
    rm -rf "$WORK"
    if (( LOCK_HELD )); then
        rm -f "$LOCK/pid"
        rmdir "$LOCK" 2>/dev/null || true
    fi
    if (( status == 0 && cleanup_status != 0 )); then
        status=$cleanup_status
    fi
    return "$status"
}
trap cleanup EXIT

[[ $N =~ ^[0-9]+$ ]] || { printf 'N must be an integer >= 3\n' >&2; exit 2; }
N=$((10#$N))
(( N >= 3 )) || { printf 'N must be an integer >= 3\n' >&2; exit 2; }
[[ -x $BINARY ]] || { printf 'binary not executable: %s\n' "$BINARY" >&2; exit 2; }
[[ -f $SNAPSHOT ]] || { printf 'usage snapshot missing: %s\n' "$SNAPSHOT" >&2; exit 2; }
mkdir -p "$ROOT/artifacts"
if pgrep -f '[d]eepseek_v4' >/dev/null 2>&1; then
    printf 'another deepseek_v4 process is already running\n' >&2
    exit 2
fi
if ! mkdir "$LOCK" 2>/dev/null; then
    printf 'prefill benchmark lock exists: %s\n' "$LOCK" >&2
    exit 2
fi
LOCK_HELD=1
printf '%s\n' "$$" >"$LOCK/pid"

parse_ttft() {
    awk '/timing time_to_first_token=/ {
        value=$0
        sub(/^.*time_to_first_token=/, "", value)
        sub(/s.*$/, "", value)
        print value
        exit
    }' "$1"
}

median() {
    printf '%s\n' "$@" | sort -n | awk '{ values[NR]=$1 }
        END {
            middle=int((NR + 1) / 2)
            if (NR % 2) result=values[middle]
            else result=(values[NR / 2] + values[NR / 2 + 1]) / 2
            printf "%.3f", result
        }'
}

sample_sd() {
    printf '%s\n' "$@" | awk '{ values[NR]=$1; sum+=$1 }
        END {
            mean=sum / NR
            for (i=1; i<=NR; i++) squared+=(values[i] - mean)^2
            printf "%.3f", sqrt(squared / (NR - 1))
        }'
}

run_model() {
    local prompt=$1
    local log=$2
    local status restore_status=0
    cp "$SNAPSHOT" "$USAGE" || restore_status=$?
    if (( status == 0 && restore_status != 0 )); then
        status=$restore_status
    fi
    if env COLI_V4_SAVE_USAGE=0 \
        "$BINARY" "$MODEL" "$prompt" --max-tokens 1 --memory-gb 96 \
        >"$log" 2>&1; then
        status=0
    else
        status=$?
    fi
    cp "$SNAPSHOT" "$USAGE"
    return "$status"
}

cd "$ROOT"
RUNS_STARTED=1
printf 'WARMUP rebaseline max_tokens=1\n'
run_model "$(<"$ROOT/.backlog/prefill_prompts/p064.txt")" "$WORK/warmup.log"

declare -a summaries=()
for prompt_name in p064 p256; do
    prompt=$(<"$ROOT/.backlog/prefill_prompts/$prompt_name.txt")
    declare -a values=()
    for (( run=1; run<=N; run++ )); do
        log="$WORK/${prompt_name}_${run}.log"
        run_model "$prompt" "$log"
        ttft=$(parse_ttft "$log")
        [[ $ttft =~ ^[0-9]+([.][0-9]+)?$ ]] || {
            printf 'invalid ttft in %s: %s\n' "$log" "${ttft:-<missing>}" >&2
            exit 1
        }
        values+=("$ttft")
        printf 'RUN %s n=%d ttft=%ss\n' "$prompt_name" "$run" "$ttft"
    done
    prompt_median=$(median "${values[@]}")
    prompt_sd=$(sample_sd "${values[@]}")
    printf 'SUMMARY %s median=%ss sd=%ss n=%d\n' "$prompt_name" "$prompt_median" "$prompt_sd" "$N"
    machine="BASELINE $prompt_name median=$prompt_median sd=$prompt_sd n=$N"
    printf '%s\n' "$machine"
    summaries+=("$prompt_name|$prompt_median|$prompt_sd|$N|$machine")
done

artifact_tmp=$(mktemp "$ROOT/artifacts/baseline.md.XXXXXX")
{
    printf '# Prefill TTFT baseline\n\n'
    printf 'Cold-cache runs with `COLI_V4_SAVE_USAGE=0`; values are seconds. SD is sample standard deviation.\n\n'
    printf '| prompt | median (s) | sd (s) | n |\n'
    printf '|---|---:|---:|---:|\n'
    for summary in "${summaries[@]}"; do
        IFS='|' read -r name value sd count _ <<<"$summary"
        printf '| %s | %s | %s | %s |\n' "$name" "$value" "$sd" "$count"
    done
    printf '\n```text\n'
    for summary in "${summaries[@]}"; do
        IFS='|' read -r _ _ _ _ machine <<<"$summary"
        printf '%s\n' "$machine"
    done
    printf '```\n'
} >"$artifact_tmp"
mv "$artifact_tmp" "$ARTIFACT"
printf 'WROTE %s\n' "$ARTIFACT"
