#!/usr/bin/env bash
set -euo pipefail

median() {
    printf '%s\n' "$@" | sort -n | awk '{ values[NR]=$1 }
        END {
            middle=int((NR + 1) / 2)
            if (NR % 2) result=values[middle]
            else result=(values[NR / 2] + values[NR / 2 + 1]) / 2
            printf "%.3f", result
        }'
}

validate_samples() {
    local sample
    for sample in "$@"; do
        [[ $sample =~ ^[0-9]+([.][0-9]+)?$ ]] || {
            printf 'invalid numeric sample: %s\n' "$sample" >&2
            return 2
        }
    done
}

print_ab() {
    local prompt_name=$1
    shift
    local off_count=$1
    shift
    local -a off_values=("${@:1:off_count}")
    shift "$off_count"
    local -a on_values=("$@")
    local off_median on_median delta
    off_median=$(median "${off_values[@]}")
    on_median=$(median "${on_values[@]}")
    delta=$(awk -v off="$off_median" -v on="$on_median" \
        'BEGIN { if (off == 0) exit 2; printf "%.2f", 100 * (on - off) / off }')
    printf 'AB %s off=%s on=%s delta=%s%%\n' \
        "$prompt_name" "$off_median" "$on_median" "$delta"
}

if [[ ${1:-} == --math ]]; then
    [[ $# -eq 4 ]] || { printf 'usage: %s --math PROMPT "OFF VALUES" "ON VALUES"\n' "$0" >&2; exit 2; }
    read -r -a off_values <<<"$3"
    read -r -a on_values <<<"$4"
    (( ${#off_values[@]} > 0 && ${#off_values[@]} == ${#on_values[@]} )) || {
        printf 'OFF and ON value counts must match and be nonzero\n' >&2
        exit 2
    }
    validate_samples "${off_values[@]}" "${on_values[@]}"
    print_ab "$2" "${#off_values[@]}" "${off_values[@]}" "${on_values[@]}"
    exit
fi

[[ $# -eq 2 ]] || { printf 'usage: %s BINARY_A_ENV BINARY\n' "$0" >&2; exit 2; }
ON_ENV=$1
BINARY=$2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
N=${N:-3}
MODEL="$ROOT/models/deepseek-v4-flash"
SNAPSHOT=/tmp/coli_usage.snapshot
USAGE="$MODEL/.coli_usage"
LOCK=/tmp/colibri-prefill-bench.lock
WORK=$(mktemp -d "${TMPDIR:-/tmp}/colibri-ab.XXXXXX")
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

read -r -a ON_ENV_ARGS <<<"$ON_ENV"
declare -a OFF_ENV_ARGS=()
for assignment in "${ON_ENV_ARGS[@]}"; do
    if [[ ! $assignment =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        printf 'invalid environment assignment: %s\n' "$assignment" >&2
        exit 2
    fi
    OFF_ENV_ARGS+=(-u "${assignment%%=*}")
done

parse_ttft() {
    awk '/timing time_to_first_token=/ {
        value=$0
        sub(/^.*time_to_first_token=/, "", value)
        sub(/s.*$/, "", value)
        print value
        exit
    }' "$1"
}

run_model() {
    local state=$1
    local prompt=$2
    local log=$3
    local status restore_status=0
    local -a state_env
    if [[ $state == on ]]; then
        state_env=("${ON_ENV_ARGS[@]}")
    else
        state_env=("${OFF_ENV_ARGS[@]}")
    fi
    cp "$SNAPSHOT" "$USAGE" || restore_status=$?
    if (( status == 0 && restore_status != 0 )); then
        status=$restore_status
    fi
    if env "${state_env[@]}" COLI_V4_SAVE_USAGE=0 \
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
warmup_prompt=$(<"$ROOT/.backlog/prefill_prompts/p064.txt")
printf 'WARMUP AB state=off max_tokens=1\n'
run_model off "$warmup_prompt" "$WORK/warmup_off.log"
printf 'WARMUP AB state=on max_tokens=1\n'
run_model on "$warmup_prompt" "$WORK/warmup_on.log"

for prompt_name in p064 p256; do
    prompt=$(<"$ROOT/.backlog/prefill_prompts/$prompt_name.txt")
    declare -a off_values=()
    declare -a on_values=()
    for (( pair=1; pair<=N; pair++ )); do
        off_log="$WORK/${prompt_name}_${pair}_off.log"
        on_log="$WORK/${prompt_name}_${pair}_on.log"
        run_model off "$prompt" "$off_log"
        off_ttft=$(parse_ttft "$off_log")
        [[ $off_ttft =~ ^[0-9]+([.][0-9]+)?$ ]] || {
            printf 'invalid OFF ttft in %s: %s\n' "$off_log" "${off_ttft:-<missing>}" >&2
            exit 1
        }
        off_values+=("$off_ttft")
        printf 'RUN %s pair=%d state=off ttft=%ss\n' "$prompt_name" "$pair" "$off_ttft"

        run_model on "$prompt" "$on_log"
        on_ttft=$(parse_ttft "$on_log")
        [[ $on_ttft =~ ^[0-9]+([.][0-9]+)?$ ]] || {
            printf 'invalid ON ttft in %s: %s\n' "$on_log" "${on_ttft:-<missing>}" >&2
            exit 1
        }
        on_values+=("$on_ttft")
        printf 'RUN %s pair=%d state=on ttft=%ss\n' "$prompt_name" "$pair" "$on_ttft"
    done
    print_ab "$prompt_name" "${#off_values[@]}" "${off_values[@]}" "${on_values[@]}"
done
