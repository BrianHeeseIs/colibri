#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_MD5=5d04890413ff539e802985ce8c727814

ext() {
    awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal) /{print} /^timing /{f=0}' "$1"
}

md5_stdin() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q
    else
        md5sum | cut -d ' ' -f 1
    fi
}

check_log() {
    local log=$1
    local expected=$2
    local actual
    actual=$(ext "$log" | md5_stdin)
    if [[ $actual == "$expected" ]]; then
        printf 'PASS golden md5=%s\n' "$actual"
        return 0
    fi
    printf 'FAIL golden expected=%s actual=%s\n' "$expected" "$actual" >&2
    return 1
}

if [[ ${1:-} == --fixture-log ]]; then
    [[ $# -eq 2 ]] || { printf 'usage: %s --fixture-log LOGFILE\n' "$0" >&2; exit 2; }
    check_log "$2" "${GOLDEN_OVERRIDE_MD5:-$DEFAULT_MD5}"
    exit
fi

[[ $# -le 1 ]] || { printf 'usage: %s [BINARY]\n' "$0" >&2; exit 2; }
BINARY=${1:-"$ROOT/c/deepseek_v4"}
MODEL="$ROOT/models/deepseek-v4-flash"
SNAPSHOT=/tmp/coli_usage.snapshot
USAGE="$MODEL/.coli_usage"
LOCK=/tmp/colibri-prefill-bench.lock
PROMPT='Write a detailed technical explanation of how a mixture-of-experts transformer routes tokens.'
WORK=$(mktemp -d "${TMPDIR:-/tmp}/colibri-golden-run.XXXXXX")
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

declare -a EXTRA_ENV_ARGS=()
if [[ -n ${EXTRA_ENV:-} ]]; then
    read -r -a EXTRA_ENV_ARGS <<<"$EXTRA_ENV"
fi
for assignment in "${EXTRA_ENV_ARGS[@]}"; do
    if [[ ! $assignment =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        printf 'invalid EXTRA_ENV assignment: %s\n' "$assignment" >&2
        exit 2
    fi
done

run_model() {
    local max_tokens=$1
    local log=$2
    local status restore_status=0
    cp "$SNAPSHOT" "$USAGE"
    if env "${EXTRA_ENV_ARGS[@]}" COLI_V4_SAVE_USAGE=0 \
        "$BINARY" "$MODEL" "$PROMPT" --max-tokens "$max_tokens" --memory-gb 96 \
        >"$log" 2>&1; then
        status=0
    else
        status=$?
    fi
    cp "$SNAPSHOT" "$USAGE" || restore_status=$?
    if (( status == 0 && restore_status != 0 )); then
        status=$restore_status
    fi
    return "$status"
}

cd "$ROOT"
RUNS_STARTED=1
printf 'WARMUP golden max_tokens=1\n'
run_model 1 "$WORK/warmup.log"
printf 'RUN golden max_tokens=60\n'
run_model 60 "$WORK/golden.log"
check_log "$WORK/golden.log" "$DEFAULT_MD5"
