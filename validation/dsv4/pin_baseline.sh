#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BINARY="$ROOT/c/deepseek_v4"
MODEL="$ROOT/models/deepseek-v4-flash"
SNAPSHOT=/tmp/coli_usage.snapshot
USAGE="$MODEL/.coli_usage"
EXPECTED_SNAPSHOT_MD5=599f3d12e9347ef30541bd6f9ba18bde
EXPECTED_GOLDEN_MD5=5d04890413ff539e802985ce8c727814
ARTIFACT="$ROOT/artifacts/metal_baseline.json"
BASELINE_ARTIFACT="$ROOT/artifacts/baseline.md"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/colibri-metal-baseline.XXXXXX")
ARTIFACT_TMP=
BASELINE_EXISTED=0
BASELINE_TRACKED=0

md5_file() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | cut -d ' ' -f 1
    fi
}

restore_snapshot() {
    cp "$SNAPSHOT" "$USAGE"
}

cleanup() {
    local status=$?
    local cleanup_status=0
    set +e
    if [[ -f $SNAPSHOT && -d $MODEL ]]; then
        restore_snapshot || cleanup_status=$?
    fi
    if (( BASELINE_TRACKED )); then
        if (( BASELINE_EXISTED )); then
            cp "$WORK/baseline.md.before" "$BASELINE_ARTIFACT" || cleanup_status=$?
        else
            rm -f "$BASELINE_ARTIFACT"
        fi
    fi
    [[ -z $ARTIFACT_TMP ]] || rm -f "$ARTIFACT_TMP"
    rm -rf "$WORK"
    if (( status == 0 && cleanup_status != 0 )); then
        status=$cleanup_status
    fi
    return "$status"
}
trap cleanup EXIT

nonengine_gb() {
    ps -axo rss=,comm= | awk '
        index($0, "deepseek_v4") == 0 { kib += $1 }
        END { printf "%.3f", kib * 1024 / 1000000000 }
    '
}

compressor_gb() {
    vm_stat | awk '
        NR == 1 { page_bytes=$8 }
        /Pages occupied by compressor/ { gsub(/\./, "", $5); pages=$5 }
        END {
            if (page_bytes != 16384 || pages == "") exit 1
            printf "%.3f", pages * page_bytes / 1000000000
        }
    '
}

baseline_median() {
    local prompt_name=$1
    awk -v prompt_name="$prompt_name" '
        $1 == "BASELINE" && $2 == prompt_name {
            for (field=3; field<=NF; field++) {
                if ($field ~ /^median=/) {
                    sub(/^median=/, "", $field)
                    print $field
                    exit
                }
            }
        }
    ' "$WORK/rebaseline.log"
}

[[ -x $BINARY ]] || { printf 'binary not executable: %s\n' "$BINARY" >&2; exit 2; }
[[ -f $SNAPSHOT ]] || { printf 'usage snapshot missing: %s\n' "$SNAPSHOT" >&2; exit 2; }
[[ -d $MODEL ]] || { printf 'model directory missing: %s\n' "$MODEL" >&2; exit 2; }
if pgrep -f '[d]eepseek_v4' >/dev/null 2>&1; then
    printf 'another deepseek_v4 process is already running\n' >&2
    exit 2
fi

snapshot_md5=$(md5_file "$SNAPSHOT")
if [[ $snapshot_md5 != "$EXPECTED_SNAPSHOT_MD5" ]]; then
    printf 'usage snapshot md5 mismatch: expected=%s actual=%s\n' \
        "$EXPECTED_SNAPSHOT_MD5" "$snapshot_md5" >&2
    exit 2
fi

mkdir -p "$ROOT/artifacts"
if [[ -f $BASELINE_ARTIFACT ]]; then
    cp "$BASELINE_ARTIFACT" "$WORK/baseline.md.before"
    BASELINE_EXISTED=1
fi
BASELINE_TRACKED=1

restore_snapshot
binary_md5=$(md5_file "$BINARY")
commit=$(git -C "$ROOT" rev-parse HEAD)
run_date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
nonengine=$(nonengine_gb)
compressor=$(compressor_gb)

printf 'PIN binary_md5=%s commit=%s date=%s nonengine_gb=%s compressor_gb=%s\n' \
    "$binary_md5" "$commit" "$run_date" "$nonengine" "$compressor"

if COLI_V4_SAVE_USAGE=0 "$ROOT/bench/golden.sh" "$BINARY" >"$WORK/golden.log" 2>&1; then
    cat "$WORK/golden.log"
else
    status=$?
    cat "$WORK/golden.log" >&2
    exit "$status"
fi
restore_snapshot

golden_line=$(awk -v expected="$EXPECTED_GOLDEN_MD5" \
    '$0 == "PASS golden md5=" expected { print; exit }' "$WORK/golden.log")
if [[ $golden_line != "PASS golden md5=$EXPECTED_GOLDEN_MD5" ]]; then
    printf 'required golden line missing: PASS golden md5=%s\n' \
        "$EXPECTED_GOLDEN_MD5" >&2
    exit 1
fi

if N=3 COLI_V4_SAVE_USAGE=0 "$ROOT/bench/rebaseline.sh" "$BINARY" \
    >"$WORK/rebaseline.log" 2>&1; then
    cat "$WORK/rebaseline.log"
else
    status=$?
    cat "$WORK/rebaseline.log" >&2
    exit "$status"
fi
restore_snapshot

p064_median=$(baseline_median p064)
p256_median=$(baseline_median p256)
[[ $p064_median =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'invalid p064 median: %s\n' "${p064_median:-<missing>}" >&2
    exit 1
}
[[ $p256_median =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'invalid p256 median: %s\n' "${p256_median:-<missing>}" >&2
    exit 1
}

final_binary_md5=$(md5_file "$BINARY")
if [[ $final_binary_md5 != "$binary_md5" ]]; then
    printf 'binary changed during baseline: before=%s after=%s\n' \
        "$binary_md5" "$final_binary_md5" >&2
    exit 1
fi

ARTIFACT_TMP=$(mktemp "$ROOT/artifacts/metal_baseline.json.XXXXXX")
cat >"$ARTIFACT_TMP" <<EOF
{
  "binary_md5": "$binary_md5",
  "commit": "$commit",
  "p064_median_s": $p064_median,
  "p256_median_s": $p256_median,
  "golden_md5": "$EXPECTED_GOLDEN_MD5",
  "nonengine_gb": $nonengine,
  "compressor_gb": $compressor,
  "date": "$run_date"
}
EOF
mv "$ARTIFACT_TMP" "$ARTIFACT"
ARTIFACT_TMP=
printf 'WROTE %s\n' "$ARTIFACT"
