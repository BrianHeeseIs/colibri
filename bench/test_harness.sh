#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIXTURE="$ROOT/bench/fixtures/run.log"

md5_stdin() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q
    else
        md5sum | cut -d ' ' -f 1
    fi
}

fail() {
    printf 'FAIL %s\n' "$*" >&2
    exit 1
}

expected_md5=$(printf '%s\n' \
    'generated_text=First generated line' \
    'Second generated line' | md5_stdin)

printf 'test golden fixture match\n'
GOLDEN_OVERRIDE_MD5="$expected_md5" \
    "$ROOT/bench/golden.sh" --fixture-log "$FIXTURE"

printf 'test golden fixture mismatch\n'
if GOLDEN_OVERRIDE_MD5=00000000000000000000000000000000 \
    "$ROOT/bench/golden.sh" --fixture-log "$FIXTURE" >/dev/null 2>&1; then
    fail 'golden mismatch returned success'
fi

printf 'test phase shares\n'
shares=$("$ROOT/bench/shares.sh" "$FIXTURE")
grep -F 'TIMING ttft=1.250s decode_wall_ms=200.000' <<<"$shares" >/dev/null ||
    fail 'timing summary missing'
grep -F 'SHARE phase=attention total_ms=50.000 pct=25.00' <<<"$shares" >/dev/null ||
    fail 'attention share wrong'
grep -F 'SHARE phase=expert_forward total_ms=80.000 pct=40.00' <<<"$shares" >/dev/null ||
    fail 'expert_forward share wrong'

printf 'test A/B math\n'
ab=$("$ROOT/bench/ab.sh" --math p064 '1 2 3' '1 1.5 2')
grep -F 'AB p064 off=2.000 on=1.500 delta=-25.00%' <<<"$ab" >/dev/null ||
    fail 'A/B summary wrong'
if "$ROOT/bench/ab.sh" --math p064 '1 bad 3' '1 2 3' >/dev/null 2>&1; then
    fail 'A/B accepted nonnumeric sample'
fi

printf 'PASS test_harness\n'
