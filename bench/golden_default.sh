#!/usr/bin/env bash
# Golden gate for the DEFAULT (champion) path.
#
# bench/golden.sh pins COLI_V4_BASELINE=1 and guards the historical deterministic reference, whose
# md5 AGENTS.md calls sacred. Since 2026-08-29 the engine's DEFAULT is the champion stack
# (E114-E119), which is deliberately NOT token-identical to that reference -- so without this file
# the shipping configuration would have no regression gate at all.
#
# The expected value lives in bench/GOLDEN_DEFAULT_MD5 (not sacred: re-record it deliberately when a
# default changes, and say so in experiments_results.md). Seed hash is VERIFIED, not merely present.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
MODEL="$ROOT/models/deepseek-v4-flash"
DURABLE="$ROOT/.backlog/lab/coli_usage.snapshot"
EXPECT_FILE="$ROOT/bench/GOLDEN_DEFAULT_MD5"
SEED_MD5=599f3d12e9347ef30541bd6f9ba18bde
BINARY=${1:-"$ROOT/c/deepseek_v4"}
TOKENS=${TOKENS:-60}
PROMPT=${PROMPT:-"Explain what a mixture-of-experts layer does in one paragraph."}

[[ -f $DURABLE ]] || { echo "FATAL: durable seed missing: $DURABLE" >&2; exit 2; }
got=$(md5 -q "$DURABLE")
[[ $got == "$SEED_MD5" ]] || { echo "FATAL: seed hash $got != $SEED_MD5" >&2; exit 2; }
pgrep -f '[d]eepseek_v4' >/dev/null && { echo "FATAL: engine already running" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/goldendef.XXXXXX")
ext(){ awk '/^generated_text=/{f=1} f&&!/^(timing|v4_rows16|v4_direct|v4_tokens|v4_profile|v4_kernels|v4_metal|v4_prefill_trace) /{print} /^timing /{f=0}' "$1"; }

cp "$DURABLE" /tmp/coli_usage.snapshot; cp "$DURABLE" "$MODEL/.coli_usage"
echo "RUN golden_default max_tokens=$TOKENS (engine defaults, no COLI_V4_BASELINE)"
if ! env COLI_V4_SAVE_USAGE=0 "$BINARY" "$MODEL" "$PROMPT" \
        --max-tokens "$TOKENS" --memory-gb 96 >"$WORK/run.log" 2>&1; then
    echo "FAIL golden_default: engine exited non-zero; log kept at $WORK/run.log" >&2
    exit 1
fi
cp "$DURABLE" "$MODEL/.coli_usage"
got_md5=$(ext "$WORK/run.log" | md5 -q)

if [[ ! -f $EXPECT_FILE ]]; then
    echo "$got_md5" > "$EXPECT_FILE"
    echo "RECORDED golden_default md5=$got_md5 -> $EXPECT_FILE"
    echo "(first run: nothing to compare against yet)"
    rm -rf "$WORK"; exit 0
fi
want=$(tr -d '[:space:]' < "$EXPECT_FILE")
if [[ $got_md5 == "$want" ]]; then
    echo "PASS golden_default md5=$got_md5"; rm -rf "$WORK"; exit 0
fi
echo "FAIL golden_default md5=$got_md5 want=$want" >&2
echo "--- generated text (kept: $WORK/run.log) ---" >&2
ext "$WORK/run.log" >&2
echo "NOTE: a changed md5 is NOT proof of breakage (AGENTS.md). Diff the text above, run" >&2
echo "      .backlog/lab/taskcheck.sh, and repeat the arm against itself before concluding." >&2
exit 1
