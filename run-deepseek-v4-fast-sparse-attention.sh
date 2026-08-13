#!/usr/bin/env bash
# Runs deepseek_v4 with the reassociated-FP sparse-attention kernel enabled.
#
# Effect: the attn_sparse phase uses hand-written NEON with 4 independent FMA
# accumulators instead of a strictly-ordered scalar reduction.
#   - attn_sparse phase: ~4.5x faster (22639 ms -> 4999 ms over 220 tokens)
#   - end to end:        ~+12% throughput (1.446 -> 1.627 tok/s steady state)
#
# TRADE-OFF: reassociating the dot-product sum changes floating-point rounding,
# which eventually flips an argmax at long context. Output is therefore NOT
# bit-identical to the default build. It is not worse -- just different -- but
# transcripts from this script are not comparable to default-build transcripts.
#
# Reference md5s (60 tokens, MoE-routing prompt):
#   default          5d04890413ff539e802985ce8c727814
#   --fast-sparse-attn  c6d8f26ef47095bf6f777c11d99df080
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "warning: fast reassociated-FP sparse attention enabled; output is NOT bit-identical to default" >&2
exec "$here/c/deepseek_v4" "$@" --fast-sparse-attn
