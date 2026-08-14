#!/usr/bin/env bash
# Runs deepseek_v4 with every REASSOCIATED-FP KERNEL enabled (--fast-kernels).
#
# The historical filename stays because docs and benchmark scripts reference it. The flag enables:
#   * attn_sparse : hand NEON, 4 independent FMA accumulators   (E37)  ~4.5x on that phase
#   * router      : hand NEON, 4 independent FMA accumulators   (E40)  ~10.8x on that phase
#
# Measured on a 60-token generation (M3 Max, warm):
#   default : router 1955.5 ms   decode_wall 38612.8 ms
#   flagged : router  181.8 ms   decode_wall 34197.5 ms   (~11.4% faster end to end)
#
# TRADE-OFF: reassociating these FP sums changes rounding, which flips an argmax at
# long context. Output is NOT bit-identical to the default build. It is not worse --
# just different -- but transcripts from this script are not comparable to default ones.
#
# Reference md5s (60 tokens, MoE-routing prompt, multi-line block extract):
#   default             5d04890413ff539e802985ce8c727814
#   --fast-kernels      7155bab905cbfa70aa06afa08f757cee
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -v COLI_V4_KERNELS ]]; then
  echo "warning: COLI_V4_KERNELS=$COLI_V4_KERNELS overrides wrapper --fast-kernels selection; active reassociated-FP kernels change output" >&2
else
  echo "warning: reassociated-FP kernels enabled (attn_sparse + router); output is NOT bit-identical to default" >&2
fi
exec "$here/c/deepseek_v4" "$@" --fast-kernels
