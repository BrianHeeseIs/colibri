# Lane: raising the decode batch dimension S (concurrent serving)

**Status: DOCUMENTED, NOT BUILT. Deferred by operator decision — not in scope for the current
pull request.** Full reasoning in `experiments_results.md` E98. This file is the runnable backlog.

## Why the lane exists
Decode runs at **S=1** — one token per forward — on a 40-core GPU. The GPU's advantage over the
CPU GROWS with S while the CPU's cost grows linearly (`validation/metal/bench_matmul 20`,
gate|up 4096->2048): CPU 0.428 ms vs `simd_exact` 0.072 ms at S=1 (5.9x), and CPU 3.012 ms vs
0.387 ms at S=8 (**7.8x**). Raising S is the largest structural inefficiency left on this hardware.

Only two mechanisms raise decode S, and one is already dead:
- **Speculation** — MEASURED and rejected. Drafts rarely fire, and rejected-suffix replay costs
  more than the drafts save. See `mem:deepseek_v4/dead_levers`.
- **Concurrent in-flight requests** — never tested. This lane.

## What it would and would not buy
It raises **aggregate** throughput (tokens/second across requests). It does **not** reduce
single-request latency and may slightly increase it. Every prior headline number in this project is
single-request tok/s, so any result here must state which metric it improves or it will be
misread against ~98 entries measured the other way.

## Known blockers, in the order they will be hit
1. **One KV slot.** `coli serve` supports greedy generation and a single active KV slot
   (`docs/deepseek-v4.md`). Multi-request batching needs KV slot management first; this is not a
   scheduling tweak.
2. **Memory.** The engine already reaches ~58 GB RSS at `--memory-gb 96` on this 128 GB host.
   Concurrent KV state adds to that. `swap` must stay at 0 or the measurement is void — the
   monitor pane in `.backlog/lab/lab_monitor.sh` displays it for this reason.
3. **The MIN_N gate is stale.** `COLI_V4_MOE_BATCHED_MIN_N=4` was tuned against the OLD kernels,
   and `MIN_N=3` measured +0.81% SLOWER then. `simd_exact` changes the GPU side of that trade by
   roughly 5x, so the crossover has almost certainly moved.

## Do this FIRST — it is cheap and it sizes the whole lane
Re-measure the MIN_N crossover with `simd_exact` enabled. No concurrency work, no KV changes.
If the optimum moves from 4 down to 2 or 1, part of prefill's small-group population converts to
the GPU immediately, and the measured slope tells you what concurrency would be worth.

Note the prefill subtlety that makes this promising: the S that matters is not the chunk size but
the number of tokens routed to a GIVEN expert. A 64-token chunk at top-6 over 256 experts averages
**1.5 rows per expert**, so most groups sit BELOW the current gate of 4.

```bash
cd /Users/cptn/workbench/ai/colibri
pgrep -f '[d]eepseek_v4' && { echo "ABORT: engine running"; exit 2; }
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot

export COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold
for M in 1 2 3 4 8; do
  COLI_V4_MOE_BATCHED_MIN_N=$M N=5 TOKENS=60 \
    PROMPT_FILE=.backlog/prefill_prompts/p256.txt ./.backlog/lab/tokps.sh \
    "min_n_$M=@=COLI_V4_MOE_BATCHED_MIN_N=$M" \
    "baseline=@=COLI_V4_MOE_BATCHED_MIN_N=4" \
    2>&1 | tee ".backlog/lab/minn_simdexact_${M}_$(date +%Y%m%d-%H%M%S).log"
done
unset COLI_V4_METAL COLI_V4_MOE_BATCHED COLI_V4_METAL_VARIANT
```
Run it in the `colibri-lab` tmux window with `.backlog/lab/lab_monitor.sh` in a second pane.
**One engine at a time** — `tokps.sh` only `pgrep`s, it does not take the
`/tmp/colibri-prefill-bench.lock` that `ab.sh`/`golden.sh` use.

## Then, if the slope justifies it
1. KV slot management for N concurrent sequences.
2. A batching scheduler that forms a forward from the in-flight set.
3. Report **aggregate** tok/s AND single-request tok/s AND TTFT, separately, at 1/2/4/8
   concurrency. Never TTFT alone -- see the mandatory both-axes rule in AGENTS.md.
4. Gate on `.backlog/lab/taskcheck.sh` per stream — batching must not change any stream's answer.
