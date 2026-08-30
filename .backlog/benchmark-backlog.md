# Backlog — gated work and parked benchmark runs

## P0 — mxfp4 expert kernel  *** REQUIRES EXPLICIT OPERATOR PERMISSION TO START ***
Operator decision 2026-08-30: **highest priority next step, but GATED. Do not begin — not even the
sizing probe — without asking again.** Full detail in the T6 section below.
One-line summary: `expert_forward` is 41.7% of decode; `neon_rows16_accumulate`
(c/deepseek_v4.c:14572) applies the block scale PER ELEMENT (2 muls + 1 add) although the scale is
constant across each 32-column group, so hoisting it would be ~1.125 ops/element instead of 3.
Ceiling if it delivered 2x on the phase: **+26% decode**. Two blockers: the unfused ordering is what
keeps the CPU path bit-identical to the Metal kernel, and only ~6% of decode expert calls reach the
NEON path at all. Estimated 2-4 h, real risk, gated on taskcheck rather than golden.

# Benchmark backlog — parked runs
Operator rule (2026-08-30): anything over ~25 min is parked here rather than run, so the operator
chooses when to spend the wall time. Shorter instruments must be used whenever they prove the point.

Every entry states: what it decides, why a cheaper instrument cannot decide it, and the exact command.

| id | cost | decides | why not cheaper |
|---|---|---|---|
| B1 | ~35 min | Does the E125 fp8 kernel gain scale with prompt length (p512)? | The 90 s profile shows phase ms but not tok/s; p512 decode needs N>=5 to clear the 5-13% noise floor. |
| B2 | ~35 min | Does the whole-prompt MoE win keep growing at p1024 (E120 showed -17.4% at p256 -> -25.9% at p512)? | Counters already show the mechanism scales; only wall-clock settles the curve. |
| B3 | ~45 min | OMP_NUM_THREADS 12 vs 16 at N>=5 on BOTH axes, if the cheap profile A/B is ambiguous. | A single profile pair may fall inside noise; only N>=5 resolves a sub-10% decode delta. |

## B1 — p512 tok/s for the champion stack
```bash
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
TOKENS=40 N=5 PROMPT_FILE=.backlog/prefill_prompts/p512.txt .backlog/lab/tokps.sh \
  'off=@=COLI_V4_FP8_ROWS16=0' 'on=@=COLI_V4_FP8_ROWS16=1'
```

## B2 — p1024 whole-prompt scaling
```bash
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
TOKENS=40 N=3 PROMPT_FILE=.backlog/prefill_prompts/p1024.txt .backlog/lab/tokps.sh \
  'wp_off=@=COLI_V4_MOE_WHOLE_PROMPT=0' 'wp_on=@=COLI_V4_MOE_WHOLE_PROMPT=1'
```

## B3 — OpenMP thread count, resolution grade
```bash
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
TOKENS=40 N=5 PROMPT_FILE=.backlog/prefill_prompts/p256.txt .backlog/lab/tokps.sh \
  't16=@=OMP_NUM_THREADS=16' 't12=@=OMP_NUM_THREADS=12'
```

## T6 — a better mxfp4 expert kernel (the largest remaining lever, NOT attempted)
`expert_forward` is 41.7% of decode after tonight's work. Both existing fp4 kernels (scalar and
NEON rows16) sit at 16-19 GB/s against a measured ~105 GB/s host read ceiling, and they are within
1.00-1.14x of each other, so **raising `COLI_V4_PIN_SLOTS` is closed** (E107 vindicated by direct
measurement, E126).

Identified inefficiency, unproven: `neon_rows16_accumulate` (c/deepseek_v4.c:14572) applies the
block scale PER ELEMENT --
    sums[g] = vaddq(sums[g], vmulq(vmulq(x, values), scales[g]))
-- two multiplies and an add, where the scale is constant across each 32-column group. Accumulating
unscaled with one FMA and applying the scale once per block would be ~1.125 ops/element instead of
3. Ceiling if it delivered 2x on the phase: **+26% decode**.

Two reasons it is not a quick win:
1. The no-FMA ordering is DELIBERATE -- it is what makes the CPU rows16 path bit-identical to the
   Metal kernel (`validation/metal/probe_rows16_parity.m`). Changing it breaks that parity and needs
   the task-level capability gate rather than golden.
2. Only ~6% of decode expert calls take the NEON path at all (16 pinned of 256 per layer), so a
   faster NEON kernel must ALSO be made reachable -- either by a cold-layout permute like E125's or
   by widening pinning, and pinning costs +66% `expert_wait` (E123).

A valid sizing test must keep the existing `vqtbl1q`/`vqtbl4q` gather and change ONLY where the
scale is applied; my first attempt replaced the gather with scalar rebuilds and measured 0.83x,
which says nothing about the hypothesis. Estimated 2-4 h with real risk.


## B4 — headline validation: COLI_V4_BASELINE=1 vs shipping defaults  (DONE 2026-08-30, E128)
**RUN AND RECORDED — see E128 in `experiments_results.md`.** Ran at N=3 not N=5 (effect >50%, far
above the 5-13% decode noise floor; 10 min instead of 17). Result: 1.3948 -> 2.1596 tok/s
**+54.83%**, TTFT -62.4%, net wall @40 tokens -57.1%, each arm deterministic 3/3.
Outcome: the chained +28.1% endpoint REPRODUCES (2.1596 vs 2.1341), but the totals differ because
`COLI_V4_BASELINE=1` also disables `KERNELS=all` (`c/deepseek_v4.c:10693`). Report as a
decomposition, never as one number. Log: `.backlog/lab/B4_headline_baseline_vs_shipping_*.log`.

**Why it is worth running.** Every figure quoted so far is a CHAIN of A/Bs (E125 +10.18%, then
E126/E127 +15.27%, composed to +28.1%). That composition has never been measured directly. This run
answers the only question a user actually asks — what do I get today versus the historical engine —
in one measurement, and would either confirm the chained claim or expose drift in it.

```bash
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
TOKENS=40 N=5 PROMPT_FILE=.backlog/prefill_prompts/p256.txt .backlog/lab/tokps.sh \
  'baseline=@=COLI_V4_BASELINE=1' 'shipping=@='
```
Expect the baseline arm to be slow (no fast kernels, no GPU prefill): TTFT ~113 s, decode ~28 s, so
budget ~20 min for the pair. Report BOTH axes with N and the determinism verdict. Note the two arms
will NOT share an output md5 — the shipping path is capability-equivalent, not token-identical, so
judge it with `.backlog/lab/taskcheck.sh`, never by hash equality.

## B5 — p512 decode generalisation of E126/E127  (~16 min, PARKED)
Operator decision 2026-08-30: backlog for now.
Third data point after p064 (+13.20%) and p256 (+15.27%). Cheaper than the old B1 estimate because
tonight's work cut p512 TTFT substantially.
```bash
cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot
OFF="COLI_V4_HEAD_ILP=0 COLI_V4_HC_OMP=0 COLI_V4_FP8_DUAL_ROWS16=0 COLI_V4_SPARSE_OMP=0 COLI_V4_INDEXER_OMP=0"
TOKENS=40 N=5 PROMPT_FILE=.backlog/prefill_prompts/p512.txt .backlog/lab/tokps.sh \
  "off=@=$OFF" 'on=@='
```
Prediction to test: the gain should sit at or slightly above the p256 +15.27%, because `attn_sparse`
scales with KV candidate count and its 4.69x therefore contributes more at longer context. If it
comes in BELOW p064's +13.20%, that prediction is wrong and worth chasing.
