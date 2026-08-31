# Backlog — gated work and parked benchmark runs

## P0 — mxfp4 expert kernel  *** CLOSED 2026-08-30, see E129. DO NOT RE-OPEN ***
Ran under operator permission. **No kernel was written — two measurements taken first closed it.**

1. **The "~6% of decode expert calls reach NEON" figure in this entry was WRONG.** It was inferred
   from "16 pinned of 256 per layer" = 6.25%. Measured with a new counter: **22.05%** (rows16 2227,
   scalar 7872, total 10099). Off by 3.5x — the inference assumed uniform routing, but pinning
   selects the HOTTEST experts, which is this engine's whole premise.
2. **The "+26% ceiling" in this entry was never reachable.** It assumed 2x on the phase. A ceiling
   arm that abandons parity entirely (scale hoisted + FMA + reassociation + 16 chains) measures
   **1.154x / 1.201x**. Against the measured coverage that is **+4.88% decode** for the cold path
   and **+6.26% even if both kernels were rewritten** — at or under the 5-13% decode noise floor,
   before the 2.15-3.58x probe-to-real dilution recorded on this host.
3. Widening accumulator chains — the mechanism that gave the LM head 2.7x in E126a — is **flat**
   here: 4/8/12/16 chains give 1.00/1.09/0.95/0.99, non-monotone, 12 consistently worse.
4. The cold path **already hoists the block scale** (`c/quant.h:1456,1481`), so the inefficiency
   named in this entry only ever existed on the rows16 kernel, which serves the minority of calls.

Kept from the attempt: the kernel-split counter (both golden gates pass with it in) and
`.backlog/lab/kbench/fp4disc.c`. Remaining decode levers are `expert_wait` (~1395 ms, structural)
and the streaming/residency path — not the expert arithmetic.

### Open, spun out of P0: two ue8m0 decoders disagree (correctness, not performance)
`mx4_scale` (`c/quant.h:1437`) gives `s=255 -> +inf`; `coli_e8m0_decode` (`c/deepseek_v4.c:13503`)
returns **NaN** for `0xff`. Same byte, same engine, different value. The MXFP4 kernels follow
`mx4_scale`. No current gate exercises it. Unresolved.

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

## T6 — a better mxfp4 expert kernel  *** SUPERSEDED / CLOSED by E129, see P0 above ***
**Everything below this banner is the PRE-MEASUREMENT reasoning and two of its numbers are now
known wrong: the "~6%" coverage (measured 22.05%) and the "+26% ceiling" (measured 1.15-1.20x, worth
+4.88%). Retained only as the record of what was believed before it was measured. Do not act on it.**

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

## B5 — p512 decode generalisation of E126/E127  *** DONE 2026-08-31, see E135 ***
**RESULT: CONFIRMED.** D = +17.65% tok/s (off 1.8593 -> on 2.1874), non-overlapping ranges, N=5.
TTFT -18.38%, net wall @40 tokens -17.70%. All ten runs across both arms shared ONE md5
d7f7c51a0421eb149360131e10450a6b, so E126/E127 bit-exactness generalises to 512 tokens (8 chunks).
Series is monotonic: p064 +13.20%, p256 +15.27%, p512 +17.65%. Full working in
`.backlog/E135-b5-p512-generalisation.md`.

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
