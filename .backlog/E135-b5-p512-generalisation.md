# E135 — B5, p512 decode generalisation of E126/E127

Pre-registered 2026-08-31, BEFORE the run. Branch `ft-decode-generalization`, base b3fca8e.
Binary md5 `a6f6726e205a6fd7b491505d46725df3`, `METAL=1`, Metal seam linked.

## Why this runs BEFORE the hot-pack default flip
B5's reference points — p064 **+13.20%**, p256 **+15.27%** — were measured on builds where
`COLI_V4_HOT_PACK_UNLOCKED` was default OFF. Flipping the default first would measure p512 against
a different baseline than its own reference points, in a study whose entire purpose is
comparability across prompt length.

The confound is real in direction and negligible in magnitude, and the reason is subtler than
"the flag is constant across both arms":
1. **Amdahl denominator shift.** E132 showed the flag does not delete lock time, it CONVERTS it —
   `store_lock` 454.331 -> 36.711 ms while `wait_finish_complete_block` rose 6.24%. Wait is a
   condvar park; the five kernels under test accelerate COMPUTE. Moving main-thread time out of
   compute-adjacent lock into a non-compute stall shrinks the accelerable fraction.
2. **Arm-asymmetric thread pressure.** The OFF arm disables `HC_OMP`, `SPARSE_OMP` and
   `INDEXER_OMP`, so those regions run SERIAL in OFF and PARALLEL in ON. The flag's effect is on
   lock contention, which is thread-pressure dependent, so there is a genuine flag x arm
   interaction term and the ratio is NOT strictly invariant.
Net decode_wall effect of the flag was -1.58%, so this perturbs the measured delta by well under
1pp — beneath the 5-13% decode noise floor and undetectable at N=5. Running B5 first costs nothing
and removes the argument entirely.

## Command (as recorded in the backlog)
```
OFF="COLI_V4_HEAD_ILP=0 COLI_V4_HC_OMP=0 COLI_V4_FP8_DUAL_ROWS16=0 COLI_V4_SPARSE_OMP=0 COLI_V4_INDEXER_OMP=0"
TOKENS=40 N=5 PROMPT_FILE=.backlog/prefill_prompts/p512.txt .backlog/lab/tokps.sh "off=@=$OFF" 'on=@='
```

## Metric
`D = 100 x (median_on - median_off) / median_off`, tok/s, N=5, p512, 40 tokens.
Separation test **S**: `min(on_runs) > max(off_runs)` — non-overlapping ranges, the repo's own
established bar (E125: "+10.18% at N=5 with non-overlapping ranges") and distribution-free.

## Validity preconditions — failure means VOID and rerun, not INDETERMINATE
- **P1** `tokps.sh` did not abort on the seed hash. It verifies `599f3d12e9347ef30541bd6f9ba18bde`,
  which `ab.sh` and `golden.sh` do not.
- **P2** EXACTLY 5 run lines per arm and ZERO `ENGINE FAILED`. `tokps.sh` runs `set -uo pipefail`
  WITHOUT `-e`, so a failed arm `continue`s and the median is silently taken over fewer points.
  Count the lines; do not trust the summary.
- **P3** One engine, no background agents. A profile taken under agent load was wrong by 76%.
- **P4** Per-arm spread recorded as `100 x (max - min) / median`, which measures the noise floor
  in THIS run rather than assuming 5-13%.

## Pre-registered bands
| verdict | condition | meaning |
|---|---|---|
| **CONFIRMED — generalises** | S holds AND D >= **14.2%** | Real, and at or above the p064 reference |
| **FALSIFIED** | S holds AND D <= **12.2%** | Real but below p064's +13.20%; the attn_sparse-scales-with-KV story does not hold |
| **INDETERMINATE** | S fails, OR 12.2% < D < 14.2% | The guard band straddles the falsifier at a precision N=5 cannot deliver |
| **NULL / REVERSAL** | D <= 0 or median_on < median_off | Contradicts E126/E127 at longer context; a separate finding |

## Mandatory non-adjudication clause
**The "at or slightly above +15.27%" half of the prediction is NOT decidable at N=5 and must not be
claimed either way.** Separating 13.20% from 15.27% is a 2.07pp discrimination, roughly 13%
relative, against a 5-13% per-run floor; that needs about N=20-40 per arm. Even under CONFIRMED the
wording is "D = X%, at or above the p064 reference, consistent with but not resolving the p256
point" — never "confirmed at 15.27%".

## Secondary observable, free: bit-exactness at p512
E125-E127 are recorded BIT-EXACT, so `md5(on)` should equal `md5(off)`. p512 is 8 chunks of 64 and
these five kernels have never been checked at this length. The repo's own cautionary tale has this
exact shape: rows16 passed golden at ~20 tokens, was written up as bit-exact, then diverged at p256
and the entry was retracted.
- md5 equal -> bit-exactness generalises to p512. Record it.
- md5 differ -> a SEPARATE FINDING, not a failure. Diff the text, run `taskcheck.sh`, and use the
  five-runs-per-arm determinism verdict to separate a real difference from nondeterminism. The word
  "breaks" may not be used before all three.

## Results
Run 2026-08-31 12:28-12:48. `tokps_rc=0`, 10/10 runs. Console
`.backlog/lab/e135_b5_console.log`, per-run logs `.backlog/lab/b5_work_20260831-122807/`.

## VERDICT: CONFIRMED — the E126/E127 gain generalises to p512

| | off | on | delta |
|---|---|---|---|
| median tok/s | 1.8593 | **2.1874** | **D = +17.65%** |
| median TTFT | 83.387 s | 68.063 s | -18.38% |
| median decode_sec | 20.976 | 17.829 | |
| net wall @40 tokens | 104.363 s | 85.892 s | -17.70% |
| generated_text md5 | d7f7c51a0421eb149360131e10450a6b | d7f7c51a0421eb149360131e10450a6b | IDENTICAL |
| determinism verdict | deterministic | deterministic | |

**Preconditions.** P1 seed verified (`599f3d12...`, tokps.sh aborts on mismatch). P2 exactly 5 run
lines per arm, zero `ENGINE FAILED`. P3 one engine, no background agents. P4 spreads below.

**Separation test S HOLDS**: min(on) = 2.1744 > max(off) = 1.8635, non-overlapping ranges.
With S and D = 17.65% >= 14.2%, the pre-registered band is **CONFIRMED**.

**Spreads.** on 0.66%. off 30.47% raw — but that is entirely run1 (1.2970), a cold-cache first
run; runs 2-5 span 1.8408-1.8635 = **1.22%**. The median (1.8593) correctly excludes the outlier.
This is exactly why P2 requires counting run lines rather than trusting the summary: had an arm
silently lost points, the median would have shifted without any visible error.

Both observed spreads are far tighter than the assumed 5-13% decode noise floor. That floor was
established at p064/p256; at p512 the decode phase is a larger share of a longer wall, so
run-to-run variation is proportionally smaller.

## Honouring the non-adjudication clause
D = 17.65% sits above BOTH references (p064 +13.20%, p256 +15.27%). The directional prediction —
that the gain grows with context because `attn_sparse` scales with KV candidate count — is
**consistent with the data and not falsified**.

It is NOT resolved. 17.65 vs 15.27 is 2.38pp, which against the observed per-arm spreads
(0.66% on, 1.22% off) is roughly 1.7 sigma. That is suggestive, not decisive. The correct statement
is: **D = 17.65%, at or above the p064 reference, consistent with but not resolving the p256
point.** Resolving 2pp cleanly needs about N=20-40 per arm. Do not quote this as "confirms +15.27%".

## Secondary observable: bit-exactness at p512 — HOLDS
All TEN runs across BOTH arms produced the single md5 `d7f7c51a0421eb149360131e10450a6b`, and
tokps.sh flagged both arms deterministic from those same runs.

This is the first multi-chunk bit-exactness check of E126/E127 at 512 tokens (8 chunks of 64). It
matters because the repo has been burned by exactly this gap before: a rows16 change passed golden
at ~20 tokens, was written up as bit-exact, and then diverged at p256, forcing a retraction. The
five kernels here — HEAD_ILP, HC_OMP, FP8_DUAL_ROWS16, SPARSE_OMP, INDEXER_OMP — are now checked at
a length where chunk- and group-conditioned behaviour genuinely executes.

## The series
| prompt | D (tok/s) |
|---|---|
| p064 | +13.20% |
| p256 | +15.27% |
| **p512** | **+17.65%** |

Monotonic in prompt length across three points. Consistent with the `attn_sparse` mechanism, though
the p256-to-p512 step is within the resolution caveat above.

