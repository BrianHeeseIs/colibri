# Benchmark backlog — runs costing >25 minutes
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
