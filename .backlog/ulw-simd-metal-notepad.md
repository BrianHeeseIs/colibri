# Ultrawork Notepad — wire the `simd` Metal matmul into the production V4 expert path
Branch: `simd-apple-metal`  (from `perf-upstream-adopt` @ f4110ca)
Started: 2026-08-26

## Goal
`coli_v4_matmul_mxfp4_simd` (`c/metal/coli_v4_matmul.metal:58`) is 6.6-7.8x CPU at S=1 but is
dispatched ONLY by `validation/metal/bench_matmul`. It has never run in production and its
OUTPUT HAS NEVER BEEN CHECKED FOR CORRECTNESS. Wire it in, gated, and measure TTFT + tok/s.

## Scenarios (the contract)
| id | class | pass condition | real-surface artifact |
|---|---|---|---|
| S1 | correctness (happy) | simd output vs CPU scalar reference: max rel err < 1e-5 on all 3 shapes | `validation/metal/probe_simd_parity` stdout |
| S2 | edge | non-multiple-of-32 I, and O not a multiple of 16, produce correct results (tail guard) | probe stdout on ragged dims |
| S3 | task-level capability | `.backlog/lab/taskcheck.sh` PASSES with the flag ON (non-bit-exact bar) | taskcheck log |
| S4 | decode perf | `.backlog/lab/tokps.sh` n>=5, median tok/s ON vs OFF, arm md5 recorded | tokps log |
| S5 | TTFT perf | `bench/ab.sh` n>=3 p064+p256, remember FASTER IS NEGATIVE | ab.sh log |
| S6 | adjacent-surface regression | `bench/golden.sh` still PASS md5=5d04890413ff539e802985ce8c727814 with flag OFF | golden log |

## Findings
- **E94 REPRODUCED AND EXCEEDED** (`./validation/metal/bench_matmul 20`, M3 Max, OpenMP 16 threads,
  engine not running):
  | shape | CPU ms | best EXACT | simd | simd vs CPU |
  |---|---|---|---|---|
  | gate\|up S=1 4096->2048 | 0.438+-0.009 | 0.404 (1.08x) | 0.066+-0.002 | **6.65x** |
  | down S=1 2048->4096 | 0.426+-0.005 | 0.190 (2.24x) | 0.055+-0.001 | **7.80x** |
  | gate\|up S=8 4096->2048 | 3.278+-0.017 | 0.417 (7.87x) | 0.360+-0.001 | **9.09x** |
  E94 recorded 5.56x / 6.34x / (S=8 n/a); this run is higher. Same binary, same command.
- **INTEGRITY GAP in the E94 evidence**: `bench_matmul.m:95` computes the mismatch count against
  `yg`, which is filled ONLY by the `ordered` dispatch (`:94`, `out=yg`). Every other variant,
  including simd (`:99`), is dispatched with `out=NULL`. So the "simd is non-bit-exact" line is
  not a measurement — and more importantly **simd has never been checked for being CORRECT at
  all.** A kernel returning garbage would benchmark exactly this fast. S1 exists to close this.
- **Mechanism hypothesis for the S=1 win**: `bench_matmul.m:46-48` dispatches `ordered` as
  `S*O` threads (2048 at S=1) but `simd` as `32 x S*O` (65536). At S=1 the GPU is thread-starved;
  simd is a 32x occupancy increase, not a smarter inner loop. Prediction that follows: the
  advantage should SHRINK as S grows — and it does (6.65x at S=1 -> 1.16x over best-exact at S=8).
  **So simd is primarily a DECODE / tok-s lever, and should be expected to do little for TTFT.**
  State this before measuring so the TTFT result cannot be retrofitted.

- **S1 RED then GREEN — and the headline result of this session.**
  `validation/metal/probe_simd_parity` (new) was RED: `simd` max_rel **1.06e-3** on production
  shapes, three decimal digits. Narrowing the UE8M0 scale band from 30 exponents to 4 does NOT
  reduce it (1.49e-3) — the error is intrinsic to the tree reduction, not adversarial data.
- **Root cause of the divergence**: the scalar reference contracts `ga += x*w` into a single
  **fma**, so the product is never separately rounded. `simd` shuffles the ALREADY-ROUNDED
  product into `simd_sum`, inserting one extra rounding per column.
- **NEW KERNEL `coli_v4_matmul_mxfp4_simd_exact`** (`c/metal/coli_v4_matmul.metal`): lane L owns
  GROUP gb+L, runs that group's 32-FMA chain privately (serial => exact), and only the outer
  `a = fma(ga_g, sc_g, a)` uses shuffles — both factors shuffled so the outer link stays one fma.
  Dependent chain ~4096 links -> ~(32+ng). Consecutive lanes read consecutive 16-byte weight
  spans (one contiguous 512 B burst/round).
  **BIT-EXACT on 12 shape/seed/scale-band combinations, ~48k values, zero mismatches.**

  | shape | CPU | ord+xc (PRODUCTION) | simd (approx) | **simd_exact (BIT-EXACT)** |
  |---|---|---|---|---|
  | gate\|up S=1 | 0.428 | 0.379 | 0.079 | **0.072  = 5.3x prod** |
  | down S=1 | 0.417 | 0.199 | 0.055 | **0.068  = 2.9x prod** |
  | gate\|up S=8 | 3.012 | 0.416 | 0.389 | **0.387  = 1.08x prod** |

  This SUPERSEDES the E94 lead: E94's best was a NON-bit-exact kernel needing a capability gate.
  simd_exact is within 7-14% of it and needs no accuracy gate at all.
- **DEAD (measured, do not retry): column-split bit-exact simdgroup reduction.** First cut of
  simd_exact put lane L on COLUMN base+L and replayed the inner sum with per-column shuffles.
  Bit-exact, but **1.374 ms vs 0.379 production = 3.6x SLOWER**: it keeps the full 4096-deep
  dependent FMA chain and adds two shuffles per link, 31/32 lanes redundant. The win comes from
  splitting on GROUPS, not columns.
- **Production kernel is `ordered_xcache` / `ordered_hot_xcache`, NOT `ordered`**
  (`c/backend_metal_v4.mm:371-378`). Speedups must be quoted against xcache.
- **The decode S=1 GPU path ALREADY EXISTS**: `moe_token_pipeline` calls
  `coli_v4_metal_expert_forward` per routed expert (`c/deepseek_v4.c:4974`, `:5022`), gated on
  `coli_v4_metal_enabled()` i.e. `COLI_V4_METAL=1`; that delegates to
  `coli_v4_metal_expert_forward_batch(batch=1)` (`.mm:979-990`, rejects block_rows!=1).
  Memory records `COLI_V4_METAL=1` as **1.118x SLOWER** than CPU — consistent with it running
  `ordered_xcache`. So the lever is: swap the matmul pipeline, re-measure that same flag.
- `COLI_V4_METAL_VARIANT` already parses `simd_cold`=2 / `simd_hot`=3 (`.mm:394-404`) but
  `expert_forward_batch` **rejects any variant != 0** (`.mm:667-670`). Half-built seam.
- The whole 9-dispatch expert chain shares ONE command buffer + encoder, committed once
  (`.mm:~730-955`). So per-expert GPU overhead is one submit+wait, not nine.

## RESULT (session end)
Shipped on branch `simd-apple-metal`. `COLI_V4_METAL_VARIANT=simd_exact_cold`, default OFF.
- Correctness: kernel parity 0/48k mismatches; SEAM differential GREEN (identical digests at
  batch 1 and 8); golden x3 PASS `5d04890413ff539e802985ce8c727814`; execution counter 0 -> 11544;
  p256 multi-chunk differential identical md5 `e44c2b5ca42288fcc5e06888a1c62497`, both arms
  deterministic.
- Perf (p256, N=2, PROVISIONAL): **+33.90% tok/s** (0.9724 -> 1.30205) and **-16.6% TTFT**
  (143.7 -> 119.9 s) vs `metal_ord`.
- **The TTFT gain contradicts the E95 pre-registration** (predicted ~0). Reported as a surprise in
  E96, not explained away. Cause: prefill is not all large-S; the batched path gates at MIN_N=4 so
  it dispatches a distribution of group sizes starting at 4, where simd_exact is strongest.
- Verdict under the pre-registered rule: **KEEP as documented opt-in, default OFF** — the CPU arm
  was not run, so the default-on branch is not evidenced and is not claimed.
- Deferred runs (p512 differential, 3-arm N=5 at p064/p256 incl. CPU, ab.sh TTFT) with exact
  commands: `.backlog/simd-exact-remaining-measurements.md`.

## Learnings
- **A harness that TIMES a variant must also CHECK it.** `bench_matmul` dispatched 5 of 6 variants
  with `out=NULL`; `simd` carried a 1.06e-3 error through many benchmark runs unnoticed.
- **Unknown enum values that silently fall back to the baseline are a measurement hazard.**
  `COLI_V4_METAL_VARIANT=<typo>` parses to 0 = production, so an A/B would compare the baseline
  with itself and report a clean match. The probe now refuses that state; it fired for real.
- **Reject vs fall back matters for measurement, not just behaviour.** Rejecting an unsupported
  layout sends the expert to the CPU, converting a kernel A/B into a GPU-vs-CPU swap.
- **FMA contraction is part of the bit-exactness contract.** `ga += x*w` contracts to one fma;
  any GPU form that shuffles the pre-rounded product adds a rounding per column. Shuffle BOTH
  factors and keep the fma.
- Split a simdgroup over GROUPS, not columns. Columns keeps the full dependent chain (3.6x slower
  than production); groups shortens it from ~4096 links to ~(32+ng).

## Todo
see TODO list in session

## Learnings
- (append as they land)
