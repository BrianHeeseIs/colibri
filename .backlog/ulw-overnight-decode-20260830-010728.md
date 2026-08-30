# Ultrawork Notepad — squeeze remaining decode performance (overnight, autonomous)
Started: 2026-08-30 (operator asleep; questions BANKED until "good morning"/"good afternoon")
Notepad lives in .backlog/ NOT /tmp — AGENTS.md line 1. Template says mktemp; project rule wins.

## Operator contract for this run
- Bank ALL questions; ask via the question tool only when greeted.
- Keep optimizing decode on this host.
- Work until genuinely blocked on operator input.
- Any benchmark >25 min => write to .backlog/benchmark-backlog.md, DO NOT run.
- Never burn wall time on a long benchmark when a short one proves the same thing.
- When experiments run out, MAY run remaining relevant long benchmarks from the backlog and record.

## Baseline (clean, post-E125, p256, 39 generated tokens)
decode_wall 20977.5 ms, accounted 99.2%, tok/s ~1.859 (N=5 median 21.253 s => 1.8350 tok/s)
  expert_forward  7557.1  36.0%
  attention       6697.6  31.9%  (attn_out 3724.7 / attn_qkv 1767.9 / attn_sparse 1139.9)
  head            1399.6   6.7%
  expert_wait     1394.6   6.6%
  shared_expert   1372.6   6.5%
  hc_norm         1092.4   5.2%
  indexer          616.7   2.9%
  compressor       557.3   2.7%
Golden gates: bench/golden.sh md5 5d04890413ff539e802985ce8c727814 (SACRED),
              bench/golden_default.sh md5 cc09015d089d9a25d10d75753f9e849a (not sacred).
Seed .backlog/lab/coli_usage.snapshot md5 599f3d12e9347ef30541bd6f9ba18bde.
Branch ft-decode-apple-metal @ 5bbcbea, pushed to fork, tree clean.

## Instrument cost (drives every choice below)
  unit test / microbench      seconds
  COLI_V4_PROFILE=1 run       ~90 s   <- isolates ONE phase; primary instrument
  golden.sh / golden_default  ~2 min / ~1 min
  tokps N=5 x2 arms p256      ~15 min <- only for the AGGREGATE claim, once
  tokps N=5 x2 arms p512      ~35 min <- BACKLOG, do not run
RULE: prove kernel deltas with the 90 s profile. Spend the 15 min N=5 once, at the end.

## Targets, ranked (Amdahl from the baseline above)
T1 head_bf16_dot NEON            6.7%  2x -> +3.5%   low cost  (AVX2-only; bf16->f32 is a shift)
T2 fp8 DUAL matvec rows16       ~4.4%  2x -> +2.2%   low cost  (direct extension of shipped E125)
T3 hc_norm parallel + scratch    5.2%  4x -> +4.1%   low-med   (zero OpenMP today, 3 mallocs/call)
T4 decode-path scratch pass       n/a  unquantified  med       (~50k malloc/free per generation)
T5 OMP_NUM_THREADS=12             n/a  unknown       zero code (V4 never adopted omp_tune.h)
T6 better fp4 expert kernel     36.0%  2x -> +22%    HIGH      (must beat existing NEON, not scalar)
T1+T2+T3 compound ~= +10.1%

## Scenarios (the contract) — apply to EVERY kernel change below
S1 CORRECTNESS: bench/golden.sh PASS (sacred md5) AND golden_default.sh PASS, with the change ON.
   If not bit-exact, that is recorded openly and gated on .backlog/lab/taskcheck.sh 5/5 instead.
S2 ENGAGEMENT: a counter proves the new path ran — 0 with the flag off, >0 with it on (E101 rule).
S3 PHASE DELTA: COLI_V4_PROFILE=1 A/B shows the TARGET phase down, other phases not up.
S4 UNIT: a test with the real shapes + adversarial values goes RED before the code, GREEN after.
S5 AGGREGATE (once, at the end): tokps N=5 p256, non-overlapping ranges on decode.
S6 NO REGRESSION: test_fp8_rows16 still PASSES; prefill TTFT not worse.

## Banked questions for the operator (ask via question tool on greeting)
(none yet)

## Now
Wave 0: ground the three targets in source, then plan agent.

## Results — night 3
T1 head ILP        head 1399.6 -> 524.4 ms (2.67x)   decode_wall 20977.5 -> 20049.1  committed 1ceeba9
T3 hc_norm omp     hc_norm 1092.4 -> 398.3 ms (2.74x) decode_wall 20049.1 -> 19339.4  committed f5778d3
T2 fp8 dual rows16 shared_expert 1372.6 -> 965.1 ms   decode_wall 19339.4 -> 19106.2  committed 0b8f36f
AGGREGATE N=5      tok/s 1.8533 -> 2.0238 (+9.20%), TTFT -6.3%, both non-overlapping, md5 identical
SESSION TOTAL      1.6655 -> 2.0238 tok/s = +21.5% (E125 +10.18%, E126 +9.20%), all bit-exact

T5 OMP_NUM_THREADS: NULL RESULT, keeping 16.
    16 threads 18987.5 ms | 12 threads 19175.6 (worse) | 10 threads 18704.8 (better)
    Spread 2.5%, inside the 5-13% decode noise floor and below the 3% adoption bar. NOT escalated to
    N=5: that is backlog item B3 at ~45 min for an effect that cannot exceed ~2.5%. This also
    contradicts omp_tune.h's P-cores-only policy for THIS workload -- the E-cores are not hurting.

## Ranking AFTER E126 (from the post-T2 profile, decode_wall 19106.2)
  expert_forward 7594.6  39.8%   <- dominant, but see the fp4 head-to-head: existing NEON is only
                                    1.03-1.14x over scalar, so coverage work caps at ~+3.5%
  attention      6702.8  35.1%   (attn_out ~3725, attn_qkv ~1768, attn_sparse ~1140)
  expert_wait    ~1395    7.3%
  shared_expert   965.1   5.1%
  head            527.4   2.8%
  hc_norm         402.0   2.1%
  indexer        ~617     3.2%
  compressor     ~557     2.9%
expert_forward + attention = 75% of decode.

## Banked questions for the operator (ask via question tool on greeting)
Q1 The fp4 expert phase is 39.8% but its existing NEON kernel is only 1.03-1.14x over scalar, and
   both sit at 16-19 GB/s of a ~105 GB/s ceiling. Writing a genuinely better fp4 kernel is the only
   large remaining lever and is a multi-hour, higher-risk project. Do you want it attempted?
Q2 Should the three new flags stay default-ON (they are bit-exact and both golden gates pass), or
   would you prefer any of them shipped off until you have reviewed?
Q3 Backlog B1/B2 (p512 and p1024 sweeps, ~35 min each) are authorized-if-idle. Do you want them run,
   or is wall-clock better spent on new kernels?

## fp4 expert kernel probe — INCONCLUSIVE, and the flaw was mine
Found the likely inefficiency by reading c/deepseek_v4.c:14572:
    sums[g] = vaddq_f32(sums[g], vmulq_f32(vmulq_f32(x, values), scales[g]));
That is TWO multiplies and an add PER ELEMENT. The block scale is constant across each 32-column
group (block_scales is already hoisted to the base+=32 loop at :14679) yet it is still multiplied
into every element. Hoisting the multiply as well would give ~1.125 ops/element instead of 3, a
2.7x arithmetic reduction. My fp8 rows16 kernel does ONE fma per element by folding its scale.
NOTE the no-FMA form is DELIBERATE there: a comment and validation/metal/probe_rows16_parity.m say
it is what makes the CPU path match the Metal kernel bit-for-bit. So this is not free.

I tried to size it in .backlog/lab/kbench/fp4bench.c (variant V2) and the test is INVALID:
  V2 measured 0.83-0.88x vs the existing NEON and diverged with worst rel 262.
Cause: I replaced the kernel's vqtbl1q/vqtbl4q gather with vst1q_u8 to memory plus sixteen scalar
array reads to rebuild each float32x4. That decode is far slower than what it replaced, and my
scale-lane indexing was wrong too. So V2 measures MY BAD DECODE, not the hoisted-scale hypothesis.
=> The hypothesis is NEITHER confirmed NOR refuted. A valid test must keep the vqtbl gather and
   change ONLY where the scale is applied, which needs the engine's static NeonRows16Tables.
Same run also re-measured scalar vs existing NEON at 1.00-1.08x (previously 1.03-1.14x), which
further confirms that raising pin coverage is not a lever.
