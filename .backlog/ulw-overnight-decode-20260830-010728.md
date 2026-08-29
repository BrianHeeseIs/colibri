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
