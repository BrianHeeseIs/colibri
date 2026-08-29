# Ultrawork Notepad — review m3-max-decode-research + find further decode avenues
Started: 2026-08-29 (post-E125)
Notepad in .backlog/ NOT /tmp, per AGENTS.md line 1 (the template says mktemp; project rule wins).

## Goal
Review .backlog/m3-max-decode-research-2026-08-29.md and identify/validate FURTHER decode levers.
I'll stop right away when: the document is corrected against E125, decode is re-profiled post-E125,
and every remaining avenue is ranked with a measured or explicitly-unmeasured basis + a kill criterion.

## Status of the document under review (written 14:41, BEFORE E125 landed ~15:20)
Its central recommendation (P0: packed-row16 FP8 + FP16 reinterpretation) is now DONE and SHIPPED.
=> The document is STALE on its own headline. Correcting it is part of this task.

### What it got RIGHT (independently confirmed by my implementation)
- rows16 + exact E4M3->FP16 reinterpretation is the winning design (line 34, 105).
- E104's "no bit-exact arm64 kernel" is refuted (line 117) - I proved this by measurement.
- "The resident layout CAN be memory-neutral ... only a temporary packing tile" (line 229).
  STRONGER THAN IT KNEW: memory-neutrality is not optional, it is REQUIRED. My first integration
  used a side-by-side packed copy; it cost +4.0 GB peak RSS (87.8 -> 91.8 GB) and REVERSED the win
  (-2.99% tok/s, 20% spread). Permuting in place gave +10.18%.
- "Producer/consumer mismatch is the critical corruption hazard" (line 229) - exactly what bit:
  coli_fp8_matmul_batch_ref:13801 validates block_rows and returns -1 for anything but 128/8, so a
  naive in-place repack makes PREFILL hard-fail.
- "cover all 256 E4M3 codes" (line 105/231) - the NaN codes 0x7F/0xFF are the real trap; the
  reinterpret maps them to a FINITE f16. I planted them in the test and proved the guard by deleting
  it and watching every real shape fail with ref=nan neon=42.59.

### What it got WRONG / over-estimated
- Derived scenario "+23% tok/s (1.67 -> 2.05)" (line 40). ACTUAL: +10.18% (1.6655 -> 1.8350),
  N=5, non-overlapping. The kernel DID hit 2.09-2.28x at real shapes and the profile confirms
  attn_out -26.7% / attn_qkv -31.3% -- but those phases also contain rope, bf16 rounding and the
  SERIAL activation QDQ, which the kernel does not touch. Its own Amdahl table (line 178) bounded
  2x-on-32.4% at 1.193x; the realised 1.102x is consistent with that bound, so the +23% line was the
  optimistic end and should be retired.
- It did not concretely trace WHICH other readers share the weight bytes; it named the hazard
  abstractly. The concrete set is: coli_fp8_matmul_batch_ref (prefill), Metal prefill attention,
  deepseek_v4_dspark.inc, and a SECOND attention implementation at :2346-2508.

## Scenarios (the contract)
S1 RE-PROFILE: a fresh post-E125 decode profile at p256 exists, accounted >=99%, showing the NEW
   phase ranking. Binary: v4_profile lines captured with decode_wall and per-phase ms.
S2 RANKED AVENUES: every remaining avenue has (a) a share-of-decode basis from S1, (b) an Amdahl
   ceiling, (c) a cost, (d) an explicit kill criterion. No number without a source.
S3 CHEAP LEVER TESTED: at least one zero-code lever measured with a real observable (power mode
   state and/or OMP thread count), reported with N and ranges, or explicitly declared
   operator-gated with the reason.
S4 ADJACENT REGRESSION: nothing I do breaks the shipped state - golden gates still PASS and
   test_fp8_rows16 still passes at the end.
S5 DOC CORRECTED: the reviewed document carries the E125 outcome so the next reader is not misled
   by a stale +23% headline.

## Now
Wave 0: power/thermal state check + parallel explores.

## Findings — wave 0
F1. POWER STATE: the document (line 323) says "Low Power Mode Yes on AC; High Power No".
    That is WRONG for the current host. `pmset -g custom` shows the AC profile at `powermode 2`
    (0=automatic, 1=low, 2=high) and `pmset -g` confirms `powermode 2` in use while on AC Power.
    => High Power Mode is ALREADY ON. The document's "P0 operational: Automatic/High Power A/B"
    candidate offers NO GAIN and should be struck. (Battery profile is powermode 0; irrelevant here.)
    Thermal: `pmset -g therm` records no thermal or performance warning => not throttled.
F2. *** BENCHMARK CONTAMINATION - METHODOLOGY LESSON ***
    My first post-E125 profile read decode_wall 28500 ms and expert_forward 13290 ms, against a
    measured N=5 median of 21253 ms and E123's expert_forward of 7528 ms. Not a real change:
    `ps -Ao pcpu -r` showed `opencode` at 235.4% CPU with load average 5.72, i.e. the two explore
    agents I had just launched were competing with the engine for the same 12P+4E cores.
    AGENTS.md enforces "one engine at a time" but says nothing about AGENT CPU LOAD during a run.
    It should: the engine takes 16 OMP threads, so any concurrent agent directly inflates decode.
    RULE ADOPTED FOR THIS SESSION: no background agents in flight while a timing run executes.
    This also retroactively explains the single anomalous OFF-arm reading (31964 ms) I flagged and
    refused to quote during the E125 A/B - same cause. The N=5 runs were clean (tight ranges:
    off [23.243,23.871], on [21.040,21.401]) because no agents were running then.
    Memory was NOT the cause: 93% free, swap 4.5 GB is historical from 18 days uptime.

## Findings — wave 1 (clean measurements)
F3. CLEAN POST-E125 PROFILE (p256, 39 tok, no agents in flight, load1 3.27 at start).
    decode_wall 20977.5 ms, accounted 99.2%. Matches the N=5 median 21253 ms => trustworthy.
      expert_forward  7557.1  36.0%   (E123: 7528.7 / 32.2%)  <-- NEW #1
      attention       6697.6  31.9%   (E123: 8787.8)  -23.8% from E125
        attn_out      3724.7  17.8%   (E123: 5046.9)  -26.2%
        attn_qkv      1767.9   8.4%   (E123: 2530.6)  -30.1%
        attn_sparse   1139.9   5.4%   unchanged
      head            1399.6   6.7%
      expert_wait     1394.6   6.6%
      shared_expert   1372.6   6.5%   (E123: 1619.8)  -15.3% (its DOWN proj uses matvec_ref)
      hc_norm         1092.4   5.2%
      indexer          616.7   2.9%
      compressor       557.3   2.7%
    E125 removed ~2337 ms of 23360 = -10.0%, which independently reproduces the measured
    +10.18% tok/s. Two different instruments agreeing is the strongest evidence in this session.
F4. *** fp4 EXPERT KERNELS MEASURED HEAD-TO-HEAD (.backlog/lab/kbench/fp4bench.c, no model) ***
    Real expert shapes, median of 15, 16 threads, calling the ENGINE'S OWN kernels:
      gate 2048x4096  scalar 0.253 ms  NEON 0.222 ms  1.14x   16.58 -> 18.89 GB/s
      up   2048x4096  scalar 0.271 ms  NEON 0.237 ms  1.14x   15.48 -> 17.70 GB/s
      down 4096x2048  scalar 0.240 ms  NEON 0.232 ms  1.03x   17.48 -> 18.08 GB/s
    => E107 IS VINDICATED, and now directly rather than by inference. "Raise pin coverage so more
    experts reach NEON" is NOT a lever: the two kernels are within 1.03-1.14x, so converting 100%
    of calls could buy at most ~1.1x on 36% of decode = ~+3.5% whole, before the +66% expert_wait
    cost that pinning also incurs. This is the OPPOSITE of the fp8 situation, where the scalar path
    had no SIMD at all and the gap was 2.1x.
    NOTE the NEON fp4 output DIFFERS from scalar (worst rel 8.35e-4) - a pre-existing rounding
    difference (the kernel does (activation*value)*scale as three rounded ops, no FMA), not
    something introduced here.
    SECOND-ORDER READING: in ELEMENTS/s, fp8-rows16 does ~66 G elem/s while the fp4 NEON does
    ~37.8 G elem/s despite fp4 packing 2 elements per byte. So the fp4 path is decode/compute bound
    and a BETTER fp4 kernel may exist - but the thing to beat is the existing NEON kernel, not the
    scalar one, which makes it a much larger and riskier project than E125 was.
F5. Structural gaps found by agents that are the SAME BUG CLASS E125 fixed (AVX2-only fast path,
    scalar on ARM), all still open:
    (a) `head_bf16_dot` (c/deepseek_v4.c:9605-9631) is SIMD only under #ifdef __AVX2__ => scalar
        bf16 multiply-add on M3. head = 1399.6 ms (6.7%), 39 calls, and it streams
        vocab 129280 x hidden 4096 x 2 B = 1.059 GB PER TOKEN at ~29.5 GB/s vs the ~105 GB/s
        ceiling. bf16 -> f32 is a 16-bit left shift, so vectorising it is trivial compared with the
        fp8 work. Best effort/º ratio remaining.
    (b) `coli_fp8_dual_matvec_ref` (:13943-14014) has NO rows16 path - E125 only covered the SINGLE
        matvec. It falls to scalar `matmul_fp8_dual` (quant.h:542) and ALSO mallocs per call. It is
        the gate/up of `shared_expert` (6.5%), so roughly two thirds of that phase is still scalar.
        This is a direct extension of already-proven code.
    (c) `hc_norm` (5.2%) has NO OpenMP anywhere (`normalized_hc_pre` :3657, `coli_v4_hc_pre` :1436,
        sinkhorn :1400) and runs twice per layer per token, fully serial, with 3 mallocs per call.
    (d) Allocation churn: ~30+ malloc/free per layer per token across block_token_pipeline (7),
        attention_token_impl (8), hc/sinkhorn (3+), compressor (5), indexer (7+), moe_token_pipeline
        (6), plus `head_argmax` allocating scores[129280] EVERY token and `final_hidden` calling
        coli_tensor_load_f32 on FOUR tensors every token. Order 50k malloc/free per generation.
        Precedent that this matters: E102/E103 removed 2 mallocs per call from the fp8 matvec.
F6. Power lever is CLOSED (F1): High Power already active. The document's P0-operational candidate
    is void.
