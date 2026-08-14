# Ultrawork Notepad — colibri M4 moe_batch (prefill lane) + startup lane
Started: 2026-08-14T01:26 Europe/Amsterdam

## Mandate
User: "keep working until you can't continue at all without my input."
Save ALL questions for tomorrow's "good morning"/"good afternoon" -> ask via question tool then.
=> DO NOT block on questions. Log them under ## Questions For User and keep going.

## Findings (carried from this session)
- ft-deepmetal @197583b. Merged+pushed: M15(2row), M15b(4row), E37 attn_sparse opt-in.
- Steady state (220 tok): decode_wall 134619ms w/ --fast-sparse-attn, 151426ms default. 1.446->1.627 tok/s.
- Leaf ranking steady: expert_forward 26.6%, attn_out 18.1%, attn_sparse 15.0%(now 3.3%), attn_qkv 8.9%,
  expert_wait 6.0%, shared_expert 5.8%, head 4.9%, router 4.8%, hc_norm 4.0%.
- c/deepseek_v4.c:4141-4162 prefill loop: attention batched over 64-token chunks
  (coli_v4_attention_window_batch_ref) BUT MoE is per-token: moe_token_pipeline() at :4154
  called once per item. No dedup of expert leases across tokens in a chunk.
- target_batch() :7528, chunk size 64 hardcoded at :7547 (offset += 64).
- expert_requests == tokens*layers*6 exactly (61662 ~ 240*43*6) => confirms no grouping today.
- MEASUREMENT PROBLEM: ttft is dominated by model LOAD (~40s), not prefill. Must isolate.

## Questions For User (ASK TOMORROW, DO NOT BLOCK)
1. (pending — none yet beyond scope confirmations)

## Now
Isolate true prefill cost from model-load cost; establish grouping factor S before building M4.

## Todo
see todowrite list

## Learnings
- 8-token gates are too short (3x this session). Use >=60 tokens for correctness gates.
- Always warm up after rebuild; cold first-run inflated numbers twice.
- generated_text is MULTI-LINE; grep '^generated_text=' compares only first line. Use awk block extract.

## Durability
Relocated from /tmp to repo (.backlog/ULW-NOTEPAD.md) 2026-08-14 ~01:30 per user:
/tmp is wiped on crash and this machine crashes. This file is COMMITTED and is the
authoritative resume point. Append only.

## Permissions granted by user
- Commit freely whenever work is verified + has results (no need to ask).
- Work autonomously; collect questions and ask them at "good morning"/"good afternoon".

## Findings 2026-08-14 01:29 (pre-plan measurements, M4)
- Fixed floor: prompt="Hi" (5 tok), max-tokens 1 -> ttft=5.278s, wall real=5.78s.
  Dense weights are resident/mmap (v4_dense_resident 6.267GiB), so ENGINE LOAD IS CHEAP.
  Earlier assumption that ttft was ~40s of model load was WRONG.
- Prefill is COLD-EXPERT-I/O heavy: 5 prompt tokens -> expert_requests=1290 (=5*43*6),
  hits=443 misses=847, hit_rate=34.3%, bytes=11.32 GB. 847 misses * 13.37MB = 11.3GB. checks out.
- IMPORTANT DESIGN CORRECTION for M4: the expert cache ALREADY dedups repeated reads
  (2nd token using expert 7 is a cache HIT, not a re-read). So grouping tokens by expert
  does NOT reduce bytes read. M4's win must come from ARITHMETIC INTENSITY:
  S=1 matvec reads 13.37MB of expert weights to do ~13.37M MACs (~1 MAC/byte);
  S=g matmul does g x the MACs for the same weight traffic. It is a bandwidth win.
- ACHIEVABLE GROUP SIZE IS THE CRUX (must measure, not assume):
  64-token chunk -> 64*6 = 384 selections over 256 experts/layer.
  Uniform-routing estimate: unique ~ 256*(1-(1-1/256)^384) ~ 199 => mean group S ~ 1.93.
  To reach S~8 you need ~341 tokens/chunk. Chunk size is HARDCODED 64 at c/deepseek_v4.c:7547
  (offset += 64) inside target_batch() :7528.
  => M4 may require RAISING the chunk size, which costs buffer memory. Real routing is skewed
  so measure actual unique-expert counts before committing to a design.

## Questions For User (ASK AT "good morning"/"good afternoon")
Q1. Prefill chunk size is hardcoded 64 (:7547). Raising it (256-512) is likely REQUIRED for M4
    to reach useful group sizes, but multiplies prefill buffer memory. Is raising it acceptable,
    and is there a memory ceiling I should respect beyond --memory-gb?
Q2. E37 shipped --fast-sparse-attn as opt-in (not bit-exact). If M4 grouping also changes FP
    combine order (likely - summing expert contributions in a different order), do you want the
    same opt-in treatment, or is prefill-only reordering acceptable by default since it affects
    the prompt pass rather than sampled output? (I will default to opt-in/bit-exact unless told.)

## Findings 2026-08-14 01:32 — prefill curve MEASURED (the M4 business case)
| prompt | tokens | ttft s | prefill s | ms/token | hit_rate |
|---|---|---|---|---|---|
| "Hi"  |   5 |   5.278 |  (floor) |    -  | 34.3% |
| p064  |  70 |  45.249 |  39.971  | 615.0 | 76.4% |
| p256  | 184 | 118.840 | 113.562  | 634.4 | 86.2% |

Marginal cost between the two points = 645.5 ms/token => prefill is LINEAR at ~0.63-0.65 s/token.
Extrapolated 799-token prompt ~ 516s (~8.6 min). Matches the plan's 708s order of magnitude.

DECISIVE: per token there are 43*6 = 258 expert ops, so ~2.50 ms per expert op, and each op
reads a 13.37 MB expert record => ~5.3 GB/s effective. That is RAM-bandwidth territory, NOT
compute. At S=1 the expert FFN is BANDWIDTH BOUND on weight reads.
=> Batching S tokens against one expert amortizes the same 13.37MB read over S tokens.
=> Expected win is ~linear in S until the compute roof. This validates M4 as the right lane.
=> AND it re-confirms that raising the 64-token chunk (to lift S) is the lever that matters.

## CORRECTION 2026-08-14 01:40 — ttft does NOT include model load
Agent bg_2ecb66f3 traced it: ttft timer starts `setup_done = spec_now()` at c/deepseek_v4.c:8175
(immediately before target_batch prefill) and stops `first_at` at :8218 (after first token emit).
EXCLUDED: engine open (:9001-9013), tokenizer load (:9018), prompt template (:9112), session
create (:9129), tokenization+embedding (:8114-8173).
=> My "5.278s floor" was NOT load. It was genuine COLD-CACHE prefill of 5 tokens (34% hit, 11.3GB).
=> Actual model load ~= wall(5.78s) - ttft(5.278s) ~= 0.5s. Startup lane M9/M8 is DEAD, confirmed.
=> Robust prefill rate = MARGINAL between the two warm points:
   (118.840-45.249)/(184-70) = 0.6455 s/token. Conclusion unchanged.

## Code map (bg_3afd8031) — authoritative
- moe_token_pipeline :3777-4016 (single token). TWO call sites:
    :4066-4067  block_token_pipeline  -> DECODE. MUST NOT CHANGE.
    :4154-4156  coli_v4_block_window_batch_ref per-item loop :4141-4162 -> PREFILL. TARGET.
- Routing coli_v4_route_bf16 :6511-6559 — single token, no batch param.
- Expert selection reorder :3839-3849 scans experts 0..n-1 ascending =>
  PER-TOKEN COMBINE ORDER IS ASCENDING EXPERT ID. Preserving that = bit-exact grouping.
- Expert forward coli_v4_expert_forward_ref :9444-9483 (fp4 matvec gate/up/down)
  rows16 variant :6452-6489 uses coli_fp4_dual_matvec_rows16_v10 (the fused gate/up I 4-row'd).
- **BATCH PRIMITIVES ALREADY EXIST** c/native_quant_batch.h:6-11
    coli_fp4_matmul_batch_ref(outputs, weight, inputs, batch)  impl :10934-10961
    coli_fp8_matmul_batch_ref(...)                             impl :10862-10932
    batch range 1..64. Header states scalar column accumulation order PRESERVED.
    NOTE: no DUAL (fused gate+up) batch variant -> batching loses that fusion. Tradeoff.
- Lease API c/expert_store.h:40-58: 1 lookup = exactly 1 release; view not copyable/shareable/
  concurrent; active lease BLOCKS eviction (miss replacement needs !slots[i].references :5475).
  Holding one view across several SERIAL token computations is explicitly allowed. => M4 is legal.
- Profiler is DECODE-ONLY (reset :8219 after first token). Prefill instrumentation points:
  start :8175, end :8181. coli_v4_profile_add :6987-6990 is NOT thread-safe (plain += ).

## Prior art (bg_66841634)
Canonical: route -> stable sort by expert -> pad to block -> per-expert GEMM -> fixed-order
weighted reduction -> scatter back (inverse permutation).
MLX (Apple, most relevant) ml-explore/mlx#2078: gather_sort/gather_qmm/scatter_unsort.
  DeepSeek V3 (256 experts) prompt: 112->154 tok/s (1.4x), 114->210 (1.8x), "~2x once enough
  tokens fill experts". Mixtral (8 experts) 3.4-3.8x. => 256-expert case is the HARD case;
  realistic target 1.4-2x, gated on tokens-per-expert.
vLLM/SGLang moe_align_block_size: stable sort, pad per expert to BLOCK_SIZE_M, -1 for empty,
  topk weights applied AFTER expert GEMM, separate moe_sum. Small batches skip sort entirely.
Determinism: upstream engines accept tolerance (assert_close), NOT bitwise — because their
  combine order changes. OURS CAN BE BITWISE because existing order is already ascending-expert.

## PLAN AGENT VERDICT 2026-08-14 01:45 (ses_002821028fferuyKQ5BiBztjK9)
Plan agent found an ERROR IN MY REASONING and it reframes M4:
  I called the expert op "bandwidth bound" at 5.34 GB/s. But 5.34 GB/s is ~20x BELOW this
  machine's DRAM bandwidth (~120 GB/s). The op is NOT memory-saturated. It is DEQUANT +
  scalar-column-accumulation bound (fp4 dequant).
  => M4 wins ONLY IF coli_fp4_matmul_batch_ref dequantizes each weight block ONCE and reuses it
     across the S columns. If it re-dequants per column, f(S) ~ S*f(1) and speedup_op ~ 1.0.
  => THIS IS THE GATE. Testable standalone with ZERO model code. If it fails we abandon M4 free.

Also useful: FFN ops = 258 * 2.502ms = 645.5 ms/token ~= 100% of the 0.6455 s/token prefill.
So overall prefill speedup ~= FFN speedup. Single clean lever.

### Gate thresholds (G0), adopted
projected FFN speedup = sum(S_i * f(1)) / sum(f(S_i)) over the REAL group-size distribution.
  GO       >= 1.25x
  GRAY     1.10-1.25x  -> minimal impl only (keep chunk 64, batch the unfused ref path)
  ABANDON  < 1.10x
  SKEW VETO: also require >=50% of selections in groups S>=2 AND speedup_op(2) >= 1.6
G3 (post-impl): SHIP if byte-identical AND canaries green AND real >=1.25x (or >=1.10x if GRAY).
  ROLLBACK if <1.10x OR byte-identity fails with unavoidable reassociation.
  EXPLICIT: do NOT ship an opt-in tolerance flag here. Byte-identical or abandon.
  (Contrast E37 --fast-sparse-attn, which we accepted as opt-in. Not repeating that for M4.)

### Wave 1 LAUNCHED (parallel, both measurement-only, no model behavior change)
- bg_32c65abb  T0a  ultrabrain     : batch primitive f(S) curve + dequant-amortization source read
                                     + which forward path prefill uses (ref :9444 vs rows16 :6452)
- bg_3ac594bd  T0b  unspecified-high+ast-grep : COLI_M4_TRACE gated prefill group-size histogram
                                     on p064/p256 + raw routing trace dump for re-binning
Next after both: T0c re-bin @64/128/256 + buffer audit, then G0 decision.

## SESSION STATE 2026-08-14 02:58 — ft-deepmetal @ a3f5afb (pushed)
### Landed tonight
- E38 M4 gated out pre-implementation (batch primitives re-dequantize per item; projected 1.24x
  vs 1.25 bar + skew veto). ZERO model code written. ~25 min of parallel measurement.
- E39 dequant hoisting: bitwise-identical, speedup_op(64) 1.479->2.793, BUT f(1) 0.796->1.537.
  Crossover S~10-12, real distribution below it. Best hybrid 1.21x vs 1.6x bar. LANE CLOSED.
  Root-cause correction: expert FFN is compute-bound on the FP32 dot products + per-item
  activation QDQ. NOT dequant, NOT bandwidth. Kills the whole M4/M5 batching family.
- E40 router: Tier 1 bit-exact = 0.997x NEGATIVE (clang already vectorizes bf16 decode; and it
  made default 3.83% slower -> stripped, baseline restored). Tier 2 opt-in = 10.76x on phase,
  decode_wall -11.4%. Default md5 golden on every run.

### Current numbers (60-token warm, M3 Max)
  default : router 1955.5 ms  decode_wall 38612.8 ms  md5 5d04890413ff539e802985ce8c727814
  flagged : router  181.8 ms  decode_wall 34197.5 ms  md5 7155bab905cbfa70aa06afa08f757cee
  --fast-sparse-attn now = attn_sparse (E37) + router (E40) reassociated kernels.

### Remaining unoptimized leaves (from E35 steady-state profile)
  head     4.9%  33.8354 ms/call, 1 call/token, 129280 vocab x 4096, ALREADY OpenMP-parallel
                 (:7258 head_argmax, #pragma omp parallel for at ~:7279). 529M MACs/token
                 => 15.7 GMAC/s aggregate. Inner dot likely same serial-chain defect.
                 NEXT TARGET.
  hc_norm  4.0%  0.3245 ms/call x 18834
  indexer  2.5%  0.8146 ms/call x 4599
  compressor 1.9% 0.3246 ms/call x 8979
  expert_forward 26.6% + attn_out 18.1% + attn_qkv 8.9% + shared_expert 5.8% -> all 4-row'd (E33)
  expert_wait 6.0% -> I/O, de-promoted (E35)

### Big lane not yet touched
  GPU offload via coli_v4_metal_expert_forward (backend_metal_v4.o exists, METAL=1 links).
  expert_forward is 26.6% of decode. E38 telemetry showed 71% of prefill selections in groups
  >=4, which is favorable for GPU batching. Large/risky lane - needs its own gate.

## Questions For User (ASK AT "good morning"/"good afternoon")
Q1. Prefill chunk size 64 (:7547) - moot now, M4 closed. WITHDRAWN.
Q2. Bit-exactness policy - ANSWERED by you: opt-in flag. Applied to E37 + E40.
Q3. NEW: --fast-sparse-attn now covers attn_sparse + router and will likely cover head next.
    The name is misleading. Rename to --fast-kernels (keeping --fast-sparse-attn as a hidden
    alias for compat), or leave as is?
Q4. NEW: Metal/GPU offload of expert_forward (26.6% of decode) is the largest remaining lane
    but is a big, risky change. Worth opening, or keep to CPU kernels?

## SESSION STATE 2026-08-14 03:51 — ft-deepmetal @ 5b1c0a3 (pushed)

### E41 REVERTED — the most important finding of the night
Fast NEON LM head: 11.71x microbench, 3.67x on the head phase... and 11.2% SLOWER decode_wall.
Mechanism MEASURED (accounted_pct 98.5% both ways, so nothing hidden):
  head          2091.3 -> 554.8   (-1536.5)
  router        1956.9 -> ~176    (-1780)
  expert_forward 10594.4 -> 12803.2 (+2208.8)   <-- the time comes back HERE
  shared_expert  2406.1 -> 2820.6   (+414.5)
Head streams ~1 GiB BF16 vocab weights/token; 3.7x faster => evicts resident expert slabs from
shared cache => expert_forward refetches from DRAM. Revert snaps expert_forward back to ~10.5s
and wall back to 34148-34514 (matches E40 reference 34197.5/34246.8). Toggle definitive both ways.

### *** STANDING RULE FOR ALL FUTURE WORK IN THIS REPO ***
PHASES ARE NOT INDEPENDENT. They contend for shared cache + memory bandwidth.
A phase-local speedup can be NET NEGATIVE.
=> ALWAYS validate on decode_wall, NEVER on the phase= number alone.
=> ALWAYS use a same-build A/B toggle (cross-build comparison hides this).
Third time this pattern bit us: E39 (hoist regressed f(1) 2x), E40 Tier 1 (regressed default
3.83%), E41 (regressed wall 11%).

### Where things actually stand (60-token warm, M3 Max)
  default : decode_wall ~38.5-38.7 s   md5 5d04890413ff539e802985ce8c727814  (bit-exact, unchanged)
  flagged : decode_wall ~34.1-34.5 s   md5 7155bab905cbfa70aa06afa08f757cee  (~11% faster)
  --fast-sparse-attn = attn_sparse (E37) + router (E40). Head NOT included (E41 reverted).

### Landed and pushed tonight
  E33 M15b 4-row matvec        1.38-1.42x cumulative decode, byte-identical
  E34 loader depth             NEGATIVE, rejected (cold-cache artifact)
  E34b cache-policy retraction methodology fix
  E35 steady-state profile     expert_wait 17.9%->6.0%, found attn_sparse
  E36 attn_sparse root cause   clang fmul+lane-extract+serial add
  E37 attn_sparse NEON         4.53x phase, opt-in (not bit-exact)
  E38 M4 gated out             zero model code, batch primitives re-dequantize
  E39 dequant hoisting         bitwise-identical but f(1) 2x worse, lane closed
  E40 router NEON              10.76x phase, opt-in, default provably untouched
  E41 LM head NEON             REVERTED (cache pollution, +11% wall)

### Remaining candidates (all now suspect under the standing rule)
  hc_norm    4.0%  0.3245 ms/call x 18834
  indexer    2.5%  0.8146 ms/call x 4599
  compressor 1.9%  0.3246 ms/call x 8979
  These are SMALL and all read/write large arrays -> high risk of the same cache-pollution
  backfire for little upside. Recommend NOT pursuing without a same-build A/B harness first.
  expert_forward 26.6% is the only phase big enough to be worth real risk -> GPU/Metal lane.

## Questions For User (ASK AT "good morning"/"good afternoon")
Q3. --fast-sparse-attn now covers attn_sparse + router. Name is misleading. Rename to
    --fast-kernels with --fast-sparse-attn kept as a hidden compat alias, or leave it?
Q4. expert_forward is 26.6% of decode and the only phase big enough to justify real risk.
    Metal/GPU offload (coli_v4_metal_expert_forward exists, backend_metal_v4.o builds) is the
    largest remaining lane but is big and risky. Open it, or stay on CPU?
Q5. Given E41, should I build a same-build A/B toggle harness (env var selecting kernel variant
    at runtime) so future kernel work can be measured without cross-build confounds? I think
    yes - it would have caught E41 in one run instead of six.

## 2026-08-14 04:00 — activation-QDQ hoist CONSIDERED AND REJECTED (arithmetic only, no code)
Found real redundancy: coli_fp4_dual_matvec_ref (c/deepseek_v4.c:11004-11010) mallocs a 16KB
activation buffer and runs coli_fp8_activation_qdq_ref on `input` EVERY call. All 6 experts in a
layer get the SAME ffn_normalized input, so 258 QDQ passes/token where 43 would do (83% redundant).
BUT: QDQ is ~12.3K ops vs 25.2M MACs per expert op = 0.049% of the work.
Hoisting all 215 redundant calls saves ~0.04% of expert_forward ~= 4 ms of a 34.3 s run.
REJECTED on arithmetic. No code written. (Same discipline as E38: do the sums before building.)

Also confirms expert_forward is in decent shape: 0.69 ms per expert op for 25.2M MACs
= 36.5 GMAC/s (73 GFLOP/s) on CPU. E33's 4-row matvec did its job. The remaining lever on
that phase is GPU offload, not more CPU micro-optimization.

## FINAL A/B (same binary, flag toggled, 3 interleaved runs each, 60-token gen)
  default  mean 38554.8 ms  sd 234.7   md5 5d04890413ff539e802985ce8c727814 (bit-exact)
  flagged  mean 34311.2 ms  sd  49.7   md5 7155bab905cbfa70aa06afa08f757cee
  => 11.01% faster (1.1237x). expert_forward identical in both (10476-10672) = no cache pollution.

## 2026-08-14 04:12 — STOPPING POINT. ft-deepmetal @ 30aa44b (+ this commit), all pushed.

### Repo state VERIFIED CLEAN
Default build: 0 metal symbols, 0 metal frameworks, 26 units, decode_wall 38605.5 ms,
golden md5 5d04890413ff539e802985ce8c727814. Working tree has no tracked modifications.

### E42 Metal audit (investigation only, nothing changed)
METAL ?= 0 is the default => the ENTIRE campaign measured CPU only.
METAL=1 builds cleanly (6 shaders -> metallib, 8 symbols, frameworks linked) but the GPU path
NEVER EXECUTES: same-build A/B COLI_V4_METAL=0 vs =1 is identical within noise, no stats output,
md5 unchanged; also identical with AUTOPIN=0.
Two causes found:
  1. COLI_V4_DEFAULT_METALLIB is baked as a CWD-RELATIVE path "build/metal-v4/deepseek_v4.metallib"
     but the file is at c/build/metal-v4/ -> never resolves when run from repo root.
     (COLI_V4_METALLIB env at backend_metal_v4.mm:631 overrides it.)
  2. coli_v4_metal_expert_forward (backend_metal_v4.mm:332) requires metal_variant()==0 AND all
     three tensors to pass coli_v4_ordered_cold_tensor_valid (:356-364) = ORDERED-COLD layout,
     but the runtime hot-packs experts into rows16 (packed_slots=722-902) which is what E33's CPU
     4-row kernel targets. Setting COLI_V4_METALLIB explicitly STILL did not make it fire.
NEXT STEP for this lane is a one-line reject-reason log inside coli_v4_metal_expert_forward,
NOT writing new kernels. Cheap and decisive.

### WHY I STOPPED
- CPU kernel lane exhausted: expert_forward already at 36.5 GMAC/s (73 GFLOP/s) post-E33;
  activation-QDQ hoist rejected on arithmetic (0.04%); remaining leaves (hc_norm 4.0%,
  indexer 2.5%, compressor 1.9%) are small AND carry the E41 cache-pollution backfire risk.
- The one lane with real upside (expert_forward 26.6% via Metal) needs a scope decision.
- Everything else pending is a QUESTION, listed below.

## QUESTIONS FOR USER — ASK ALL OF THESE AT "good morning"/"good afternoon"
Q3. --fast-sparse-attn now enables attn_sparse + router (and would have covered head). Name is
    misleading. Rename to --fast-kernels with --fast-sparse-attn kept as a hidden compat alias?
Q4. Metal lane: the backend exists and builds but never fires. Want me to diagnose it (add a
    reject-reason log, fix the metallib path, see if it can run on the rows16 layout)? This is
    the only remaining lane with >10% upside.
Q5. Build a same-build A/B harness (env var selecting kernel variant at runtime) so future kernel
    work is measured without cross-build confounds? E41 took 6 runs to catch; this would have
    caught it in 1. I recommend yes.
Q6. Should the opt-in reassociated kernels become the DEFAULT? They are 11.01% faster and the
    output difference is rounding-level, not quality-level. Currently default is bit-exact and
    the win is opt-in only.

## 2026-08-14 04:16 — T0c CLOSED (was the last genuinely-unfinished item)
Re-binned the preserved routing trace (.backlog/m4_traces/*.gz) at chunk 64/128/184:
  chunk  meanS  %sel S>=8   projCUR  projHOIST  projBEST(hybrid min)
     64   3.96      46.9%    1.186x     1.008x   1.217x
    128   5.08      58.4%    1.203x     1.081x   1.246x
    184   7.65      72.9%    1.224x     1.182x   1.283x
Tripling chunk = +5.4% projected for 2.9x buffer memory. Saturates because speedup_op(64)=1.479
is the kernel ceiling. And 1.283x charges NOTHING for gather/scatter, hybrid dispatch, or E41
cache contention. => G0 ABANDON verdict CONFIRMED, not overturned. M4 stays dead.

M5 status corrected: its telemetry gate (>=30% selections in S>=8) IS MET (46.9% @64, 72.9% @184).
M5 is therefore BLOCKED ON THE METAL PATH NOT FIRING (E42), not on routing data. That is a user
scope decision (Q4), so M5 is not actionable tonight.

## TODO LIST RECONCILED
All M4 downstream tasks (T1a/T1b/T2a/T3a-c/G3) are CANCELLED-BY-GATE, not skipped: G0 decided
ABANDON, and T2a was precisely the model code the gate existed to avoid writing. Not writing it
is the gate working, not work left undone.
T1a's intent (a byte-identical harness) is already satisfied in practice: golden md5
5d04890413ff539e802985ce8c727814 + the multi-line block extractor, used as the gate in E37/E40/E41.

================================================================================
# ULW SESSION 2 — 2026-08-14 12:05 — Metal cache-heating + cache preservation
================================================================================

## USER REQUEST (verbatim intent)
1. Take cache heating into account for the Metal tests BEFORE writing Metal off.
2. Look into ways of preserving pre-heated cache between runs to save time.
   Implement if feasible WITHOUT affecting test scores. Maybe a ram disk.

## WHY (1) IS A REAL CONCERN — user is right, I may have mis-concluded E43
E43 measured METAL=1 as 2.67x slower on expert_forward (5188.6 -> 13841.8 ms).
BUT: I have been burned by cold-cache artifacts TWICE this campaign already
  - E34: loader-depth "2.3x win" was 100% cold-cache artifact on the first sweep row
  - E41-adjacent: cold first-run-after-rebuild inflated numbers repeatedly
The METAL=1 runs REQUIRED a full rebuild (rm -f c/*.o) immediately before them.
=> The METAL=1 measurement may include cold page-cache / cold expert-cache effects
   that the METAL=0 run (done later, warmer) did not pay.
ALSO: metal reject histogram showed layout=5455 rejects => 43.1% of experts fall back
to CPU. A mixed GPU/CPU run may thrash the expert cache differently than pure CPU.
=> MUST re-measure with rigorous interleaved warm A/B, same binary, multiple runs.

## KNOWN COLD/WARM MECHANICS IN THIS ENGINE (established earlier)
- expert cache: 164 slots/layer x 43 layers = 7052 records, 13.37 MB each, ~87.81 GiB
- .coli_usage (29127 bytes) drives autopin; I freeze/restore it around every run
- hit_rate rises 34% (5-tok cold) -> 76% (70-tok) -> 86% (184-tok) -> 88.4% (220-tok)
- cache is only 24.7% full after a 17-token run => short runs are ALL compulsory misses
- model load itself is cheap (~0.5s); ttft excludes it (timer :8175->:8218)
- 60-token run reads ~24 GB; 220-token reads ~96 GB
=> Between-process, the OS page cache holds the safetensors shards; that is the thing
   a ram disk / preload could preserve. The in-process expert slab cache dies with the
   process.

## SCENARIO CONTRACT (binding, both artifacts required per scenario)
S1. Metal fair-fight: interleaved warm A/B, same binary, N>=3 each, cache-state equalized.
    PASS = a defensible verdict on METAL=1 vs 0 with variance reported.
    Artifact: raw per-run decode_wall/expert_forward + metal reject counters.
S2. Cache-preservation mechanism identified + measured for time saved.
    PASS = measured warmup-time delta with mechanism on vs off.
    Artifact: wall-clock of a fixed benchmark sequence with/without.
S3. Score neutrality: preservation must NOT change measured results.
    PASS = golden md5 5d04890413ff539e802985ce8c727814 unchanged AND
           decode_wall within run-to-run variance of the no-mechanism baseline.
    Artifact: md5 + 3-run decode_wall distributions both ways.

## Now
Survey skills, fire parallel explores, then plan agent.

## Findings 12:14 — live system facts (none of the agents can see these)
  RAM                128.0 GiB   (Apple M3 Max)
  model on disk      155 GB across 48 safetensors shards
  free disk          81 GiB of 926 (95% full)
  page cache now     File-backed 31.3 GiB, free 66.3 GiB

*** MODEL (155 GB) IS LARGER THAN RAM (128 GiB). ***
=> A full RAM disk for the model is IMPOSSIBLE. Rules out the user's "maybe a ram disk"
   in its simplest form. Partial/selective residency is the only RAM-disk-shaped option,
   and the disk is 95% full so there is no room for a second copy anyway.
=> The realistic levers are:
   (a) a persistent server process that keeps the in-process expert cache hot across
       requests (eliminates re-heating entirely) - bg_baca106c is checking if serve mode exists
   (b) selective page-cache warming of the hottest shards (66.3 GiB free could hold a chunk)
   (c) reducing what must be heated at all
   NOTE: if the engine uses F_NOCACHE/direct IO (COLI_V4_DIRECT / v4_ssd_io), the OS page
   cache may be deliberately BYPASSED today, which would make (b) useless or even
   counterproductive. Must confirm before building anything.

## Findings 12:14 — my quality-test harness bug (task B), NOT a model failure
All 20 outputs came back empty because I filtered STDOUT for '^generated_text=', but that
line goes to STDERR. STDOUT carries the clean generated text directly.
Manual check: `... "The capital of France is" --max-tokens 24` -> stdout is
  `The capital of France is **Paris**.<|end of sentence|>`
=> Fix: read stdout directly (no awk needed), or merge 2>&1 and use the block extractor.
Task B is NOT invalidated, just re-run needed.

## AGENT RESULTS 12:30 — all three in

### bg_229920ec Metal buffer lifecycle — my E43 verdict is MOSTLY architectural, but I DID miss warmup
AMORTIZABLE (first-call only), which a 30-token run charges unfairly:
  - lazy coli_v4_metal_init: device + metallib load/compile + probe pipeline + queue (:423-426, :740-768)
  - chain pipelines built on FIRST expert dispatch, then memoized (:367-376, :475-478)
  - scratch buffers allocated on first need, reused by capacity (:277-285)
  - ONE MTLBuffer per registered slab via newBufferWithBytesNoCopy = ZERO-COPY weights (:201-221)
NOT AMORTIZABLE (every single dispatch):
  - input memcpy (:552), output memcpy (:710)
  - weight buffer resolution x6 tensors (:557-568) - zero-copy IF slab registered else memcpy
  - ONE command buffer + ONE encoder created per expert (:575-577)
  - *** ONE SYNCHRONOUS [command_buffer waitUntilCompleted] PER EXPERT (:698-699) ***
=> ~240 expert dispatches/token x 60 tokens = ~14,400 BLOCKING GPU round trips. That is the
   architectural cost and no amount of cache heating removes it. Explains 1.614 ms/dispatch.
COUNTERS AVAILABLE: COLI_V4_METAL_STATS=1 -> reject histogram + metal_dispatches;
   COLI_V4_METAL_PROFILE=1 -> per-stage ms_per_expert + zero_copy_tensors vs copy_fallback_tensors.
   WARNING: PROFILE=1 inserts 9 commit/wait per dispatch (:592-606) so it DISTORTS timing.
   Use STATS for timing runs, PROFILE only for stage attribution.

### bg_327551ae macOS cache warming — options 2 and 3 are STRUCTURALLY DEAD
  - RAM disk: hdiutil docs - RAM disks use WIRED memory, cannot page out. 100GB on 128GB machine
    = severe pressure. AND benchmark-invalid: changes storage medium vs prod APFS/SSD.
  - posix_fadvise: DOES NOT EXIST on macOS. madvise(MADV_WILLNEED) is for MAPPED ranges only;
    our engine uses pread not mmap => not applicable.
  - F_RDADVISE: advisory only, forwarded to VNOP_IOCTL, FS decides. Hint not guarantee.
  - vmtouch: works (mincore/msync/mlock) but pinning bounded by RLIMIT_MEMLOCK; changes eviction
    behavior => benchmark-invalid unless prod also pins.
  - *** DECISIVE: our engine sets F_NOCACHE (COLI_V4_DIRECT default ON) for expert reads, so the
    OS unified buffer cache is DELIBERATELY BYPASSED. Page-cache warming cannot help expert IO
    at all. ***
  - Measurement tools if ever needed: mincore, vmtouch -v, vm_stat, footprint, fs_usage, purge.

### CONCLUSION ON USER REQUEST 2
Only viable mechanism = PERSISTENT SERVE PROCESS (option 1). Proven working:
  identical prompts: hit_pct 75.2 -> 98.3 -> 98.3, wall 53.1 -> 44.2 -> 43.3 s
  BUT that improvement is CONFOUNDED by KV prefix reuse (same prompt on same slot).
  DISTINCT prompts: hit_pct 75.2 -> 91.1 -> 93.6 -> 95.0 (real heating) but
    wall 52.7 -> 48.0 -> 54.0 -> 69.4 and tok/s 1.491 -> 1.457 -> 1.369 -> 1.091 (DEGRADES)
  => second confound: single KV slot ACCUMULATES context across turns (truncate-and-extend),
     so attention cost grows every request. Must reset between benchmark requests.
=> Serve mode IS the answer but the harness must control BOTH confounds.

## USER ADDED REQUEST (12:30)
"I do want to record warmed cache performance too though, make sure to test, collect and document"
=> Both operating points are first-class deliverables:
   COLD  (one-shot, per-process): hit_rate ~75-88%
   WARM  (serve, steady state):   hit_rate ~98.3%
   Must document decode throughput at BOTH, with confounds controlled.

## T3 DONE 12:34 — build_toggle.sh
.backlog/metal/build_toggle.sh builds BOTH variants from clean object state (encodes the
rm -f c/*.o trap that produced wrong conclusions twice):
  c/deepseek_v4.cpu    metal_syms=0   objects=26
  c/deepseek_v4.metal  metal_syms=13  objects=27
Default c/deepseek_v4 restored to the .cpu variant. Doc scaffolds created.

## T4 DONE 12:40 — GOLDEN GATE PASSED + COLD operating point pinned
Interleaved 3-run A/B, one-shot, 60-token, CPU binary. CSV: .backlog/results/T4_cold_oneshot.csv
  default  decode_wall mean 39194.0 ms sd 107.2  hit_rate 77.97%  md5 5d048904 GOLDEN x3
  fast     decode_wall mean 35284.3 ms sd 247.7  hit_rate 78.51%  md5 7155bab9
  => --fast-kernels is 9.97% faster COLD.
CAVEAT: absolute numbers ~2% above yesterday's reference (38554.8 / 34311.2) for BOTH configs
=> consistent machine drift, not a regression. The A/B RATIO is the durable quantity
   (9.97% today vs 11.01% yesterday). Always compare within a session.
*** COLD OPERATING POINT = hit_rate ~78% *** (this is what every one-shot benchmark measures)
