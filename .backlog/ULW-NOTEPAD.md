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
