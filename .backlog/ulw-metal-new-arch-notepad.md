# Ultrawork Notepad — new Metal architecture (S=N batched MoE) for DeepSeek-V4 on M3 Max

## Goal (verbatim from user)
1. Detailed implementation plan for the NEW metal architecture, max perf on THIS host
2. Investigate core premise + design + build plan + ALL experiments to measure/record perf
3. symlink plan .md into .omo/plans/
4. Atlas + Momus + Oracle adversarial review, iterate until ALL pass
5. Branch ft-metal-new-arch, implement code + tests
6. Run benchmarks, record results, decide what works
7. Keep designing new experiments until CERTAIN max perf reached

## Now
Phase 0: verify the CORE PREMISE before planning anything.

## Findings

### F1. HOST (M3 Max)
16 CPU (12P+4E), 40 GPU cores, 137 GB unified, 16 KB pages.
Metal: threadgroup mem 32768 B, max 1024 thr/tg, GPUFamily Apple 9, Metal3, rec working set 115.4 GB.

### F2. PREMISE OVERTURNED - GPU HAS 3.08x THE BANDWIDTH
/tmp/bwprobe.m, 4 GB MTLResourceStorageModeShared buffer, same physical memory both sides:
  CPU streaming read  120.5 GB/s  (12 threads, GCD, 4-way unrolled)
  GPU streaming read  370.8 GB/s  (iter0 70.8 = warmup; 370.8/370.0 steady)
  ratio 3.08x ; 370.8 = ~93% of M3 Max 400 GB/s spec -> credible saturation
RESULTS.md S12 cause #3 "unified memory gives the GPU no bandwidth advantage ...
both processors read the same DRAM" is FALSE ON THIS HOST. CPU cannot saturate the fabric.

### F3. BUT neither path is DRAM-bound today - the arithmetic
expert record = gate 4.19 + up 4.19 + down 4.19 + scales 0.79 = 13.37 MB
measured per-expert: CPU 2.919 ms, GPU 3.264 ms
  => effective 13.37MB/2.919ms = 4.58 GB/s (CPU) and 4.10 GB/s (GPU)
Both are ~26-90x BELOW streaming bandwidth (120 / 370 GB/s).
So per-expert time is NOT set by DRAM. It matches SSD: QD1 measured 5.227 GB/s
(E52), 13.37MB/5.227GB/s = 2.56 ms ~ the observed 2.9 ms.
=> MISS path is SSD-bound; GPU cannot help there (data still crosses the SSD).
=> The GPU opportunity is the HIT path + amortising dispatch, NOT raw FLOPs.

### F4. THE BATCHING PRIZE, quantified
prefill 799 tok: tokens routed to a given expert per layer = 799*6/256 = 18.7 avg.
Today those are 18.7 SEPARATE S=1 forwards over the SAME 13.37 MB of weights
=> ~18.7x redundant DRAM traffic per expert per layer.
S=N batching reads the weights ONCE per expert per layer.
Combined with F2 (GPU 3.08x bandwidth), the ceiling is large - but ONLY on cache hits.
Hit rate measured at --ram 96: 57.7% (PREWARM=0) / 67.7% (=1).

### F5. BACKLOG LINE REFS ARE STALE (verified)
- "moe_token_pipeline at deepseek_v4.c:3579" -> actually at 4639; 3579 is a gate-bias
  line inside moe_token (starts 3559)
- "three seams (3167,3730,3763)" -> FOUR call sites: 3621, 4620, 4831, 4877
- coli_v4_expert_forward_ref has THREE definitions: 7582, 7661, 10899
Ground everything in current code, never the backlog line numbers.

### F6. BATCHING THESIS VALIDATED BY DIRECT MEASUREMENT (my own probes, no engine)
Real gate/up shape I=4096 O=2048 MXFP4 blk=32. us per token, one matrix:
  S    CPU     GPUv1(simple)  GPUv2(tiled+uchar4)   best GPU   verdict
  1    397.0   376.9          968.5                 376.9      GPU 1.05x
  2    325.5   290.1          304.8                 290.1      GPU 1.12x
  4    320.3   256.3          210.6                 210.6      GPU 1.52x
  8    219.8   212.4          163.5                 163.5      GPU 1.34x
  16   317.0   147.7          128.9                 128.9      GPU 2.46x
- CPU does NOT scale with S (1.25x from S=1->16; S=16 regressed vs S=8, likely cache thrash)
- GPU scales 2.9x (376.9 -> 128.9 us/tok)
- GPU wins at EVERY S tested, advantage GROWS with S
- BOTH my probes wall at ~114-130 GFLOP/s = ~3% of M3 Max fp32 peak
  => kernel quality, not memory, is the current limit. Large headroom remains.
  => absolute GPU numbers here are a LOWER BOUND.
- v2 (tiled+uchar4) scales better (7.51x) but is WORSE at S=1 (tiling overhead).
  Design implication: variant-select on S, or make the tile degrade gracefully at S=1.

### F7. WHY THE SHIPPED BACKEND LOST - now coherent
Shipped Metal = S=1. My S=1 kernel edge is only 1.05x. Add per-call command buffer +
waitUntilCompleted + (claimed) upload, and 17.8% coverage -> net 1.118x SLOWER. Consistent.
The fix is therefore NOT kernel micro-tuning; it is raising S. Confirmed by measurement.

### F8. Probe artifacts (reusable)
/tmp/bwprobe.m       CPU vs GPU streaming bandwidth on one unified buffer
/tmp/sweep_mxfp4.m   GPU MXFP4 S-sweep, simple 1-thread-per-row
/tmp/sweep2.m        GPU MXFP4 S-sweep, tiled + uchar4 + threadgroup-staged X
/tmp/cpu_mxfp4.c     CPU MXFP4 S-sweep, 12 threads GCD

### F9. STRUCTURE: prefill is ALREADY batched; only MoE is not
- target_batch (:8879) is LAYER-outer, CHUNK-inner: `for layer ... for offset+=64 { chunk=min(64,...) }`
- attention ALREADY batches across tokens: coli_fp8_matmul_batch_ref(..., batch)  (:2592,:2632,:2644,:2772)
- MoE does NOT: :5110 `for item<batch` -> :5141 moe_token_pipeline(..., tokens[item])  = S=1
- batch capped at 64 in THREE places: :8884 (chunk), :4996 (block), :2551 (attention)
- decode = S=1 (:9877 target_token). Speculation DOES batch via target_batch (:9742), batch=proposals+1 (<=25)
=> NO prefill restructure needed. The batch already exists; MoE fails to exploit it.

### F10. THE PIECES ALREADY EXIST
- route cache holds FULL batch routing: cache->idx[items*topk], ->w[], ->unique_experts[], ->n_unique
  built at :4514-4546, allocated :3791-3794. Used ONLY for loading, never for compute.
- coli_fp4_matmul_batch_ref (:12389) - batched FP4 matmul reference ALREADY EXISTS (unused by MoE)
- coli_fp8_matmul_batch_ref (:12317) - used by attention
- c/bench_moe_batch.c benchmarks the batched FP4 path already
- SIBLING ENGINES already do batched expert MoE: inkling.c (Apple GPU), backend_vulkan.c
- route cache only exists when COLI_V4_PREFILL_PREFETCH=1 (default OFF) -> must build it unconditionally
  for the batched path, or gate the batched path on it.

### F11. MEASURED N (the decisive design input) - p064 prefill, 86 layer-chunk records
  chunk=6  : 36 selections / 24.3 unique -> N=1.48
  chunk=64 : 384 selections / 93.7 unique -> N=4.10   (min uniq 73 -> N=5.3 ; max 188 -> N=2.0)
  only 37% of the 256-expert space touched per chunk
  uniform-random would touch ~202 of 256 -> actual 94 means routing is HEAVILY CONCENTRATED
=> N=4.10 avg. Below llama.cpp's ne21>=32 mat-mat threshold, but my probe gives GPU 1.52x at S=4.
=> prize = 4.1x fewer dispatches (94 vs 384) + 4.1x less weight traffic, on top of GPU 3.08x bandwidth.

### F12. INDEPENDENT LEVER: the chunk cap of 64 is a CODE CONSTANT, not hardware
Raising it raises N. Attention's recurrent parts (:2603 compressor, :2688 ring) are a SEQUENTIAL
inner loop over items - raising batch does not break them, it just iterates more; the MATMULs batch.
Estimated: chunk=128 -> N~5.9 ; chunk=256 -> N~9.0 (needs measurement, not assumption).

### F13. PRODUCTION REFERENCES (librarian)
- MLX gather_qmm: tile BM=BN=BK=32, group_dims(32,wn,wm), threadgroup Xs[BM*BK_pad]/Ws[BN*BK_pad].
  SORTING IS EXTERNAL: benchmark does argsort(indices) -> sorted_indices, then inv_order unpermute.
- llama.cpp MUL_MAT_ID: mat-mat path only when has_simdgroup_mm && ne00>=64 && ne21>=32.
  matvec constants N_R0_MXFP4=2, N_SG_MXFP4=2. mat-mat NR0=64 NR1=32 NK=32.
  kernel_mul_mm_id_map0 builds per-expert token id lists. instantiations ne20=1,2,4,5,6,8,10,16,22.
- Megatron/vLLM canonical pipeline: count per expert -> offsets -> permute -> grouped GEMM -> unpermute.
  vLLM moe_align_block_size for block-aligned grouping.
- Apple: threadExecutionWidth=32, threadgroup mem 32KB, simdgroup_float8x8/half8x8/bfloat8x8.
=> At N=4.1 we are in llama.cpp's MATVEC regime (N_R0=2 rows/thread), NOT the mat-mat regime.
   Design must target the matvec-with-token-batching shape, not a simdgroup_matrix GEMM.

### F14. NUMERICS CONTRACT (bit-exactness) - and a STRUCTURAL PROOF that batching is safe
SHIPPED unit is COLI_V4_UNIT_EXPERT_ROWS16 (Makefile.deepseek-v4.units:23-24), so the live
coli_v4_expert_forward_ref is :7661 with COLI_FP4_ROWS16_KERNEL ->
coli_fp4_dual_matvec_rows16_v10 (gate+up) and coli_fp4_matvec_rows16_v10 (down).

** matmul_mxfp4 (quant.h:1427) ALREADY HAS AN S PARAMETER **
  signature: matmul_mxfp4(y, x, q4, e8s, int S, int I, int O)
  loop nest : for(o+=4) { for(s<S) { for(g<ng) { for(k<16) ... } a+=g*scale } y[s*O+o]=a }
  => each (s,o) output accumulates INDEPENDENTLY of every other s.
  => BATCHING OVER S CANNOT CHANGE ANY SINGLE OUTPUT'S ARITHMETIC. Bit-exact BY CONSTRUCTION.
  This is a structural proof, not an empirical hope. It is the core correctness argument.

Accumulation order that MUST be preserved (per output element):
  - group size 32; ng=(I+31)/32
  - within group: serial k=0..15, even column then odd: g+=x[base+2k]*lo; g+=x[base+2k+1]*hi
  - across groups: serial g=0..ng-1, a += g_partial * scale
  - NO flat reduction, NO simd tree (that is the `simd` variant, not bit-exact)

fp8 activation QDQ (:11980-12001): amax over block, fmaxf(amax,1e-4), exponent=ceil_log2(amax/448),
clamp [-127,127], values clamped to +-448, e4m3 encode->decode * scale.
  ** SCALE SCOPE = PER TOKEN, PER 128-COLUMN BLOCK ** (batch path :12406-12410 calls QDQ per item)
  => a batched kernel MUST compute scales per token independently. Never across tokens.

UE8M0 has TWO DISAGREEING decoders:
  quant.h:1424 mx4_scale  = bit trick (uint32)s<<23  -> s=0 gives +0.0,     s=255 gives +inf
  st.h:756    ue8m0_to_f32= ldexpf(1,v-127)          -> v=0 gives 2^-127,   v=255 gives NaN
  The MoE matmul path uses quant.h's mx4_scale. MATCH THAT ONE.

bf16 round (:11949): if exponent!=0xff { tie=(bits>>16)&1; bits += 0x7fff+tie } bits&=0xffff0000

swiglu (:1641): if limit>0 { gate=fmin(gate,limit); up=fmax(-limit,fmin(up,limit)) }
                out = gate*sigmoid_stable(gate)*up

MoE combine (:4720-4727, :4886-4897): selected experts SORTED ASCENDING BY EXPERT ID, summed into
an fp32 accumulator with NO per-add rounding, then + shared_output, then ONE bf16 round.
  ** CRITICAL FOR GATHER-BY-EXPERT: fp32 addition is NOT associative. If I iterate unique experts
     and scatter into per-token accumulators, I MUST visit experts in ASCENDING EXPERT ID order so
     each token's 6 contributions land in the same order as today. unique_experts[] is built in
     ENCOUNTER order (:4545) - it must be SORTED before use. **

### F15. BANDWIDTH PREMISE HARDENED (answers the DCE objection)
/tmp/bw2.m rewrites the probe so the GPU kernel does an UNCONDITIONAL out[gid]=sum write and both
sides are checksum-verified against the known exact sum.
  sizeMB   CPU GB/s   GPU GB/s   ratio   checksum
   512      118.7      271.3     2.29x   MATCH
  1024      115.8      350.6     3.03x   MATCH
  2048      113.1      374.0     3.31x   MATCH
  4096      115.0      367.3     3.19x   gpu MATCH, cpu saturated
  8192      117.8      379.1     3.22x   gpu MATCH, cpu saturated
- GPU checksum == expected EXACTLY at every size => loads were NOT dead-code eliminated. Real.
- ratio stable ~3.2x for >=2GB (well beyond any cache), so it is DRAM bandwidth, not cache.
- CPU checksum divergence at >=4GB is float32 accumulator saturation, fully explained:
  805306368 == 48 * 2^24 exactly (12 threads x 4 accumulators, each pinned at 2^24). It is a
  checksum precision artifact ONLY; the timing is unaffected (same loop, same bytes touched).
=> The 3.08x figure stands, and is if anything conservative (3.2x at large sizes).

## ===== SESSION 2: AUTONOMOUS BUILD/EXPERIMENT (no questions until good morning/afternoon) =====
Standing order: keep building + experimenting to improve perf on THIS host, learn from each
result, document ALL tests/theory/results in experiments_results.md, until out of ideas.

STATE AT RESUME:
- branch ft-metal-new-arch @ 8096040, tracked tree clean
- plan docs/plans/metal-batched-moe-architecture.md (692 lines) APPROVED 3/3
  (Momus OKAY, Oracle "PASS. Implement it.", reviewer3 "Yes. The plan is executable.")
- c/deepseek_v4 is now the METAL=1 build md5 f93118b1 (was 1b658b15 METAL=0)
  golden PASSES on it: md5=5d04890413ff539e802985ce8c727814, Metal inactive unless COLI_V4_METAL=1
- NEXT: T-1 = CPU-only grouped scheduler (the K0 kill test). NOT a CPU-speed gate;
  it measures (a) token-exactness of grouping, (b) non-kernel overhead fraction vs 0.167.

### F16. ATTENTION LANE probed (E59) - promising but unscoped
validation/probes/attn_fp8_sweep.m, real shapes, FAIR CPU baseline (256-LUT + 4-way ILP, 12 thr):
  S=64: wq_a 3.26x, wq_b 6.90x (GPU 572 GF/s vs CPU 83), wkv 4.42x, wo_b 7.56x -> per-layer 6.0x
  S=1 : GPU LOSES (0.16-0.89x). crossover ~S=16. Same batch-decides-the-sign story.
FIRST RUN SAID 45x - that was MY strawman CPU (scalar e4m3 per element). Engine uses a 256 LUT
(:12350). Fixed -> 6.9x. Never publish the 45x.
SCOPING: projections = 3.40 s of the 14.4 s attention block = 7.8% of the 43.5 s wall.
  6.0x on them alone = 1.070x full prefill. NOT ENOUGH ALONE (gate is 1.12x).
DECISIVE UNKNOWN: the other 11.0 s of attention (QK^T/softmax/AV/RoPE/DSA indexer/recurrent
compressor :2603/:2688). Dense+batchable -> lane >> MoE lane. Recurrent -> lane capped ~1.07x.
=> B3 ATTENTION ATTRIBUTION IS NOW THE CRITICAL MEASUREMENT. Do not scope attention work before it.

### F17. ATTENTION INTERNAL STRUCTURE (read from source, :2546-2777) - decides the lane's size
coli_v4_attention_window_batch_ref stages:
  BATCHED (GPU-favorable, measured 6.0x in E59):
    :2592 wq_a matmul   :2632 wq_b matmul (the big one, 1024->32768)
    :2644 wkv matmul    :2772 wo_b matmul
  PER-ITEM / SEQUENTIAL (NOT batched today):
    :2611 coli_v4_compressor_step   - RECURRENT (carries kv_state/score_state)
    :2619 coli_v4_indexer_step      - per item
    :2663-2675 RoPE apply x64 heads + bf16 round, per item
    :2692-2694 KV ring memcpy per item  <-- CAUSAL: token i writes KV that later tokens read
    :2727 coli_v4_sparse_attention_ref  - THE ATTENTION CORE, per item, 64 heads x head_dim 512
    :2732-2738 RoPE apply on attended x64 heads, per item
    :2764 wo_a matmul PER GROUP (not batched over items)
KEY STRUCTURAL FACT: :2694 makes the per-item block CAUSALLY SEQUENTIAL within a chunk - token i
attends to KV written by tokens < i. So the attention core canNOT be batched across tokens the way
the projections are. Parallelism inside one token is 64 heads x selected keys, which IS large.
=> The lane's size depends on how the 14.4 s splits between (a) the 4 batched matmuls ~3.4 s
   (measured 6.0x available) and (b) the per-item block ~11.0 s (needs a DIFFERENT strategy:
   per-token GPU dispatch of 64-head sparse attention, or CPU SIMD, or nothing).
=> B3 must time :2611 / :2619 / RoPE / :2727 / :2764 SEPARATELY. Without that split, any attention
   proposal is guesswork - and this project has killed 4 hypotheses for exactly that mistake.

### F18. MXFP4 kernel was OCCUPANCY-limited (E60)
v1/v2 dispatched 1D over O only = 2048 threads on 40 GPU cores. fp8 attention probe dispatched 2D
(rows x S) and hit 572 GF/s. Re-dispatching MXFP4 2D (validation/probes/sweep3_2d.m):
  S=4 81.6 GF/s (205.6us/tok) | S=16 147.3 (113.9) | S=32 275.1 (61.0) | S=64 421.6 (39.8)
=> 3.2x past the old 130 GF/s wall. Format was never the limit.
=> KERNEL IS NOT THE BOTTLENECK. N IS. N=4.10 sits in the FLAT part of the curve.
=> chunk-cap lever (T9) is promoted to the PRIMARY lever of the MoE lane.
CAUTION: my coupon-collector extrapolation of N vs tokens saturates unique at ~95, but the MEASURED
max at chunk=64 is 188. Model contradicts data -> N at larger chunks is UNMEASURED. T9 must read it
from the engine's unique= log. Do not attach a perf claim to the extrapolation.

### F19. DESIGN IMPROVEMENT - decouple the MoE batch from the attention chunk
Today target_batch (:8879) is `for layer { for offset+=64 { attention -> MoE } }`, so MoE's batch is
FORCED to equal attention's 64-token chunk. That is what pins N=4.10 in the flat part of the E60
kernel curve.
MoE does not need attention's batch. Restructure to:
   for layer { for offset+=64 { ATTENTION only, buffer ffn_norm[token] } ; ONE grouped MoE over ALL
   tokens of the layer }
=> raises the MoE batch to WHOLE-PROMPT scope WITHOUT touching any of the 7 batch-cap sites,
   including the __m256 sums[64] stack-overflow hazard at :12355, and without disturbing attention's
   recurrent compressor (:2611) or causal KV ring (:2694).
=> buffer cost is tiny: 799 tokens x 4096 x 4B = 13.1 MB.
TRADEOFF: n_unique per MoE call also rises, so more waves vs capacity 164 - the plan's wave design
already covers it. UNMEASURED: actual N and n_unique at whole-prompt scope; must be read from the
engine's unique= log (same caution as E60).

### ===== SESSION 2 RESULTS (E58-E64) =====
E58 premise overturned: GPU 3.1x CPU bandwidth (hardened, checksum-verified). Prefill ALREADY
     layer-outer/chunk-inner with 64 tok in flight; only MoE discards it. S=1 is the real fault.
E59 attention lane probed: dense fp8 projections 6.0x GPU-favorable at S=64 (572 GF/s vs 83).
     CAUTION: first run said 45x - my CPU baseline was a strawman (scalar e4m3 vs engine's LUT).
E60 MXFP4 kernel was OCCUPANCY-limited not format-limited: 1D dispatch (2048 thr on 40 cores).
     2D (O,S) -> 421.6 GF/s at S=64. KERNEL IS NOT THE BOTTLENECK, N IS.
E61 T-1 CPU grouped MoE SHIPPED behind COLI_V4_MOE_GROUPED: BIT-EXACT (same golden md5),
     K0 PASS (overhead 9.72% < 16.7%), +4.15% p064 / +2.09% p256 (3/3 pairs each).
     Gain HALVES with length - same decay as prefetch, same cause: chunk cap pins N=4.10.
E62 measured N at whole-prompt scope: 4.14 -> 7.99 (1.93x) by decoupling MoE batch from the
     attention chunk. My coupon-collector model had predicted ~16 - WRONG by 2x. Measured instead.
     Early layers touch 226/256 experts (NOT concentrated); later ones 126-151.
E63/E64 B3 attention attribution, residual driven 26.6% -> 0.21%:
     proj_wo_a 25.66% (WAS HIDDEN - biggest stage) | proj_out 19.23% | proj_wq_b 17.47%
     sparse_core 17.28% | compressor_indexer 14.06% | wq_a 3.23% | wkv 2.01% | rest <1%
     ALL FIVE projections = 9.838s = 67.6% of attention = 22.6% of the prefill wall.
     At E59's 6.0x -> 1.232x full prefill. Even worst shape (3.26x) -> 1.186x. CLEARS 1.12x gate.
     alloc-churn hypothesis TESTED AND WRONG (2.2ms, 0.02%).
     Not batchable: sparse_core + compressor = 31.3% of attention / 10.5% of wall.

### PRIORITY CHANGE
ATTENTION PROJECTIONS (1.23x, resident weights, no SSD/lease/wave machinery) now OUTRANK the
batched-MoE plan (~1.10-1.15x projected). MoE grouped path already landed and is kept (free,
bit-exact). Next implementation target = GPU the five fp8 projections.

### ENV KNOBS ADDED THIS SESSION (all default OFF, all measurement-safe)
  COLI_V4_MOE_GROUPED=1        CPU grouped MoE scheduler (bit-exact, +4.3%)
  COLI_V4_MOE_GROUPED_STATS=1  moe_group_overhead_ns / moe_chunk_ns / waves / fallbacks / groups
  COLI_V4_MOE_GROUPED_DUMP=1   per-chunk sorted unique expert ids (offline union analysis)
  COLI_V4_ATTN_STATS=1         12-slot prefill attention attribution, residual 0.21%

### NEXT
1. bit-exactness feasibility probe for a Metal fp8 dense matmul vs coli_fp8_matmul_batch_ref
   (must match the CPU accumulation order exactly; check the portable non-AVX2 reference).
2. if exact -> seam + host glue for the five projections (no gather/waves/leases needed).
3. still open: F19 decoupling for MoE (1.48x on expert kernel cost, measured N 7.99).
