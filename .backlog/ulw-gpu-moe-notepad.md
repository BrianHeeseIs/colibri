# Ultrawork Notepad — wire grouped MoE wave to dispatch batched (S~4) expert matmuls through the Metal seam
Started: 2026-08-15

## Goal
Plan of record: docs/plans/metal-batched-moe-architecture.md
Expected: ~1.06-1.07x incremental over shipped 1.189x/1.179x -> ~1.26x total.

## HARD CONSTRAINTS (violating any = failure)
- golden md5 `5d04890413ff539e802985ce8c727814` is SACRED. Never adjust expected; fix code.
- ab.sh sign trap: delta = 100*(on-off)/off -> FASTER IS NEGATIVE. 1.12x == delta <= -10.71%.
- ab.sh runs `--max-tokens 1`, parses ONLY `time_to_first_token`. p064/p256 are PROMPT FILES.
- Seed: restore /tmp/coli_usage.snapshot (md5 599f3d12e9347ef30541bd6f9ba18bde) BEFORE and AFTER
  every engine run; always COLI_V4_SAVE_USAGE=0.
- ONE engine at a time (`pgrep -f '[d]eepseek_v4'`). engine+non-engine <= ~100 GB.
- Metal: NO fp64. fastMath MUST be off (-fno-fast-math / MTLMathModeSafe). threadgroup 32768 B.
- New feature ships DEFAULT OFF behind its own env flag.
- Never build trace/experimental variants in-tree; use scratch copy, verify c/deepseek_v4 md5 after.

## KNOWN FACTS (measured, do not re-derive)
- MoE = 58.0% of p064 metric, 51.7% of p256.
- N = 1.84 rows/expert visit whole-prompt; E58e: chunk-64 gives N~4.1.
- E58d GPU vs CPU: S=1 1.05x | S=4 1.52x | S=16 2.46x. "Fault is S=1, not the kernel."
- Grouped path is CPU-ONLY: no Metal seam call in deepseek_v4.c:5040-5800.
- COLI_V4_METAL=1 GPU MoE path = 1.118x SLOWER (dispatches at S=1).
- MoE effective 2.28 GB/s vs 6.83 GB/s disk -> compute-bound, not I/O bound.
- Probe->in-situ dilution observed: 2.39x, 2.15x, 3.58x.

## Plan (exhaustive, atomic)
(pending plan agent)

## Scenarios (the contract)
(pending)

## Now
Firing explore agents on the 3 knowledge gaps blocking a precise plan.

## Todo
(pending plan agent)

## Findings
- GPU MoE kernel: `coli_v4_moe_expert_fp4_ordered_cold` @ c/metal/coli_v4_moe.metal:170
  bufs: input, gate_q4, gate_scales, up_q4, up_scales, down_q4, down_scales, output, ColiV4MoeParams
  threadgroup: input_qdq[COLI_V4_MOE_MAX_HIDDEN] + gate_up[2*COLI_V4_MOE_MAX_INTERMEDIATE] = full 32KB
- Host dispatch site: c/backend_metal_v4.mm:343 (newFunctionWithName)
- Grouped wave calls coli_v4_prefill_trace_expert_forward -> coli_v4_expert_forward_ref
- coli_v4_expert_forward_ref declared at deepseek_v4.c:8179 / 8258 / 11518 (unit-gated variants)

## Learnings
- Four scope errors this session came from dividing a counter by a wall of different scope.
  ALWAYS state the scope of a counter before dividing it.

## Explore agents dispatched (parallel, background)
- bg_935c61a6 / ses_ffb71f37fffeGE24xtxXQA4Jza — grouped MoE wave structure, insertion point, group==expert? layout
- bg_b9d05026 / ses_ffb71a902ffeyiURFfEblQXAd4 — Metal MoE backend, ColiV4MoeParams, S>1 batching, weight transfer, reusable kernels
- bg_9da55bba / ses_ffb715a28ffeWP5jqm62UWOX7s — bit-exactness contract of coli_v4_expert_forward_ref (qdq, mxfp4 decode, accum order, swiglu, routing weights, AVX2 hazards)

## Gate before ANY implementation
Re-verify the S=4 GPU advantage AT PRODUCTION DIMENSIONS with REAL MXFP4 weights, CPU vs GPU in the
SAME process on the SAME buffer (the discipline that caught A7). E58d's 1.52x@S=4 was an out-of-situ
probe; standing dilution is 2.15-3.58x. If in-situ S=4 does not beat CPU, the build is dead and I
report that instead of building it.

## FINDINGS from 3 explores (all file:line verified)

### Insertion point (explore 1)
- `coli_v4_moe_grouped_batch` @ deepseek_v4.c:5115+
- Grouping ALREADY gathers routes per expert:
  `counts[group]` :5235 | `offsets[group..group+1]` :5237-5240 | `route_slots[offset]=route` :5241-5244
- **N = offsets[group+1] - offsets[group]** (rows for one expert)
- Loop nest: for wave :5293 -> for expert-in-wave :5348 -> **for offset :5350-5351**
- **THE S=1 DEFECT**: forward is called ONCE PER ROUTE @ :5356-5359
  `coli_v4_prefill_trace_expert_forward(expert_output, &wave_views[current], inputs + item*dimension, route_weights[route], config->swiglu_limit)`
- Rows are NOT contiguous: gathered `route = route_slots[offset]`, `item = route/topk` :5352-5353,
  row ptr = `inputs + item*dimension` (inputs is batch-major, each row contiguous len=dimension)
- Accumulate: `routed_acc[item*dimension+idx] += expert_output[idx]` :5366-5369
- Final: `outputs[...] = bf16_round(routed_acc + shared_outputs)` :5377-5381
- Live variant = COLI_V4_UNIT_EXPERT_ROWS16 @ :8258-8296 (units file :24). COLI_V4_UNIT_EXPERT is TEST-ONLY.
- Fallback triggers: wave_size<2 :5246-5259 | expert acquire fail :5334-5339

### Metal side (explore 2) - THE BIG ONE
- **Fused kernel `coli_v4_moe_expert_fp4_ordered_cold` is DEAD CODE**: `__attribute__((unused))`
  @ backend_metal_v4.mm:338, never dispatched. Its 32KB threadgroup limit is IRRELEVANT.
- Real path = multi-kernel chain @ backend_metal_v4.mm:658-959 (qdq, gate, up, swiglu, weighted,
  down qdq, down, bf16) in ONE command buffer.
- **MXFP4 kernels ALREADY S-PARAMETERISED**: `struct ColiV4MatmulDims { int S, I, O, rb, ng; }`
  @ coli_v4_matmul.metal:21. `coli_v4_matmul_mxfp4_ordered_xcache` uses 2D grid (O,S),
  `s = gid.y` :156-160, writes `y[s*O + o]` :177.
- **HOST HARDCODES S=1** @ backend_metal_v4.mm:824-827 gate_dims={1,hidden,intermediate,..},
  down_dims={1,intermediate,hidden,..}
- Seam API is scalar, no batch arg: `coli_v4_metal_expert_forward(out, expert, input, route_weight,
  swiglu_limit)` @ backend_metal_v4_seam.h:17-19
- Weights zero-copy via slab `newBufferWithBytesNoCopy` :208-212, falls back to memcpy if not
  256B-aligned :223-237. fp8 cache (4096) is a SEPARATE attention thing :443-449.
- Elementwise dispatches are per-row sized: `dispatchThreads(intermediate,1,1)` :885-909,
  `dispatchThreads(hidden,1,1)` :937-938 -> must become *S.

=> **THIS IS NOT "WRITE A BATCHED KERNEL". IT IS "STOP PASSING 1".**

### Bit-exactness contract (explore 3)
Live CPU order (deepseek_v4.c:8258-8295):
 1. `coli_fp4_dual_matvec_rows16_v10(gate, up, ...)`  (NEON rows16)
 2. `coli_bf16_round_array(gate)`, `coli_bf16_round_array(up)`
 3. `coli_v4_swiglu(activated, gate, up, intermediate, swiglu_limit)`
 4. `activated[i] = coli_bf16_round(activated[i] * route_weight)`
 5. `coli_fp4_matvec_rows16_v10(output, &expert->down, activated)`
 6. `coli_bf16_round_array(output)`
- Activations QDQ'd to **FP8 E4M3, block 128** before EVERY fp4 matvec (`coli_fp8_activation_qdq_ref`
  :12599-12621). Weights e2m1 nibble + e8m0 per-32 scale.
- rows16 NEON accum: `sums = vadd(sums, vmul(vmul(x, values), scales))` :13235-13249,
  columns ascending, **NO FMA** (comment :13191-13195).
- swiglu: `gate*sigmoid_stable(gate)*up` with clamps :1641-1653; `sigmoidf_stable` uses **expf** :1360-1367
- bf16_round :12568-12577. route_weight applied BEFORE down proj, float, then bf16.

### HAZARD + WHY IT IS ALREADY SOLVED
`expf` in sigmoid is a classic CPU/GPU bit-exactness hazard. BUT experiments_results.md:742 records
the existing Metal expert path shipped **"with correctness perfect"** at 1.118x slower. So the
numerics were ALREADY made bit-exact at S=1. **Must re-verify, not assume.**

### KEY INSIGHT: BATCHING IS NUMERICALLY NEUTRAL
Each (s,o) output accumulates over the same columns in the same ascending order regardless of S.
QDQ block-128 never crosses a row boundary. So S=1 -> S=N must not change a single bit BY
CONSTRUCTION. Any diff = a real bug, not a tolerance question.

## G1 GATE: **PASSED** (commit a6170ea, E83)
probe: validation/probes/mxfp4_s_scaling.m — CPU vs GPU, one process, same buffers, MTLMathModeSafe

| shape | S=1 | S=2 | S=4 | S=8 | S=16 |
|---|---|---|---|---|---|
| gate/up O=2048 I=4096 | 0.398x | 0.834x | **1.680x** | 3.216x | 4.379x |
| down O=4096 I=2048 | 0.385x | 0.732x | **2.413x** | 1.933x | 4.432x |
(ratio = CPU/GPU, >1 means GPU faster. 3 quiesced reruns: gate/up 1.546-1.752x, down 1.338-2.413x)

CPU fidelity verified: LLDB hit rows16 fns; disasm has NEON tbl.16b/fmul.4s/fadd.4s; block_rows=1
guard returns -1. NOT the scalar fallback. md5 c/deepseek_v4 unchanged.

### *** DESIGN-CRITICAL CONSEQUENCE (plan did not anticipate) ***
GPU LOSES at S=1 (0.40x) AND at S=2 (0.73-0.83x). Break-even is between S=2 and S=4.
Measured routing reality: N=1.84 whole-prompt, ~4.1 at chunk-64 => MANY groups are N=1 or N=2.
=> P2 MUST apply an N THRESHOLD: dispatch to GPU only when N >= threshold (start at 4),
   else keep the existing per-row CPU path. Unconditional batching would SHIP A REGRESSION.
   Make the threshold an env override (e.g. COLI_V4_MOE_BATCHED_MIN_N, default 4) so M1 can tune it
   without a rebuild.
Also note: probe measures MATMUL ONLY. Real path adds QDQ + swiglu + elementwise around it, so the
end-to-end gain will be strictly less than these ratios. Do not project 1.68x onto the wall.

# ============================================================
# HALT POINT — opencode restart requested. RESUME FROM HERE.
# ============================================================

## Progress against plan
- [x] G1 GATE  PASSED — S=4 GPU beats CPU (gate/up 1.680x, down 2.413x). Commit a6170ea (E83).
- [x] G2 GATE  PASSED — `EXTRA_ENV="COLI_V4_METAL=1" ./bench/golden.sh` -> PASS sacred md5.
- [x] R1 RED   captured: `Undefined symbols: "_coli_v4_metal_expert_forward_batch"`
- [x] P1 GREEN 0 ULP at S=1,2,6,16 (0/4096, 0/8192, 0/24576, 0/65536). Golden re-PASSED after fix.
- [ ] P2  <-- NEXT. NOT STARTED.
- [ ] GOLD / M1 / M2 / C1|RB

## UNCOMMITTED WORK (on disk, survives restart) — HEAD is a6170ea
    M  c/backend_metal_v4.mm       (+94/-61: S-aware scratch, QDQ, 2D matmuls, elementwise, per-row weighted-bf16)
    M  c/backend_metal_v4_seam.h   (+5: batch seam declaration)
    ?? validation/probes/moe_batched_seam_check.m   (the 0-ULP probe)
    ?? .backlog/ulw-gpu-moe-notepad.md              (this file)
NOTE: other `??` entries (.backlog/quality/, bench/kernel_gap, validation/glm52-metal/, etc.) are
PRE-EXISTING and NOT mine. Do not clean them.
Binary md5 now 0c59e4350190608185fb288a5c7c3949 (was b6d102b538157f202ea9b0edc3992ee0). Change is
EXPECTED — backend_metal_v4.mm was edited and rebuilt. Golden still PASSES.

## *** TWO FINDINGS THAT MUST SHAPE P2 — DO NOT SKIP ***

### 1. N THRESHOLD IS MANDATORY (from G1)
GPU LOSES below S=4:  S=1 -> 0.40x,  S=2 -> 0.73-0.83x,  S=4 -> 1.68-2.41x.
Measured routing: N=1.84 whole-prompt, ~4.1 at chunk-64 => MANY groups are N=1 or 2.
P2 MUST gate on N: dispatch to GPU only when N >= COLI_V4_MOE_BATCHED_MIN_N (default 4), else keep
the existing per-row CPU path at deepseek_v4.c:5356-5359. Unconditional batching SHIPS A REGRESSION.
Make MIN_N an env override so M1 can sweep it without rebuilding.

### 2. block_rows LAYOUT SPLIT (discovered by the P1 agent — OPEN QUESTION)
The existing SCALAR Metal seam accepts ONLY `block_rows=1` (cold layout). The live CPU production
path requires `block_rows=16` (rows16). The agent tried widening scalar acceptance and it BROKE
GOLDEN; it reverted and instead made the NEW batch seam handle rows16 via `ordered_hot_xcache`.
IMPLICATIONS TO VERIFY IN P2/GOLD — I have not confirmed these:
  a) If production experts are packed rows16, the scalar Metal seam would REJECT them, meaning the
     COLI_V4_METAL=1 path may have been largely INACTIVE in production. That would explain why it
     measured only 1.118x slower rather than catastrophically slower, and it means the batch seam is
     doing something genuinely NEW (rows16 on GPU) rather than just batching an existing path.
  b) The 0-ULP probe compared batch(rows16 layout) against scalar(cold layout) with logically
     equivalent weights. It did NOT compare against the CPU rows16 kernel
     (`coli_fp4_dual_matvec_rows16_v10` / `coli_fp4_matvec_rows16_v10`).
     => GOLD with COLI_V4_MOE_BATCHED=1 is therefore the FIRST real test that batched GPU output
        matches what the CPU path produces end-to-end. Treat a GOLD failure as expected-possible,
        not as a shock, and debug it as a layout/ordering issue.

## Exact next command when resuming (P2)
Wire c/deepseek_v4.c `coli_v4_moe_grouped_batch`, expert-in-wave loop at :5348:
  - N = offsets[group+1] - offsets[group]
  - if COLI_V4_MOE_BATCHED on AND N >= MIN_N: gather N rows (route=route_slots[offset] :5352,
    item=route/topk :5353, src = inputs + item*dimension) into a contiguous staging buffer,
    call coli_v4_metal_expert_forward_batch once with per-row route_weights, then
    scatter-accumulate routed_acc[item*dimension+idx] += out_row[idx] preserving :5366-5369.
  - else: existing per-row path at :5356-5359 VERBATIM.
  - fallback paths (wave_size<2 :5246-5259, acquire-fail :5334-5339) MUST bypass batching and still
    bump coli_v4_moe_wave_fallbacks (:5047).
Then GOLD:
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
  EXTRA_ENV="COLI_V4_METAL=1 COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1" ./bench/golden.sh ./c/deepseek_v4
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
Then M1 (note the ab.sh isolation trap — baseline flags go in the PARENT shell):
  export COLI_V4_METAL=1 COLI_V4_MOE_GROUPED=1
  N=5 ./bench/ab.sh "COLI_V4_MOE_BATCHED=1" ./c/deepseek_v4
  unset COLI_V4_METAL COLI_V4_MOE_GROUPED

## P2 DESIGN DECISION (evidence-based, supersedes earlier assumption)

### Layout reality
`view->block_rows = 1` by DEFAULT (deepseek_v4.c:7065, :11756). Promoted to 16 ONLY for packed
hot-pinned slots (:7667, `policy->packed[hot_slot_index(...)]`, 16 pin slots/layer out of 256
experts). So COLD block_rows=1 experts DOMINATE prefill volume; rows16 is the pinned minority.

### Batch seam accepts BOTH layouts
backend_metal_v4.mm:701+ : `block_rows != 1 && block_rows != 16` -> reject; validates via
`coli_v4_ordered_tensor_valid`; selects kernel via `coli_v4_get_chain_pipelines(block_rows, ...)`.
Scalar seam (:982) guards block_rows==1 then DELEGATES to the batch seam with S=1.

### TRANSITIVE BIT-EXACTNESS PROOF for block_rows==1  (this is why P2 is safe)
- G2: `COLI_V4_METAL=1` golden PASS. In that config the scalar seam accepts ONLY block_rows=1 and
  delegates to the batch seam at S=1 => the GPU chain ALREADY matches the CPU path end-to-end in
  production for cold experts.
- P1: batch(S=N) == batch(S=1) at 0 ULP (S=1,2,6,16).
- => batch(S=N) == CPU for block_rows==1. PROVEN, no new probe needed.

### rows16 is NOT proven
The scalar seam REJECTS block_rows=16, so G2 never compared GPU vs CPU rows16. Excluding rows16 from
batching costs little (pinned minority) and avoids shipping an unproven numeric path.

### DECISION
P2 gates on: flag ON && N >= MIN_N (default 4) && expert->gate.block_rows == 1.
rows16 experts keep the existing per-row CPU path. Seam self-validates; any nonzero return falls back
to the per-row loop so output can never be corrupted.

## P2 COMPLETE + GOLD PASS (both states)

### Build-config trap CAUGHT (would have invalidated everything)
`METAL ?= 0` in c/Makefile.deepseek-v4:97 — Metal is OPT-IN.
`make deepseek-v4` builds WITHOUT the seam: COLI_V4_METAL_SEAM undefined, batched symbol 0 refs,
feature silently compiled out. **CORRECT BUILD: `make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`**
(the suggested_commands memory has been corrected accordingly).

### Implementation (c/deepseek_v4.c, all behind #ifdef COLI_V4_METAL_SEAM)
- `COLI_V4_MOE_BATCHED` (default OFF) + `COLI_V4_MOE_BATCHED_MIN_N` (default 4) read via the existing
  `coli_v4_moe_grouped_once` pthread_once.
- staging buffers allocated ONCE per call (routes x dimension), freed in cleanup, only when flag on.
- gate: flag ON && N >= MIN_N && gate/up/down all block_rows==1.
- gather -> single `coli_v4_metal_expert_forward_batch` -> scatter-accumulate into routed_acc.
- nonzero seam return falls through to the per-row loop (never drops a group).

### Evidence
- build METAL=1: exit 0, **0 warnings**, seam defined 26x, batched symbol linked.
- GOLD flag OFF: `PASS golden md5=5d04890413ff539e802985ce8c727814`
- GOLD flag ON : `PASS golden md5=5d04890413ff539e802985ce8c727814`
- **Path-executed proof** (p064, --max-tokens 1): batched=off metal_dispatches=**0** ttft=35.470s;
  batched=on metal_dispatches=**1013** ttft=**31.582s**. metal_fp8_dispatches=429 in BOTH
  (attention lane untouched). Single-run signal ~1.123x.
  This matters: golden alone could have passed trivially without the path ever firing.

## M1 RESULT + ORACLE REVIEW

### M1 isolated increment (N=5, baseline exported in PARENT so OFF inherits it)
| prompt | off | on | delta | incremental |
|---|---|---|---|---|
| p064 | 35.842 s | 31.975 s | **-10.79 %** | **1.121x** |
| p256 | 94.251 s | 83.221 s | **-11.70 %** | **1.133x** |
Total stack vs bare baseline: 42.528 -> 31.975 s on p064 = **~1.33x**.
NOTE: this BEAT the ~1.06x projection by ~2x. The S=4 probe UNDER-predicted, the opposite of this
host's usual 2.15-3.58x dilution. Unexplained — treat as an open question, not a banked law.

### Oracle review verdict: NO BLOCKING CORRECTNESS BUGS
Verified by Oracle with line refs:
1. scatter/accumulate is order- AND value-identical to the per-row path (route_slots preserves
   ascending route order; batched gather/scatter replay the same order). Duplicate items safe.
2. seam-failure fallback is safe: the seam writes only `batch_outputs` scratch, never `routed_acc`,
   so falling through to the per-row loop cannot double-add.
3. bounds safe: `0 < group_n <= routes`; `gathered_item` in `[0,batch)`.
4. alloc-failure degradation and the #ifdef'd frees are correct (`free(NULL)` fine).
5. NO data race: call path is serial (`target_batch` -> `coli_v4_block_window_batch_ref` ->
   `coli_v4_moe_grouped_batch`), no OpenMP region around the loop, buffers are per-call heap.
6. `batched_done` #ifdef split is safe.

### Oracle finding 7 — DECLINED, with reason
Oracle: batched branch does not AND-in `coli_v4_metal_enabled()`, so `COLI_V4_MOE_BATCHED=1` can
dispatch Metal while `COLI_V4_METAL=0`.
DECLINED because:
- It cites no failing success criterion (golden PASSES, no wrong output, no leak, no race).
- It CONTRADICTS shipped precedent: the attention lane gates on `coli_v4_metal_fp8_enabled()`, which
  reads **`COLI_V4_METAL_ATTN`** — its own flag, not the global one. Per-feature gating is the
  established convention here.
- Adopting it would DISABLE the feature in the exact configuration M1 was measured in, and would drag
  the 1.118x-slower scalar MoE path into every measurement as a confound.
Recorded as a note in `mem:deepseek_v4/metal` so it is not "fixed" by a future agent.

### Oracle finding 5 — accepted, DEFERRED
Wrap the two new static getters in `#ifdef COLI_V4_METAL_SEAM` (they are unused in non-Metal builds;
repo disables -Wunused-function so it does not warn today, but strict builds would).
DEFERRED: **must not rebuild while M2 is running** — swapping c/deepseek_v4 mid-run would corrupt the
in-flight measurement. Apply after M2 completes, then re-verify golden.

## UPSTREAM/FORK SURVEY — RANKED ADOPTION PLAN

### CORRECTION to an earlier claim in this notepad
I wrote "upstream has NEVER touched Metal". **Wrong.** Upstream has `c/backend_metal.mm` (no _v4)
for the OLDER colibri/inkling engines — PRs #72 (Metal backend), #763, #757. What IS unique to us is
`c/backend_metal_v4.mm`, the **V4-specific** Metal backend. Upstream has no Metal for deepseek_v4.

### TIER 1 — adopt now (cherry-picks CLEAN, verified locally)
| commit | gain | note |
|---|---|---|
| `8c40fbd` max_tokens is a ceiling, clamp to context | fixes CONTEXT_EXCEEDED **in the engine** | supersedes the client-side clamp I put in the web bridge; engine knows the exact prompt count |
| `a20c7aa` skip OMP spin-wait tuning on macOS (=PR #957, issue #707) | issue #707 reports a **2.2x decode regression** on macOS from this | our exact platform |
| `1684e89` size the OpenMP team around the expert loaders | thread/loader balance | pairs with the loader pool |

### TIER 2 — high value, needs manual port (cherry-pick CONFLICTS)
| item | evidence | why |
|---|---|---|
| **issue #900** hot rows16 pack under `state->mutex` | #900: **6.87 s extra TTFT, 5.70 s of it lock-held**, +~1.2 s from the block_rows=16 path | **WE CARRY THIS BUG** — verified: `hot_pack_slot_locked` :7793 (comment: "state->mutex is held"), called :7911/:8084; `hot_pack_matrix` does pack_rows16 + 2 full memcpys x gate/down/up (~12.6 MB) under the global mutex. 0 matches for the fix. AUTOPIN is on by default so it fires. **Highest-value single item found.** |
| PR **#983** (preferred) vs **#1023** | #983 adds per-slot pack mutexes, defers in-place rows16 conversion to an exclusive lease, ships a regression test; #1023 is a smaller prepare/commit split | librarian verdict: #983 more correct, #1023 smaller surface |
| `48db957` expert-loader pool, runtime-tunable depth | disk N+lanes overlaps CPU expert N | conflicts in deepseek_v4.c |
| `7b50437` + `b78431f` dashboard telemetry (EMAP/HITS/TIERS/PROF/HWINFO + `matmul_sec`) | issue #890 was "100% other" because matmul was never timed | the real contract for the web UI I hand-rolled |

### TIER 3 — technique transfer only (different engine or arch)
- PR **#763** lifts the Metal attention `S<=4` cap for prefill (`COLI_METAL_PREFILL=1`):
  prefill attention **35.9s -> 9.0s**, prefill wall **96.4s -> 67.3s**. OLD engine, but the S-cap
  lesson is exactly our S>=4 batching finding, independently confirmed.
- PR **#757** overlap shared experts with the GPU round: decode 1.75 -> 2.46 tok/s, prefill 10.3 -> 6.4 s.
- Issue **#428** Metal fused-decode `r_top8` first: +6.7%, identical output.
- Issue **#250** proposes Apple **AMX** for dynamic expert matmul (closed, unimplemented).

### NOT APPLICABLE
- PR **#1024** FUSED3 AVX2 expert matmul (40% less matmul time, 2.08->2.38 tok/s): guarded
  `#if defined(__AVX2__)`, **no ARM/NEON variant**. x86 only.
- PR **#934** ARM NEON `matmul_e8` — arm64 but `fmt=6 E8/IQ3` only, no MXFP4/e2m1.
- PR **#1017** `coli_v4_route_bf16` in `moe_token` — single-token path only; our metric excludes decode.
- CUDA PRs (#935/#936/#1031), #964 backend registry (architectural, not perf).

### FORKS — mostly mirrors
Sampled newest 1000 forks. Only 5 have real divergence: `simsim314` (GGUF+scheduler, +12),
`gohlerdev/BetterColibri` (+48, io_uring, TC_INT4), `slimedragonair` (Vulkan expert cache, +34),
`oncoapop` (Vulkan staged upload, +6), `cloudQuant` (+2, V4 serving).
**No substantive Apple-GPU fork.** `codearranger/colibri:metal-backend` is the only verified Metal
fork and it is the OLD-engine backend already merged upstream as PR #72.

### Apple Silicon datapoints for context (decode tok/s)
M3 Max 128GB 0.35-0.45 (#47) | M4 Pro Metal 0.30 (#107) | M3 Ultra 1.45 @82% hit (#180) |
M5 Max Metal 1.83 (#87) | M5 Max 2.06 (#103).
**Our M3 Max measured 1.52-1.56 tok/s decode** via the REPL — competitive with M5 Max numbers.

# ============================================================
# HALT — host going to standby. RESUME FROM HERE.
# ============================================================

## Branches
- `ft-metal-new-arch` @ 633024a — shipped work + dashboard telemetry. GOLDEN PASS verified.
    633024a lab: serve Brain and Profiling from live engine telemetry
    e9deae0 feat(v4): emit dashboard telemetry (TIERS/EMAP/HITS/HWINFO/PROF)  [ports 7b50437 + b78431f]
    242ad58 fix(omp): skip spin-wait tuning on macOS (#707)                   [cherry-pick -x a20c7aa]
- `ft-opensourceftw1` — upstream adoption branch (CURRENT).
    <1684e89 pick> deepseek_v4: size the OpenMP team around the expert loaders [cherry-pick -x]
    4edd865 lab: drop the bridge's client-side max_tokens clamp
    c4f056c v4: max_tokens is a ceiling — clamp to context (#975)             [cherry-pick -x 8c40fbd]

## Gates PASSED (evidence captured)
- golden @ 099e7ce baseline .................. PASS 5d04890413ff539e802985ce8c727814
- golden after telemetry port ................ PASS (same md5)
- golden after 8c40fbd ....................... PASS (same md5)
- golden after 1684e89 ....................... PASS (same md5)  <- highest bit-exactness risk, survived
- a20c7aa binary-identity (F1) ............... IDENTICAL (same-dir test)
- 8c40fbd GREEN .............................. "[V4] max_tokens 5000 clamped to 511 (context 512 - prompt 1)" + ACCEPT
- 8c40fbd EDGE ............................... "CONTEXT_EXCEEDED prompt_tokens=80 requested=1 capacity=64" (still rejects, clamp not over-permissive)
- [OMP] banner ............................... "13 compute threads (16 logical CPUs minus 3 expert-loader workers)"

## THE ONE OPEN ITEM: W3.3 A/B gate for 1684e89 — DID NOT RUN
`.backlog/lab/gate_20260815-154737.log` stops after both WARMUP lines; no RUN/AB lines, exit was
silent. **Two bugs to fix before retrying:**
1. MY runner `.backlog/lab/run_gate.sh` pipes ab.sh through `grep -E '^(RUN|AB) '`, which HIDES any
   error ab.sh prints. Remove or widen that filter first — the failure is currently invisible.
2. Root cause of the early exit is UNDIAGNOSED. Reproduce with ab.sh run directly and unfiltered:
     export COLI_V4_MOE_GROUPED=1 COLI_V4_METAL_ATTN=1 COLI_V4_MOE_BATCHED=1
     N=3 ./bench/ab.sh "COLI_NO_OMP_TUNE=1" ./c/deepseek_v4     # unfiltered, watch the output
   Suspect ab.sh's ON_ENV validation or its own engine/precondition check.

### What that gate decides
OFF = tuning ACTIVE (13 threads, the pick).  ON = COLI_NO_OMP_TUNE=1 (16 threads, old behaviour).
delta = 100*(on-off)/off, FASTER IS NEGATIVE.
**delta <= -1.5% on either prompt => the old behaviour is faster => REVERT 1684e89.**
Prediction (F3, confirmed): `COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER` has **9 guards in
deepseek_v4.c but 0 occurrences in the build config**, so this commit reserves 3 of 16 CPUs for a
loader pool that is NOT compiled into our binary, on a prefill-dominated metric. A regression is the
expected outcome. 1684e89 is isolated as its own commit precisely so `git revert` is surgical.

## Still pending after that
W0.6 TTFT baseline N=5 | W4.1 adjacent regression (GROUPED/METAL_ATTN/BATCHED within +/-1.5pp)
W5 #900 PACK_SHARE measurement (scratch COLI_V4_PREFILL_TRACE build; divide by TTFT_MS, NOT pct_wall)
W6 port #983 only if PACK_SHARE >= 5.0% | CLOSE relaunch bridge

## Host facts (corrected)
RAM is **137.4 GB** (an earlier plan claimed 128 — wrong). Swap was 95% used with the bridge live,
eased to 9341M used / 3970M free after stopping it. Rebuild ALWAYS with
`make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`; verify with
`nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch` > 0.
