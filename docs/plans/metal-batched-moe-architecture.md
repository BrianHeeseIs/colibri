# Batched (S=N) Metal MoE Architecture for DeepSeek-V4 — Implementation Plan

**Target host:** Apple M3 Max — 40 GPU cores, 16 CPU (12P+4E), 137 GB unified, 16 KB pages,
Metal 3 / GPUFamily Apple 9, 32768 B threadgroup memory, 1024 max threads/threadgroup,
threadExecutionWidth 32, recommended max working set 115.4 GB.

**Status:** PLAN — not yet implemented. Branch to be created: `ft-metal-new-arch`.

**Supersedes:** the wave plan in `experiments-backlog.md` (T1–T12), which is marked SUPERSEDED and
whose line references have drifted (see §1.4).

---

## 0. Executive summary

The shipped Metal backend is **correct but 1.118x slower**. The prior verdict attributed this to
four structural causes and concluded no backend tuning could change the sign. **One of those four
causes is false on this host, and I measured it.** The real fault is narrower and fixable: the seam
is S=1, so the GPU never reaches the batch size at which it wins.

This plan raises S by batching tokens per expert. It reuses kernels that are already S=N-capable.

**Honest ceiling, stated first (revised after adversarial review).** Two independent reviews
computed the Amdahl bound from the measured p064 attribution and both landed in the same place:

- expert-path gain at the measured `g_hit = 1.52` and hit rate `h = 0.577..0.677`:
  `1 / ((1-h) + h/1.52)` = **1.25x .. 1.30x on the expert path alone**
- untouched share of the prefill wall: miss read/first-touch **43.47 %** + attention **33.18 %** +
  norms/HC/head **5.22 %** + residual **1.55 %** = **83.42 %**
- **full-prefill ceiling ≈ 1.20x .. 1.25x**, *before* any new gather/sort/unpermute cost

**Therefore the original 1.3x acceptance gate was above the theoretical ceiling and is withdrawn.**
The revised gate is in §6/§9. The bandwidth advantage, the traffic reduction and the S=4 microbench
gain are **not independent multiplicative levers** — they are three views of the same effect, and
the earlier draft double-counted them.

---

## 1. Core premise — investigated, not inherited

### 1.1 The claim that is FALSE

`deepseek-v4-RESULTS.md` §12 cause #3: *"Unified memory gives the GPU no bandwidth advantage. Both
processors read the same DRAM."*

Measured (`/tmp/bwprobe.m`, one 4 GB `MTLResourceStorageModeShared` buffer, same physical pages):

| path | streaming read |
|---|---|
| CPU (12 P-cores, GCD, 4-way unrolled) | **120.5 GB/s** |
| GPU (saturating dispatch) | **370.8 GB/s** |
| ratio | **3.08x** |

370.8 GB/s is ~93 % of the M3 Max's 400 GB/s spec — a credible saturation figure. The CPU simply
cannot saturate the fabric. **The GPU has a ~3.1x bandwidth advantage on this host.**

**Scope of this number (revised).** It is *qualitative* evidence that kills the old "GPU has zero
bandwidth advantage" claim. It is **not** a multiplier in the performance model: the CPU side is a
scalar GCD scan and the GPU side is an ideal hot linear read, so the ratio is an upper bound on a
best case that the MoE path never reaches (the MoE path runs at 4.58 GB/s effective — §2.4). It is
**excluded from all speedup arithmetic in §6/§7.**

**Hardened against the dead-code-elimination objection** (`/tmp/bw2.m`): the GPU kernel was rewritten
to perform an *unconditional* `out[gid] = sum` write, and both sides are checksum-verified against the
known exact sum. The GPU checksum matches **exactly** at every size, proving the loads were not
eliminated. The ratio is stable across sizes far larger than any cache:

| buffer | CPU GB/s | GPU GB/s | ratio | checksum |
|---|---|---|---|---|
| 512 MB | 118.7 | 271.3 | 2.29x | match |
| 1 GB | 115.8 | 350.6 | 3.03x | match |
| 2 GB | 113.1 | 374.0 | 3.31x | match |
| 4 GB | 115.0 | 367.3 | 3.19x | GPU match |
| 8 GB | 117.8 | 379.1 | 3.22x | GPU match |

The CPU checksum diverges at ≥4 GB purely from float32 accumulator saturation
(`805306368 == 48 × 2^24` exactly: 12 threads × 4 accumulators each pinned at 2^24). That is a
checksum artifact only — identical loop, identical bytes touched, timing unaffected.

### 1.2 The claims that remain TRUE

- **S=1 is real.** The seam is `coli_v4_metal_expert_forward(float *out, const ColiExpertView*,
  const float *input, float route_weight, float swiglu_limit)`
  (`c/backend_metal_v4_seam.h:17`). One token in, one token out, always. Host glue at
  `c/backend_metal_v4.mm:415-715` builds **one blocking command buffer per token-expert**.
- **~Zero weight reuse at S=1.** ~13.37 MB of expert weights per ~0.5 MFLOP.
- **Prefill did not help** — because prefill walked the same S=1 seam, not because prefill lacks
  tokens (see §1.3, which corrects this).

### 1.3 What the four-cause analysis got structurally wrong

It claimed *"the engine walks prefill token by token through the same S=1 seam"* and treated that as
immovable. **The engine does not walk prefill token by token.** Verified:

- `target_batch` (`c/deepseek_v4.c:8879`) is **layer-outer, chunk-inner**:
  `for (layer) { for (offset += 64) { chunk = min(64, batch-offset); ... } }`
- **Attention already batches across those tokens**: `coli_fp8_matmul_batch_ref(qa, &wq_a, inputs,
  batch)` at `:2592`, `:2632`, `:2644`, `:2772`.
- **Only MoE fails to**: `:5110 for (item < batch)` → `:5141 moe_token_pipeline(..., tokens[item])`.

So up to **64 tokens are already in flight at every layer**, and MoE is the only consumer that
discards that. This is the single most important structural fact in the plan: **S=N needs no prefill
restructure.**

### 1.4 Stale references in the source document (corrected)

| `experiments-backlog.md` claim | verified reality |
|---|---|
| `moe_token_pipeline` at `deepseek_v4.c:3579` | it is at **:4639**; :3579 is a gate-bias line inside `moe_token` (:3559) |
| "three seams" (3167, 3730, 3763) | **four** call sites: **3621, 4620, 4831, 4877** |
| one `coli_v4_expert_forward_ref` | **three** definitions (7582, 7661, 10899); shipped is **:7661** under `COLI_V4_UNIT_EXPERT_ROWS16` |

**Rule for this plan: ground every reference in current code. Never trust the backlog's line numbers.**

---

## 2. The prize, quantified

### 2.1 Why S=1 loses — measured on the real shape

`/tmp/sweep_mxfp4.m`, `/tmp/sweep2.m`, `/tmp/cpu_mxfp4.c`. Real gate/up shape I=4096, O=2048,
MXFP4 block 32. **Microseconds per token for one matrix:**

| S | CPU | GPU (simple) | GPU (tiled+uchar4) | best GPU | verdict |
|---|---|---|---|---|---|
| 1 | 397.0 | 376.9 | 968.5 | 376.9 | GPU 1.05x |
| 2 | 325.5 | 290.1 | 304.8 | 290.1 | GPU 1.12x |
| 4 | 320.3 | 256.3 | 210.6 | 210.6 | **GPU 1.52x** |
| 8 | 219.8 | 212.4 | 163.5 | 163.5 | GPU 1.34x |
| 16 | 317.0 | 147.7 | 128.9 | 128.9 | **GPU 2.46x** |

- **CPU does not scale with batch** (1.25x from S=1→16; S=16 regressed vs S=8, likely cache thrash).
- **GPU scales 2.9x** (376.9 → 128.9 us/token).
- **GPU wins at every S tested**, and the advantage grows.

At S=1 the kernel edge is only **1.05x** — trivially erased by per-call command-buffer construction
and `waitUntilCompleted`. Combined with **17.8 % GPU coverage**, the shipped **1.118x slower** is
fully explained. **The fix is raising S, not micro-tuning kernels.** This is now measured.

Both probes wall at **~114–130 GFLOP/s ≈ 3 % of this GPU's fp32 peak**, so the GPU column above is a
**lower bound**; an optimised kernel has substantial headroom.

### 2.2 The achievable N — measured, not assumed

The engine already logs this (`:5086`). p064 prefill, prefetch ON, 86 layer-chunk records:

| chunk | selections | unique experts | **avg N** |
|---|---|---|---|
| 6 | 36 | 24.3 | 1.48 |
| **64** | **384** | **93.7** (min 73, max 188) | **4.10** |

Only **37 %** of the 256-expert space is touched per chunk. Uniform-random routing over 384 draws
would touch ~202 experts; the actual 94 means **routing is heavily concentrated** — which favours
batching.

**The prize at chunk=64:**
- **4.1x fewer dispatches** (94 instead of 384)
- **4.1x less weight traffic** (each expert's 13.37 MB read once, not 4.1 times)
- stacked on the **3.08x** GPU bandwidth advantage
- landing at S≈4, where the measured GPU advantage is **1.52x**

### 2.3 The independent lever

The batch cap of 64 is a code constant — but in **seven+ places, not three** (draft 1 was wrong,
and dangerously so):

| site | nature |
|---|---|
| `:8884` | `target_batch` chunk stride |
| `:4996` | block batch rejects `>64` |
| `:2551` | attention batch rejects `>64` |
| `:12317` | **FP8** batch matmul rejects `>64` |
| `:12389` | **FP4** batch matmul rejects `>64` |
| **`:12355`** | **`__m256 sums[64]` with `for (item < batch)` — removing the guard alone is a STACK BUFFER OVERFLOW** |
| `:1232` | resource planner reserves scratch for exactly 64 rows |
| `:8808`, `deepseek_v4_dspark.inc:21` | DSpark 128-entry rings — chunk 256 needs explicit DSpark-ON verification |

Raising the cap is therefore **not** a three-constant edit. Raising it raises N. Attention's recurrent parts (`:2603` compressor, `:2688` ring
buffer) are a **sequential inner loop over items** — raising the cap makes them iterate more, it does
not break them; the matmuls batch. Estimated chunk=128 → N≈5.9, chunk=256 → N≈9.0. **UNMEASURED — T9 measures it.**
**Do not ship on projected chunk=128/256 wins.** Note also that chunk=256 likely pushes `n_unique`
past the per-layer slot capacity (§4.8) and would fall back unless expert waves land first.

### 2.4 What batching cannot fix

Per-expert time today is 2.919 ms (CPU) for 13.37 MB = **4.58 GB/s effective**, which is 26–80x
below either processor's streaming bandwidth. That matches SSD (E52: QD1 5.227 GB/s → 2.56 ms).
**The cache-miss path is SSD-bound and no compute placement changes it.** The GPU prize applies to
cache **hits** (57.7–67.7 % at `--ram 96`) and to dispatch amortisation. Any claim of end-to-end
speedup must be read against that ceiling.

---

## 3. Assets that already exist (reuse, do not rebuild)

| asset | location | state |
|---|---|---|
| **S=N-capable bit-exact matmul kernels** | `c/metal/coli_v4_matmul.metal` `ordered_xcache` | `Dims{S,I,O,rb,ng}`, dispatch `MTLSizeMake(O,S,1)`. **Source of the 7.26x@S=8 result.** |
| **Full batch routing** | `route_cache->idx[items*topk]`, `->w[]`, `->unique_experts[]`, `->n_unique` (built `:4514-4546`, alloc `:3791-3794`) | Exists; used **only for loading**, never compute. Gated on `COLI_V4_PREFILL_PREFETCH` (default OFF). |
| **Batched FP4 matmul reference** | `coli_fp4_matmul_batch_ref` `:12389` | Exists, unused by MoE. Benchmarked by `c/bench_moe_batch.c`. |
| **Zero-copy weight path** | `newBufferWithBytesNoCopy`; slab 16 KB-aligned (`posix_memalign :7288-7300`), tensor offset 256 B-aligned, registered `:7317-7321` | Working, **0 fallbacks** reported. |
| **Build seam** | `COLI_V4_METAL_SEAM`, `Makefile.deepseek-v4:92-118` | METAL=0 → 26 objects, METAL=1 → 27. |
| **Sibling implementations** | `inkling.c` (Apple GPU batched expert MoE), `backend_vulkan.c` | Proven patterns to transpose. |

**What is S=1: the seam signature and the host glue.** But the rewrite surface is **larger than
that** (draft 1 understated it): the batch block allocates only *scalar* `ffn_normalized` /
`ffn_branch` scratch (`:5005`), and HC-post / FFN-norm / MoE / FFN-post are **interleaved per token**
inside `:5110`. A batched MoE requires **splitting that loop into batch phases** and allocating
batch-sized FFN input/output buffers. T8 owns that restructure.

---

## 4. Bit-exactness contract (hard constraint)

Shipped unit is `COLI_V4_UNIT_EXPERT_ROWS16` (`Makefile.deepseek-v4.units:23-24`); the live reference
is `coli_v4_expert_forward_ref` `:7661` using `coli_fp4_dual_matvec_rows16_v10` +
`coli_fp4_matvec_rows16_v10`.

### 4.1 Structural proof that batching is safe

`quant.h:1427`:
```c
matmul_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I, int O)
  for (o += 4) { for (s < S) { for (g < ng) { for (k < 16) ... } a += g_partial * scale }
                 y[s*O + o] = a }
```
**Each `(s,o)` output accumulates independently of every other `s`.** Therefore batching over S
**cannot introduce cross-token mixing in the reference loop**.

**Scope of this proof (revised after review).** It establishes *token-separability of the reference*.
It does **NOT** prove a Metal kernel is bit-exact: it says nothing about FMA contraction, fast-math,
denormal flushing, threadgroup scheduling, accumulator register width, scale decode, or bf16 round
placement. **0-ULP probes remain mandatory and are the actual gate** (T1/T4/T5/T6/T7). Do not cite
this proof as a substitute for measurement.

### 4.2 Order that must be preserved per output element

- group size **32**; `ng = (I+31)/32`
- **within** a group: serial `k = 0..15`, **even column then odd column**
  (`g += x[base+2k]*lo; g += x[base+2k+1]*hi`)
- **across** groups: serial `g = 0..ng-1`, `a += g_partial * scale`
- **NO flat reduction, NO simd tree.** (That is the `simd` variant — explicitly not bit-exact.)

### 4.3 Scale scope — the batching trap

fp8 activation QDQ (`:11980-12001`) computes its scale **per token, per 128-column block**; the batch
path (`:12406-12410`) calls QDQ **separately per item**. A batched kernel **must compute scales per
token independently**. Sharing a scale across tokens silently changes results.

### 4.4 UE8M0 — two decoders that disagree

| decoder | v=0 | v=255 |
|---|---|---|
| `quant.h:1424 mx4_scale` — bit trick `((uint32)s << 23)` | `+0.0` | `+inf` |
| `st.h:756 ue8m0_to_f32` — `ldexpf(1, v-127)` | `2^-127` | `NaN` |

The MoE matmul path uses **`quant.h`'s bit trick. Match that one.**

### 4.5 Combine order — non-associative

Routed experts are summed **ascending by expert id** in an fp32 accumulator with **no per-add
rounding**, then `+ shared_output`, then **one** bf16 round (`:4720-4727`, `:4886-4897`).

fp32 addition is **not associative**. A gather-by-expert path must visit experts in **ascending id
order per token**. `unique_experts[]` is built in **encounter order** (`:4545`) and **must be sorted**
before use. This is the highest-likelihood correctness bug in the whole design.

### 4.6 Route-weight position — a second order dependency (MISSED IN DRAFT 1)

The live path multiplies by `route_weight` and bf16-rounds **BEFORE** the down matvec
(`c/deepseek_v4.c:7690-7693`):
```c
for (size_t i = 0; i < intermediate; i++)
    activated[i] = coli_bf16_round(activated[i] * route_weight);   /* <-- BEFORE */
result = coli_fp4_matvec_rows16_v10(output, &expert->down, activated);
```
A batched kernel that applies `route_weight` after the down projection, or that skips this
intermediate bf16 round, is **not bit-exact** even with a perfect reduction order. Also pinned:
the routed sum completes **first**, then `+ shared_output`, then **one** final bf16
(`:4886-4897`) — the shared expert is never folded in early.

### 4.7 Other pinned numerics

- bf16 round `:11949` — RNE with tie, exponent-all-ones skip
- swiglu `:1641` — `gate = fmin(gate, limit)`, `up = fmax(-limit, fmin(up, limit))`

### 4.8 Expert lease capacity — the blocker draft 1 missed

**The lease *lifetime* is safe.** Victim selection excludes any slot with `references != 0`
(`c/deepseek_v4.c:7241`); the miss path sets `references = 1` before releasing the mutex and reading
(`:7324`); the Metal buffer wraps the slab with `deallocator:nil` (`backend_metal_v4.mm:201`) and the
blocking wait precedes the caller's release (`:696`). **No eviction can overwrite a slab whose lease
is held through `waitUntilCompleted`.**

**The lease *capacity* is not safe.** `coli_v4_prefill_store_capacity()` returns `slots_per_layer`
(`:6899`), computed as `cache_bytes / (layers * record_bytes)` (`:6690`). At `--ram 96` that measured
**164 experts per layer**. But the measured `n_unique` at chunk=64 reaches **188** (§2.2, max over 86
records), and layers 0–2 already fall back today at 188/184/188.

**Consequence: "one command buffer per chunk holding all `n_unique` leases" is not universally
achievable.** The design must therefore specify, and T6 must implement:

1. **Preflight**: compare `n_unique` against `coli_v4_prefill_store_capacity()` before acquiring.
2. **Expert waves**: if `n_unique > capacity`, split the chunk's experts into `ceil(n_unique/capacity)`
   waves, each wave one command buffer, each wave's leases released before the next acquires.
   Wave boundaries must not break the ascending-expert-id combine (§4.5) — accumulate into a
   persistent per-token fp32 accumulator across waves, in ascending id order.
3. **Partial-acquisition rollback**: if acquisition fails mid-wave, release everything acquired so
   far and fall back to the S=1 path for that layer-chunk. No leaked leases.
4. **Whole-layer fallback**: preserved and tested, not assumed.

### 4.9 SSD/compute overlap must not be destroyed

Today the prefill loader **overlaps future SSD reads with current expert compute** and waits per
expert (`:4030`, `:4593`). A naive "acquire all `n_unique` experts, then submit one command buffer"
**serialises every miss ahead of all GPU compute**, destroying that overlap. Since the miss path is
SSD-bound (§2.4) this could make the batched path *slower* than today even with a faster kernel.

**Required design (T6):** keep the loader's overlap by pipelining — issue the GPU wave for experts
already resident while the loader continues fetching the next wave. This is a scheduling
requirement, not an optimisation, and it is the single most likely reason the batched path
underperforms.

---

## 5. Design

### 5.1 Regime selection — matvec, not simdgroup GEMM

llama.cpp switches to its mat-mat path only when `ne00 >= 64 && ne21 >= 32` (≥32 tokens per expert),
otherwise matvec with `N_R0_MXFP4 = 2`, `N_SG_MXFP4 = 2`. **At the measured N = 4.10 we are firmly in
the matvec regime.** The design therefore targets a **grouped gather-matvec**, not a
`simdgroup_matrix` GEMM. MLX's `gather_qmm` uses BM=BN=BK=32 tiles with **external** sorting —
consistent with doing the permutation host-side.

### 5.2 Pipeline (canonical, per Megatron/vLLM)

```
route (already in route_cache)
  -> SORT unique_experts ASCENDING              (correctness, §4.5)
  -> count tokens per expert
  -> prefix-sum -> offsets
  -> permute token rows into expert-grouped order (CSR)
  -> for each unique expert, ONE grouped matvec over its N rows
  -> unpermute (inv_order)
  -> combine ascending expert id, fp32, +shared, ONE bf16
```

### 5.3 New seam

```c
int coli_v4_metal_expert_forward_batch(
    float *out,                     /* [S*O] token-major                         */
    const ColiExpertView *experts,  /* [n_unique], SORTED ASCENDING by expert id  */
    int n_unique,
    const int   *tok_expert_idx,    /* CSR: (token,slot) -> index into experts[]  */
    const int   *tok_expert_off,    /* [S+1] offsets                              */
    const float *route_w,           /* [S*topk] per-token per-expert weights      */
    const float *input,             /* [S*I] token-major                          */
    int S, int I, int O,
    float swiglu_limit);
```
The existing S=1 seam is **retained as fallback**. A nonzero return means "CPU path", exactly as
today.

### 5.4 Dispatch model

Today: one command buffer + one `waitUntilCompleted` **per token-expert** (384 per chunk).
Target: **one command buffer per chunk**, `n_unique` (~94) kernel dispatches inside it, **one**
`waitUntilCompleted`. The lease invariant is preserved because the single blocking wait still
precedes every `coli_expert_release` for that chunk.

### 5.4b Route-cache un-gating is NOT a one-line change

`COLI_V4_PREFILL_PREFETCH` currently gates **four distinct behaviours**:

| behaviour | site |
|---|---|
| route-cache enablement + verification | `:3761` |
| full-batch route construction | `:5072` |
| worker-pool init + **SSD loader startup** | `:3990` |
| **cached-route substitution that suppresses normal per-token routing** | `:4686` |

Naively enabling it changes CPU-fallback behaviour and starts SSD prefetch workers. It is also
**not free**: that feature failed its own 15 % gate at **2.70 %** on p256, and route-ahead alone
consumes **8.45 %** of the prefill wall (E54). Treat it as a **cost the batched path inherits**, not
a free asset.

**Required design (T8):** separate *route-metadata generation* from *loader enablement*, from
*verification*, and from *cached-route CPU substitution*. The batched path needs only the first.

### 5.5 Applicability

| path | batchable | note |
|---|---|---|
| prefill | **YES** — up to 64 today | primary target |
| speculation verify | **YES** — `target_batch :9742`, batch ≤ 25 | secondary |
| decode | **NO** — S=1 (`:9877`) | unchanged; documented as control |

---

## 6. Wave plan

### Wave 0 — **T-1: CPU-only grouped scheduler (THE KILL TEST). Do this before anything else.**

Both reviewers independently proposed the same cheapest-earliest detector, and it reorders the plan.

**Do not write a single line of Metal until this passes.** Implement the gather/permute/grouped/
unpermute pipeline **entirely on CPU**, reusing the *existing* exact batched references
(`coli_fp4_matmul_batch_ref :12389`, `coli_fp8_matmul_batch_ref :12317`). No GPU, no new kernel, no
seam change. Then measure warm prefill.

**Why this is the right first move:** it exercises the *entire* risky surface — gather, ascending-id
combine, per-token scales, lease capacity/waves, loader-overlap scheduling, the batch-block phase
split — while the numerics stay on the already-bit-exact CPU path. If grouping does not pay on CPU,
it will not pay on GPU either, and the Metal rewrite is dead for ~1 day of work instead of ~2 weeks.

**T-1 acceptance:** golden md5-stable AND token-exact; warm prefill delta recorded interleaved.
**T-1 KILL (K0): if CPU grouping yields < 1.05x warm prefill, STOP. Do not proceed to Metal.**

```
Wave 0:             T-1 CPU grouped scheduler  [KILL TEST]
Wave 1 (parallel):  T0 baseline+harness | T1 failing oracle (RED) | T2 seam+stub
Wave 2 (parallel):  T3 gather/permute   | T4 batched kernel       | T5 per-token QDQ scales
Wave 3:             T6 host glue (one command buffer)
Wave 4:             T7 ascending-id combine
Wave 5:             T8 wire at :5110-5143   -> T12 regression gate
Wave 6 (parallel):  T9 chunk-cap lever   | T10 decode/spec applicability
Wave 7:             T11 full experiment sweep + record

Critical path: T1 -> T4 -> T6 -> T7 -> T8 -> T9 -> T11
Serialisation: T4/T5 both touch the .metal source -> sequential within Wave 2 if same file.
```

| id | task | depends | acceptance (binary) |
|---|---|---|---|
| **T0** | Baseline + host-discipline harness | — | METAL=1 → 27 objects; baseline JSON `{binary_md5, prefill_us_p064, decode_tok_s, rss_gb}`; golden md5 `5d04890413ff539e802985ce8c727814`; harness **exits nonzero on md5 mismatch** |
| **T1** | Failing batched-MoE oracle (TDD RED) | — | probe compiles, **FAILS** (no batched path); self-check: S=1 oracle == `coli_v4_expert_forward_ref` **0 ULP** |
| **T2** | Batched seam signature + CPU-fallback stub | — | METAL=1 → 27 objects; golden md5 stable; stub returns nonzero → CPU fallback |
| **T3** | Gather/permute (sort, counts, offsets, CSR) | T1 | sorted uniques **strictly ascending**; `permute∘unpermute == identity`; per-token slots == topk |
| **T4** | Batched FP4 grouped-matvec kernel | T1,T2 | == oracle **0 ULP at S=1 AND S=8**; threadgroup mem ≤ 32768 B; `bench_matmul.m` reproduces **≥7x@S=8** — **turns T1 GREEN** |
| **T5** | Per-token per-128col fp8 QDQ + UE8M0 bit-trick | T1,T2 | 2-token differing-dynamic-range probe → independent scales, 0 ULP; UE8M0 v=0/255 match `quant.h` |
| **T6** | Host glue: one command buffer + zero-copy slab | T2,T3,T4,T5 | seam returns 0; 0 ULP @S=8; **dispatch count == n_unique** (~94, not 384); **exactly ONE** `waitUntilCompleted` per chunk |
| **T7** | Ascending-expert-id fp32 combine | T3,T6 | token routed {5,2,9,1} visits **1,2,5,9**; bit-identical to S=1 `moe_token_pipeline` (0 ULP) |
| **T8** | Wire at `:5110-5143` (incl. batch-block phase split + route-metadata separation) | T6,T7 | `bench/golden.sh` prints `PASS golden md5=5d04890413ff539e802985ce8c727814`; `python3 c/tests/test_deepseek_v4_tiny.py` exit 0; `N=3 ./bench/ab.sh` interleaved p064 delta **≥1.12x** (revised — see §9 K3); engine `phys_footprint`+non-engine ≤ ~100 GB |
| **T9** | Chunk-cap lever 64→128→256 | T8 | each value builds + token-exact; record `{chunk, measured_N, tok_s, rss_gb}`; winner by tok/s subject to RSS ≤ ~100 GB |
| **T10** | Decode/spec applicability | T8 | decode documented unchanged (control); spec verify token-exact; delta **recorded honestly incl. negative** |
| **T11** | Full experiment sweep + record | T8,T9,T10 | every row filled and reproducible from the recorded command; negatives published |
| **T12** | Bit-exact regression gate | T8 | all probes 0 ULP + golden md5-stable + tiny oracle; script exits 0 **only** if all green |

---

## 7. Experiments to run and record

Every experiment: **interleaved A/B**, assert **binary md5**, **freeze `.coli_usage`**, record
hardware + commit + exact command + raw log, and report `phys_footprint`/RSS/gap/compressor.

| id | variable | fixed | metric | pass criterion |
|---|---|---|---|---|
| **E1** | S ∈ {1,4,8,16} | real gate/up shape | GPU us/token, GFLOP/s | GPU scales ≥2.4x @S=16 (matches measured 2.46x) |
| **E2** | batched vs S=1 seam | p064 prefill | prefill ttft, interleaved n≥3 | batched **≥1.12x** (ceiling is 1.20–1.25x; 1.3x was withdrawn as above-ceiling) |
| **E3** | chunk ∈ {64,128,256} | batched path | measured N, tok/s, RSS | N rises with chunk; RSS ≤100 GB; token-exact at every value |
| **E4** | dispatch count | chunk=64 | dispatches/chunk | **== n_unique (~94, not 384)** |
| **E5** | weight traffic | chunk=64 | bytes read/chunk | ≈**4.1x** less than S=1 |
| **E6** | realised bandwidth | grouped kernel | GB/s | ≥300 GB/s (toward the measured 370.8 ceiling) |
| **E7** | spec verify batched | `target_batch` | spec tok/s, acceptance | token-exact; delta recorded (may be negative) |
| **E8** | decode | S=1 path | tok/s | **unchanged** (control) |
| **E9** | cold vs warm | `coldwarm2.sh` | tok/s both states | batched wins **warm**; cold is SSD-bound and expected flat (§2.4) |
| **E10** | hit vs miss split | instrumented | tok/s on hits only | isolates the §2.4 ceiling; quantifies the real headroom |
| **E11** | kernel headroom | grouped kernel | GFLOP/s vs fp32 peak | current probes are 3 % of peak — measure how far an optimised kernel goes |
| **E12** | SSD wait time | batched vs S=1 | ms of loader wait per chunk | **must not grow** vs baseline (guards §4.9) |
| **E13** | CPU-only grouping (T-1) | no GPU | warm prefill delta | the K0 kill test — ≥1.05x to proceed |

---

## 7b. Per-task QA — exact commands and expected output

Every task's QA is a copy-pasteable command with a binary observable. `$MODEL=models/deepseek-v4-flash`,
`$P64="$(cat .backlog/prefill_prompts/p064.txt)"`.

| task | command | PASS is exactly |
|---|---|---|
| **T-1** | `./bench/golden.sh ./c/deepseek_v4` then `N=3 ./bench/ab.sh "COLI_V4_MOE_GROUPED=1" ./c/deepseek_v4` | `PASS golden md5=5d04890413ff539e802985ce8c727814` AND `AB p064 ... delta=` ≥ 5 % improvement |
| **T0** | `cd c && make -f Makefile.deepseek-v4 METAL=1 2>&1 \| tail -3; ls build/metal-v4/*.air \| wc -l` | link succeeds; `nm c/deepseek_v4 \| grep -c metal` > 0; `md5 -q c/deepseek_v4` recorded to `artifacts/metal_baseline.json` |
| **T1** | `clang -fobjc-arc -O2 -framework Metal -framework Foundation validation/metal/probe_batched_moe.m -o /tmp/pb && /tmp/pb` | exits **nonzero**, prints `RED: no batched path`; and `S=1 self-check 0 ULP` |
| **T2** | `cd c && make -f Makefile.deepseek-v4 METAL=1 && ls *.o \| wc -l` and `make -f Makefile.deepseek-v4 deepseek-v4-clean && make -f Makefile.deepseek-v4 && ls *.o \| wc -l` | `27` then `26`; golden md5 unchanged |
| **T3** | `clang -O2 validation/metal/test_permute.c -o /tmp/tp && /tmp/tp` | prints `sorted ascending OK`, `permute∘unpermute == identity OK`, exit 0 |
| **T4** | `/tmp/pb` (from T1) and `./validation/metal/bench_matmul 20` | `/tmp/pb` exits **0** with `0 ULP at S=1` and `0 ULP at S=8`; bench prints gate/up S=8 speedup ≥ 7x |
| **T5** | `clang -fobjc-arc -O2 -framework Metal -framework Foundation validation/metal/probe_fp8_twotoken.m -o /tmp/p5 && /tmp/p5` | `token0 scale != token1 scale` where required, `0 ULP`, exit 0 |
| **T6** | `COLI_V4_METAL=1 COLI_V4_METAL_STATS=1 ./c/deepseek_v4 $MODEL "$P64" --max-tokens 1 --memory-gb 96 2>&1 \| grep metal_dispatches` | `metal_dispatches=` ≈ sum of per-chunk `n_unique` (~94×86), **not** 384×86 |
| **T7** | `/tmp/pb --combine-order` | prints visit order `1,2,5,9` for routing `{5,2,9,1}`; near-cancellation case `0 ULP` |
| **T8** | `./bench/golden.sh ./c/deepseek_v4` ; `python3 c/tests/test_deepseek_v4_tiny.py --binary ./c/deepseek_v4 --fixture ./c/deepseek_v4_tiny` ; `N=3 ./bench/ab.sh "COLI_V4_MOE_GROUPED=1" ./c/deepseek_v4` | `PASS golden md5=5d04890413ff539e802985ce8c727814`; tiny oracle exit 0; `delta` ≥ 12 % |
| **T9** | for `C in 64 128 256`: rebuild with cap C, `./bench/golden.sh`, `N=3 ./bench/ab.sh` | golden PASS at **every** C; CSV row `{chunk,N,tok_s,rss_gb}` per C; no `sums[64]` overflow (ASan clean) |
| **T10** | `V4_DRAFT=4 V4_NGRAM=1 ./bench/golden.sh` and spec A/B | token-exact vs S=1 verify; delta recorded incl. if negative |
| **T11** | `./validation/dsv4/metal_sweep.sh` | every E1–E11 row present in `docs/experiments/metal-batched-moe.md`, each with its command |
| **T12** | `./validation/metal/gate.sh` | exits 0 **only** if all probes 0 ULP + golden md5 + tiny oracle pass; exits nonzero otherwise |

**Note on `bench_matmul.m`:** it **exists** at `validation/metal/bench_matmul.m` (7684 B) with a
prebuilt `validation/metal/bench_matmul`. A review claimed it was missing; that claim was checked
and is incorrect.

## 7c. Verification fixtures that the naive tests would MISS

| failure mode | why a naive test misses it | required fixture |
|---|---|---|
| **Combine visits experts in encounter order** | random routing usually gives sums whose fp32 order barely matters; golden may never hit an order-sensitive case | construct expert outputs with **near-cancellation**: two experts whose contributions are large and nearly opposite, so ascending-order and encounter-order fp32 sums differ **before** the final bf16. Assert exact equality with the S=1 path. |
| **fp8 scale shared across tokens** | two tokens with *similar* dynamic range encode the **same** exponent, hiding a shared-scale indexing bug | two tokens whose block maxima **straddle a `ceil_log2(max/448)` boundary** (e.g. just below vs just above a power of two). Assert **both the scale bytes and the QDQ outputs** differ correctly. |
| **Lease exhaustion path** | typical layers have `n_unique` < capacity, so waves never trigger | force a layer with `n_unique > capacity` (or lower `--ram` to shrink `slots_per_layer`) and assert wave split + rollback + no leaked leases |
| **Loader overlap regression** | end-to-end timing can hide it if the kernel got faster | measure SSD wait time separately (E12) and assert it did not grow vs baseline |

## 8. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Combine visits experts in encounter order, not ascending | **High** | **FATAL** (silent wrong output) | T3 sorts; T7 asserts; test with deliberately shuffled routing |
| ULP drift from reduction-order change | Med | **FATAL** | §4.1 structural proof + T1 0-ULP gate before any wiring |
| fp8 scale shared across tokens | Med | **FATAL** | T5 per-token per-128col; 2-token differing-range probe |
| UE8M0 `st.h` vs `quant.h` mismatch (v=0/255) | Med | silent wrong output | T5 boundary-value probe pinned to `quant.h` |
| threadgroup mem > 32768 B at larger chunk | Med | kernel fails to launch | T4 static assert; T9 guards |
| RSS > ~100 GB at chunk=256 | Med | host thrash, invalid runs | T9 residency gate; cap by residency, not speed |
| route_cache un-gating changes prefetch behaviour | Med | regression | T8 enables for the batched path only; CPU fallback preserved |
| N too small to pay (measured 4.10) | **Med** | plan yields <1.3x | K3 kill criterion; T9 chunk lever is the counter-move |
| Miss path dominates so end-to-end gain is small | **High** | headline underwhelms | §2.4 + §0 ceiling stated up front; E10 isolates hits; report honestly |
| **`n_unique` (max 188) exceeds slot capacity (164)** | **High** | one-buffer design fails | §4.8 preflight + expert waves + rollback; T6 acceptance |
| **Acquire-all destroys SSD/compute overlap → net SLOWER** | **High** | plan inverts its own goal | §4.9 pipelined waves; **E12** measures SSD wait directly |
| Route-cache un-gating starts SSD workers / changes CPU path | Med | confounded or regressed baseline | §5.4b: separate metadata generation from loader enablement |
| Raising cap overflows `__m256 sums[64]` (`:12355`) | **High** | **stack corruption** | T9 must fix `:12317`, `:12389`, `:12355`, `:1232` and verify DSpark rings |

---

## 9. KILL criteria

- **K1** — any probe cannot reach **0 ULP** vs the shipped reference after 2 focused attempts →
  **HALT**. Do **not** relax the bit-exact contract.
- **K2** — `golden.sh` output diverges on any token → revert the wire; batched path stays behind the
  fallback.
- **K0** — **T-1 CPU grouped scheduler < 1.05x warm prefill → STOP. Do not start the Metal work.**
- **K3** — E2 batched prefill **< 1.12x** → not worth the complexity; keep opt-in, do not
  default-enable. (Rationale: measured ceiling is 1.20–1.25x, so 1.12x is ~half the available
  headroom. The original 1.15x floor and 1.3x target were set before the Amdahl bound was computed;
  1.3x is **above** the ceiling and was withdrawn.)
- **K4** — RSS breaches ~100 GB and cannot be capped without losing the win → cap chunk, abandon the
  256 lever.
- **K5** — dispatch count does not fall to ~`n_unique` → gather/permute is broken; block T11.

---

## 10. Success criteria

1. All probes **0 ULP** vs the shipped CPU reference.
2. `bench/golden.sh` token-exact and md5-stable; tiny oracle ≥ baseline.
3. **E2 ≥ 1.12x** prefill, interleaved, md5-asserted (against a measured ceiling of 1.20–1.25x).
4. **E4** dispatch count ≈ `n_unique` (~94 vs 384).
5. Chunk winner chosen by tok/s **subject to** RSS ≤ ~100 GB.
6. Experiment log committed; every number reproducible; negatives published.
7. Decode unchanged; speculation finding recorded honestly.

---

## 11. Reproduction of the evidence in this plan

```bash
# premise: CPU vs GPU bandwidth on one unified buffer
clang -fobjc-arc -O2 -framework Metal -framework Foundation /tmp/bwprobe.m -o /tmp/bwprobe && /tmp/bwprobe 4096

# S-sweep on the real gate/up shape
clang -fobjc-arc -O2 -framework Metal -framework Foundation /tmp/sweep_mxfp4.m -o /tmp/sweep_mxfp4 && /tmp/sweep_mxfp4
clang -fobjc-arc -O2 -framework Metal -framework Foundation /tmp/sweep2.m      -o /tmp/sweep2      && /tmp/sweep2
clang -O3 /tmp/cpu_mxfp4.c -o /tmp/cpu_mxfp4 && /tmp/cpu_mxfp4

# measured N (engine's own log)
env COLI_V4_PREFILL_PREFETCH=1 COLI_V4_SAVE_USAGE=0 ./c/deepseek_v4 models/deepseek-v4-flash \
    "$(cat .backlog/prefill_prompts/p064.txt)" --max-tokens 1 --memory-gb 96 2>&1 \
  | grep 'routeahead active'
```
