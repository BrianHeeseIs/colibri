# DeepSeek-V4 Metal Backend — Experiments Backlog

Companion to `experiments_results.md` (what has been run) — this file is **what remains to be
run**. Branch `ft-deepmetal`. Source: plan agent output 13:48, grounded in three parallel
exploration passes (V4 GPU seam + slot lifecycle, GLM Metal host patterns, Apple/llama.cpp/MLX
batch-1 matvec technique).

**North star (user, verbatim):** *"the goal now is not benchmarking all configurations, it's
testing the metal setup and measuring if it improves speed. After the work is done, tested and
verified we can run the other tests against less ram to document performance."*

So the deliverable is **Metal actually running inside the V4 engine with a measured end-to-end
tok/s delta** — not more kernel microbenchmarks. Everything below is ordered to reach that as
early as possible (**T5 is the money milestone**), then improve it.

---

## Critical path

```
T1 ─┐
    ├─► T3 ──► T4 ──► T5 ◀── MONEY: Metal proven running + first measured delta
T2 ─┘                 │
                      ├──► T10 (conditional, only if weight upload dominates)
                      │
                      ├──► T6  ──────────────► T7 ──► T8 ──┐   approximate track (fast)
                      │                                     ├──► T9 (rule F verdict + RESULTS)
                      └──► T11 ──► T12 ────────────────────┘   EXACT track (raise the ceiling)
```

Wave 1 (T1, T2) runs in parallel — disjoint files (build/C vs `.metal`/probe).
**Serialization constraints:** T7/T8 both edit `c/backend_metal_v4.mm`. T6/T11/T12 all edit
`c/metal/coli_v4_moe.metal`. Within each set, run sequentially; the two sets are independent of
each other.

**The two tracks are a deliberate race.** T6 makes the approximate path fast; T11/T12 make the
exact path fast. T9 picks the winner on measured evidence. If the exact track closes the batch-1
gap, the accuracy tradeoff in rule F evaporates and `ordered` ships as default.

---

## Binding design decisions (resolved, do not relitigate)

| id | decision | rationale |
|---|---|---|
| **A** | Land **ordered_cold first**, prove the seam end-to-end, *then* multiply variants | The seam/sync/buffer-management is the risk, not the kernel math. A bit-exact variant is the ideal oracle for de-risking it. |
| **B** | v1 = **one command buffer per (token, expert)**, blocking wait | Minimal restructuring at a serial seam. Token-level batching is a *perf* optimization (T8), deferred until a delta exists to improve on. |
| **C** | v1 = **copy-upload weights into grow-only scratch** | Base `slot->slab` is plain `malloc` (`:5238-5245`) — not 16 KiB aligned, so GLM's `wrap()` would copy anyway. Zero-copy is T10, gated on profiling. |
| **D** | New **`COLI_V4_METAL_SEAM`** macro (C-safe), *not* `COLI_METAL` | The 26 C amalgamation units must never see Metal headers. Blocking `waitUntilCompleted` lives **inside** `coli_v4_metal_expert_forward`, so it returns only after GPU completion — `coli_expert_release` cannot race a refill. |
| **E** | A/B must **interleave ON/OFF/ON/OFF** on one warm binary, report **paired** deltas | `RESULTS.md:723-726` already warns absolute tok/s drifts with thermal + memory pressure. Today's PREWARM re-run drifted 25% from the overnight arm on identical config — proof the warning is real. |
| **F** | Default = **`simd_cold` iff end-to-end token-exact on all 4 prompts**, else `ordered_cold` | Decide by measurement, not preference. `ordered_cold` stays as the permanent correctness oracle and fallback regardless. |

**Why F is a real question and not a formality.** Per-expert output *is* bf16-rounded inside the
reference chain, and bf16 does **not** absorb simd's error (E12: ~2/4096 finals off by 1 bf16
ULP). But the MoE accumulator sums 6 experts + shared in fp32 and rounds **once** at the end
(`:3775-3777`). Whether a 1-ULP per-expert difference survives that and flips a *token* is an
empirical question — greedy decoding only diverges if it crosses an argmax boundary.

---

## Wave 1 — start immediately (parallel, no dependencies)

### T1 · Build ABI + seam wiring (inert fallback)
Introduce the seam so the engine can call Metal **without** exposing Metal headers to the 26 C units.
- `c/Makefile.deepseek-v4`: add `CFLAGS += -DCOLI_V4_METAL_SEAM` under `ifeq ($(METAL),1)` only
- **new** `c/backend_metal_v4_seam.h` (plain C, includes only `expert_store.h`):
  `coli_v4_metal_enabled()`, `coli_v4_metal_variant()`,
  `coli_v4_metal_expert_forward(float *out, const ColiExpertView*, const float *input, float weight, float swiglu_limit)`
- `c/backend_metal_v4.mm`: read `COLI_V4_METAL` / `COLI_V4_METAL_VARIANT` once at init;
  `coli_v4_metal_expert_forward` is a **stub returning -1** (forces CPU fallback)
- `c/deepseek_v4.c`: wrap **all three** seams — `:3763-3766`, `:3730-3733`, `:3167-3170` — as
  `#ifdef COLI_V4_METAL_SEAM if (enabled && forward(...)==0) {} else #endif result = coli_v4_expert_forward_ref(...);`
  Release stays *after*.

**Verify**
```
cd c && make -f Makefile.deepseek-v4 METAL=1 2>&1 | tail -5            # links, 27 objects
cd c && make -f Makefile.deepseek-v4 deepseek-v4-clean && make -f Makefile.deepseek-v4   # METAL=0, 26 objects
python3 tests/test_deepseek_v4_tiny.py && python3 tests/test_deepseek_v4_prefix.py
COLI_V4_METAL=1 python3 tests/test_deepseek_v4_tiny.py                 # stub -> CPU fallback, still exit 0
```
**Pass:** both builds succeed (27 / 26 objects); default object manifest byte-identical; S4 token-exact with the env ON *and* OFF.
**Category:** `unspecified-high` · **Skills:** `git-master` · **Blocks:** T3, T4

### T2 · Fused `ordered_cold` MoE kernel + bit-exact probe
Author `coli_v4_moe_expert_fp4_ordered_cold` in **new** `c/metal/coli_v4_moe.metal`, fusing the
7-step chain: `qdq → gate GEMV → up GEMV → bf16 → swiglu(limit) → ×weight → bf16 → qdq → down GEMV → bf16`.
Reuse the proven **two-level** reduction (serial within a 32-col group, then `a += ga*sc` serially
across groups). *A flat reduction over I is not equivalent.*
**new** `validation/metal/probe_fused_moe.m` — RED first (references the missing kernel), GREEN after.

**Verify**
```
cd c && make -f Makefile.deepseek-v4 METAL=1 deepseek-v4-metal-lib
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  validation/metal/probe_fused_moe.m -o validation/metal/probe_fused_moe
./validation/metal/probe_fused_moe
```
**Pass:** `BIT-EXACT` (diff 0/2048) at **I=4096, O=2048, ng=128** across all seeds, exit 0; RED transcript captured before the kernel landed.
**Category:** `ultrabrain` · **Skills:** — · **Blocks:** T3

---

## Wave 2 — after T1 + T2

### T3 · Host glue — real `ordered_cold` GPU forward
Implement `coli_v4_metal_expert_forward` for `ordered_cold`: grow-only shared scratch `ensure()`;
copy-upload gate/down/up weights + scales; **one** command buffer (qdq → gate/up → swiglu → qdq →
down) with barriers; `waitUntilCompleted`; read back into `out`.
Mapping is **gate=w1, down=w2, up=w3** (`fill_tensor_view` `:5268-5271`) — easy to transpose by accident.

**Verify**
```
cd c && make -f Makefile.deepseek-v4 METAL=1 2>&1 | tail -3
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  validation/metal/probe_glue_expert.m -o validation/metal/probe_glue_expert
./validation/metal/probe_glue_expert
```
**Pass:** bit-exact (0/4096) vs `coli_v4_expert_forward_ref` at real dims; ASan/MallocScribble clean; returns −1 cleanly when the device is unavailable.
**Category:** `deep` · **Skills:** `debugging` · **Blocks:** T4

---

## Wave 3 — after T3

### T4 · Enable synchronous v1 seam + S4 under Metal
Route the real GPU forward at all three seams for `ordered_cold`. Add a **dispatch counter**
(`coli_v4_metal_dispatches()`) — without it, a silent CPU fallback would masquerade as a
successful Metal run and quietly invalidate every subsequent measurement. Confirm the blocking
wait precedes **every** `coli_expert_release`.

**Verify**
```
COLI_V4_METAL=1 COLI_V4_METAL_VARIANT=ordered_cold python3 tests/test_deepseek_v4_tiny.py
COLI_V4_METAL=1 COLI_V4_METAL_VARIANT=ordered_cold python3 tests/test_deepseek_v4_prefix.py
COLI_V4_METAL=0 python3 tests/test_deepseek_v4_tiny.py
```
**Pass:** S4 token-exact ON **and** OFF; dispatch counter > 0 under ON; METAL=0 manifest unchanged.
**⚠ Do NOT run `make deepseek-v4-tiny-check`** — it runs `deepseek-v4-clean` first and rebuilds with `ARCH=` empty (baseline, not `-mcpu=native`), destroying the benchmarked binary mid-campaign (see `experiments_results.md` E21).
**Category:** `ultrabrain` · **Skills:** `debugging` · **Blocks:** T5

---

## Wave 4 — MONEY MILESTONE

### T5 · End-to-end interleaved A/B, `ordered_cold` — first measured delta
4 prompts × 128 tok (matching the PREWARM harness so results are comparable to existing
`RESULTS.md` data), warm cache, `--ram 96`, on `models/deepseek-v4-flash`.
**Interleave ON/OFF/ON/OFF per prompt** to cancel thermal/memory drift. Capture tmux transcript +
`.backlog/metal_ab_ordered_cold.csv` (paired per-prompt tok/s, delta, dispatch count).
Also profile **GPU time vs upload time** — this is the go/no-go input for T10.

**Pass:** CSV produced; dispatches > 0 on every ON run; paired delta reported per prompt; output tokens identical to OFF.
Expected sign: modest — `ordered` is at parity on gate/up, 2.2x on down.
**Category:** `deep` · **Skills:** — · **Blocks:** T6, T10

---

## Wave 5 — the real perf play

### T6 · `simd_cold` variant + accuracy characterization + A/B uplift
Add `coli_v4_moe_expert_fp4_simd_cold` (`simd_sum` across K — order-breaking, and the accepted
industry tradeoff: llama.cpp gates at NMSE ≤1e-7, MLX at 1e-3). Characterize at real dims
(expect raw maxRel ~4.03e-04) and **count bf16-ULP diffs in the per-expert output**. Re-run the
T5 A/B. Record **per-prompt end-to-end token-exactness** — this is the D1/F evidence.

**Pass:** accuracy logged (maxRel + ULP count); `.backlog/metal_ab_simd_cold.csv` shows gate/up uplift; token-exactness verdict recorded per prompt (not a hard gate — it *is* the evidence).
**Category:** `ultrabrain` · **Skills:** `debugging` · **Blocks:** T7, T9

---

## Wave 5b — raise the BIT-EXACT ceiling (parallel track to T6)

**Strategic point.** T6 makes the *approximate* path fast. T11/T12 make the *exact* path fast.
They are competing answers to the same question, and running both is what lets T9 (rule F) pick
on evidence. If T11+T12 close the batch-1 gap, **rule F resolves to `ordered` and the accuracy
tradeoff disappears entirely** — that is a strictly better product than shipping a variant that
is 1 bf16 ULP off on ~0.05% of outputs.

Both levers are **order-preserving**: they change data movement and thread→row mapping, never the
sequence of floating-point additions. Bit-exactness must therefore survive them — and that is the
gating assertion, not a nice-to-have.

⚠ **Serialize against T6** — T6, T11, T12 all edit `c/metal/coli_v4_moe.metal`.

### T11 · Multiple output rows per thread / simdgroup
**The single most promising unused lever.** Currently one thread computes one output row, so the
staged activation vector is re-read for every row. llama.cpp's Metal matvec assigns **4 output
rows per simdgroup** (2 simdgroups/threadgroup); MLX uses `results_per_simdgroup = 4`. Both
amortize the activation read across N rows, raising arithmetic intensity.

Applied here: each thread holds N accumulators (`a[0..N-1]`) and walks the shared activation
once, reading N rows' worth of weights per step. Each row's own two-level accumulation
(serial within a 32-col group, then `a += ga*sc` across groups) is **untouched** — rows are
independent, so interleaving them cannot change any single row's summation order.

Expected effect: cuts activation traffic ~N× on top of the threadgroup staging already won in
E14/E15. This directly attacks the batch-1 gate/up shape where `ordered` currently only reaches
parity with CPU.
Try N = 2, 4, 8 and report the curve — 4 is the upstream consensus but this kernel dequantizes
inline, so the register-pressure optimum may differ.

**Verify**
```
./validation/metal/probe_fused_moe            # MUST stay BIT-EXACT at I=4096,O=2048,ng=128
./validation/metal/bench_matmul 20            # mean±sd, all three real shapes
```
**Pass:** bit-exact preserved at real reduction length (non-negotiable — if it breaks, the
thread→row mapping is wrong, not the math); measured speedup reported per N with mean±sd;
best N recorded. A speedup that costs bit-exactness is a **failure**, not a tradeoff.
**Category:** `ultrabrain` · **Skills:** `debugging` · **Depends:** T2 · **Serializes with:** T6, T12

### T12 · Vectorized loads (`uchar4` / packed nibble)
Weights are currently read one `uchar` at a time (one byte = two FP4 values). Both llama.cpp and
MLX use wide loads. Switching to `uchar4` (or a `packed_uchar4` device read) fetches 8 FP4 values
per load — fewer, wider memory transactions on a kernel already shown to be **memory-bound**
(E14: the GPU was losing to CPU on effective bandwidth, not compute).

Order-preserving **provided the unpack order is exactly preserved**: within each byte, low nibble
= even column, high nibble = odd column; across the 4 bytes, ascending column order. This is
precisely the detail that silently corrupts results if reversed — and the existing probe catches
it, which is why the bit-exact assertion is the gate.

Also consider `float4` loads for the staged activation (threadgroup memory is already 16 B aligned).

**Verify**
```
./validation/metal/probe_fused_moe            # MUST stay BIT-EXACT
./validation/metal/probe_matmul               # tail + ragged cases still exact (I=48, O=17)
./validation/metal/bench_matmul 20
```
**Pass:** bit-exact preserved **including the tail/ragged shapes** (a `uchar4` load is the classic
way to over-read past a non-multiple-of-4 row end — `rb` for I=4096 is 2048, divisible by 4, but
I=48 → rb=24 and O=17 exercise the edges); speedup reported mean±sd.
**Category:** `ultrabrain` · **Skills:** `debugging` · **Depends:** T2 · **Serializes with:** T6, T11

### Rejected — break accumulation order by construction
`simd_sum` reduction across K · split-K · tensor-op / `simdgroup_matrix` tiles. These are exactly
how upstream buys its speed, and exactly why upstream does not maintain bit-exactness (llama.cpp
gates at NMSE ≤1e-7, MLX at 1e-3). They belong to the `simd` variant (T6), not the `ordered` one.

### Also not scheduled
A hand-ported **bit-exact `expf`**. Deliberately excluded: E11/E12 showed the FP8 requantize
absorbs the sigmoid transcendental completely, so it would defend against an error that never
reaches the output.

---

## Wave 6 — after T6 (T7 → T8 serialize; both edit `backend_metal_v4.mm`)

### T7 · Hot variants (`ordered_hot`, `simd_hot`)
rows16: `packed[(tile*stride+col)*16+lane]`, tile=row/16, lane=row%16, stride=rb (weights) / ng (scales).
**Output rows must be padded to a multiple of 16**; buffer is `((O+15)/16)*16*stride`. Getting this
wrong is a heap OOB — already cost one `Trace/BPT trap 5`. Wire the hot store's `aligned_slab`
(`:5902-5911`) as a candidate for true `newBufferWithBytesNoCopy` zero-copy.
**Pass:** `ordered_hot` bit-exact at real dims; ASan clean, no OOB; A/B CSV; zero-copy either proven or explicitly falling back to copy **with a logged reason**.
**Category:** `ultrabrain` · **Skills:** `debugging`

### T8 · Token-level batching (one command buffer / 6 experts, qdq-once)
Batch all 6 routed experts per token into one command buffer; qdq the input **once per token**
into device memory and share it across experts. Threadgroup budget forces this: gate+up for
intermediate=2048 is 16 KB and a qdq'd 4096 input is another 16 KB — together they exactly hit the
32 KB cap with zero headroom. Single `waitUntilCompleted` before **all six** releases.
**Pass:** batched output token-identical to the per-expert path for the same variant; further uplift in `.backlog/metal_ab_batched.csv`; race-free (ASan).
**Category:** `ultrabrain` · **Skills:** `debugging`

---

## Wave 7 — final

### T9 · Variant sweep + D1 verdict + RESULTS delta
*(depends: T6, T7, T8, T11, T12)*
Interleaved A/B across {ordered_cold, simd_cold, ordered_hot, simd_hot, batched, **ordered+multirow
(T11)**, **ordered+vecload (T12)**} on one warm binary. Apply rule **F** — and note that F's
premise changes if T11/T12 land: if the optimised **exact** path matches or beats `simd_cold`
end-to-end, choose `ordered` and the 1-bf16-ULP question becomes moot rather than merely
acceptable. Write the delta into `.backlog/deepseek-v4-RESULTS.md` — **every parity
claim must state the reduction length it was measured at** (the E12 lesson). Document the shipped
default env.
**Pass:** RESULTS delta with paired per-prompt table; default justified by the F rule + token-exactness evidence.
**Category:** `writing` (+ `deep` for the sweep) · **Skills:** —

---

## Conditional

### T10 · Keyed weight cache + aligned allocator — *only if T5 profiling shows upload dominates*
Cache GPU buffers keyed on `ColiExpertKey` to skip re-upload across tokens/layers within
residency. If cache-miss upload still dominates: switch base slab `malloc` → `posix_memalign`
(`:5238-5245`), or route through the hot store's `aligned_slab`, enabling true zero-copy.
**Pass:** token-exact preserved; measured upload drop; net tok/s up vs T6; METAL=0 manifest unchanged.
**Category:** `deep` · **Skills:** `debugging`

---

## Deferred side threads (user-directed order: after Metal is verified)

| item | state | note |
|---|---|---|
| **PREWARM A/B re-run at lower RAM** | arm 0 complete at ram64 (4/4 gate-clean, mean 0.1595 tok/s); arm 1 partial (1/4, **0.1544 vs 0.1353 = +14.1%** on the cold prompt); a ram96 run was started then stopped | User explicitly deferred this until Metal is done. Both arms **must** be re-run on one machine state — the frozen overnight arm A is not a valid comparator. |
| **PREWARM verdict** → `RESULTS.md` + `cache-design.md` S7 | pending the above | Kill condition still live: if hit rate doesn't improve, the limit is **pin policy, not persistence**. |
| **`libexec/colibri/deepseek_v4` refresh** | pending (D3) | `c/coli` prefers the local `c/` build, so this never affected any measurement — but anything invoking `bin/coli` still gets the stale engine (sha `3e34b4ed` vs `7406b1fd`). |
| **Stale Makefile docs** | open | `c/Makefile:333-336` and the `else`-branch message both still claim V4 needs "x86-64 Linux/Windows or aarch64 Linux", omitting Darwin — despite `:347-349` enabling it. This is what misled me in E20/E21. |

## Teardown checklist (QA hygiene — must be clean before "done")
- [ ] no `coli serve` / `deepseek_v4` processes left running; port 8090 free
- [ ] tmux sessions used for A/B killed (`metalab`, and `colibri-lab` left as found)
- [ ] `models/deepseek-v4-flash/.coli_usage` restored from `/tmp/coli_usage.snapshot` (29127 B)
- [ ] `c/deepseek_v4_tiny/.coli_usage` (test side-effect) removed
- [ ] scratch binaries in `/tmp` and `c/metal/_moe_combined.metal` (probe-only artifact) cleaned

---

---

# ⚠️ SUPERSEDED — read this section first (2026-08-13, post-verdict)

Everything above was written *before* the end-to-end measurement. The measurement changed the
conclusion, so the wave plan above is **historical**. See `experiments_results.md` E24/E24b/E25 and
`.backlog/deepseek-v4-RESULTS.md` §12 for the results.

## Verdict
Metal was built, proven bit-exact at production reduction length, wired into the engine with a
race-free synchronous seam, and measured. **It is slower:**
- decode (8 tok): **1.118x slower** (best of three architectural fixes: 2.30x → 1.17x → 1.118x)
- prefill (799 tok): **1.063x slower**
- output byte-identical in every paired run

## Why the plan above cannot fix it
Four causes stack against the GPU, and none is a backend problem:
1. **Every expert call is S=1** — the seam passes one input vector and one output vector. No token
   dimension exists to batch over.
2. **Prefill does not help** — the engine walks prefill token-by-token through the same S=1 seam
   (206,916 expert_requests ≈ 799 × 43 × 6). Arithmetic intensity never rises, so the genuine
   **7.26x batch-8 microbenchmark win is structurally unreachable**.
3. **Unified memory** — no GPU bandwidth advantage; both processors read the same DRAM.
4. **~Zero weight reuse** — ~8 MB of weights per ~0.5 MFLOP, top-6 of 256 experts.

## The real fix (not scoped here)
Batch the token dimension **inside** the expert call: gather all tokens routed to expert E in a
layer, issue one S=N matmul. That is an **engine/scheduler change** in `moe_token_pipeline`
(`deepseek_v4.c:3579`) — exactly the shape the 7.26x microbenchmark measured. No backend tuning
changes the sign until that exists.

## Corrected status of the remaining items

| item | status | honest assessment |
|---|---|---|
| **T8 token-level batching** | **REINSTATED — I dropped this for the wrong reason** | I dropped it saying it "doesn't address the bottleneck" while believing the bottleneck was occupancy. P0a fixed occupancy by other means, so that reasoning no longer applies. T8 amortises submit+wait across 6 experts per token. The remaining gap is only ~0.345 ms/expert (3.264 vs 2.919); if per-command-buffer overhead is ~0.1–0.2 ms, amortising 6x could recover **roughly half** of it. **Plausible but unquantified** — it does not change S=1, so it probably narrows rather than flips. Worth measuring before anything else on this list. |
| P1 rows16 coverage | open | Only **17.8%** of forwards reach the GPU on a long run (rows16-packed pinned experts rejected by cold-layout validation; `packed_slots=491`, `pin_slots_per_layer=16`). Raises GPU *share*, not per-call speed. Amplifies whatever sign the per-call result has — currently that means it would make things *worse*, not better. |
| P2 simd_cold | open | Order-breaking reduction, ~5.4x faster in isolation at batch-1. Would narrow the gap but cannot fix S=1 bandwidth. Also reopens the accuracy question (1 bf16 ULP on ~0.05% of outputs) for a result that still loses. |
| T11 multi-row per thread | open | Order-preserving, upstream consensus is 4 rows/simdgroup. Raises arithmetic intensity **within** a call — the only listed lever that partially attacks the actual problem. Best remaining candidate after T8. |
| T12 uchar4 vectorized loads | open | Order-preserving, attacks a memory-bound kernel. Modest. |
| T9 verdict write-up | **DONE** | `RESULTS.md` §12. |

## What this campaign did establish (durable, reusable)
- Every V4 MoE arithmetic primitive **bit-exact on GPU**, several exhaustively (`mx4_scale`,
  `e4m3fn` 256/256), all at **production reduction length** (gate/up I=4096 ng=128; down I=2048 ng=64).
- Fused single-dispatch expert kernel bit-exact: **12,288 outputs, 3 real seeds, 0 differing bits**.
- Host glue bit-exact **0/4096** vs `coli_v4_expert_forward_ref`, with a working cap-fallback.
- Race-free synchronous seam at all 3 sites; lease invariant verified line-by-line.
- Zero-copy weight path (16 KiB alignment, `newBufferWithBytesNoCopy`, **0 fallbacks**).
- `fp8_qdq` rewritten to a 128-thread `fmax` tree and **still bit-exact** (`fmax` is
  associative/commutative for non-NaN, unlike a sum — proven by probe, not asserted).
- Build integrity: 26 objects METAL=0 / 27 METAL=1, manifests byte-identical, tiny oracle
  token-exact with Metal ON and OFF.

## Three traps worth carrying into any future GPU work here
1. **A dispatch counter is mandatory.** The first working seam passed every test while silently
   falling back to CPU — Metal was never initialized at runtime. Without `metal_dispatches`,
   every subsequent number would have been a lie.
2. **Profile-mode attribution ≠ production attribution.** A per-stage profile reported
   `matmul gate` 1.1597 ms vs `matmul up` 0.0452 ms — 25.7x apart on *identical shapes* — because
   profile mode issues ~10 command buffers instead of 1 and inflates whichever runs first. The
   stages summed to *less* than the CPU total, which is self-refuting.
3. **Fixtures must match production reduction length.** A parity result that was bit-exact at
   reduction length 64 was 4.03e-04 off at 4096.

## Footnote: the ragged-O test gap is not a live gap
The §12d correction notes that no ragged-O case was run at I=4096. For completeness: **neither
real V4 output dimension is ragged** — gate/up O=2048 and down O=4096 are both exact multiples of
16 (2048/16=128, 4096/16=256). The untested "ragged-O at real reduction length" combination
therefore **cannot occur in this model**. The small-dim ragged coverage (O=17) is defensive
hardening for hypothetical shapes. Do not treat the correction as an outstanding action item.
