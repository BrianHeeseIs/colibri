# DeepSeek-V4 Metal Backend — Experiments & Results

Branch `ft-deepmetal`. Host: Apple M3 Max, 40 GPU cores, Metal 4, 128 GiB unified (137.4 GB
decimal), macOS 26.6.1. Model: `models/deepseek-v4-flash` (48 shards, hidden=4096,
moe_intermediate=2048, 256 routed experts, top-6, 43 layers, swiglu_limit=10.0).

**Reading guide.** Every experiment below states its method, its raw numbers, and its verdict.
Where a later experiment overturned an earlier claim, the retraction is kept inline rather than
edited out — the retractions are the most useful part of this document.

---

## Context: why this work exists

An earlier verdict said a Metal backend for V4 should be DEFERRED, on three stated blockers:
V4's MoE math differs from the existing GLM Metal path (B1), CPU determinism makes exactness a
hard requirement (B2), and a tensor-format allowlist appeared to exclude V4's weights (B3).

The user overrode DEFER and asked for the backend to be built and measured. All three blockers
were dissected and found **costly, not fatal** — B3 in particular turned out to be a red herring
(the allowlist is real but irrelevant: V4 uses `COLI_TENSOR_FP4_NATIVE_BLOCK` + `COLI_SCALE_UE8M0`
with `block_columns=32`, which is not the code path the allowlist gates).

One structural fact drove the whole design: **V4's FP4 `block_columns` = 32 = the device's
`threadExecutionWidth`.** One quantization block maps exactly onto one SIMD group.

---

## Part 1 — Feasibility probes

### E1. Metal toolchain availability
**Method.** `xcodebuild -downloadComponent MetalToolchain`, then compile a trivial `.metal` both
offline (`metal` → `.air` → `metallib`) and at runtime (`newLibraryWithSource:`).
**Result.** Toolchain installed (687.9 MB, exit 0). `metal 32023.883`, target
`air64-apple-darwin25.6.0`. **Both** compilation paths work.
**Verdict.** No toolchain blocker. Runtime source compilation is available as a fallback that
needs no install path — this later became load-bearing.

### E2. FP4 (E2M1) + UE8M0 decode on GPU
**Method.** Decode FP4 weights with UE8M0 scales on GPU, compare bit patterns against the CPU
decoder over all 8 test vectors.
**Result.** **Bit-exact, 8/8.**
**Verdict.** The quantization format itself is not a source of divergence.

### E3. Reduction determinism (tree vs serial)
**Method.** Run a `simd_sum` tree reduction 64 times on identical input; separately run a
serial-chain kernel and compare to CPU.
**Result.** `simd_sum` is **bit-reproducible across all 64 runs** (rel err 6.481e-06 vs CPU).
The serial-chain kernel is **bit-identical to CPU** (`0x493c59f0`).
**Verdict.** Both a fast-approximate and an exact kernel are achievable. This is what made a
runtime-selectable `ordered` / `simd` pair possible.

### E4. FMA contraction — the hypothesis that was WRONG
**Method.** The planning pass ranked FMA contraction as the #1 bit-exactness risk (if CPU and
GPU contract `a*b+c` differently, no amount of ordering control helps). Probed across
`MTLMathMode` Safe / Relaxed / Fast, with both benign inputs (1+k·2⁻¹²) and inputs at float
epsilon (1+k·2⁻²³).
**Result.**
- Benign inputs: CPU split == CPU explicit `fma` == GPU on **all three math modes** (`0x4100480d`).
- Epsilon inputs: a 1-ULP gap appeared (`0x41000009` vs `0x4100000a`), but `contract(fast)` ==
  `contract(off)` on CPU and Fast == Safe on GPU — **math mode changed nothing**.
- Also learned: `fastMathEnabled` is deprecated since macOS 15; `MTLMathMode` {Safe=0, Relaxed=1,
  Fast=2} is the current API.

**Verdict — RECORDED AS A MISS.** FMA contraction is **not** the bit-exactness blocker. The plan
ranked it first; probing found it irrelevant. The real blocker (E6) was something nobody had
ranked. *Reasoning identified the wrong suspect; measurement found the right one.*

---

## Part 2 — Primitive ports (the arithmetic floor)

Every primitive was ported RED→GREEN: the probe was written and run **before** the kernel
existed (confirming it fails for the right reason), then the port was written.

### E5. Five core primitives
**Method.** `validation/metal/probe_primitives.m` — compare GPU vs CPU bit patterns.
Exhaustive where cheap (256/256 for anything keyed on a `uint8`).

| primitive | coverage | result |
|---|---|---|
| `mx4_lut` | 16/16 | **bit-exact** — incl. index 8 = **negative zero** (`0x80000000`), distinct from +0 |
| `mx4_scale` | **256/256 exhaustive** | **bit-exact** — incl. s=0 → +0 and s=255 → +inf |
| `e4m3fn_decode` | **256/256 exhaustive** | **bit-exact** — incl. NaN encodings |
| `bf16_round` | 32/32 | **bit-exact** — incl. NaN, ±Inf, denormals, round-to-even ties |
| `sigmoid` | 32 | **FAILS 4/32** |

**Key detail.** `mx4_scale(s)` must be the bit-trick `as_type<float>(uint(s)<<23)`, **not**
`ldexp` — they diverge at s=0 (+0 vs 2⁻¹²⁷). Separately, `e8m0_decode` **is** `ldexp(1, s-127)`.
Two superficially similar scale decoders with different semantics; substituting one for the
other silently corrupts results.

### E6. The real blocker: sigmoid's transcendental
**Method.** Sweep 8192 points over [-40, 40], compare `precise::exp`-based GPU sigmoid vs libm.
**Result.**
- **18.87% of values differ** (1546/8192), max **2 ULP**, max rel err **1.611e-07**
- Two distinct causes: (A) GPU **flushes denormal results to zero** — `exp(-88)` gives CPU
  6.05e-39, GPU +0; (B) `precise::exp` ≠ libm `expf`.

**Why it matters.** `sigmoidf_stable` is on the **SwiGLU hot path** (`deepseek_v4.c:1394`), so it
executes for every expert activation — not a corner case.

**Verdict.** This, not FMA, is the reason end-to-end bit-identity cannot be assumed. It also
forced a naming change: the variants were renamed **`exact`→`ordered`** and **`fast`→`simd`**,
because "exact" would have been a false claim in the source tree.

### E7. FP8 activation QDQ — the hardest primitive
**Method.** Port `coli_fp8_activation_qdq_ref` (steps 1 and 5 of the expert chain). Test vector
deliberately spanned every sub-path: wide values, the subnormal E4M3 encode region (<0.015625),
the near-448 clamp, values near the 1e-4 max floor.
**Result.** **Bit-exact: scales 0 bad, outputs 0/768, maxULP 0.**
Four sub-primitives came along: `ceil_log2_positive` (frexp, incl. the fraction==0.5 boundary),
`e8m0_decode`, `e4m3fn_encode` (**both** arms — subnormal, and normal with round-to-nearest-even
including the `rounded==16` carry into the exponent), `e4m3fn_decode`.
**Verdict.** The most intricate primitive in the chain reproduces exactly, including its rounding.

### E8. Matmul `matmul_mxfp4`
**Reference correction found here.** `c/quant.h` contains **two** implementations. The AVX2 one is
`#ifdef __AVX2__` and **never compiles on Apple silicon** — the arm64 ground truth is the scalar
path at `quant.h:1401-1412`. Its accumulation is **two-level** and the order is load-bearing:
`ga` accumulates serially within a 32-column group, then `a += ga*sc` with the scale applied
**once per group, after it closes**. A flat reduction over I is *not* equivalent.

**Result.**
| variant | shapes | result |
|---|---|---|
| `ordered` | S=2 I=64, S=2 I=48 (**tail**), S=1 I=32 | **BIT-EXACT, maxULP 0** |
| `simd` | same | diverges, maxRel 7.44e-06 / 4.59e-06 |

The tail case (I=48) matters: it exercises the `glen` clamp and the guarded odd-column path,
which is where ports usually break silently. `simd`'s error agreed with the independently
measured `simd_sum` figure (6.481e-06) — two independent measurements agreeing.

### E9. SwiGLU
**Preserved asymmetry** (deliberate in the CPU source, easy to "tidy" away): `gate` is clamped
**only from above** (`fmin`), `up` is clamped **both sides**; product associates left-to-right.
**Result.** Over 4096 gate values in [-100, 100]: 19.65% differ (no clamp) / 19.48% (limit=7),
260 denormal-flush cases, **maxRel (non-flush) 2.64e-07**.
Clamping barely moves the rate — as expected, since divergence clusters at *negative* gate and
gate is not clamped from below.

**On the 260 flush cases.** Their relative error is 100%, which is true and completely
misleading. The absolute error is what matters: gate=-88 → CPU ≈ -5.3e-36 vs GPU 0. Harmless by
~36 orders of magnitude against activations of order 1. They are counted separately, never
merged into the relative-error statistic.

### E10. rows16 "hot" layout
**Method.** `packed[(tile*stride+col)*16+lane]`, tile=row/16, lane=row%16, applied to **both**
weights (stride=rb) and scales (stride=ng).
**Result.** **hot == cold == cpu, bit-identical** on all shapes: O=32, O=48, O=16, and O=17
(**ragged tile**).
**Cost of learning it.** A `Trace/BPT trap 5` (heap OOB). rows16 requires output rows **padded to
a multiple of 16** — the packed buffer is `((O+15)/16)*16 * stride`, not `O*stride`. This is
precisely why the engine's fixtures use `STORAGE_HIDDEN=128` (padded from 64). **Any host glue
allocating hot weight/scale storage must pad O to a multiple of 16 and zero the pad.**

---

## Part 3 — End-to-end expert forward

### E11. Full 7-step chain, tiny dims
**Chain.** `fp8_qdq → matmul(gate,up) → bf16 → swiglu → ×route_weight → bf16 → fp8_qdq →
matmul(down) → bf16`.
**Method.** Independent CPU and GPU chains from identical input, hidden=64/intermediate=128,
12 seeds, both clamp modes, route weights 0.257–1.183.
**Result.** step1 qdq bit-exact; step2 gate bit-exact; step4 swiglu diverges 9/128 (maxRel
1.1e-07); **end-to-end output diff = 0/64 on 12/12 seeds** — for `ordered` **and** for `simd`.

**Mechanism (verified, not assumed).** swiglu's output is immediately re-quantized to FP8 E4M3
before the down matmul. The transcendental error sits far below the E4M3 quantization step, so
both CPU and GPU swiglu outputs map to the **same FP8 codes**. The requantization *absorbs* the
divergence.

### E12. The retraction — absorption at REAL dims
**Method.** Re-measure the absorption claim at production shapes (4096→2048, 2048→4096) instead
of the 64-wide toy fixture, and check what survives each quantization stage.

| shape | simd raw diff | maxRel | after bf16 | after fp8 |
|---|---|---|---|---|
| gate/up 4096→2048 | 1634/2048 | **4.03e-04** | **1/2048 survives** | 0 |
| down 2048→4096 | 3149/4096 | **2.64e-04** | **2/4096 survives** | 0 |

**RETRACTION.** The E11 claim "simd is bit-exact end-to-end, 12/12" **does not generalise and is
withdrawn.** `simd_sum` error grows with **reduction length**: the toy fixture reduced over 64
elements, production reduces over 4096 — 64x longer — and the error grew from 7.4e-06 to
**4.03e-04, ~54x worse**. The toy fixture was *structurally incapable* of exposing this.

**Corrected mechanism.** Absorption is only **half** true. FP8 requantize (~6% granularity)
absorbs it fully. **bf16 (~0.2% granularity) does not.** Step 7 is a bf16 round straight to the
output, so ~2/4096 final values differ by 1 bf16 ULP.
- `simd`: **not** bit-exact end-to-end at production scale.
- `ordered`: unaffected — it reproduces CPU accumulation order regardless of reduction length,
  and stayed bit-exact at all real shapes.

**METHOD LESSON (the most portable result here).** Numerical fixtures **must match production
reduction lengths**. Correctness-at-tiny-dims does not imply correctness-at-real-dims for
anything involving accumulation. Every parity claim in this project must now state the reduction
length it was measured at.

---

## Part 4 — Performance at real dims

CPU baseline is the real engine path: 16-thread OpenMP `matmul_mxfp4`.

### E13. First pass (best-of-5) — later found too noisy
| shape | CPU | ordered | hot | simd |
|---|---|---|---|---|
| gate/up S=1 | 0.418 | 1.618 | 0.720 | **0.075** |
| down S=1 | 0.383 | 0.299 | 0.421 | **0.044** |
| gate/up S=8 | 2.941 | 0.758 | 1.088 | **0.097** |

Read at the time as: bit-exact `ordered` is *slower than CPU* at batch-1 (0.26x), `hot` is not a
consistent win, `simd` is 5–30x.

### E14. Diagnosis of the batch-1 deficit
Not compute-bound — a **memory-access** problem, in two parts:
1. **Coalescing.** In the cold kernel, thread `o` reads `w[o*rb + …]`, so adjacent threads read
   addresses rb=2048 B apart — worst-case uncoalesced. rows16 fixes exactly this, and the data
   agreed (0.720 vs 1.618 = 2.2x).
2. **Redundant activation traffic (the larger one).** Every thread read the **entire** input
   vector from device memory: 2048 threads × 16 KB = **~32 MB of redundant reads against a 4 MB
   weight matrix** — x traffic was 8x the weights.

**Fix.** Stage x in **threadgroup memory** once per threadgroup (16 KB of the 32 KB budget).
This preserves bit-exactness *perfectly* — identical values, identical accumulation order; it is
purely a data-placement change.

**Structural constraint discovered.** Bit-exactness **forbids splitting a row's reduction across
threads** (it changes the association and therefore the rounding), so `ordered`'s thread count is
capped at O. Parallelism cannot be raised further without abandoning exactness — x-staging works
*within* that cap by cutting traffic ~8x instead of adding threads.

### E15. D1 settled (n=20, mean±sd, first iteration dropped as warmup)
| shape | CPU | ord | hot | **ord+xc** | hot+xc | simd |
|---|---|---|---|---|---|---|
| gate/up S=1 | 0.449±0.065 | 0.883±0.432 | 0.638±0.045 | **0.407±0.018** | 0.501±0.032 | **0.075±0.030** |
| down S=1 | 0.437±0.008 | 0.208±0.012 | 0.252±0.010 | **0.199±0.034** | 0.259±0.039 | **0.064±0.038** |
| gate/up S=8 | 3.114±0.213 | 0.449±0.027 | 0.514±0.044 | **0.429±0.041** | 0.487±0.015 | **0.374±0.041** |

`ord == cpu` **bit-exact on all three shapes.**

**RETRACTION.** The E13 reading "bit-exact is 0.77x, slower than CPU at batch-1" is **withdrawn**
— it came from best-of-7 on an unstable variant. With n=20 the bit-exact path is 0.407 vs CPU
0.449.

**x-staging did two things**, and the second is arguably more important:
- speed: batch-1 gate/up 0.883 → 0.407; batch-8 gate/up 0.449 → 0.429
- **stability: batch-1 sd 0.432 → 0.018 (~24x tighter).** Raw `ordered` was too unstable to ship.

**Honest reading (deliberately not overclaimed).**
- gate/up S=1: CPU [0.384, 0.514] vs ord+xc [0.389, 0.425] — **error bars overlap. This is
  parity, not a win.**
- down S=1 (**2.20x**) and gate/up S=8 (**7.26x**) are clear wins; bars do not overlap.
- `simd` remains **5.4x faster than ord+xc at batch-1** (0.075 vs 0.407) — and batch-1 *is*
  decode, where a chat workload lives.

**Where this leaves the choice.** No longer "exact vs regression" but "exact-at-parity vs 5.4x
at 1-bf16-ULP on ~0.05% of decode outputs". For scale: FP4 weight quantization already carries
~6e-2 error — **~230x larger** than simd's 2.6e-04. Bit-exactness buys *reproducibility against
the CPU reference*, not accuracy against ground truth.

---

## Part 5 — Build integrity

### E16. Metal wiring does not perturb the default build
**Method.** Forced dry run (`make -n -B`), METAL off vs on, plus durable object manifests.
**Result.** METAL=0 → 27 commands, **26 objects, 0 Metal artifacts**. METAL=1 → 39 commands,
**27 objects**, 5 Metal artifacts. Delta is exactly **one** object: `backend_metal_v4.o`.
Manifests before == after (SHA `e7b07731…`).

**Why 26→27 is the decisive number.** The engine compiles `deepseek_v4.c` **once per unit** (26
units, each `-DCOLI_V4_UNIT_*`). `backend_metal_v4.mm` must compile **exactly once** and never be
swept into that loop, or you get 26 duplicate ObjC++ objects and duplicate-symbol link failures.
26→27 proves it compiles once.

**RETRACTION recorded.** An earlier check of this claimed "object list byte-identical" by
comparing `make -n` output of a `.orig` Makefile against the live one. Both lists were **empty** —
the `.orig` aborts on Darwin (rc=2) and the live run said "up to date". **Two empty sets were
compared and called a pass.** Withdrawn and redone properly. (Also established: the `.orig` was a
stale snapshot, 44 lines adrift from HEAD — never a valid baseline.)

### E17. Two silent-failure bugs worth recording
1. **`COLI_METAL` guard on `.metal` files.** Guarding shader sources made them preprocess to an
   *empty translation unit* under runtime `newLibraryWithSource:` (where the macro is undefined).
   Compilation **succeeds** on nothing, and the failure only surfaces later as "entry point
   missing". Proof: `.air` was 2384 B without `-DCOLI_METAL`, 4272 B with. The guard belongs on
   host `.mm`/`.h`, which the C build actually sees; a `.metal` file is only ever read by the
   Metal compiler, so the guard defends against a scenario that cannot occur.
2. **Guard scope swallowing a kernel.** `swiglu.metal` had its sigmoid `#ifndef` closed by an
   `#endif` at **EOF**, so the whole file — struct and kernel included — sat inside the guard.
   Standalone it compiled (macro undefined); after `decode.metal` defined the macro in a combined
   source, the entire kernel silently vanished. Diagnosed by preprocessing: `swiglu refs after
   preprocess = 0`.

Both share a shape: **a clean compile that produced nothing.**

### E18. Probe exit semantics
`probe_primitives` originally gated on total bit-equality, so it was **permanently red** (sigmoid
legitimately diverges). A permanently-red test trains people to ignore it and would mask a real
regression. It now asserts: the four bit-exact groups stay bit-exact; sigmoid stays within its
documented bound (≤2 ULP, ≤5e-07 rel); denormal-flush count ≤ 2.
The first attempt at that bound was *also* wrong — it applied ULP distance to the denormal-flush
class and read `maxULP 4320708`. Flush is now counted separately and excluded from ULP/rel stats.

---

## Current status

**Suite: 9/9 green** — `probe_primitives`, `probe_matmul`, `probe_swiglu`, `probe_fp8qdq`,
`probe_hot`, `probe_fp4_runtime`, `probe_reduction_parity`, `parity_v4`, `probe_expert`.

**Proven.** Every V4 MoE arithmetic primitive on GPU; `ordered` matmul bit-exact at real dims;
cold and rows16-hot bit-identical; build wiring inert when Metal is off.

**Not yet done.** The four **fused** single-dispatch MoE kernels the benchmark rows expect
(`coli_v4_moe_expert_fp4_{ordered|simd}_{cold|hot}`) — the constituent pieces are proven, the
fused entries are not written. Host glue is a stub. The live GPU seam is unwired pending the
`slot->slab` decision (now settled: **synchronous v1**). **No end-to-end model measurement has
been taken** — every number above is kernel-level.

## Side thread: PREWARM A/B (deprioritized by user, kept for the record)
Re-run of the `COLI_V4_PREWARM` 0-vs-1 A/B at `--ram 64`. Arm 0 completed 4/4 gate-clean
(mean **0.1595 tok/s**); arm 1 reached 1/4 (**0.1544** vs arm 0's 0.1353 on the same cold prompt,
**+14.1%** — the first evidence the prewarm branch does anything, since it was unreachable before
the per-turn flush commit). A `--ram 96` re-run was started and then stopped when the user
redirected to Metal.
Validity note carried forward: this re-run's arm 0 was ~25% slower than the frozen overnight arm A
(946s vs 712s on prompt 1) purely from machine-state differences — which is exactly why both arms
must be run on one machine state, and why the frozen arm A must not be compared against a
fresh arm B.

---

## E19. Ship-candidate bit-exactness (gap found in my own work, then closed)

**The gap.** E15 benchmarked `ord+xc` for *speed* and reported `ord==cpu bit-exact` on the same
line — but that mismatch check only ever ran against the plain `ordered` kernel's output buffer.
**The xcache variants had never been checked for correctness**, and they are the ones that would
ship. Caught during a routine regression pass, not by a test.

**Method.** Extend `probe_matmul` to assert the xcache variants bit-exact, and — per the E12
method lesson — include a case at the **production reduction length** (I=4096, ng=128 groups),
not just toy fixtures. `hot_xcache` needed a dedicated runner (rows16 packing + O padded to a
multiple of 16).

**Result.**

| variant | shape | result |
|---|---|---|
| `ordered_xcache` | S=2 I=64 O=8 | BIT-EXACT 0/16 |
| `ordered_xcache` | S=2 I=48 O=8 (tail) | BIT-EXACT 0/16 |
| `ordered_xcache` | **S=1 I=4096 O=64 (REAL, ng=128)** | **BIT-EXACT 0/64** |
| `ordered_hot_xcache` | S=2 I=64 O=32 | BIT-EXACT 0/64 |
| `ordered_hot_xcache` | **S=1 I=4096 O=64 (REAL)** | **BIT-EXACT 0/64** |
| `ordered` (cold) | **S=1 I=4096 O=64 (REAL)** | **BIT-EXACT 0/64** — regression intact |
| `simd` | S=2 I=64 / I=48 | MISMATCH, maxRel 7.44e-06 / 4.59e-06 (expected) |

**Verdict.** The threadgroup-staging optimisation preserves bit-exactness **at production
reduction length**, confirming it is purely a data-placement change. Both exact ship candidates
(`ordered_xcache`, `ordered_hot_xcache`) are now validated on correctness *and* measured on speed.

**Process note.** A benchmark that prints a correctness verdict for one variant while timing
five is a trap — the verdict reads as if it covers the row it sits on. Correctness assertions
belong in the probe, not the benchmark.

## Suite status after E19
**9/9 green**, and all five matmul kernels are exposed through the **offline metallib** path
(`.air` 9232 B → `.metallib` 25661 B), not just runtime source compilation.

---

## E20. S4 — CPU regression check (both halves PASS)

**Build half.** METAL=0 → 26 objects, 0 Metal artifacts; durable object manifests before ==
after (SHA `e7b07731…`). Metal adds exactly one object (`backend_metal_v4.o`) and only under
METAL=1.

**Functional half.** The underlying test scripts were invoked directly rather than via
`make deepseek-v4-tiny-check` (safe: the binary is newer than its source, so the 26-unit rebuild
the target performs was unnecessary).

> **RETRACTED CLAIM.** This section originally asserted that `make deepseek-v4-tiny-check` hits a
> "Darwin SKIP stub" and validates nothing on macOS. **That was wrong.** See E21 — the gate does
> support Darwin, the real target fires, and the SKIP stub is in a dead `else` branch. The claim
> came from grepping the SKIP line without checking which branch was live.

| test | result |
|---|---|
| `tests/test_deepseek_v4_tiny.py` | **exit 0** — teacher-forcing + greedy **token-exact**; compressed, long, session, CLI (>512-tok prompt) and serve SUBMIT/DATA/DONE protocol all PASS |
| `tests/test_deepseek_v4_prefix.py` | **exit 0** — prefix reuse / repeat / reset all PASS |

**Verdict.** CPU engine semantics are unaffected by the Metal work.

**Fixture note.** `c/deepseek_v4_tiny/{config,ref}.json` show modified vs HEAD, but the diff is
**only version strings** (`transformers` 5.14.1→5.15.0, `torch_version`). The reference
logits/tokens are unchanged — a regeneration under newer transformers reproduced identical
numerics, which is mildly reassuring in its own right.

**Side effect for teardown.** The tiny test writes `c/deepseek_v4_tiny/.coli_usage`
(selections 6228 → 6318).

### Scenario contract status
| id | scenario | status |
|---|---|---|
| S1 | fused MoE kernel matches CPU expert ref at real dims | **pending** (kernels not yet written) |
| S2 | error characterised at real reduction length | **partial** — matmul done (E19), fused pending |
| S3 | `slot->slab` reuse cannot race the GPU | **pending** (synchronous v1 decided, not yet built) |
| S4 | CPU-only build unchanged | **PASS** (this section) |
| S5 | end-to-end Metal ON vs OFF tok/s on the real model | **pending** — the actual user ask |

---

## E21. Is macOS actually excluded from the V4 build? (No — and a real hazard found)

**Question.** The skip message reads "V4 runtime requires x86-64/aarch64 Linux or Windows/MSYS2".
This host is aarch64 — so is the gate on architecture, or on OS?

**Method.** Stop reading the Makefile and ask `make` what it resolves (`make -n`, `make -p`).

**Result — macOS is NOT excluded.**
```
COLI_V4_SUPPORTED := 1      DARWIN := darwin      AARCH64 := arm64
X86_64 :=                   LINUX  :=             IS_WIN :=
```
`make -n deepseek-v4-tiny-check` expands the **real** recipe (generate → clean → build → run both
test scripts), rc=0. The gate at `c/Makefile:343-350` grants support to `AARCH64 && LINUX` **and,
separately, to `AARCH64 && DARWIN`**. The SKIP stub at :524 sits in the `else` branch of
`ifeq ($(COLI_V4_SUPPORTED),1)` and is **dead code on this host**.

**So the real issue is stale documentation, not a platform gate:**
1. The explanatory comment at `c/Makefile:333-336` still claims the engine runs on "x86-64
   Linux/Windows and aarch64 Linux" — it never mentions Darwin, even though the code three lines
   below enables it.
2. The `else`-branch message repeats the same stale claim, and is the string a developer will
   grep for and believe.
3. `git diff` confirms the Darwin branch is **committed**; the only uncommitted change to
   `c/Makefile` is the 2-line `METAL=$(METAL)` passthrough.

**My error, recorded.** I grepped the SKIP string, matched it, and concluded the target was inert
on macOS — without checking whether that branch was live. The stale message was *designed* to be
misleading in exactly this way, but the failure was mine: `make -n` answers this in one command
and I reached for `grep` instead.

**Hazard uncovered by asking make instead of reading it.** The real target's first action is
`deepseek-v4-clean`, which deletes `deepseek_v4` **and all 31 unit objects** (the clean list
includes `backend_metal_v4.o`, so it is Metal-aware), then rebuilds with `ARCH=` **empty** —
`PORTABLE_ARCH` is defined as `$(if $(X86_64),x86-64-v3,)`, so on arm64 it resolves to nothing,
i.e. a **baseline build rather than `-mcpu=native`**. Running the target mid-campaign would have
destroyed the benchmarked binary and replaced it with a potentially slower one, silently
invalidating comparisons against every number measured so far.

**Actionable follow-ups** (documentation defects, not code defects):
- Update the comment at `c/Makefile:333-336` to include aarch64 Darwin.
- Update the `else`-branch message so it stops asserting something false on a supported host.

---

## E22. Industry practice: is bit-exact quantized matvec on Apple GPU actually achievable?

**Why this was asked.** E15 left a genuine fork: the bit-exact path reaches only *parity* with CPU
at batch-1 (decode), while the reordered path is 5.4x faster. Before accepting that tradeoff it
was worth knowing whether anyone else solves this, or whether parity-at-batch-1 is the known
ceiling.

**Method.** Literature/source review of the two production Metal LLM backends (llama.cpp, MLX)
plus Apple's own guidance, focused on the exact shape in question: 4-bit quantized matvec,
batch=1, I=4096→O=2048.

**What upstream actually does at batch=1**

| project | structure | reduction |
|---|---|---|
| llama.cpp Metal | 1 simdgroup = 32 lanes, 2 simdgroups/threadgroup, **4 output rows per simdgroup** (`mul_vec_q_n_f32_impl`, `ggml-metal.metal` L3547-3632) | per-thread partials → `simd_sum` |
| MLX | `dispatch_qmv()` routes M=1 to qmv; `num_simdgroups=2`, `results_per_simdgroup=4`, `packs_per_thread=1-2` (`quantized.h` L751-815) | final `simd_sum` |

**The decisive finding.** *No upstream fast Metal quantized-matvec path preserves CPU summation
order.* Both buy their speed precisely by parallelizing across K and reducing with `simd_sum` /
split-K. Correctness is validated by **tolerance, not bitwise equality**:
- llama.cpp: NMSE with `max_nmse_err()` default **1e-7** (`test-backend-ops.cpp` L1145-1172)
- MLX: differential tolerances; recent GPU-vs-reference sweeps gated at **1e-3**

**How that reframes this project's numbers.** The `simd` variant's real-dim error is **2.64e-04
relative** — comfortably inside MLX's 1e-3 production gate. So reordered reduction is not a
compromise invented here; it is the industry-standard engineering choice, and this project is
currently holding itself to a *stricter* standard than either upstream backend.

**Order-preserving levers that remain unused** (these keep bit-exactness):
1. **Multiple output rows per thread/simdgroup** — upstream consensus is 4. Amortizes the staged
   activation read across N rows. Rows are independent, so per-row accumulation order is
   untouched. Not yet tried here; the single most promising remaining lever.
2. **Vectorized loads** (`uchar4`, packed nibble) — order-preserving if unpack order is exact.
3. **Wider row tiling per threadgroup** — occupancy only, does not touch single-row order.

**Order-breaking, therefore excluded from the exact variant:** simdgroup reduction across K,
split-K, `simdgroup_matrix`/tensor-op tiles. Apple's own guidance for custom quantized formats
(WWDC26 session 330) points at dequantizing into threadgroup memory / cooperative tensors — but
explicitly for *throughput*, not for exact accumulation order.

**Apple occupancy guidance, applied.** Threadgroup size should be a multiple of
`threadExecutionWidth`; `dispatchThreads` vs `dispatchThreadgroups` is about nonuniform
threadgroups, not occupancy. The relevant read for this kernel: 2048 threads is not the binding
constraint — memory reuse is. That is consistent with E14, where the bottleneck turned out to be
redundant activation traffic rather than thread count.

**Verdict.** Bit-exact *and* fast is not something upstream achieves on Apple silicon. Parity at
batch-1 for the exact path is therefore a credible ceiling, not a defect — but levers 1 and 2 are
untried here and are now scheduled (backlog T11/T12) precisely because closing that gap would
make the accuracy question disappear rather than merely be accepted.

---

## E23. Codebase reconnaissance: the `slot->slab` hazard is smaller than the design doc implied

**Why this was asked.** The live GPU seam was blocked on a stated hazard: `slot->slab` is
overwritten in place for a different expert, so a refill racing an in-flight GPU read would make
the GPU consume *another expert's weights* — silently wrong output, no crash.

**Method.** Direct source reconnaissance of the seam, the expert-store slot lifecycle, and the
threading model.

**Findings (file:line):**

| aspect | finding |
|---|---|
| seam | `moe_token_pipeline()` `c/deepseek_v4.c:3579`; GPU seam `:3763-3768` — calls `coli_v4_expert_forward_ref`, then `coli_expert_release` `:3766`, then accumulates `:3767-3768`. **Three** sites total: `:3763-3766`, `:3730-3733` (dual-loader), `:3167-3170` (serial `moe_token`) |
| slot struct | `V4ExpertSlot {expert, references, used, slab, aligned_slab}` `:5078-5084` |
| **refcount** | base eviction requires `!slots[i].references` `:5227-5232`; hot path likewise `:5886-5896` |
| overwrite | `slot->slab = malloc(record_bytes)` `:5238-5245` — **not page-aligned**; hot path has `aligned_slab` `:5902-5911` |
| threading | **no OpenMP in the MoE expert loops** — experts are processed serially (`:3707-3736`, `:3744-3768`). Loader pthreads (`:3360`, `:3473`, `:3484`, `:3508`) do refill **I/O only**. Store mutex guards lookup/release metadata only |
| accumulation | per-expert output is bf16-rounded inside the ref chain, but the MoE accumulator sums 6 experts + shared in **fp32** and rounds **once** at the end (`:3775-3777`) |

**The hazard is already 90% mitigated by existing code.** Lease refcounting means an in-use slot
*cannot* be evicted — eviction explicitly skips any slot with `references != 0`. The residual
window is therefore narrow and precise: **only if the GPU is still reading after
`coli_expert_release`**. Synchronous v1 collapses to a single requirement — the GPU must complete
before that release call — rather than the architectural change the design note implied.

**Consequence for the accuracy question.** The fp32 accumulate-then-round-once structure
(`:3775-3777`) matters for rule F: a 1-bf16-ULP per-expert difference is summed in fp32 across 6
experts and rounded once, so whether it survives to flip a *token* under greedy decoding is an
empirical question — it only matters if it crosses an argmax boundary. That is why the D1 verdict
is deferred to an end-to-end token-exactness measurement rather than decided from kernel-level ULP
counts.

**Host-side patterns available for reuse** (`c/backend_metal.mm`, the existing GLM backend):
device/queue/library init `:621-633`; return 0 to signal CPU fallback `:1094-1099`;
`MTLResourceStorageModeShared` `:473`; grow-only scratch `ensure()` `:584-589`; `wrap()`
zero-copies **only** if pointer and length are 16 KiB aligned, else copies `:613-619`;
`newBufferWithBytesNoCopy` for registered slabs `:691-710`; and — directly relevant — `moe_submit`
already batches gate GEMV + up GEMV + SILU + down GEMV into **one command buffer with barriers**
`:1198-1285`, with `moe_finish` waiting and scatter-adding `:1288-1305`. That is the exact fused
pattern V4 needs, already proven in-tree.

**Practical consequence.** Because the base slab is plain `malloc`, zero-copy upload is not
available on the base path (GLM's `wrap()` would silently copy anyway). v1 therefore uses an
explicit copy into grow-only scratch, with zero-copy via `aligned_slab` deferred behind a
profiling gate.

---

## E24. T5 — END-TO-END MEASUREMENT: Metal is currently 2.1–2.3x SLOWER

**This is the answer to the question the whole campaign exists to answer, and it is negative.**

**Method.** Real model (`models/deepseek-v4-flash`), `--ram 96`, 8 tokens, identical prompt,
`COLI_V4_SAVE_USAGE=0` with `.coli_usage` restored from the frozen snapshot before each arm so
both arms seed from identical history (29127 B / 14190 selections).

| arm | ttft | after_first | dispatches | output |
|---|---|---|---|---|
| Metal OFF | **10.037s** | **6.731s** | 0 | `The capital of France is **Paris**.` |
| Metal ON (`ordered_cold`) | 21.194s | 15.466s | 2396 | `The capital of France is **Paris**.` |

**2.11x slower on time-to-first-token, 2.30x slower on generation.** Output byte-identical, so
the correctness chain holds all the way to real routing — the regression is purely performance.

**A first smoke run also mutated `.coli_usage` (14190 → 18318 selections).** That run was
discarded and the snapshot restored; the table above is the clean paired measurement. Worth
recording because it is exactly the confound the PREWARM work identified, and it appeared
immediately in a context where it had not been anticipated.

### Why the kernel benchmarks lied — two confirmed causes

**Cause 1: occupancy collapse from the fusion design (the big one).**
The glue dispatches `dispatchThreads(N)` with `threadsPerThreadgroup(N)` — i.e. **exactly one
threadgroup**, at `backend_metal_v4.mm:266-267`. One threadgroup occupies **one** of the 40 GPU
cores; 39 sit idle. The microbenchmarks in E13/E15 dispatched O=2048 threads spread across many
threadgroups, which is where the 1.1–7.3x came from.

This is not an implementation slip — it is **forced by the fusion decision**. The fused kernel
keeps every intermediate (qdq'd input, gate, up) in threadgroup memory, and threadgroup memory is
private to one threadgroup. Fusing the 7 steps therefore *requires* single-threadgroup execution.
The 32 KB budget that made fusion look elegant is the same constraint that caps it at 1/40 of the
device.

**Cause 2: 32 GB of redundant weight copying per 8-token run.**
Per dispatch at real dims: gate 4.19 MB + up 4.19 MB + down 4.19 MB + scales 0.79 MB =
**13.37 MB**. Times 2396 dispatches = **32.0 GB** copied host→scratch, which the GPU then re-reads
from unified memory. The underlying weights are only ~12 MB per expert; they are re-copied on
every single use because the glue has no cache keyed on expert identity.
This was a deliberate v1 simplification (decision C: copy-upload, because the base slab is plain
`malloc` and not 16 KiB aligned). Profiling now says it is not viable — T10 moves from
*conditional* to *required*.

**Cause 3 (indicated, NOT yet root-caused): only 58% GPU coverage.**
`expert_requests=4128` (= 43 layers x 6 top-k x 16 steps), `hits=2394`, `misses=1734`,
`dispatches=2396`. Dispatches track *hits* almost exactly, and the shortfall (~1732) matches
*misses* to within 2. All three seam sites are confirmed wired, so the likely explanation is the
glue's strict byte-validation (`backend_metal_v4.mm:87-93`, requiring
`data_bytes == rows*row_bytes` and `scale_bytes == rows*groups`) rejecting tensor views produced
by the hot `aligned_slab` path (`deepseek_v4.c:5902-5911`) rather than the base `slab` path.
**Unconfirmed.** Needs a diagnostic that logs which validation branch returns -1. Do not state
this as fact until measured.

### What this changes

The plan assumed fusion was the right shape and that variants (T6 simd, T7 hot, T8 batching)
would layer speed on top. **That assumption is now falsified**: no variant fixes single-threadgroup
occupancy, and T8 (batching 6 experts into one command buffer) would not either — it reduces
command-buffer count but each expert still runs in one threadgroup.

The correct pivot is the opposite of fusion: **multi-dispatch using the already-proven,
high-occupancy `ordered_xcache` matmul kernels** (measured 1.10x / 2.20x / 7.26x at real dims,
bit-exact at I=4096), with the fused single-threadgroup kernel retained purely as the bit-exact
oracle it has already served as. That trades on-chip intermediate reuse for ~40x more parallelism —
and the E14 diagnosis already showed this workload is memory-bound, not on-chip-reuse-bound.

**Honest framing for the record:** the kernel work was not wasted — every primitive and the
ordered matmul are bit-exact at production reduction length, and the seam, glue, counter and race
invariant are all proven. What was wrong was a single architectural choice (fuse everything into
one threadgroup), and it was only detectable end-to-end. Kernel microbenchmarks measured the
matmul in isolation with full occupancy; the fused kernel never had that occupancy. That gap
between "the kernel is fast" and "the engine is faster" is the entire lesson of this section.

### E24b. Cause 3 root-caused, and a zero-copy blocker found

**rows16 packing is LIVE on the real model.** `hot_pack_slot_locked` (called from `lookup_hot`
when `hot_is_pinned` is true) repacks pinned experts into the rows16 layout, and views are then
filled by `hot_fill_view` — a *different* function from the base `fill_tensor_view`. The smoke run
confirms it is active: **`v4_rows16 packed_slots=491`**.

My glue validates for the **cold** layout only (`block_columns == 32`, `data_bytes ==
rows*row_bytes`). A rows16-packed view will not satisfy that, so it returns -1 and the caller falls
back to CPU. That is the most likely mechanism behind the 58% coverage — and it reframes T7 (hot
variants) from "optional perf polish" to **required for coverage**, since a meaningful fraction of
the hot path is pinned/packed by design (`pin_slots_per_layer=16`).

**Zero-copy blocker: 4 KiB vs 16 KiB.** The hot store allocates with
`posix_memalign((void**)&slot->slab, 4096, capacity)` — 4 KiB alignment. Apple Silicon uses
**16 KiB pages**, which is precisely why GLM's `wrap()` requires pointer AND length to be 16 KiB
aligned before it will use `newBufferWithBytesNoCopy` (`backend_metal.mm:613-619`). So the existing
`aligned_slab` is *not* sufficient for true zero-copy; the allocation constant would have to move
4096 → 16384.

This matters because on Apple Silicon the weights are **already in unified memory**. Copying them
into a Metal scratch buffer is pure waste — 32 GB of it per 8-token run. The fix is not a smarter
cache, it is not copying at all: align the slab to 16 KiB and wrap it in place.

### Revised plan (supersedes the T6→T9 ordering in experiments-backlog.md)

| priority | change | why |
|---|---|---|
| **P0** | Multi-dispatch instead of one fused threadgroup | Recovers ~40x parallelism. Reuse the already-proven `ordered_xcache` kernels (1.10x/2.20x/7.26x, bit-exact at I=4096). One command buffer, barriers between stages — GLM's `moe_submit` pattern (`backend_metal.mm:1198-1285`) is the in-tree precedent. |
| **P0** | 16 KiB-align the expert slab + `newBufferWithBytesNoCopy` | Eliminates 32 GB/run of memcpy outright rather than caching it. |
| **P1** | Accept rows16 views in the glue (was T7) | Required for coverage, not polish — 491 slots are packed on the real model. |
| **P2** | simd variant (was T6) | Only meaningful once occupancy is fixed; currently it would also run at 1/40 occupancy. |
| **dropped** | T8 token-level batching | Reduces command-buffer count but each expert still runs in one threadgroup, so it does not address the actual bottleneck. |

The fused kernel is **retained as the bit-exact oracle** — it earned that role (12,288 outputs,
3 real seeds, 0 differing bits) and remains the reference any faster path must match.

---

## E25. P0b + the prefill test: Metal cannot win through this seam. Structural, not fixable by tuning.

### P0b results (zero-copy + qdq occupancy)
Landed: expert slab aligned to 16384 (Apple Silicon page size), `newBufferWithBytesNoCopy` with
per-tensor offsets, cached wrapper keyed on slab pointer, **zero copy-fallbacks**. `fp8_qdq`
rewritten from one-thread-per-block to one 128-thread threadgroup per block with an `fmax` tree
reduction — and it stayed **bit-exact** (`fmax` is associative/commutative for non-NaN, unlike a
sum, so a tree is legitimate here; the probe proved it rather than the argument being trusted).

Per-stage profile (`COLI_V4_METAL_PROFILE=1`):

| stage | ms/expert | % |
|---|---:|---:|
| host weight memcpy | **0.0215** | 0.52 |
| matmul gate | 1.1597 | 27.83 |
| matmul up | 0.0452 | 1.08 |
| matmul down | 0.5537 | 13.29 |
| fp8_qdq in / down_input | 0.0366 / 0.0245 | 1.5 |
| swiglu / weighted_bf16 / bf16 x2 | ~0.081 | 2.0 |
| submit+wait | 2.2444 | 53.87 |

**Two corrections to earlier claims, both mine:**
1. **Copy was never the bottleneck** — 0.52%. I estimated ~0.84 ms from 18.69 GiB; actual is
   0.0215 ms, wrong by ~40x. On unified memory those "copies" were largely cache-resident. P0b was
   therefore the right *engineering* change (32 GB of pointless traffic removed) but it bought
   almost no time, and I predicted otherwise.
2. **The profile's GPU attribution does not reconcile with production and must not be quoted as a
   decomposition.** Stages sum to 1.922 ms — *below* CPU's 2.919 ms — so if the split were literal,
   Metal would already win. Profile mode issues ~10 command buffers instead of 1, inflating
   whichever dispatch lands first; that is why `matmul gate` reads 1.1597 ms against
   `matmul up` 0.0452 ms on **identical shapes** (25.7x). The gate "excess" alone (2.66 s/run)
   exceeds the entire measured gap (0.825 s), which is self-refuting. The host memcpy figure IS
   reliable (CPU timer, granularity-independent); the GPU stage split is not.

Trajectory across all three fixes: **2.30x slower → 1.17x → 1.118x slower.** Clear diminishing returns.

### The decisive experiment: prefill
Hypothesis: batch-1 decode is weight-bandwidth-bound with ~zero weight reuse (each expert's ~8 MB
used once per token), and on unified memory the GPU has no bandwidth advantage over the CPU — so
GPU should only win where arithmetic intensity rises, i.e. prefill. The microbenchmarks supported
this: batch-1 gate/up = parity, batch-8 gate/up = 7.26x.

Measured, 799-token prompt, paired, frozen history:

| arm | ttft | dispatches | coverage |
|---|---|---|---|
| Metal OFF | **708.143s** | 0 | — |
| Metal ON | 753.030s | 36740 | 36740 / 206916 = **17.8%** |

**HYPOTHESIS FALSIFIED — 1.063x slower on prefill as well.**

**Why.** 799 tokens x 43 layers x 6 top-k = 206,142 ≈ the 206,916 observed expert_requests. The
engine walks prefill **token by token through the same per-expert seam**. Every call is S=1.
Prefill is not batched at the expert level — it is simply *more* batch-1 calls. Arithmetic
intensity never rises, so the batch-8 advantage is **unreachable through this seam**.

The seam signature makes this structural, not incidental:
```c
int coli_v4_metal_expert_forward(float *out, const ColiExpertView *expert,
                                 const float *input, float route_weight, float swiglu_limit);
```
One input vector, one output vector. **There is no token dimension to batch over.**

### Verdict on the original question

**Does Metal make DeepSeek-V4 faster on this hardware? No — not through an expert-level seam.**
Best achieved: 1.118x slower on decode, 1.063x slower on prefill, with correctness perfect
throughout (byte-identical output in every paired run).

The causes are structural and stack against the GPU:
1. **S=1 per call** — no arithmetic intensity; the 7.26x batch-8 result cannot be reached.
2. **Unified memory** — the GPU has no bandwidth advantage; both processors read the same DRAM.
3. **~Zero weight reuse** — top-6 of 256 experts, ~8 MB of weights per ~0.5 MFLOP. Pure bandwidth.
4. **Per-call dispatch + synchronous wait** — real overhead the CPU path simply does not pay.

Remaining backlog items would improve the *margin*, not the *sign*: P1 (rows16 coverage — currently
only 17.8% of forwards reach the GPU) and T11/T12 (order-preserving occupancy levers) make the GPU
faster and more-used, but none of them changes S=1 or unified-memory bandwidth.

**What would actually be needed:** batch the token dimension *inside* the expert call — i.e. gather
all tokens routed to expert E in a layer and issue one S=N matmul. That is an **engine-level
change** to the MoE scheduler (`moe_token_pipeline`), not a backend change, and it is exactly the
shape the 7.26x microbenchmark was measuring. That is the honest recommendation, and it is a
substantially larger piece of work than this campaign scoped.

### What the campaign did establish (not wasted)
- Every V4 MoE arithmetic primitive is **bit-exact on GPU**, several exhaustively, all at production
  reduction length (I=4096 / 2048).
- A fused single-dispatch expert kernel bit-exact across 12,288 outputs, 3 real seeds.
- A working, race-free seam (synchronous v1) with the lease invariant verified at all three sites.
- A dispatch counter that caught a silent CPU fallback which would have made every number a lie.
- Build integrity preserved throughout: default build byte-identical, 26 objects, token-exact.
- The measurement discipline itself: frozen history, paired interleaving, and stating reduction
  length — each of which caught a real error in this campaign.

---

## E26. PREWARM A/B — KILL verdict, and a process gap worth fixing

Full detail in `RESULTS.md` §13 and `cache-design.md` §7. Summary:

**Result.** `--ram 64`, both arms one machine state, frozen history, zero gate failures:
paired mean **−29.7%** (per-prompt −18.2 / −27.3 / −32.7 / −38.7%), worsening monotonically.

**The interesting part.** Hit rate *improved* +9.98pp (57.728 → 67.708%), so the anticipated
failure mode — warming the wrong experts — did **not** occur. Persistence and pin selection both
work. The economics are what fail: 412 misses avoided x 13.37 MB = ~5.51 GB saved, against
`warmed=688 bytes=8.566GiB` = ~9.20 GB read eagerly. **Net ~+3.69 GB more I/O**, landing during
prefill. Prediction only has to be wrong ~40% of the time for 688 speculative 13.37 MB reads to
lose against on-demand loading.

So §7's gate resolved **both ways**: the flush/persistence mechanism is justified and stays; the
eager-warming consumer is killed. Those were always two different things, and the gate as written
only measured one of them.

**Two false leads recorded** (see §13e): a *partial* arm1 on an older binary under compression
contamination showed **+14.1%** — the opposite sign — and would have justified keeping a 29.7%
regression had it been reported. And an entire `--ram 64` run was invalidated by ambient load
(non-engine 43–44 GB, outside §11's 18–31 GB band): compressor hit 25.0 GB with `gap_gb=20.1`.
Rerun at 21.9 GB non-engine went 901s → 793s (**+13.6%**) with `gap_gb` 0.4.

## E27. Process gap: probe binaries can go stale and report phantom failures

Final teardown showed `probe_fp8qdq` FAILING. It was not a regression: the probe *source* had been
correctly updated (15:31) to `dispatchThreadgroups(NB) x threadsPerThreadgroup(BLK)` for the
rewritten kernel (15:33, now one 128-thread threadgroup per block with an `fmax` tree), but the
compiled binary on disk dated from **02:45** — before the ABI change. Rebuilt: BIT-EXACT 0/768.

**No make target rebuilds `validation/metal/probe_*`.** A kernel ABI change therefore leaves stale
binaries that fail for a reason unrelated to correctness — or worse, could *pass* stale and hide a
real break. Recommended: add a `metal-probes` target, or always rebuild before trusting the suite.
The suite is 10/10 green with every probe rebuilt from current source.

---

## E28. M1 `spec_keep` — KILLED (branch ft-spec_keep)

First experiment of the performance campaign; kill-test pre-registered in
`performance-boost-research.md` before the run.

**Design.** 5 arms x 4 coding prompts x 64 tok, ram96, paired, frozen history,
`COLI_V4_SAVE_USAGE=0`, engaged-counter = `v4_dspark attempts`, text_sha per row.

| arm | config | mean tok/s | paired Δ | acceptance by prompt |
|---|---|---|---|---|
| base | no speculation | 0.4474 | — | — |
| d4k0 | DRAFT=4 (self-disabling default) | 0.4381 | −2.1 % | 0/0/75/0 % |
| d4k1 | DRAFT=4 + PARTIAL_KEEP=1 | 0.4098 | **−8.1 %** | 31/0/54/0 % |
| d2k1 | DRAFT=2 + KEEP | 0.4277 | −4.3 % | 40/0/69/0 % |
| d8k1 | DRAFT=8 + KEEP | 0.4101 | −7.9 % | 25/0/34/0 % |

Token-exact: p1 text_sha identical across ALL arms (single unique sha). Gates: all `ok`.

**Verdict: KILL.** The kill condition ("no (D,KEEP) beats baseline ≥5 %") fired maximally —
nothing beat baseline. More KEEP = more wasted verify+replay on undraftable text; the adaptive
self-disable is doing exactly its job. The historical +18.5 % (24-tok Q&A, older binary) does
not transfer to this regime — a regime-dependence finding worth more than the method.

**Process notes.** One harness bug cost one run: `local E=$(date +%s) W=$((E-S))` under
`set -u` — arithmetic expansion referenced `E` before binding. Split declarations; re-ran from
scratch. M12's priority drops accordingly (its premise weakened).

---

## E29. M11 `phase_profiler` — DONE, and the profile overturns the cost model

Runtime `COLI_V4_PROFILE=1` profiler (branch ft-phase_profiler): 98.0 % of flash decode wall
attributed (99.1-99.3 % tiny), zero output when unset, token-exact with profiler ON, 26-object
default build intact. Overhead check inconclusive-but-benign: the profiled run was FASTER than
the unprofiled pair (6.776 vs 7.362 s) — ambient variance dominates any profiler cost.

**Flash decode profile (7 tokens, 968 ms/token, ram96):** attention **38.7 %**, expert_forward
**28.2 %** (1.06 ms/call), shared_expert 8.8 %, expert_wait 8.7 %, head 3.6 %, router 3.4 %,
hc_norm 2.9 %, indexer+compressor 3.7 %, embed 0.06 %, rope **0.02 %**.

**RETRACTION.** The campaign doc's "expert chain ≈88 % of decode" was wrong: the 2.81 ms/expert
figure had been derived by dividing total wall by dispatch count, silently absorbing attention,
router, shared expert and head. Real expert_forward is 1.06 ms/call; the routed chain is 36.9 %
and **attention is co-dominant at 38.7 %** — with no dedicated method until now (M16 added).

**Consequences, all via pre-registered gates:** M2 rope_cache KILLED (0.02 % < 3 % gate — the
gate fired exactly as designed, cheaper than the branch would have been); M3 gutted (embed half
dead at 0.06 %); M16 `attention_dissect` added; sequencing rewritten (attention and expert
kernels are the two lanes; prefill batch M4 unchanged).

---

## E30. M15 `cpu_expert_kernel` — microbench gate PASSED (~29 % kernel, bit-exact)

`c/tests/bench_mxfp4.c`, n=20 mean±sd (first dropped), real dims, vs the shipping scalar path
(`quant.h:1401-1412`):

| shape | baseline | v1_notail | v2_2row |
|---|---|---|---|
| gate/up decode 4096→2048 | 0.401 ±0.022 ms | 0.320 ±0.008 (**−20.2 %**) | **0.286 ±0.011 (−28.7 %)** |
| down decode 2048→4096 | 0.418 ±0.022 | 0.319 ±0.013 (−23.7 %) | **0.298 ±0.011 (−28.7 %)** |
| gate/up S=8 | 3.036 ±0.233 | 2.214 ±0.139 (−27.1 %) | **1.962 ±0.182 (−35.4 %)** |

**All variants BIT-EXACT** against baseline output (the gate — a faster variant that changed one
bit would be rejected, not accepted as a tradeoff).

- **v1_notail**: whole 32-column groups have no odd-column tail, so the per-iteration
  `if (i+1 < base+glen)` bounds test is pure overhead — hoisted into a 16-byte fast path.
  Order-identical; same adds in the same sequence.
- **v2_2row**: two output rows per pass, so the activation vector is walked ONCE for two rows.
  This is the weight/activation-reuse thesis from §0.2 confirmed on CPU, and it is exactly the
  llama.cpp/MLX "multiple results per simdgroup" idea (4 rows) applied to the scalar path.
  Rows are independent ⇒ per-row two-level accumulation order untouched ⇒ bit-exact.

**Projected engine effect:** expert_forward is 28.2 % of decode (M11), so ~29 % of that ≈ **8 %
decode** — just under the pre-registered 10 % PASS bar, so the engine-level result decides it.

**Bigger prize spotted:** the same two techniques should transfer to the **FP8 rows8 matvec**
(`:10337-10405`) used by attention's wq_a/wq_b/wkv/wo and by shared_expert — i.e. potentially the
38.7 % + 8.8 % slices, not just the 28.2 %. M16's sub-attribution will size that directly.

---

## E31. M16 step-1 — attention dissected: the output projection is the biggest line item in decode

Nested sub-phases added to the M11 profiler (branch `ft-attention_dissect`). Reconciliation is
exact: 911.685 + 1.340 + 96.162 + 1668.940 + 4.642 = **2682.769 = parent attention**. Build 26
objects, METAL=1 links, profiler silent when unset, tiny oracle token-exact with profiler ON,
flash output exactly `The capital of France is **Paris**.`, accounted 98.0 %.

| attention sub-phase | ms/decode | % of attention | % of decode | ms/call |
|---|---|---|---|---|
| **attn_out** (wo_a/wo_b) | **1668.9** | **62.2 %** | **25.0 %** | **5.54** |
| attn_qkv (wq_a/wq_b/wkv) | 911.7 | 34.0 % | 13.6 % | 3.03 |
| attn_sparse | 96.2 | 3.6 % | 1.4 % | 0.32 |
| attn_kv_assembly | 1.3 | 0.05 % | 0.02 % | 0.004 |
| attn_other | 4.6 | 0.2 % | 0.07 % | — |

### The finding that reorganises the campaign
Grouping by KERNEL rather than by phase:

| kernel family | % of decode |
|---|---|
| **FP8 matvec** (attn_out 25.0 + attn_qkv 13.6 + shared_expert 9.0) | **47.6 %** |
| **FP4 matvec** (routed experts) | 28.7 % |
| **all quantized matvec** | **76.3 %** |
| everything else (wait, head, router, norms, indexer, compressor, sparse…) | 23.7 % |

**76 % of decode is quantized matvec** — and E30 already measured **−28.7 %, bit-exact**, on
exactly that shape (tail-branch hoist + two-output-rows-per-activation-walk). If the technique
transfers to both the FP4 expert path and the FP8 rows8 path (`:10337-10405`), the arithmetic is
~21.9 % decode reduction ≈ **1.28× purely from bit-exact kernel work**.

Note this also *raises* M15's value above its original scoping: M15 was written against the
28.7 % FP4 slice; the profile shows the FP8 slice it did not target is **larger** (47.6 %).
M15 is therefore rescoped from `cpu_expert_kernel` to cover both matvec families.

**Attention was never "attention" — it is 96 % matvec.** The sparse-attention scalar loop that
M10/M16 assumed was the target is 1.4 % of decode. That assumption is now dead.

---

## E32. M15 `matvec_kernels` — **WIN: −20.4 % decode, token-exact** (branch ft-matvec_kernels)

Applied the two E30-proven techniques to BOTH quantized matvec families: tail-branch hoist for
whole 32-column groups, and paired-row processing so the activation vector is walked once per two
output rows. Plus fused the gate/up dual matvec (was two separate S=1 walks, `:10461-10464`).
Touched `c/quant.h` (MXFP4 + FP8 rows8) and `c/deepseek_v4.c` (dual-matvec fusion).

**Independently re-measured by me** (agent's numbers reproduced within noise):

| phase | before | after | delta |
|---|---|---|---|
| attn_out | 1668.9 ms | 1114.6–1120.6 | **−32.9 %** |
| attn_qkv | 911.7 | 583.8–589.1 | **−36.0 %** |
| expert_forward | 1920.8 | 1408.1–1426.4 | **−25.7 %** |
| shared_expert | 599.2 | 374.1–376.5 | **−37.6 %** |
| **decode wall** | **6684.2** | **5321.8–5338.9** | **−20.4 %** |
| ttft | 10.116 s | 8.529–8.628 s | −15 % |

**≈1.26× decode throughput (1.047 → 1.315 tok/s equivalent), bit-exact.**
`generated_text` is byte-identical (`The capital of France is **Paris**.`) — bit-exactness proven
at the only level that matters, on the real model.

**The win landed exactly where predicted.** Matvec phases account for 1595.7 ms of savings against
a 1362.4 ms wall reduction (117 %) — i.e. no time was displaced into unmeasured phases; if
anything the other phases got marginally cheaper too. This is the difference between "the number
moved" and "the number moved for the reason we said".

**Shared-header risk checked, not assumed.** `quant.h` is included by `colibri.c` and `kimi_k3.c`
as well as V4. All four quant tests pass (`test_native_quant`, `test_ue8m0`, `test_e8_kernel`,
`test_i4_acc512`), plus the 5 Metal probes that assert against the CPU reference
(`probe_primitives/matmul/fp8qdq/hot/fused_moe`), plus V4 tiny-oracle and KV-prefix suites.
26-object default build; METAL=1 still links.

### Why this worked when Metal didn't
Same underlying thesis — **reuse the walk** — but applied where the machine actually is. Metal
tried to win by moving 12.6 MB of weights per S=1 call to a device with no bandwidth advantage
and a per-dispatch tax; this walks the *activation* once for two output rows on the CPU that is
already resident, changing traffic without changing a single accumulation order. The profiler is
what pointed here: the pre-M11 cost model would have sent this effort at rope caching (0.02 %)
or the sparse-attention scalar loop (1.4 %).

---

## E33. M15b `matvec_rows4` — **cumulative 1.38–1.42× decode, bit-exact** (branch ft-matvec_rows4)

Raised the row-block from 2 → 4 per activation walk (llama.cpp/MLX both use 4 results per
simdgroup; we had left that on the table). Only `c/quant.h` changed (MXFP4 + scalar FP8; the AVX2
rows8 path already processes 8 rows per column and was left alone). Ragged `O` handled with
clamped indices + guarded stores; agent verified `O={1,2,3,4,5,7,17,129,130}`.

| phase | pre-M15 | 2-row | 4-row | total vs pre-M15 |
|---|---|---|---|---|
| attn_out | 1668.9 | 1114.6 | 869–883 | **−47.1 %** |
| attn_qkv | 911.7 | 589.1 | 430–441 | **−51.6 %** |
| shared_expert | 599.2 | 376.5 | 283–288 | **−52.0 %** |
| expert_forward | 1920.8 | 1408.1 | 1240–1262 | **−34.3 %** |
| **decode wall** | **6684.2** | 5338.9 | **4693–4838** | **−27.6 to −29.8 %** |
| ttft | 10.116 s | 8.628 | **7.860–7.904 s** | −22 % |

**Cumulative ≈1.38–1.42× decode throughput (1.047 → 1.42–1.49 tok/s equivalent), output
byte-identical throughout.** Two independent runs (mine 4693.1, agent's 4837.6) bracket ~3 % run
variance; the conservative figure is quoted in claims.

All canaries green: `test_native_quant`/`test_ue8m0`/`test_e8_kernel`/`test_i4_acc512` (shared
`quant.h` is used by `colibri.c` and `kimi_k3.c`), 5 Metal probes, V4 tiny-oracle + KV-prefix,
26-object build, METAL=1 links.

### The important part: the bottleneck MOVED
`expert_wait` went **415 ms (6.2 % of decode) → 779–864 ms (16.6–17.9 %)** and is now the #3
phase. It did not get slower in absolute terms — **compute got ~37 % cheaper, so SSD/loader
latency that compute used to hide is now exposed.** Textbook Amdahl shift, and it is the whole
argument for having built the profiler first: without per-phase attribution this would look like
"the optimization caused an I/O regression" instead of "the optimization revealed the I/O wall".

New ranking: matvec 59.4 % · **expert_wait 17.9 %** · everything else 22.7 %.
That promotes M6/M7 (expert prefetch + cache policy) from "medium" to the next lane, on data
rather than on the plan's original guess.


---

## E34. `loader_depth` — **NEGATIVE, hypothesis rejected** (no code change kept)

`COLI_V4_EXPERT_LOADER_COUNT` is a compile-time constant (3). Since `expert_wait` had risen to
17.9 % of decode, tested whether the loaders were queue-depth starved by sweeping N = 3/6/10.

**The first sweep looked like a 2.3x win and was entirely an artifact.** N=3 ran first on a cold
page cache after a full 26-unit rebuild and reported `decode_wall=10706 ms`,
`expert_forward=3898 ms` — but the identical committed N=3 build had measured 4693-4838 ms with
`expert_forward≈1240 ms` minutes earlier. Loader count cannot triple compute time; N=6 and N=10
simply inherited the warmth N=3 paid for.

Re-measured N=3 **warm**, same protocol:

| loaders | decode_wall ms | expert_wait ms |
|---|---|---|
| 3 (warm, run1) | 4761.3 | 834.0 |
| 3 (warm, run2) | 4771.6 | 757.0 |
| 6 | 4750.8 | 833.1 |
| 10 | 4721.8 | 839.9 |

All within ~1 %; `expert_wait` statistically identical. **Loader parallelism is not the
constraint — rejected, nothing merged.** Also re-confirms the M15b baseline reproduces (4761/4772
sits inside the earlier 4693-4838 band).

Implication: the residual wait is **dependency-bound, not bandwidth-bound**. You cannot fetch an
expert until the router has chosen it, so more threads cannot help. Only *knowing earlier*
(prediction) or *missing less often* (cache policy) can. That points at M7/M6, not at parallelism.

**Process note:** this is the second time in the campaign that a first-run measurement was
cold-cache inflated. Warm-up is now mandatory before any timing row.

### E34b. Retraction + methodology fix: short runs cannot measure cache policy

While scoping M7 I asserted the cache policy was "performing worse than uniform-random" because
`hit_rate=57.7 %` sat below the `164/256 = 64.1 %` residency capacity. **That was my error and is
retracted.** The uniform-random baseline assumes a *saturated* cache. Ours is not:

- cache capacity = 164 slots/layer x 43 layers = **7052 expert records**
- bytes read 23.33 GB / 13.37 MB = **1745 distinct loads == 1745 misses** (1 load per miss, no churn)
- => cache ends the run **24.7 % full**

So nearly every miss in a 17-token run is **compulsory**, not a policy failure. With only a quarter
of slots populated, 57.7 % is evidence that autopin's 16 pins/layer plus routing skew are working,
not failing.

**Two consequences:**
1. Any M7/M6 cache experiment must first run long enough to **saturate** the cache; measured on an
   8-token benchmark it would be noise-fitting cold-start behaviour.
2. `expert_wait = 17.9 %` of decode is **cold-start-inflated** and overstates steady-state I/O for
   long generations. The matvec results (E31-E33) are unaffected - they are pure compute - but the
   *ranking* that promoted the I/O lane was built on a cold-start number and must be re-derived
   from a saturated run before any I/O work is justified.

---

## E35. Steady-state profile (220 tokens) — the short-run profile was misleading twice

Ran 220 tokens to saturate the expert cache, per the E34b methodology fix.

**Cache reaches steady state:** `hit_rate 57.7 % -> 88.406 %` (61662 requests, 7149 misses).

**`expert_wait` collapses 17.9 % -> 6.0 % of decode** (119.1 -> 41.3 ms/token, -65 %). The I/O
lane promoted in E33 was chasing a cold-start artifact; **de-promoted, M6/M7 not pursued.**
Every other phase is within 3 % per-token of the short run, so the 8-token benchmark remains valid
for *compute* work — it is only the I/O share it distorts.

**`attention` is a PARENT phase.** Its children sum to 64048 ms vs its own 63909 ms, which is why
a naive sort showed cumulative >100 %. Corrected non-overlapping top level (sums to the reported
98.7 % accounted):

| top level | ms | % |
|---|---|---|
| attention (parent) | 63908.9 | 42.2 |
| expert_forward | 40261.2 | 26.6 |
| expert_wait | 9048.5 | 6.0 |
| shared_expert | 8739.8 | 5.8 |
| head | 7410.0 | 4.9 |
| router | 7212.1 | 4.8 |
| hc_norm | 6110.7 | 4.0 |
| indexer | 3746.1 | 2.5 |
| compressor | 2914.5 | 1.9 |

**Leaf ranking — and the finding:**

| leaf | ms | % | optimized |
|---|---|---|---|
| expert_forward | 40261.2 | 26.6 | v4 4-row |
| attn_out | 27418.7 | 18.1 | v4 4-row |
| **attn_sparse** | **22639.5** | **15.0** | **never touched** |
| attn_qkv | 13547.3 | 8.9 | v4 4-row |
| expert_wait | 9048.5 | 6.0 | I/O, de-promoted |
| shared_expert | 8739.8 | 5.8 | v4 4-row |
| head | 7410.0 | 4.9 | no - 33.84 ms/token, 1 call/token |
| router | 7212.1 | 4.8 | no |

`attn_sparse` (2.4041 ms/call x 9417) is **the largest unoptimized leaf in the engine** and was
invisible in every 8-token profile because I had only ever grepped attn_out/attn_qkv. Steady
throughput 1.446 tok/s (691.4 ms/token).

**Next lane is attn_sparse, on evidence.**

---

## E36. `attn_sparse` root cause — clang emits vector-multiply + lane-extract + SERIAL scalar adds

`attn_sparse` is 15.0 % of steady-state decode (22639.5 ms / 9417 calls / 2.4041 ms per call) and
is the largest never-optimized leaf. Hot path is `coli_v4_sparse_attention_ref` (`c/deepseek_v4.c:3061`).
Shape: 64 heads x 512 head_dim, MQA (`num_key_value_heads=1`), `index_topk=512`, `sliding_window=128`.

Build uses **`-O3` with no `-ffast-math`** (verified across the full CFLAGS continuation), so LLVM
may not legally reassociate the FP reduction at `:3086`.

**Disassembled what it actually emits** (`clang -O3 -mcpu=apple-m1`):

```
ldp   q1, q2, [x10, #-32]     ; vector loads
fmul.4s v1, v1, v5            ; vector MULTIPLY only
mov   s5,  v1[3]              ; extract lane 3 -> scalar
mov   s17, v1[2]              ; extract lane 2 -> scalar
mov   s18, v1[1]              ; extract lane 1 -> scalar
...                            ; then a strict serial scalar fadd chain
```

Greps for accumulators (`fmla.4s`/`fadd.4s` targets) and for a horizontal reduce (`faddp`/`addv`)
both return **empty**. So the compiler pays vector load + vector multiply cost, **adds** 3
lane-extract `mov`s per 128-bit vector, and **still** serializes the accumulation. The dependency
chain on the single `float score` is fully intact and is the binding constraint (~4-cycle FMA
latency per element, no ILP).

The sibling AXPY at `:3106` has no reduction dependency and vectorizes cleanly - it is not the problem.

**Lever:** hand-written NEON with 4 independent `fmla.4s` accumulators + one final horizontal
reduce breaks the latency chain and deletes the lane-extract overhead. Secondary: `:3069`
`malloc`/`free` of `scores` runs **once per call** (9417 mallocs per 220-token run), and the kv rows
are walked twice (score pass, then value pass).

**Caveat that must be tested, not assumed:** multiple accumulators change FP summation order, so
this is *not* automatically bit-exact. The campaign gate is token-exact output; that must be
verified empirically, with an order-preserving fallback if it fails.

---

## E37. `attn_sparse` NEON kernel — 4.53x on the phase, **opt-in** because it is not bit-exact

Acted on the E36 root cause. Hand-written NEON with **4 independent `fmla.4s` accumulators** plus a
single horizontal reduce, replacing the compiler's fmul-then-extract-then-serial-add pattern. Also
hoisted the per-call `malloc`/`free` of `scores` to a stack buffer with a heap fallback for
oversized `topk` (order-safe, so it applies to both paths).

**Microbench** (n=30, `-O3 -mcpu=native`, no fast-math): 3.68x-4.74x across topk in {64,128,256,512}.

**Engine, steady state (220 tokens, 9417 identical calls):**

| | before | after | |
|---|---|---|---|
| attn_sparse | 22639.5 ms | **4999.3 ms** | **4.53x** (2.4041 -> 0.5309 ms/call) |
| decode_wall | 151426.7 ms | **134619.3 ms** | **-11.1 %** |
| throughput | 1.446 tok/s | **1.627 tok/s** | **+12.5 %** |

attn_sparse alone accounts for -17640 ms of the -16807 ms wall delta, so the gain is fully
attributable to this kernel.

### It is NOT bit-exact, and the 8-token gate hid that
The short gate passed (`**Paris**.` identical) because a ~3.87e-7 score delta cannot flip an argmax
in 8 tokens. At 60 tokens it does. Verified by rebuilding **without** the change and running both
builds twice:

| build | 60-token md5 | deterministic |
|---|---|---|
| baseline (change stashed) | `5d04890413ff539e802985ce8c727814` | yes (run1 == run2) |
| NEON reassociated | `c6d8f26ef47095bf6f777c11d99df080` | yes (run1 == run2) |

Both builds are individually deterministic and they disagree — so the divergence is real and
attributable to FP reassociation, not to engine nondeterminism. (Note: `generated_text` is
multi-line; comparing it with `grep '^generated_text='` only compares the first line and is wrong.
Use the full-block extraction.)

**Resolution: opt-in via `--fast-sparse-attn`.** Default keeps the strictly ordered reduction and
is byte-identical to the previous build (md5 verified). The flag prints
`v4_sparse_attn mode=fast-reassociated warning=output-not-bit-identical-to-default` so no transcript
is ambiguous. Selector is a write-once file-scope static set before any thread is created;
non-arm64 always takes the ordered path. Wrapper: `run-deepseek-v4-fast-sparse-attention.sh`.

**Methodology lesson (third of the campaign):** an 8-token gate is too short to catch FP-ordering
regressions, just as it was too short for `attn_sparse` cost (understated ~7x) and too short for
cache behaviour (E34b). Correctness gates must run at the length where the effect can appear.

---

## E38. M4 `moe_batch` — **gated OUT before any model code; pivot identified** (branch ft-moe_batch)

Plan-gated optimization: during prefill, group a 64-token chunk's tokens by routed expert, lease
each expert once, run its FFN as a batched S=group matmul. Two measurement tasks ran in parallel
BEFORE any implementation, with go/no-go thresholds fixed in advance
(GO >=1.25x, GRAY 1.10-1.25x, ABANDON <1.10x, plus a skew veto requiring speedup_op(2) >= 1.6).

### T0b — real routing is MUCH more clustered than uniform (good news)
`COLI_M4_TRACE`-gated instrumentation, measured on real prompts:

| prompt | layer-chunks | unique experts/chunk | mean S | max S | % selections S>=2 | S>=4 |
|---|---|---|---|---|---|---|
| p064 (70 tok) | 86 | 58.99 | 3.560 | 63 | 87.5 % | 67.2 % |
| p256 (184 tok) | 129 | 92.91 | 3.961 | 64 | 90.4 % | 70.9 % |

Full 64-token chunks touch only **~93 unique experts vs the ~199 uniform-random expectation**,
mean group **S = 4.1**. My pre-registered uniform estimate of S~1.93 was pessimistic by ~2x.

### T0a — but the batch primitive cannot exploit it (the killer)
`coli_fp4_matmul_batch_ref` / `coli_fp8_matmul_batch_ref` **re-dequantize per batch item.**
Loop nesting in `c/quant.h:1431-1450`:

```
for (o ...)                     /* output rows */
  for (s = 0; s < S; s++)       /* batch  <-- dequant lives INSIDE */
    for (g ...) { c0 = mx4_scale(...);
      for (k ...) g0 += x * mx4_lut[weight]; }
```

Measured on M3 Max (16 OMP threads, real dims gate/up 2048x4096, down 4096x2048, no fast-math):

| S | fp4 f(S) ms | speedup_op | | S | fp8 f(S) ms | speedup_op |
|---|---|---|---|---|---|---|
| 1 | 1.023 | 1.000 | | 1 | 1.221 | 1.000 |
| 2 | 1.896 | **1.079** | | 2 | 2.235 | **1.093** |
| 4 | 3.240 | 1.263 | | 4 | 4.052 | 1.205 |
| 8 | 6.299 | 1.300 | | 8 | 7.336 | 1.332 |
| 64 | 44.264 | 1.479 | | 64 | 55.477 | 1.409 |

speedup_op saturates at ~1.48x instead of approaching S. The residual gain is loop/cache
overhead amortization, **not** dequant amortization.

### G0 decision: ABANDON as specified
Projected FFN speedup over the MEASURED distribution
(`sum(S_i*f(1)) / sum(f(S_i))`, p256, buckets S=1/2.5/5.5/11/27.1):
**1.2406x** -> GRAY, below the 1.25 GO bar. Skew veto also fires (speedup_op(2)=1.079 vs 1.6).
**No model code was written.** Cost of this decision: ~25 min of parallel measurement.

Note the veto's premise was itself wrong - it assumed singletons dominate, and they do not
(mean S = 4.1). The binding constraint is the primitive, not the routing.

### The pivot this exposes (E39 lane)
Because `f(S)` is dominated by re-dequantization, the correct target is **the primitive, not the
call site**. Hoisting the dequant out of the batch loop (dequantize a weight block once into a
small float buffer, then run S dot products against it) does **not** change arithmetic order, so
it is bit-exact-safe. Projected against the same measured distribution:

| marginal item cost after hoisting | projected prefill speedup |
|---|---|
| 1/2 of f(1) | **1.60x** |
| 1/3 of f(1) | **1.99x** |

vs 1.24x for grouping on the current primitive. Same routing data, far better lever, smaller
blast radius (`c/quant.h` only, the file already 4-row optimized in E33).

**Artifacts kept:** `c/bench_moe_batch.c` (standalone primitive benchmark) and the
`COLI_M4_TRACE`-gated routing histogram in `c/deepseek_v4.c`. Default build verified byte-identical
(60-token golden md5 `5d04890413ff539e802985ce8c727814`, 0 `m4_trace` symbols in the binary).

## E39. Dequant hoisting — **bitwise identical, better scaling, still NO-GO** (lane closed)

E38's pivot hypothesis: `f(S)` saturates because the batch kernel re-dequantizes per item, so
hoisting the weight decode out of the batch loop should push `speedup_op(S)` toward S.
Prototyped in `c/bench_moe_batch.c` (no kernel/call-site edits).

**Redundant work confirmed** — per (output row, 32-col group), each extra batch item repeats:
1 `mx4_scale` decode, 16 packed-byte loads, 32 nibble extractions, 32 `mx4_lut` lookups. Only the
32 multiply/adds are genuinely per-item. Prototype decodes 4 rows at once into 1552 B of
worker-local scratch.

**Bit-exactness: PROVEN.** 10 cases across S={1,2,4,8,64}, 475,136 output floats:
**0 bitwise differences, max abs 0, max rel 0.** Accumulation order is unchanged (32 serial
`x * decoded_weight` per output, then the group scale), so hoisting is arithmetic-order-preserving.

| S | current ms | hoisted ms | hoisted vs current | hoisted speedup_op |
|---|---|---|---|---|
| 1 | 0.796 | 1.537 | **0.518x** | 1.000 |
| 2 | 1.427 | 2.053 | 0.695x | **1.498** (was 1.079) |
| 4 | 2.648 | 3.116 | 0.850x | **1.973** (was 1.263) |
| 8 | 5.114 | 5.367 | 0.953x | 2.291 |
| 12 | 7.685 | 7.396 | 1.039x | 2.494 |
| 32 | 20.714 | 18.734 | 1.106x | 2.625 |
| 64 | 39.392 | 35.223 | 1.118x | **2.793** (was 1.479) |

Scaling improved exactly as predicted — but `f(1)` regressed 0.796 -> 1.537 ms (scratch
write/read plus changed traversal is pure overhead when there is nothing to amortize).
Crossover is **S ~ 10-12**; the measured distribution's dominant buckets are all below it.

Projected prefill speedup on the measured distribution (47472 selections):

| strategy | projected |
|---|---|
| grouping, current kernel | 1.179x |
| grouping, hoisted kernel | 1.010x |
| grouping, hybrid `min()` dispatch | **1.207x** |

Bar was 1.6x. Best achievable ~1.21x for substantial complexity. **LANE CLOSED.**

### The finding that redirects the campaign
My dequant hypothesis was **wrong**. Measured cause: *"FP32 dot-product work and per-item
activation QDQ dominate; nibble/LUT decoding is cheap."* The expert FFN is **compute-bound on the
dot products themselves**, not on weight dequantization and not on memory bandwidth (E38 had
already shown 5.34 GB/s is ~20x below DRAM). Consequently:
- Batching cannot help much — there is little redundant work to amortize.
- The only levers on expert FFN are **fewer FLOPs** or **better SIMD on the dot product**, and
  E33's 4-row matvec already took the latter.
This closes M4/M5-style batching as a family and points the remaining work at the still-scalar
leaf phases (router, head, hc_norm) and at GPU offload.

---

## E40. `router` kernel — **10.76x on the phase, opt-in; default provably untouched**

`router` was 4.8 % of steady-state decode doing only 256x4096 = 1.048 M MACs per call in 0.766 ms
(~1.37 GMAC/s). Same defect as E36/E37: `c/deepseek_v4.c` router hot loop accumulates into a single
`float sum`, a serial FP dependency chain clang may not legally reassociate without fast-math.
(For scale: the LM head does 529 M MACs/token in 33.8 ms because it is OpenMP-parallel.)

### Tier 1 (bit-exact) was a NEGATIVE result and was removed
Hypothesis: `route_bf16_decode` is a pure elementwise widening and therefore order-independent, so
NEON-widening the decode while keeping the accumulation strictly serial should win for free.
**Measured 0.997x microbench, 1.005x in-engine — nothing.** Apple clang already auto-vectorizes that
decode. Worse, the default build's decode_wall measured 3.83 % slower with Tier 1 present
(38721.3 -> 40205.2 ms). It was stripped entirely; three warm re-runs then restored baseline
(router mean 1960.3 vs 1961.3; decode_wall mean 38714.2 vs 38721.3), confirming Tier 1 caused it.
**The bottleneck is purely the serial accumulation chain — nothing else.**

### Tier 2 (opt-in, reassociated) is the win
4 independent NEON FMA accumulators + horizontal reduce, gated behind the EXISTING
`--fast-sparse-attn` flag (no second flag introduced; internal selector generalized to
`coli_v4_fast_reassociated_kernels()`), write-once file-scope static set during CLI parse,
ordered path always on non-arm64.

| microbench (M3 Max, no fast-math, 7 rounds) | ms/call | speedup |
|---|---|---|
| current | 0.729328 +/- 0.020374 | 1.000x |
| Tier 1 (bit-exact) | 0.731344 +/- 0.014628 | 0.997x |
| Tier 2 (reassociated) | 0.067984 +/- 0.001292 | **10.728x** |

Engine, 60-token warm runs, independently re-verified by me:

| build | router ms | decode_wall ms | md5 |
|---|---|---|---|
| default | 1955.475 | 38612.758 | `5d04890413ff539e802985ce8c727814` (**golden, bit-exact**) |
| `--fast-sparse-attn` | **181.814 (10.76x)** | 34197.534 (**-11.4 %**) | `7155bab905cbfa70aa06afa08f757cee` |

Diff `+120/-4`: a dispatch guard plus a new fast function. The original scalar loop and its three
`malloc`s are untouched, proven by the golden md5 on every default run. Canaries green
(`test_native_quant`, `test_ue8m0`, `test_e8_kernel`, `test_kv_prefix`, `test_deepseek_v4`,
tiny oracle, KV-prefix); x86_64 non-arm route path cross-compiles.

**Flag scope note:** `--fast-sparse-attn` now enables the whole reassociated-kernel set
(attn_sparse + router). Name kept for compatibility; wrapper script and docs updated to say so.

---

## E41. LM head NEON kernel — **11.7x on the kernel, 11% SLOWER overall. REVERTED.**

`head` was 4.9 % of steady-state decode (33.8354 ms/call, one call/token, vocab 129280 x hidden 4096
= 529 M MACs/token). `head_bf16_dot` (`c/deepseek_v4.c:7345`) has an AVX2 path that deliberately
vector-multiplies then does eight SEPARATE scalar adds to preserve summation order, and **no arm64
path at all** — Apple Silicon fell through to a plain scalar loop.

Applied the E40 playbook: 4 independent NEON FMA accumulators behind the existing
`--fast-sparse-attn` opt-in selector. (E40's lesson was applied up front — no "bit-exact vectorize
the decode" tier, since that was proven worthless there.)

**Kernel microbench: ordered 386.939 +/- 1.357 ms -> NEON4 33.043 +/- 0.103 ms = 11.71x.**
**In-engine phase: head 2080 -> 565 ms = 3.67x.**

### And total decode got 11.2 % SLOWER
Three warm runs per config, toggling ONLY the fast head (fast router + fast attn_sparse left on in both):

| flagged config | head ms | decode_wall ms |
|---|---|---|
| with fast head | 565 | **38142.3** (mean of 3) |
| without fast head | 2062 | **34307.9** (mean of 3) |
| E40 reference (pre-E41) | 2080 | 34197.5 / 34246.8 |

### Root cause — measured, not inferred
`accounted_pct` stays 98.5 % in both configs, so nothing is unmeasured. The time reappears in the
biggest phase:

| phase | default | flagged w/ fast head | delta |
|---|---|---|---|
| head | 2091.3 | 554.8 | **-1536.5** |
| router | 1956.9 | ~176 | **-1780** |
| **expert_forward** | 10594.4 | **12803.2** | **+2208.8** |
| **shared_expert** | 2406.1 | **2820.6** | **+414.5** |

Reverting the fast head snaps `expert_forward` straight back to 10548.7 / 10494.0 and the wall to
34514.0 / 34148.9 — matching the E40 reference. The toggle is definitive in both directions.

**Mechanism:** the LM head streams ~1 GiB of BF16 vocab weights per token. Running it 3.7x faster
pushes that GiB through the shared cache 3.7x faster, evicting the resident expert slabs, so
`expert_forward` — the single largest phase — must refetch from DRAM. The kernel gain is real and
is handed back with interest to a bigger phase.

**REVERTED. Nothing from E41 ships.**

### Campaign-level lesson (applies to all future work here)
**A phase-local speedup can be net-negative.** On this machine the phases are not independent: they
contend for shared cache and memory bandwidth. Any future kernel win MUST be validated on
`decode_wall`, never on its own `phase=` number — and ideally with a same-build A/B toggle, since
cross-build comparisons hide this effect. This is the third time this campaign that a
"clearly good" local change was wrong (E40 Tier 1 regressed the default 3.83 %; E39 hoisting
regressed f(1) 2x; now E41 regressed the wall 11 %).

---

## E42. Metal/GPU backend audit — **builds, links, and never executes** (investigation only, no changes)

Assessed the largest remaining lane (`expert_forward`, 26.6 % of decode) before proposing GPU work.

**What exists:** 7 shaders under `c/metal/` (incl. `coli_v4_moe.metal`, all touched 2026-08-13), a
seam header `c/backend_metal_v4_seam.h`, an Objective-C++ backend `c/backend_metal_v4.mm`, and
**three integration points at exactly the hot path** — `c/deepseek_v4.c:3357`, `:4041`, `:4087` —
each shaped as `if (coli_v4_metal_enabled() && coli_v4_metal_expert_forward(...) == 0) done = 1;`
with a CPU fallback.

**Findings:**
1. **`METAL ?= 0` (c/Makefile.deepseek-v4:97).** The default build links **zero** Metal symbols.
   Every measurement in this campaign was CPU-only.
2. `METAL=1` **builds cleanly**: compiles all 6 shaders to `build/metal-v4/deepseek_v4.metallib`
   and links `-framework Metal -framework Foundation` (8 Metal symbols).
3. **It never executes.** Same-build A/B with `COLI_V4_METAL=0` vs `=1`:

| config | expert_forward ms | decode_wall ms |
|---|---|---|
| METAL=0 | 10555.0 / 10464.5 | 38581.0 / 38312.8 |
| METAL=1 | 10569.4 / 10583.5 | 38663.9 / 38577.7 |

Identical within noise, no `COLI_V4_METAL_STATS=1` output, output md5 unchanged
(`5d04890413ff539e802985ce8c727814`). Also unchanged with `COLI_V4_AUTOPIN=0`
(expert_forward 5206.5 vs 5204.0 on a 30-token run).
4. **Two concrete causes identified:**
   - **Metallib path bug:** the build hardcodes
     `-DCOLI_V4_DEFAULT_METALLIB='"build/metal-v4/deepseek_v4.metallib"'` — a path relative to CWD.
     The file actually lives at `c/build/metal-v4/`, so running from the repo root resolves to a
     nonexistent path. (`COLI_V4_METALLIB` at `backend_metal_v4.mm:631` can override it.)
   - **Layout precondition:** `coli_v4_metal_expert_forward` (`backend_metal_v4.mm:332`) requires
     `coli_v4_metal_variant() == 0` and all three tensors to pass
     `coli_v4_ordered_cold_tensor_valid` (`:356-364`) — i.e. the **ordered-cold** layout. The
     runtime hot-packs experts into rows16 (`v4_rows16 packed_slots=722-902`), which is the layout
     E33's CPU 4-row kernel was built for. Setting `COLI_V4_METALLIB` explicitly still did not make
     it fire, so at least one precondition rejects every expert in this configuration.

**[CORRECTED 2026-08-14 by E43 — the conclusion below was WRONG.]** The "never executes" finding
was an artifact of a stale build: `make` does **not** recompile the 26 C units when the `METAL` flag
changes, so the `#ifdef COLI_V4_METAL_SEAM` call sites stayed compiled out while the Metal object
still linked (hence 8 Metal symbols and zero effect). `make clean` does not remove the objects
either. A true rebuild requires `rm -f c/*.o`. With that, Metal **does** execute — see E43.

**Original (superseded) conclusion:** GPU offload of `expert_forward` is not a greenfield build — substantial machinery
already exists but is inert. The next step is diagnosing *which* precondition rejects (add a
one-line reject-reason log), not writing new kernels. Recorded, no code changed; default build
restored to METAL=0 and re-verified against the golden md5.

### E38c (T0c). Chunk re-bin — larger chunks do NOT rescue M4

The one open question from E38 was whether raising the hardcoded 64-token prefill chunk
(`c/deepseek_v4.c:7547`) would lift group sizes enough to clear the bar. Re-binned the captured
routing trace (184 prompt tokens x 43 layers = 7912 token-layer rows, preserved at
`.backlog/m4_traces/`) at several chunk sizes, capping groups at 64 since the batch primitives do.
Projections use the measured `f(S)` curves from E38 (current kernel) and E39 (hoisted kernel).

| chunk | groups | mean S | max S | % selections S>=8 | proj current | proj hoisted | proj hybrid `min()` |
|---|---|---|---|---|---|---|---|
| 64 | 11986 | 3.96 | 64 | 46.9 % | 1.186x | 1.008x | 1.217x |
| 128 | 9349 | 5.08 | 64 | 58.4 % | 1.203x | 1.081x | 1.246x |
| 184 (whole prompt) | 6204 | 7.65 | 64 | 72.9 % | 1.224x | 1.182x | **1.283x** |

**Tripling the chunk buys +5.4 % of projected speedup (1.217 -> 1.283) for 2.9x the prefill buffer
memory.** The curve saturates because `speedup_op(64) = 1.479` is the hard ceiling of the current
kernel and mean S climbs slowly against 256 experts. Even the whole-prompt-as-one-chunk case only
reaches 1.283x, and that is a **zero-overhead projection** — it charges nothing for gather/scatter,
nothing for the hybrid two-kernel dispatch, and nothing for the E41 cache-contention risk that has
already turned one "certain" win into an 11 % regression.

**G0 verdict stands: M4 abandoned.** T0c closes the last open question against it rather than for it.

**Side effect — M5's gate is technically met but blocked.** M5 (`gpu_gather_moe`) was gated on
">=30 % of selections in groups S>=8"; measured **46.9 % at chunk 64**, rising to 72.9 % at 184. So
the routing data supports GPU-side grouping. However E42 established the Metal expert path never
executes today, so M5 is blocked behind diagnosing that, not behind routing telemetry.

---

## E43. Metal expert path — **it works, it is bit-exact, and it is 2.67x SLOWER** (lane closed)

Scoped by the user to "diagnose only, then report". Added env-gated relaxed-atomic reject counters
to `coli_v4_metal_expert_forward` (`c/backend_metal_v4.mm`) and made the metallib path absolute in
`c/Makefile.deepseek-v4` (it was CWD-relative; note `coli_v4_metal_init` has a compile-from-source
fallback, so that path bug was never fatal).

### First: E42's premise was a build artifact
`make` does not recompile the 26 C units when `METAL` changes, and `make clean` does not remove the
objects. So `make METAL=1` relinked the Metal backend while every `#ifdef COLI_V4_METAL_SEAM` call
site stayed compiled out — 8 Metal symbols present, zero calls made. Proven by the counters:
`ok=0` **and** `library kind=none`, i.e. the function was never invoked, not rejected.
**A true switch requires `rm -f c/*.o`.** I fell into this twice; it is now documented.

### With a true METAL=1 build, Metal fires
Same-build A/B, 30-token generation, warm:

| | expert_forward | decode_wall | metal counters |
|---|---|---|---|
| `COLI_V4_METAL=0` | 5188.6 ms | 18904.0 ms | `ok=0` (never called) |
| `COLI_V4_METAL=1` | **13841.8 ms** | **26726.4 ms** | `ok=7187 layout=5455 kind=metallib` |

- **56.9 %** of experts (7187/12642) dispatch to the GPU; **43.1 %** fall back to CPU.
- Rejections are **exclusively** `block_rows/block_columns` — the rows16 hot-packed layout.
  Every other field (format, pointers, rows, columns, row_bytes, groups) matches. So the layout
  mismatch is *fixable*, not fundamental.
- **Output is byte-identical** (`e451c131bc0e30aab4dbfffee2780ea4` both ways). The GPU path is
  numerically correct.
- **But it is 2.67x slower on the phase and 1.41x slower end-to-end.**

### Why — and why fixing the layout would make it WORSE
Per-GPU-dispatch cost works out to **~1.614 ms for one 13.37 MB expert**, versus **0.410 ms** for
the same expert on the CPU path. The GPU is ~4x slower *per expert*, dominated by per-dispatch
buffer binding rather than by the matmul. Since the rejected 43.1 % currently take the *fast* CPU
path, "fixing" the rows16 layout mismatch would push more work onto the slower path and make the
regression bigger.

**Verdict: lane closed.** Making Metal competitive would require batched multi-expert dispatch to
amortize the ~1.2 ms overhead — i.e. the M5 `gpu_gather_moe` design. E38 measured mean group size
S=4.1 at chunk 64 (7.65 at whole-prompt), so even perfect grouping amortizes ~1.2 ms over ~4-8
experts while the CPU path already runs at 36.5 GMAC/s. Poor expected value for a large rewrite.

Diagnostics and the metallib path fix are kept (inert by default). Default build re-verified:
26 objects, 0 Metal symbols, golden md5 `5d04890413ff539e802985ce8c727814`.

---

## E44. Per-kernel A/B harness + flag rename (tooling)

Built in response to E41, which cost **six benchmark runs across two rebuilds** to isolate a
phase-local win that was an 11 % wall regression — because toggling one kernel required a rebuild
and cross-build comparison hid the cache-contention effect.

**`COLI_V4_KERNELS`** now selects reassociated kernels individually at runtime from a bitmask
registry: `attn_sparse`, `router`, plus `all` / `none`. Parsed once before any thread is created
into a write-once file-scope static — no per-call `getenv`, no mutex, no mutable global in the hot
path. Non-arm64 always takes the ordered path. Every run prints `v4_kernels active=...` so a
benchmark transcript is self-describing. Unknown names **hard-fail** (exit 2) rather than silently
no-op, since a typo would otherwise let someone "measure" a kernel that never ran — the exact
failure this tool exists to prevent.

CLI: `--fast-kernels` is now the documented flag; `--fast-sparse-attn` is kept as an undocumented
alias so the wrapper script and existing invocations keep working. `COLI_V4_KERNELS` takes
precedence over the CLI flag.

Verified independently (60-token, warm, fresh `rm -f c/*.o` build):

| case | router ms | attn_sparse ms | md5 | active |
|---|---|---|---|---|
| default | 1932.6 | 2929.0 | `5d048904…` **golden** | `none` |
| `--fast-kernels` | 183.3 | 651.5 | `7155bab9…` | `attn_sparse,router` |
| `--fast-sparse-attn` (alias) | 185.4 | 657.8 | `7155bab9…` | `attn_sparse,router` |
| `COLI_V4_KERNELS=router` | **179.5** | **2967.0** | `e8041076…` | `router` |
| `COLI_V4_KERNELS=attn_sparse` | **1935.4** | **625.5** | `c6d8f26e…` | `attn_sparse` |
| `COLI_V4_KERNELS=bogus` | — | — | exit 2, names listed | — |

The two single-kernel rows are the proof: one kernel fast and the other slow **in the same binary**.
Note `COLI_V4_KERNELS=attn_sparse` reproduces md5 `c6d8f26ef47095bf6f777c11d99df080` — exactly the
E37 flagged result from before the router kernel existed, three experiments ago.

Adding a third kernel is one registry line (`X(HEAD, "head", 2)`), which automatically extends
parsing, `all`, the error message, and the active-set log. That is how E41's head kernel should be
re-tested if we revisit it.

---

## E45. Warm-cache measurement via persistent serve — **cache preserved, but 40x noisier; not a benchmarking win**

User asked (a) whether cache heating had unfairly penalised Metal, (b) for a way to preserve a
pre-heated cache between runs, and (c) that warm-cache performance be recorded as a first-class
number. This experiment answers (b) and (c).

### The only viable preservation lever is a persistent process
- Model is **155 GB on disk vs 128 GiB RAM** - it does not fit, so a full RAM disk is impossible.
- `hdiutil` RAM disks use **wired** memory, and swapping the storage medium would invalidate the
  benchmark against production APFS/SSD anyway.
- `posix_fadvise` does not exist on macOS; `madvise(MADV_WILLNEED)` applies to mapped ranges and
  the engine uses `pread`, not mmap. `F_RDADVISE` is advisory only; `vmtouch`/`mlock` change
  eviction behaviour and are benchmark-invalid.
- **Decisive:** expert reads set `F_NOCACHE` (`COLI_V4_DIRECT` defaults on), so the OS unified
  buffer cache is *deliberately bypassed*. Page-cache warming cannot help expert I/O at all.
=> Only keeping the engine process alive preserves anything. Serve mode does exactly that
(`SERVE=1`, one engine + one session, requests forever, READY in 0.333 s).

### Two confounds had to die first
1. **KV prefix reuse.** Re-sending a prompt lets the engine skip prefill entirely.
2. **My own false hypothesis:** I believed a single KV slot accumulates context across turns.
   It does not. `kv_prefix_reuse()` (`c/kv_prefix.h:116-127`) is **all-or-nothing** - reuse only
   when the entire prior fed sequence is a prefix of a *longer* new prompt; anything else returns
   0 and forces a **full attention reset + prefill from position 0** (`c/deepseek_v4.c:8477-8506`).
   The real confound was accounting: `hit_pct` spans the whole generate call **including prefill**
   (`:9174-9203`) while `tok_s` divides by `decode_sec`, which starts **after** prefill
   (`:9204-9217`). Rising hit_pct never implied warmer decode.

**Protocol adopted:** cycle a fixed set of 4 distinct prompts, repeat 4 cycles. Distinct within a
cycle forces a reset every turn; the same prompt recurring each cycle makes cycle k vs cycle 1 a
controlled comparison of the *same* workload at a hotter cache.
**Proof it worked:** the engine's `DONE` line carries a 10th field, `session->prefix_reused`
(`c/deepseek_v4.c:9213-9217`). It was **0 on all 16 requests** - every turn genuinely did a full
reset. (Finding that field also fixed a harness bug: the documented 9-field `DONE` is wrong, and
the engine's own comment says readers must accept `len(fields) >= 7`.)

### Result: the cache heats, and it buys nothing for decode
| | requests | hit_pct | tok/s |
|---|---|---|---|
| heating | 1-4 | 75.2 -> 91.1 -> 93.6 -> 95.0 | - |
| plateau | 5-16 | **95.41 % (sd 0.38)** | **1.449 (sd 0.157)** |
| coldest single request | 1 | 75.2 % | 1.517 |

Heating 75.2 % -> 95.4 % changed decode throughput by **-4.5 %, inside a 10.9 % noise band**.
Consistent with the accounting above: the cold misses are concentrated in **prefill**, and decode
was already hitting.

### And serve mode is unusable as a benchmark accelerator
| | relative sd |
|---|---|
| one-shot `decode_wall` (n=3) | **0.27 %** |
| serve plateau `tok_s` (n=12) | **10.9 %** |

**Serve mode is ~40x noisier.** Matching one-shot precision would need ~472 samples per config
(~6.5 h). Probable cause: the hot policy repins on an interval (`repin_interval=6`) and actively
repacks rows16 - `packed_slots=1894` here versus 722-902 in one-shot runs.

**Decision: do NOT adopt serve mode for benchmarking.** It preserves the cache but costs far more
precision than it saves wall-clock. One-shot remains the measurement vehicle, and the Metal
re-test (E46) must therefore use one-shot, not serve.

**Recorded operating points (decode-only throughput):**
- **cold** one-shot 60-token, hit ~78 %: **1.531 tok/s** (39194.0 ms, sd 107.2)
- **warm** serve plateau, hit ~95.4 %: **1.449 tok/s** (sd 0.157, noisy)

---

## E46. Metal fair re-test (warmed, amortized) — **CONFIRM: 2.84x slower. Batching gate FAILS 10x.**

The user challenged E43's "Metal is slower" verdict on the grounds that cache heating / warmup may
have unfairly penalised it, and asked whether a non-per-dispatch (batched) design could rescue it.
Both questions are answered here with pre-committed criteria.

### Method fixes over E43
- **One-shot, not serve.** E45 showed serve mode is ~40x noisier (10.9 % vs 0.27 % rel sd), so it
  would have swamped the effect.
- **Same binary, env toggle.** `c/deepseek_v4.metal` run with `COLI_V4_METAL=0` vs `=1` - a true
  same-build A/B. Verified equivalent: pure `c/deepseek_v4.cpu` and `.metal` with `METAL=0` produce
  the identical md5 `e8c7d8e6...`.
- **300 tokens, not 30** - 10x more work so lazy `coli_v4_metal_init`, first-dispatch pipeline
  build, scratch allocation and per-slab `newBufferWithBytesNoCopy` all amortize away.
- Interleaved runs; `COLI_V4_METAL_STATS=1` for timing (PROFILE distorts and was used only for
  stage attribution, in a separate pass).

### Verdict: CONFIRM (pre-committed: OVERTURN <=1.05x, CONFIRM >1.10x)
| | CPU (METAL=0) | Metal (METAL=1) | ratio |
|---|---|---|---|
| expert_forward | 59132.1 ms | 167856.6 ms | **2.84x slower** |
| decode_wall | 218256.8 ms | 331186.9 ms | **1.52x slower** |

**The warmup hypothesis is falsified**: at 30 tokens (E43) it was 2.84x/1.41x; at 300 tokens it is
**2.84x/1.52x**. Ten times more work made Metal *worse*, not better.

**Isolating the GPU from the hybrid** (46.7 % of experts reach the GPU; 53.3 % are rejected to CPU
on the rows16 `block_dims` mismatch): `c_gpu = (ef_metal - c_cpu * N_fallback) / N_gpu`
= **3.5475 ms/expert vs c_cpu 0.7185 ms/expert = 4.94x slower judged on its own.**

**Correctness note:** at 300 tokens `METAL=1` output **diverges** from CPU
(`b55c21ec...` vs `e8c7d8e6...`), though both are internally deterministic and were *identical* at
30 tokens - the same slow-accumulation pattern seen in E37/E41. So Metal is not bit-exact at length.

### The batching gate (user's proposal) — FAILS DECISIVELY
Proposal: stop paying one synchronous `waitUntilCompleted` per expert; enqueue the independent
experts of a (token,layer) - or a whole prefill chunk - into ONE command buffer.
*The user's refcount concern was already handled*: the expert store refuses to evict a slot with
live references (`c/deepseek_v4.c:5609-5614`), so holding N leases is safe by construction.

`COLI_V4_METAL_PROFILE=1` (forwards=7060) gives the split:

| | ms/expert |
|---|---|
| **weights zero-copy** | `zero_copy_tensors=42360`, **`copy_fallback_tensors=0`** |
| matmul gate + down | 2.053 |
| all other stages | 0.093 |
| command-buffer submit+wait (9 injected) | 1.985 |

Solving for round-trip R and real GPU compute C two independent ways:
- from the 8 *extra* profile commits: R = **0.073 ms**, C = **3.475 ms**
- from `submit_stage / 9`: R = **0.221 ms**, C = **3.327 ms**

**Dispatch overhead is only 2-6 % of the cost. GPU compute is ~3.3-3.5 ms/expert.**
Gate was: build if C <= 0.10 ms (decode) or C <= 0.35 ms (prefill). Measured C is **~10x above even
the permissive prefill threshold**. Sanity check: a *perfect* 6-way batch yields
`(0.073 + 6*3.475)/6 = 3.487 ms/expert`, still **4.9x slower than CPU**.

**=> The previously-scheduled E46 (prefill GPU batch) and E47 (dspark widening) are CANCELLED.**
Batching amortizes a cost that isn't there.

### Root cause, and what would actually be required
25.2 M MACs in ~3.33 ms = **~15 GFLOP/s on an M3 Max GPU** - roughly two orders of magnitude below
the hardware. The bottleneck is **the mxfp4/UE8M0 shaders themselves**, not the dispatch model, not
buffer copies (zero-copy already), and not warmup. Making Metal competitive would require rewriting
those kernels - a far larger project than batching, with the CPU path already at 36.5 GMAC/s.

**Metal lane closed on evidence, for the second and final time.**

---

## E47. Warm vs cold operating points — **cache heating does not speed up decode**

Requested: record warmed-cache performance as a first-class number, not just cold.

Protocol per E45: cycle 4 distinct prompts x 4 cycles in one persistent serve process. Distinct
prompts force a full attention reset every turn, **proven** by `session->prefix_reused = 0` on all
16 requests in *both* runs. Plateau = requests 5-16 (cache saturates after ~4).

| cell | hit_pct | tok/s (decode-only) | rel sd |
|---|---|---|---|
| COLD one-shot 60 tok, default | 77.97 | **1.531** | 0.3 % |
| COLD one-shot 60 tok, `--fast-kernels` | 78.51 | **1.700** | 0.7 % |
| MID one-shot 300 tok, default | 90.29 | 1.370 | 3.2 % |
| WARM serve plateau, default | **95.41** | 1.449 | 10.9 % |
| WARM serve plateau, `--fast-kernels` | **95.57** | 1.503 | 11.1 % |

**Heating curve (hit_pct): 77.97 -> 88.4 (E35, 220 tok) -> 90.29 -> 95.41.**

### The result: heating buys nothing for decode
Driving the expert cache from 78 % to 95.4 % leaves decode throughput **flat within noise**. The
controlled comparison (serve request 1 at 75.2 % vs plateau at 95.4 %, identical prompts and token
count) is **-4.5 %, inside a 10.9 % band**. Cause is structural, not a measurement artifact: the
cold misses are concentrated in **prefill**, and `hit_pct` spans prefill+decode while `tok_s`
divides by `decode_sec` only (`c/deepseek_v4.c:9174-9217`). Decode was already hitting.

**Caveat on the one-shot rows:** 60/300-token runs have different *average context lengths*, so
they are not a pure cache-temperature series - the MID row is slower than COLD largely because
attention cost grows with context, not because a warmer cache hurt. Only the serve rows isolate
temperature at fixed token count.

### fast-kernels holds up, but warm cannot resolve it
11.1 % faster cold (0.3-0.7 % sd, decisive) vs 3.7 % warm (±11 % noise, **not resolvable**). This is
why E45 rejected serve as a benchmarking vehicle: it is the right tool for *observing warm
behaviour*, and the wrong tool for *measuring kernel deltas*.

### Bug found while measuring: `COLI_V4_KERNELS` was a no-op in serve mode
`v4_serve_main` reads env directly and never calls `v4_cli_parse`, where the variable is parsed
(`:8149`). The first warm+fast run therefore silently measured the **default** kernels - it looked
like fast-kernels gave no warm benefit. Fixed (commit 587c309): serve now parses the same env with
the same validator, calls `coli_v4_kernels_set_active`, and emits `v4_kernels active=` so serve
transcripts are self-describing. Golden md5 unchanged. **An env var that works on one entry path
and silently no-ops on another is exactly the failure class E44's typo-rejection was built to
prevent - it just had a hole in it.**

---

## E48. The 158x gap resolved — **my "shaders are slow" claim was WRONG; batching gate now PASSES**

E46 closed the Metal lane on the claim that the mxfp4/UE8M0 shaders run at ~15 GFLOP/s. That claim
rested on per-stage numbers containing an unexplained contradiction: `matmul gate` 1.369 ms vs
`matmul up` 0.0087 ms for **identical dispatches** (same pipeline, same 2048x4096 shape,
`backend_metal_v4.mm:618-637`). A 158x gap between identical work cannot be "slow shaders".

**Discriminator** (user-approved cheap test): re-run the PROFILE pass at 100 tokens (1.98x the
dispatches of the 30-token run). First-touch costs amortize; real compute does not.

| stage | ms/expert @30tok | @100tok | total @30 | @100 | behaviour |
|---|---|---|---|---|---|
| matmul gate | 1.3689 | 0.6126 (**0.45x**) | 9.7 s | 8.6 s (~flat) | **one-time pool, amortizing** |
| matmul down | 0.6842 | 0.3119 (**0.46x**) | 4.8 s | 4.4 s (~flat) | **one-time pool, amortizing** |
| matmul up | 0.0087 | 0.0072 | 0.06 s | 0.10 s (scales) | true per-dispatch compute |
| submit+wait | 1.9848 | 2.2287 | 14.0 s | 31.2 s (scales) | per-dispatch |

**Verdict:** gate/down absorb a **first-touch cost** (GPU page-mapping / CPU-write coherency of the
`newBufferWithBytesNoCopy` expert slabs) on whichever dispatch touches an expert's pages first;
`up` runs immediately after gate on the *same* expert, pages already mapped, so it shows the truth:
**16.8 MFLOP / 0.00715 ms = 2.35 TFLOP/s. The shaders are FINE.** E46's root-cause claim is
retracted (its *measured outcome* — Metal slower as currently dispatched — still stands).

### Batching gate re-run with corrected C
C = 3 matmuls + aux stages ~= **0.086 ms**; R ~= 1.528 ms (from E43's 1.614 ms non-profile dispatch).
Gate was: build if C <= 0.10 (decode) / 0.35 (prefill). **C = 0.086 -> GATE PASSES.**

| design | ms/expert | vs CPU 0.719 |
|---|---|---|
| today (1 sync/expert) | 1.614 | 2.2x slower |
| decode batch=6 | (R+6C)/6 = **0.341** | **2.1x FASTER** |
| prefill batch=64 | (R+64C)/64 = **0.110** | **6.5x FASTER** |

### Caveat that must be verified before building
The one-time pool (~13 s across the first ~14k dispatches) is first-touch on ~7k slabs. If slot
EVICTION re-dirties pages, part of it recurs with cache misses. Evidence points at per-slab-one-time
(the pool FELL slightly while misses nearly doubled), but a longer NON-profile run must confirm
`c_gpu` actually approaches R+C before the projection is trusted.

### Incidental find: SIGKILL was code-signing, not memory
Both runs of this test initially died with `Killed: 9`. Crash report: `EXC_CRASH SIGKILL (Code
Signature Invalid), Taskgated Invalid Signature`. Cause: `cp` onto an EXISTING executable rewrites
the vnode in place and macOS keeps a per-vnode signature cache -> next exec is killed. Fix: `rm`
before `cp` (fresh inode). `build_toggle.sh` hardened accordingly.

---

## E49. First-touch caveat verification — **FAILS. The ~2 ms residual RECURS. Batching dead again.**

E48 passed the batching gate contingent on one unverified assumption: that the first-touch pool
(GPU page-mapping/coherency of expert slabs) is one-time-per-slab. The user approved verifying
before building. **The verification falsified the assumption.**

**Method — marginal cost, not average.** Cross-length *averages* are confounded (c_cpu itself
drifted 0.41 -> 0.73 ms across run lengths as rows16 hot-packing shifts). The *marginal* cost
between a 100- and a 500-token run cancels every one-time pool by construction. Four serial
non-profile runs, same `.metal` binary, `COLI_V4_METAL_STATS=1`:

| | METAL=0 ef | METAL=1 ef | ok | layout |
|---|---|---|---|---|
| 100 tok | 18106.4 | 59285.7 | 16413 | 14289 |
| 500 tok | 96609.6 | 253612.4 | 56340 | 77562 |

Marginal dispatches = 103200 = exactly 400 tokens x 258 (prefill cancels — sanity proof the method
works). Marginal `c_cpu` = **0.761 ms**; marginal Metal mixed = 1.883 ms at GPU fraction f = 0.387;
solving: **marginal `c_gpu` = 3.66 ms** — versus the 1.62 ms (R+C) prediction if first-touch were
one-time, and statistically identical to the 300-token *average* of 3.55.

**=> A recurring ~2.05 ms/dispatch cost exists that is neither sync (R=1.53) nor compute (C=0.086).**
Mechanism consistent with **CPU-write -> GPU-read coherency**: every cache miss rewrites a 13.37 MB
slab from the CPU, and the GPU must re-pull those dirty pages across the fabric. The CPU never pays
this — it wrote the bytes, they are warm in its own cache. (Conservative note: the CPU fallbacks in
the Metal run are the rows16-packed fast-kernel experts, so true c_gpu is if anything higher.)

**Batching verdict re-run with the recurring residual:** batching removes only R.
`batched-6 = (R + 6*(C+2.05))/6 = 2.39 ms/expert` vs CPU 0.761 -> **3.1x SLOWER. Dead.**
Prefill batching dies harder — prefill is majority misses, so the coherency tax is at its worst.

**What this leaves for fast-Metal on this machine:** the coherency tax is structural to the
current design (CPU writes shared slabs in place, GPU reads them). The one architecture that
removes it is a **GPU-resident expert cache**: experts uploaded ONCE per miss via async blit into
`MTLStorageModePrivate` buffers (GPU-optimal, never CPU-rewritten), evicted GPU-side. Upload cost
~13.37 MB per miss on the loader thread, hidden behind compute like today's SSD loads. That is a
substantially larger build than batching - it is the real M5 - and it is the ONLY remaining path.

Sequence of verdicts on this lane, each overturning the last's *mechanism* while the outcome held:
E43 "slow, dispatch-bound" -> E46 "slow, shader-bound" -> E48 "shaders fine, first-touch + sync" ->
**E49 "recurring coherency tax; only GPU residency removes it."** The measured outcome never moved:
Metal as currently architected loses to this CPU path.

---

## E50. Quality validation of `--fast-kernels` — **identical factual accuracy, 10/10 both configs**

Gate the user set for ever flipping the reassociated kernels to default: validate quality first.
10 prompts with objectively verifiable answers (capitals, arithmetic, physical constants, known
facts), 24 tokens each, scored by identical substring match on STDOUT (an earlier attempt filtered
stdout for a stderr-only pattern and returned 20 empty results — harness bug, fixed).

| config | correct |
|---|---|
| default (bit-exact) | **10/10** |
| `--fast-kernels` | **10/10** |

(The raw table shows 9/10 for both: the `lang_py` row was truncated to 60 chars BEFORE the substring
check, cutting "**Python**" to "**Py". Re-run untruncated: both configs answer Python. Scoring
artifact, corrected.)

On these prompts the two configs produce **visibly identical text**. The reassociated kernels'
output divergence (different md5 at 60+ tokens) has no measurable effect on short-form factual
accuracy. This satisfies the user's stated bar for considering a default flip; the flip itself
remains a user decision.

---

## E51. Prefill I/O stage-0 gates — **GATE A kills the deep-queue design and finds the real bottleneck**

Plan-gated lane (route-ahead + deep-queue burst prefetch). Stage-0 instruments built and run
BEFORE any engine code, per campaign law.

### Layout intel (bench/layer_contig.py, 11008 records verified)
**Each layer lives wholly in ONE shard** (`model-(L+2)-of-00048`); within a layer **all 256 scale
groups form one contiguous run and all 256 weight groups another** — 21930/21930 same-stream
neighbors byte-contiguous, zero violations. No FLOCK-packed whole records (scale stream separate).
Multi-record coalescing within a stream is trivially possible by layout.

### GATE A: QD sweep on real offsets (bench/qd_sweep.c) — STOP
| QD | GB/s |
|---|---|
| 1 | 5.227 |
| 4 | **7.028 (saturation)** |
| 8 | 7.029 |
| 16-32 | 6.66-6.63 (declining) |

`ratio_qd8_qd1 = 1.34 < 1.5` → **pre-committed STOP fired. The deep-queue burst design is dead**
— this SSD gives QD1 5.2 GB/s and saturates at ~7.0 GB/s by QD4. Latency scales linearly with QD
(p50 2.5 → 58.7 ms), so deep queues only buy queueing delay.

### But the sweep exposed the actual bottleneck (fresh instrumented p064 run)
| quantity | value |
|---|---|
| prefill wall (ttft, cold) | 42.70 s |
| payload read | 53.6 GB in 4257 direct reads |
| **effective I/O rate in-engine** | **1.25 GB/s** |
| same SSD, same pattern, QD1 (microbench) | 5.23 GB/s |
| raw I/O time at QD1 | 10.2 s |
| warm compute (18060 ops x 0.41 ms) | 7.4 s |
| ideal serialized | 17.7 s |
| ideal overlapped at QD4 | **~7.6 s** |
| **unexplained by I/O or compute** | **25.0 s (59 % of wall)** |

**The engine achieves 24 % of its own SSD's QD1 bandwidth during prefill.** The bottleneck is not
queue depth and never was — it is the per-op pipeline: route→handoff→load→compute serialized per
expert through a 3-slot condvar loader, slab alloc + mutex publish + hot-packer contention per
miss, plus buffered (non-direct) scale reads. Route-ahead remains the enabler (know the layer's
~93 experts before the loop), but the design revision is **continuous overlap at QD2-4**, not
deep queues.

**Revised honest projection:** p064 42.7 s → ~10 s (~4.3x); p256 113.7 s → ~25 s (~4.5x).
Baseline re-pinned this session: p064 43.554 s (sd 0.249), p256 113.735 s (sd 0.864), golden md5
confirmed. Engine work is HALTED pending user decision, since the pre-committed gate said stop
and proceeding on revised evidence is a scope decision, not a technical one.

---

## E52. Prefill route-ahead + overlap loader — **works, is bit-exact, gives 7-9.5 %. And E51's diagnosis was WRONG.**

Built the pivoted design (route-ahead + continuous overlap at QD2-4, after GATE A killed deep
queues). It works correctly and it under-delivers, because the bottleneck is not what E51 said.

### What was built (commits 1bfaf9b, 29561ce — both default OFF)
- **Route-ahead** (T6): after batched attention, before the per-token loop, route every token in the
  chunk and cache `idx/w` + the layer's deduplicated unique-expert set (mean 86.7/layer, range
  73-107). Verified bit-exact vs re-routing: **tokens=3010 ranks=18060 mismatches=0**.
- **Continuous-overlap loader** (T7): dedicated worker pool (QD default 4, clamped 1-8), separate
  from the global 3-worker decode pool, continuous refill rather than burst, per-expert waits,
  publishes through the existing hot path. **prefetched=3689 hits=3689** (every prefetch used),
  **leased_eviction_attempts=0**, max_inflight=4. The token loop consumes the cached routes
  (`:4446` re-routes only under VERIFY).
- **Golden md5 `5d04890413ff539e802985ce8c727814` holds with the gate ON and OFF** — pure I/O
  reordering, as contracted.

### Measured (p064, cold, ttft)
| config | ttft | vs its own baseline |
|---|---|---|
| OFF, default kernels | 43.006 s | — |
| **prefetch ON, default kernels** | **40.006 s** | **7.0 %** |
| OFF, `--fast-kernels` | 39.352 s | — |
| **prefetch ON + `--fast-kernels`** | **35.597 s** | **9.5 %** |

**GATE B (>=15 % on the feature) is NOT met: 7.0-9.5 %.**

### Why — E51's "84 % is SSD wait" was a bad inference, now retracted
E51 computed I/O time by subtracting *warm decode* compute (0.41 ms/op) from prefill's 2.50 ms/op
and attributing the remainder to I/O. That remainder is mostly **slow compute**, not I/O. Three
independent measurements say so:
1. **QD sweep in-engine**: QD1 41.966 s, QD4 39.571 s, QD8 39.990 s. Moving queue depth buys 2.4 s
   and then nothing. If I/O were 84 % of the wall this curve would be steep.
2. **Arithmetic**: 53.6 GB at the measured QD4 saturation (7.03 GB/s) is **7.6 s of a 43 s wall**,
   and it is fully overlappable.
3. **Residual per-op**: subtracting even the full I/O gives **1.96 ms/expert-op** (default kernels)
   versus **0.41-0.76 ms** warm — 3-5x higher.

**The real cause: in decode most experts are rows16 hot-PACKED and use the fast fused kernel; in
prefill they are cold/UNPACKED and take the slower fallback path.** Prefill is
**compute-bound on the unpacked-expert kernel**, not I/O-bound. That is why `--fast-kernels`
(router only) moved prefill 8.5 % on its own, and why deeper queues did nothing.

### Incidental finding
Layers 0-2 need **188/184/188** unique experts against a slot capacity of **164**, so those three
layers fall back to on-demand entirely (`v4_prefill_loader fallback`). 40 of 43 layers use the
prefetch path.

**Status: engine work paused at the pre-committed gate.** The feature is correct, bit-exact, and
default-OFF; it is worth 7-9.5 % if kept. The larger prize is now clearly the unpacked-expert
compute path, which is a different change.

---

## E53. Unpacked-expert kernel hypothesis — **DEAD at the gate. Ratio 1.099x, not the 4.8x claimed.**

Third mechanism hypothesis for the prefill/decode per-op gap, killed by a pre-committed gate
before any engine code was written.

### The hypothesis
Expert dispatch (`c/deepseek_v4.c:7188-7191`) falls back to `coli_v4_expert_forward_v17_fallback`
unless all three tensors have `block_rows == 16`; only *pinned* slots are ever packed
(`hot_pack_slot_locked`, gated on `hot_is_pinned` at :6853/:6919). Coverage is ~876-891 pack events
against 7052 slots. The fallback primitives (`quant.h:1427-1476`, `:1478-1549`) walk 4 output rows
with **scalar** accumulators; the rows16 arm64 kernel (`:12179-12203`) walks **16 output rows with
four `float32x4_t` NEON vectors**. So: prefill runs cold/unpacked on a scalar kernel, decode runs
hot/packed on a NEON kernel — plausibly the whole 4.8x.

### The measurement (bench/kernel_gap.c, real shapes 2048x4096 / 4096x2048, n=100 warm)
| path | ms/expert | GFLOP/s |
|---|---|---|
| fallback (4-row scalar) | 4.427 +/- 0.079 | 11.38 |
| rows16 (16-row NEON) | 4.027 +/- 0.099 | 12.51 |
| **ratio** | **1.099x** | — |

Kill criterion was **< 1.5x ⇒ stop**. **1.099x. Hypothesis dead.**
Bit-exactness confirmed as documented: max abs and max rel delta **0**, 0/4096 outputs differ.

**Packing also fails independently.** Pack cost 3.846 +/- 0.080 ms/expert (13.90 GB/s against the
4x-record_bytes traffic model), so `N_breakeven = pack_ms / (fallback_ms - rows16_ms) = 9.62 uses`.
A prefill chunk gives an expert only ~4.4 uses (64 tokens x 6 / ~87 unique). **Packing costs more
than it repays.** Both candidate fixes are dead on the same measurement.

### The contradiction the bench exposed (more useful than the verdict)
Single-threaded kernel cost is **4.0-4.4 ms/expert**, yet the engine measures **1.96 ms/op in
prefill and 0.41 ms/op in warm decode** — the engine is 2.3x/10.8x *faster than one thread*.
A thread cannot beat itself: **the engine parallelises the expert matmul** (CFLAGS carry
`$(V4_OMPC) -lomp`). Consequences:
1. The **ratio** stands — both kernels were measured identically, so the kernel is not the gap.
2. The **absolute** microbench numbers do not model the engine and must not be used to project any
   engine-level saving. Any future kernel work must be measured in-engine or with matched threading.

### Where this leaves the prefill gap
Prefill runs a **23.6 % miss rate** (4257/18060) versus 5-12 % in steady-state decode, and the
87.81 GiB expert cache *fills during prefill* — roughly 5.8 M first-touch page faults, ~8.6 s of
pure fault cost on the p064 timeline. Miss-handling (slab first-touch, slot select, publish under
the store mutex) is the remaining suspect.

**But I have now been wrong about mechanism three times** — E46 "shader-bound" (retracted by E48),
E51 "84 % SSD wait" (retracted by E52), and now E53 "unpacked kernel". Each time the *outcome*
measurement held and the *explanation* did not. **The next step must be direct instrumentation of
the miss path, not a fourth hypothesis.** No engine code was written for this lane.

---

## E54. Neutral in-engine attribution of prefill — **the measurement that should have come first**

After three wrong mechanism hypotheses, instrumented the engine directly instead of theorising.
`COLI_V4_PREFILL_TRACE` (compile-gated, inert by default, 31.8-32.7 ns/sample overhead, golden md5
unchanged) times every prefill stage and reports a table that **sums to the wall with a 1.55 %
residual**.

### Top-level, prefetch OFF, p064 (43.544 s wall)
| stage | ms | % |
|---|---|---|
| **MoE** | 26,147 | **60.05** |
| **Attention** | 14,449 | **33.18** |
| FFN norm | 983 | 2.26 |
| attention norm | 979 | 2.25 |
| HC post (x2) | 272 | 0.62 |
| head | 37 | 0.09 |
| residual | 676 | 1.55 |

### Inside MoE (store stages; nested worker totals overlap main-thread work)
| stage | ms | % of wall | calls | mean |
|---|---|---|---|---|
| **miss read + first touch** | 18,928 | **43.47** | 4,257 | 4.446 ms |
| miss pack | 1,330 | 3.06 | 483 | 2.755 ms |
| hit pack | 900 | 2.07 | 6,704 | 134 us |
| miss relock/publish | 669 | 1.54 | 4,257 | 157 us |
| hit mutex wait | 461 | 1.06 | 13,803 | 33 us |
| miss mutex wait | 231 | 0.53 | 4,257 | 54 us |
| miss slab alloc | 38 | 0.09 | 4,159 | 9 us |
| miss slot select | 20 | 0.05 | 4,257 | 5 us |

### What this KILLS (every prior suspect, quantified)
| suspect | lane | measured share |
|---|---|---|
| mutex serialization | E52-era | **1.59 %** |
| packing cost | E53 | 5.12 % |
| unpacked kernel speed | E53 | **0 %** (already killed at 1.099x) |
| prefill/decode thread-scaling gap | my 4th | **0 %** — prefill 0.6947 ms/op @39.28 % efficiency vs decode 0.7059 ms/op @39.51 %. **Expert compute is identical in both phases**; the "4.8x per-op gap" was an artefact of my aggregate arithmetic, not a real effect. |

### What it REVEALS
1. **Expert miss read + first touch is 43.5 % of prefill** (4257 misses x 4.446 ms). Confirmed as
   the single largest MoE component — the one thing prefetch actually targets.
2. **Attention is 33.2 % of prefill (14.4 s, 168 ms per layer-chunk call x 86)** and has been
   touched by **no lane in this entire campaign**. It is the largest untouched block in prefill.
3. **Route-ahead costs 3,386 ms (8.4 %)** — the overhead I added. MoE drops 26,147 -> 19,507
   (-6,640) but route-ahead gives back 3,386, netting the measured ~7 %.

### Measured feature stack (p064 ttft, cold)
| config | ttft |
|---|---|
| default | 42.780 s |
| `--fast-kernels` | 38.942 s |
| prefetch ON | 40.452 s |
| **prefetch ON + `--fast-kernels`** | **36.392 s (-14.9 %)** |

(I predicted the fast router would recover ~3.1 s of route-ahead overhead and lift prefetch to
~14.5 %; measured, prefetch saves ~2.3-2.6 s regardless of router speed. **Another projection that
did not survive contact** — the 14.9 % above is the two features stacked, measured, not modelled.)

**Rule earned the hard way: attribute inside the engine before proposing a mechanism.** Three
hypotheses and four suspects died to a table that took one instrumented run.

---

## E55. GATE B, properly measured — **FAILS on p256 (2.70 % vs 15 %), and the gain DECAYS with length**

I had only ever measured p064. The gate was specified on **p256**. Closing that gap changed the
picture, so the omission mattered.

`bench/ab.sh`, interleaved OFF/ON, n=3 per prompt, cold-cache each run:

| prompt | OFF | ON | delta |
|---|---|---|---|
| p064 (70 tok) | 43.065 s | 40.393 s | **-6.20 %** |
| **p256 (184 tok)** | **111.442 s** | **108.429 s** | **-2.70 %** |

**GATE B required >= 15 % on p256. Measured 2.70 %. Failed by a wide margin.**

### The decay is the real finding
The benefit **halves** as the prompt grows (6.20 % -> 2.70 %), which is the opposite of the naive
expectation (longer prompt = more misses = more to prefetch). E54's attribution explains it exactly:
attention is **33.18 % of p064** and is O(n^2) in context, so at p256 it consumes a larger share of
a larger wall while the MoE-read saving stays roughly fixed in absolute terms. A saving aimed at
expert I/O is a shrinking slice of a wall increasingly dominated by attention.

**For long prompts the lever is ATTENTION, not expert I/O.** That is now measured, not inferred.

### Directional check (T11) — PASSES
| stage | OFF % | ON % |
|---|---|---|
| MoE | 60.05 | **48.68** (down, as targeted) |
| attention | 33.18 | 35.48 (share up only because the wall shrank; absolute 14449 -> 14216 ms) |
| route-ahead | 0 | 8.45 (the cost the feature adds) |

Golden md5 `5d04890413ff539e802985ce8c727814` holds with the feature **ON** and **OFF**.

### Disposition
Per user decision the feature is **KEPT, default OFF** — it is bit-exact, zero-risk when unset, and
worth 6.2 % on short prompts / 2.7 % on medium ones. But GATE B failed on its own terms and that is
recorded as such: this is a kept-despite-failing-gate call made explicitly, not a passed gate.

### Harness defect found and fixed
`ab.sh` and `golden.sh` both carried the same latent bug as `rebaseline.sh` — `local status`
declared uninitialized then used in arithmetic under `set -u`. `ab.sh` aborted on it immediately;
`golden.sh` never reached that path. All three fixed; harness tests still pass; golden re-verified
after the patch.

---

## E56. Closing the T11/T12 gaps — bit-exactness at the *gate* lengths, and why interleaving saved the verdict

Re-examining the checklist skeptically found one item I had marked done on weaker evidence than it
required.

### The gap: "golden md5 stable across prompt lengths" was never actually tested
`bench/golden.sh` runs a **single hardcoded prompt** ("Write a detailed technical explanation of how
a mixture-of-experts transformer routes tokens."). Passing it proves bit-exactness for *that* prompt
only. The gates were fought on **p064 (61 words) and p256 (165 words)** — neither had ever been
checked for output equality. Determinism at one length does not imply it at another: the feature
changes *routing/prefetch order*, and the plausible failure mode is length-dependent.

Built the missing canary (golden's exact `ext()` extraction + snapshot/restore cold-state handling,
`--max-tokens 32`):

| prompt | OFF vs ON | md5 |
|---|---|---|
| p064 | **identical** | `12f5fac018e335613d4018e595e70703` (201 B) |
| p256 | **identical** | `4b146c969f31b97cefb5f5cfb251b207` (196 B) |

**Bit-exactness now holds at both gate lengths, not just golden's prompt.** The claim is finally
backed by the evidence it always needed.

### T12: default path verified, not assumed
The gate is fail-safe by construction:
```c
const char *enabled = getenv("COLI_V4_PREFILL_PREFETCH");
coli_v4_prefill_routeahead_enabled_value = enabled && *enabled && atoi(enabled) != 0;
```
Unset -> `NULL` -> OFF. Empty string -> OFF. `"0"` -> OFF. Any malformed value that is not a nonzero
integer -> OFF. Residual cost on the default path is one `getenv` at init. No perf regression: the
OFF arm (43.065 s) is *faster* than the pinned pre-feature baseline (43.554 s).

### The drift, and why GATE B's verdict survives it
The ab.sh OFF arms did not match the pinned baseline:

| prompt | pinned | ab OFF | drift |
|---|---|---|---|
| p064 | 43.554 | 43.065 | -1.12 % |
| p256 | 113.735 | 111.442 | **-2.02 %** |

The machine was ~1-2 % faster than when the baseline was pinned. This is exactly the hazard
interleaving exists to defeat — and had I compared ON against the **stale pinned baseline** instead
of the paired OFF arm:

| prompt | naive (ON vs pinned) | true (interleaved) | overstated |
|---|---|---|---|
| p064 | -7.26 % | -6.20 % | 1.17x |
| **p256** | **-4.67 %** | **-2.70 %** | **1.73x** |

I would have published a **1.73x inflated** p256 number. It still would have failed the 15 % gate, so
the verdict is unchanged — but the honest margin is 2.70 %, not 4.67 %.

**Rule earned: never diff against a baseline measured at a different time. Pair every A with its B.**

---

## E57. STOCK vs RECOMMENDED, measured end-to-end — **the +80.7 % composition is WRONG, and the sign is inverted**

The comparison nobody had run: `(--ram 48, no speculation)` against `(--ram 96, V4_DRAFT=4 V4_NGRAM=1)`
in **one interleaved session**. RESULTS.md quotes n-gram at +18.5 % and `--ram 48->96` at +33.8 %
cold / +52.5 % warm; I composed those to a **modelled +58.6 % cold / +80.7 % warm** and explicitly
labelled it unverified. It is now verified, and it is wrong.

### Design
ABBA (A1 stock, B1 reco, B2 reco, A2 stock), fresh engine per arm, `NTOK=64`, 4 coding prompts,
cold + warm, compressor gated **before and after** every prompt, frozen `.coli_usage`
(`599f3d12`, 29127 B) restored **and md5-verified before and after each arm**.

### Result — the recommended config is SLOWER
A2 vs B2, both on a quiet machine, both **8/8 gate-clean**:

| pass | # | stock (ram48, no spec) | reco (ram96, n-gram) | delta |
|---|---|---|---|---|
| cold | 1 | 0.3122 | 0.2936 | −6.0 % |
| cold | 2 | 0.4267 | 0.3765 | −11.8 % |
| cold | 3 | 0.6154 | 0.5766 | −6.3 % |
| cold | 4 | 0.5714 | 0.4384 | −23.3 % |
| **cold mean** | | **0.4814** | **0.4213** | **−12.5 %** |
| warm | 1 | 0.3879 | 0.3516 | −9.4 % |
| warm | 2 | 0.4238 | 0.4156 | −1.9 % |
| warm | 3 | 0.6154 | 0.5714 | −7.1 % |
| warm | 4 | 0.5565 | 0.5378 | −3.4 % |
| **warm mean** | | **0.4959** | **0.4691** | **−5.4 %** |

**0/4 wins for the recommended config in BOTH passes. 8/8 paired losses.**

| | modelled | measured | miss |
|---|---|---|---|
| cold | +58.6 % | **−12.5 %** | −71.1 pp |
| warm | +80.7 % | **−5.4 %** | −86.1 pp |

### Why the composition failed
The two source figures came from **different workloads**, and composing across them was invalid —
which I flagged as an assumption and which is now disproven, not merely doubted:

- n-gram's **+18.5 %** was measured on **10 fixed prompts on ONE topic, 24 tokens** — maximally
  repetitive, which is the ideal case for prompt-lookup drafting. This experiment uses **4 diverse
  coding prompts**, where drafts miss. §2 already recorded the mechanism: each rejected suffix
  forces a recurrent-attention replay, and that tax cancels the saved forward passes.
- the RAM figures came from a different session at 128 tok/prompt.

### CONFOUND, stated plainly
This arm moves **two variables at once** (48->96 GB *and* speculation off->on). It answers
*"is the recommended config faster than stock on this workload?"* — **no, it is 5-12 % slower** —
but it does **not** attribute that loss to RAM or to speculation individually. Splitting them needs
two more arms (`ram96 + no-spec`, `ram48 + spec`). **Do not cite this as evidence against `--ram 96`
alone or against n-gram alone.**

### The ABBA order prevented publishing a false positive
Ambient load was measured directly, on **identical config** (A1 vs A2, `--ram 48`, no spec):

| pass | busy (27.9 GB non-engine) | quiet (9.7 GB) | effect |
|---|---|---|---|
| cold | 0.3718 | 0.4814 | **+29.5 %** |
| warm | 0.4340 | 0.4959 | **+14.2 %** |

§11 documented a 22 % ambient swing; cold measured **29.5 %** here. Because the user freed memory
mid-run, A1 ran busy and B2 ran quiet. Comparing **A1 warm vs B2 warm gives +8.1 %** — a plausible
"directionally confirms the model, just smaller" headline. The truth from the matched-ambient pair
is **−5.4 %**. Without the second stock arm that false positive was publishable.

### Arm B1 died exactly as §11/§4d predict
B1 ran `--ram 96` before the desktop was freed (~27.9 GB non-engine, ~122 GB total). **7 of 8 rows
gate-failed**, compressor peaking at **73.1 GB** — the engine's own cache being squeezed. After the
desktop was freed (12.6 GB non-engine, ~107 GB total) B2 ran **8/8 clean** with gap ~0.1 GB. This is
the first time `--ram 96` has held gate-clean under a real workload on this host, and it confirms
§11's rule is about **engine + non-engine total**, not the engine budget alone.

### Harness defects found and fixed
1. `coldwarm2.sh` **hardcoded `V4_DRAFT=4 V4_NGRAM=1`** at the serve launch — the no-speculation arm
   was impossible to express. Parameterized (`V4DRAFT`/`V4NGRAM`, defaults preserve prior behaviour).
2. `coldwarm2.sh` **never froze `.coli_usage`** — no `COLI_V4_SAVE_USAGE=0`, no snapshot restore —
   while `prewarm_ab.sh:17-19` documents that as mandatory for multi-arm A/Bs. Caught ~2 minutes into
   the first launch and **restarted**; added `SAVEUSAGE` (default 0) plus before/after md5 assertion.
3. `libexec/colibri/deepseek_v4` was **stale** (Aug 13 `29ba3ead`) vs `c/deepseek_v4`
   (`1b658b15`) — the live §10 hazard. Every `bin/coli` harness would have measured old code.
4. `coldwarm2.sh`'s inline report crashes on tagged CSVs (`int('96_recoB1')`). Cosmetic; the
   standalone `ab_stock_reco_report.py` parses correctly.

### Standing
n=4 per pass, one workload, one host. The **direction** is solid (8/8 paired losses, zero gate
failures in either compared arm) but the magnitude is n=4. **The +80.7 % figure is retracted.**

---

## E58. Investigating the Metal verdict's own premises — **one of the four stated causes is FALSE on this host**

The Metal lane was closed by §12 with four structural causes and the conclusion that *"no backend
tuning changes the sign"*. Before planning any new architecture I tested the premises rather than
inheriting them. **One is false, a second is structurally wrong, and the remaining two hold.**

### E58a. "Unified memory gives the GPU no bandwidth advantage" — FALSE

`validation/probes/bw2.m`, one `MTLResourceStorageModeShared` buffer, same physical pages, GPU kernel
writes unconditionally (`out[gid] = sum`) so loads cannot be dead-code eliminated, and **both sides
are checksum-verified against the known exact sum**:

| buffer | CPU GB/s | GPU GB/s | ratio | checksum |
|---|---|---|---|---|
| 512 MB | 118.7 | 271.3 | 2.29x | match |
| 1 GB | 115.8 | 350.6 | 3.03x | match |
| 2 GB | 113.1 | 374.0 | **3.31x** | match |
| 4 GB | 115.0 | 367.3 | 3.19x | GPU match |
| 8 GB | 117.8 | 379.1 | 3.22x | GPU match |

The GPU checksum equals the expected value **exactly at every size**, so the loads really happened.
370–379 GB/s is ~93 % of the M3 Max's 400 GB/s spec. **The GPU pulls ~3.1x the bandwidth the CPU
can**; the CPU cannot saturate the fabric.

The CPU checksum diverges at ≥4 GB purely from float32 accumulator saturation —
`805306368 == 48 × 2^24` exactly (12 threads × 4 accumulators, each pinned at 2^24). Checksum
artefact only; identical loop, identical bytes, timing unaffected.

### E58b. "Prefill walks token by token" — STRUCTURALLY WRONG

`target_batch` (`c/deepseek_v4.c:8879`) is **layer-outer, chunk-inner**:
`for (layer) { for (offset += 64) { chunk = min(64, batch-offset) } }`, and **attention already
batches across those tokens** (`coli_fp8_matmul_batch_ref(..., batch)` at `:2592/:2632/:2644/:2772`).
**Only MoE discards the batch**: `:5110 for (item < batch)` → `:5141 moe_token_pipeline(...,
tokens[item])`. Up to **64 tokens are already in flight at every layer**. S=N therefore needs **no
prefill restructure** — the thing that made this look like a large project.

### E58c. The two causes that DO hold

- **S=1 is real.** The seam is one token in / one token out (`backend_metal_v4_seam.h:17`), with one
  blocking command buffer per token-expert (`backend_metal_v4.mm:415-715`).
- **Nothing is DRAM-bound today.** 13.37 MB per expert at 2.919 ms = **4.58 GB/s effective**, 26–80x
  below either processor's streaming bandwidth, and matching SSD QD1 (5.227 GB/s → 2.56 ms). **The
  miss path is SSD-bound and no compute placement fixes it.**

### E58d. Why S=1 loses, measured on the real shape

`validation/probes/{sweep_mxfp4.m,sweep2.m,cpu_mxfp4.c}`, gate/up I=4096 O=2048 MXFP4 blk32,
**microseconds per token**:

| S | CPU | GPU simple | GPU tiled+uchar4 | best GPU | verdict |
|---|---|---|---|---|---|
| 1 | 397.0 | 376.9 | 968.5 | 376.9 | GPU 1.05x |
| 2 | 325.5 | 290.1 | 304.8 | 290.1 | GPU 1.12x |
| 4 | 320.3 | 256.3 | 210.6 | 210.6 | **GPU 1.52x** |
| 8 | 219.8 | 212.4 | 163.5 | 163.5 | GPU 1.34x |
| 16 | 317.0 | 147.7 | **128.9** | 128.9 | **GPU 2.46x** |

**The CPU does not scale with batch (1.25x); the GPU scales 2.9x.** At S=1 the GPU edge is only
1.05x — trivially erased by per-call command-buffer construction and `waitUntilCompleted`, which
with 17.8 % coverage fully explains the shipped **1.118x slower**. **The fault is S=1, not the
kernel.** Both probes wall at ~114–130 GFLOP/s ≈ **3 % of fp32 peak**, so those GPU figures are a
*lower bound*.

### E58e. The decisive input, measured not assumed

The engine already logs unique-expert counts (`:5086`). p064 prefill, 86 layer-chunk records:

| chunk | selections | unique experts | **avg N** |
|---|---|---|---|
| 6 | 36 | 24.3 | 1.48 |
| **64** | **384** | **93.7** (min 73, max 188) | **4.10** |

Only **37 %** of the 256-expert space is touched per chunk. Uniform-random routing would touch ~202;
the actual 94 means routing is **heavily concentrated**, which favours batching. At chunk=64 the
prize is **4.1x fewer dispatches** (94 vs 384) and **4.1x less weight traffic** — landing at S≈4,
where the GPU measures **1.52x**.

### E58f. The honest ceiling (this killed my own first target)

Amdahl on the measured p064 attribution:
- expert path at `g_hit = 1.52`, hit rate `h = 0.577..0.677`: `1/((1-h) + h/1.52)` = **1.25–1.30x**
- untouched share: miss/first-touch 43.47 % + attention 33.18 % + norms/HC/head 5.22 % + residual
  1.55 % = **83.42 %**
- **full-prefill ceiling ≈ 1.20–1.25x**

My original acceptance gate was **1.3x — above the ceiling**. Withdrawn; the gate is now 1.12x.
Equally important: **the 3.08x bandwidth, the 4.1x traffic reduction and the 1.52x microbench are
NOT independent multipliers** — they are three descriptions of one effect, and the first draft
double-counted them.

### E58g. Stale references in the source document (all verified)

| `experiments-backlog.md` claim | reality |
|---|---|
| `moe_token_pipeline` at `:3579` | it is at **:4639**; :3579 is a gate-bias line in `moe_token` (:3559) |
| "three seams" | **four**: 3621, 4620, 4831, 4877 |
| one `coli_v4_expert_forward_ref` | **three** (7582, 7661, 10899); shipped is **:7661** (`COLI_V4_UNIT_EXPERT_ROWS16`) |
| batch cap "in three places" | **seven+**, incl. `__m256 sums[64]` (`:12355`) where removing the guard alone is a **stack buffer overflow**, the planner (`:1232`), and DSpark's 128-entry rings |

### Outcome
Plan written to `docs/plans/metal-batched-moe-architecture.md`, revised across **four adversarial
review rounds** (Momus, Oracle, and a third code-grounded reviewer), and approved **3/3**. Every
round found a real defect, including in my own fixes — the sign of `bench/ab.sh`
(`100*(on-off)/off`, so a speedup is **negative**, meaning my gates would have passed a slowdown),
a wave/loader deadlock, and a `loader_reserve` that could exceed a capacity which floors at 6.

---

## E59. Attention lane feasibility — **dense projections are 6.0x GPU-favorable, but only 7.8 % of the wall**

E54 established attention is **33.18 %** of prefill (14.4 s of 43.5 s) and untouched by any campaign.
Unlike MoE it looked structurally ideal for the GPU: it **already batches to S=64**
(`coli_fp8_matmul_batch_ref(..., batch)` at `:2592/:2632/:2644/:2772`) and its weights are **resident**
(dense set in RAM), so it has **none** of MoE's blockers — no SSD bound, no lease capacity, no
eviction risk. Probed it directly (`validation/probes/attn_fp8_sweep.m`).

### Real V4 attention shapes, dense fp8 (E4M3 + F32 128x128 block scales)

| shape (rows x cols) | S | CPU ms | GPU ms | GPU/CPU | CPU GF/s | GPU GF/s |
|---|---|---|---|---|---|---|
| `wq_a` 4096->1024 | 64 | 8.72 | 2.67 | 3.26x | 61.6 | 200.8 |
| **`wq_b` 1024->32768** | 64 | 51.82 | 7.51 | **6.90x** | 82.9 | **572.1** |
| `wkv` 4096->512 | 64 | 3.10 | 0.70 | 4.42x | 86.5 | 382.1 |
| `wo_b` 1024->4096 | 64 | 8.34 | 1.10 | 7.56x | 64.3 | 486.3 |
| **per-layer total** | 64 | **71.98** | **11.98** | **6.0x** | | |

At S=1 the GPU **loses** (0.16x–0.89x); crossover is around S=16. Same story as everywhere else in
this project: **batch size, not kernel quality, decides the sign.**

### Methodology note — I had to fix my own strawman first
The first run reported **up to 45x**. That was an artefact: my CPU reference called a scalar
`e4m3_decode()` per element, while the engine precomputes a **256-entry LUT** (`:12350`) and uses
SIMD. Re-running with a LUT + 4-way ILP dropped CPU from ~12 GF/s to **62–87 GF/s** and the ratio
from 45x to **6.9x**. The 45x number was never real and is recorded here only as a caution.

### Scoping the prize — and why this is NOT yet a win
- dense projections, CPU: 71.98 ms x 43 layers = **3.10 s per 64-token chunk**
- p064 = 70 tokens = chunk(64) + chunk(6) -> **~3.40 s**
- that is **24 % of the 14.4 s attention block**, and **7.8 % of the 43.5 s prefill wall**

Applying the measured 6.0x to *only* these projections:
```
saved = 3.40 * (1 - 1/6.0) = 2.84 s
full-prefill speedup = 43.5 / (43.5 - 2.84) = 1.070x
```
**7.0 %. Not enough on its own**, and below the 1.12x gate the batched-MoE plan already commits to.

### The decisive unknown
**What is the other 11.0 s of the attention block?** Candidates: QK^T, softmax, AV, RoPE, the DSA
sparse indexer, and the recurrent compressor (`:2603`) / KV ring (`:2688`).

- If those are dense and batchable -> the attention lane is **substantially larger than the MoE lane**
  (which measures 1.52x at the real N=4.10).
- If they are recurrent/sequential -> they are **not batchable at all** and this lane is capped near
  1.07x.

**These two outcomes differ by an order of magnitude, and nothing measured so far distinguishes them.**
Per the rule this project earned the hard way — *attribute inside the engine before proposing a
mechanism* (E48/E52/E53/E54 each died for violating it) — **B3 attention attribution is now the
critical next measurement.** No attention work should be scoped until the 11.0 s is broken down.

---

## E60. The MoE GPU kernel was **occupancy-limited, not format-limited** — and that reframes the lane

E58d measured my MXFP4 probes walling at ~114–130 GFLOP/s ≈ 3 % of this GPU's fp32 peak, and I
recorded those GPU figures as a *lower bound*. E59's attention probe then hit **572 GFLOP/s** on
dense fp8 with an otherwise similar per-element decode. That 4.4x gap was the clue.

**Cause: dispatch geometry, not the quantisation format.** The MXFP4 probes dispatched **1D over
output rows only — 2048 threads on a 40-core GPU**. The fp8 attention probe dispatched **2D
`(rows × S)`**, up to 2.1 M threads. Re-dispatching MXFP4 as 2D (`validation/probes/sweep3_2d.m`):

| S | v2 (1D) GFLOP/s | **v3 (2D) GFLOP/s** | v3 us/token |
|---|---|---|---|
| 1 | ~45 | 23.0 | 729.0 |
| 4 | ~80 | 81.6 | 205.6 |
| 8 | ~103 | 122.4 | 137.0 |
| 16 | ~130 | 147.3 | 113.9 |
| 32 | — | **275.1** | 61.0 |
| 64 | — | **421.6** | **39.8** |

**3.2x past the old wall**, and us/token falls to 39.8 (vs v2's best 128.9). MXFP4 was never the
problem.

### What this changes
**The kernel is not the bottleneck — N is.** The measured N = 4.10 (E58e) sits in the *flat* part of
this curve (205.6 us/token at S=4), while the real wins live at **S=32–64** (61.0 / 39.8 us/token).
Improving the kernel further buys little at N=4.10; **raising N is the whole game**.

That promotes the chunk-cap lever (plan T9) from "nice extra" to **the primary performance lever of
the MoE lane** — and the cap is a code constant in seven+ places, one of which (`__m256 sums[64]`,
`:12355`) is a stack-overflow hazard if raised naively.

### Honest limitation — my extrapolation of N is NOT trustworthy
I modelled unique-experts-vs-tokens by calibrating a coupon-collector curve to the single measured
point (64 tokens → 93.7 unique). It predicts unique **saturating at ~95**, which would give N ≈ 16
at 256 tokens and N ≈ 50 at 799. **But the measured per-layer maximum at chunk=64 was already 188**,
which the model cannot produce. The model is overfitted to the mean and contradicts the observed
spread (min 73, max 188).

**So the N-vs-chunk curve is UNMEASURED.** Any claim that chunk=256 yields N≈16 is speculation of
exactly the kind this project has repeatedly punished (E48/E52/E53/E54, and the +80.7 % composition
in E57). **T9 must measure N at each chunk value directly from the engine's own `unique=` log
before any performance claim is attached to it.** Recorded here as a hypothesis with a named
measurement, not as a result.

---

## E61. T-1 CPU-only grouped MoE — **K0 PASSES, bit-exact, +4.3 % — and the gain decays with length**

The plan's Wave-0 kill test, implemented behind `COLI_V4_MOE_GROUPED=1` (+422 lines,
`c/deepseek_v4.c`). No GPU, no new kernel: the MoE loop is reordered **expert-outer / token-inner**,
with the unique expert list **sorted ascending**, capacity preflight, wave sizing with the
`effective_reserve` clamp, and rollback. Default OFF is untouched.

### Correctness — the gate that actually matters

| run | result |
|---|---|
| grouped **OFF** (default path) | `PASS golden md5=5d04890413ff539e802985ce8c727814` |
| grouped **ON** | `PASS golden md5=5d04890413ff539e802985ce8c727814` |

**Identical md5 — bit-exact.** Confirming it was not a silent fallback (the failure mode that once
made a non-running Metal seam look successful, §12e):

```
moe_groups         5,073     grouped path genuinely executed
moe_waves             89
moe_wave_fallbacks     0     no degenerate or rollback fallbacks
```

Every number cross-checks against independent measurements:
- **5073 / 86 chunk-calls = 59 groups per chunk**, exactly the mean of the measured `n_unique`
  (93.7 at chunk=64, 24.3 at chunk=6 → 59).
- capacity 164 − reserve 16 = **wave size 148**, so only chunks with `n_unique > 148` need a second
  wave — and `89 − 86 = 3` extra waves is exactly the count of chunks above that line (max 188).

### K0 gate

```
moe_group_overhead_ns / moe_chunk_ns = 2,323,935,774 / 23,902,732,376 = 0.0972  (9.72%)
K0 threshold                          = 0.167  (= 1 - 1/1.20)
```
**PASS.** The scheduler costs 9.72 % of the MoE phase, well inside the available headroom, so it
cannot eat the GPU win. **Metal work is authorised.**

### E13 — interleaved A/B, n=3, sign-correct

`bench/ab.sh` prints `100*(on-off)/off`, so **faster is negative**:

| prompt | off | on | delta | speedup | pairs won |
|---|---|---|---|---|---|
| p064 | 42.399 s | 40.639 s | **−4.15 %** | **1.043x** | 3/3 |
| p256 | 110.125 s | 107.826 s | **−2.09 %** | **1.021x** | 3/3 |

**A free, bit-exact +4.3 % on CPU with no GPU involved** — pure scheduling and locality.

### The decay is the finding
The gain **halves** from p064 to p256 (4.15 % → 2.09 %) — the *same* pattern as the prefill-prefetch
feature (E55: 6.20 % → 2.70 %), and for the same reason. Chunking is capped at 64, so a longer
prompt gets **more chunks, not more tokens per expert**: N stays pinned at 4.10 while attention
(O(n²)) grows and dilutes the MoE share.

**This is the direct experimental confirmation of E60's conclusion: N is the bottleneck, not the
kernel and not the scheduler.** Grouping at N=4.10 is worth ~4 %; the E60 kernel curve shows the
real wins at S=32–64. Raising N — not optimising anything currently in the path — is the next move.

---

## E62. Measuring N at whole-prompt scope — **decoupling the MoE batch would nearly double N (4.14 → 7.99)**

E60 concluded N is the bottleneck and E61 confirmed it experimentally (grouping is worth only ~4 %
at N=4.10). E60 also **flagged my own coupon-collector extrapolation as untrustworthy**. This
measures it instead.

### Method
Added `COLI_V4_MOE_GROUPED_DUMP=1` (measurement-only, behind an env, default path verified still
`PASS golden md5=5d04890413ff539e802985ce8c727814`) which dumps each chunk's **sorted unique expert
ids**. Ran p256 (184 tokens → 3 chunks × 43 layers = 129 records) and computed, offline, the
**union across a layer's chunks** — i.e. what the expert set would be if the MoE batch were the
whole prompt instead of one 64-token chunk.

### Result — 43 layers

| scope | mean unique per layer | **mean N** |
|---|---|---|
| chunk (today, 64-token) | sum 541/526/546… | **4.14** |
| whole-prompt (184-token) | union 226/224/230… | **7.99** |
| | | **1.93x higher** |

The chunk-scope figure **4.14** independently reproduces the 4.10 measured in E58e from a different
code path — a good cross-check.

Applying E60's v3 kernel curve (interpolated, not bucketed):
```
N=4.14  -> 203.2 us/token
N=7.99  -> 137.2 us/token
=> 1.48x better per-token expert cost from the SAME kernel, purely by batching more tokens
```

### My extrapolation was wrong, and flagging it was the right call
E60's model predicted **N ≈ 16 at 256 tokens**. Measured: **N = 7.99 at 184 tokens** — the model
**over-predicted by roughly 2x**. It assumed unique saturates near ~95, whereas the true
whole-prompt union reaches **226 of 256** on early layers. Had I designed against the model instead
of measuring, the wave count and capacity budget would both have been badly wrong.

### Structure discovered: concentration varies by depth
- **layers 0–2**: union **226–230** experts — routing is nearly *uniform*, barely concentrated
- **layers 3–5**: union **126–151** — substantially concentrated

So the "routing is heavily concentrated" conclusion from E58e is **true on average but false for the
early layers**. Any capacity or wave planning must use the *per-layer* figure, not the mean.

### Capacity consequence — waves become the norm, not the exception
```
whole-prompt union up to 226 ; capacity 164 ; wave size 164-16 = 148
=> 226/148 = 2 waves on early layers (vs 1 wave for most chunk-scope calls today)
weights touched per layer = 226 x 13.37 MB = 3.02 GB  (fits the cache comfortably)
```
This is exactly the regime the plan's §4.8 wave machinery was designed for, and E61 already proved
that machinery works (`moe_waves=89, moe_wave_fallbacks=0`).

### Verdict
Decoupling the MoE batch from the attention chunk (notepad F19) is **worth ~1.48x on expert kernel
cost and needs no change to any of the seven batch-cap sites** — attention keeps its 64-token chunk,
its recurrent compressor and its causal KV ring untouched. It is the highest-value next change in
the MoE lane, and it is now backed by measurement rather than a model.

---

## E63. B3 prefill-attention attribution — **the dense projections are 41.8 % of attention, and the lane clears the gate**

E59 measured the four dense attention projections at **6.0x GPU-favorable** but estimated them at
only 24 % of the attention block, giving 1.07x — below the gate. It flagged the other ~11 s as the
decisive unknown. This measures it.

### First: the existing profiler could not answer this
`COLI_V4_PROFILE` already defines `attn_qkv / attn_sparse / attn_kv_assembly / attn_out /
compressor / indexer / rope`. Running it returned **all zeros with `calls=0`** — those
`profile_add` sites are at `:1947-2097`, inside the **single-token decode** path. The **batched
prefill attention** function has none. So B3 required new instrumentation
(`COLI_V4_ATTN_STATS=1`, measurement-only, default path re-verified at
`PASS golden md5=5d04890413ff539e802985ce8c727814`).

### Result — p064, 86 attention calls

```
attn_total_ms   = 14545.1     <- E54 independently measured 14.4 s. CROSS-VALIDATED.
attn_parts_ms   = 10674.2
attn_residual   =  3870.9  (26.61 %)
```

| stage | ms | % of attention | batchable? |
|---|---|---|---|
| `proj_out` (wo_b) | 2683.9 | 18.45 % | **yes** (E59: 7.56x) |
| `proj_wq_b` | 2626.3 | 18.06 % | **yes** (E59: 6.90x) |
| `sparse_core` | 2512.4 | 17.27 % | no — per item |
| `compressor_indexer` | 2060.2 | 14.16 % | **no — recurrent** |
| `proj_wq_a` | 476.4 | 3.28 % | **yes** (E59: 3.26x) |
| `proj_wkv` | 298.0 | 2.05 % | **yes** (E59: 4.42x) |
| `rope` | 17.0 | 0.12 % | — |
| **residual** | **3870.9** | **26.61 %** | unattributed |

**Dense projections total 6.085 s = 41.8 % of attention = 13.4 % of the prefill wall** — not the
24 % / 7.8 % E59 estimated. E59's probe under-counted because `coli_fp8_matmul_batch_ref` performs
**per-token fp8 QDQ inside itself** (`:12336-12342`), which the standalone probe omitted.

### What the lane is worth

| GPU speedup on projections | saves | full-prefill |
|---|---|---|
| 3.26x (worst single shape) | 4.22 s | 1.102x |
| **6.00x (E59 measured aggregate)** | **5.07 s** | **1.126x** |
| 7.56x (best single shape) | 5.28 s | 1.132x |

**At the measured 6.0x the projections alone give 1.126x, which clears the plan's 1.12x gate** —
without touching `sparse_core` at all. And this is a *different* part of the wall from the MoE lane
already shipping 1.043x (E61), so they should compose (upper bound ~1.17x — **to be measured, not
assumed**; E57 is the standing lesson on multiplying independent-looking deltas).

### Two honest limitations
1. **Residual is 26.61 %.** E54 closed its attribution to a **1.55 %** residual; 26.6 % is too
   coarse to call finished. Unmeasured inside it: qnorm, KV assembly, the second RoPE apply, the
   per-group `wo_a` matmul, and allocation churn (the function `calloc`s ~9 buffers per call, 86
   times). Some of that residual may itself be cheap to remove.
2. **`compressor_indexer` = 14.16 % is recurrent** (`:2651` carries `kv_state`/`score_state`) and
   `sparse_core` = 17.27 % is per-item behind a causal KV-ring dependency (`:2694`). Together
   **31.4 % of attention is structurally hard to batch** — that part of E59's worry was justified.

---

## E64. Complete attention attribution (residual 0.21 %) — **`wo_a` was hidden, and the lane is worth 1.23x**

E63 left a **26.6 % residual**, far coarser than the 1.55 % E54 achieved, and I flagged it as too
coarse to call finished. Closing it changed the conclusion materially.

### Two hypotheses tested, one wrong
I suspected **allocation churn** — the function `calloc`s ~9 buffers per call including two of
**8.4 MB** (`q`, `attended`), 86 times. Measured: **`alloc_free` = 2.2 ms = 0.02 %**. **Hypothesis
wrong.** `qnorm` was likewise negligible at 0.63 %.

The residual was almost entirely **one unmeasured stage**.

### Full attribution — p064, 86 calls, residual **0.21 %**

```
attn_total_ms = 14553.0     attn_parts_ms = 14522.9     residual = 30.1 ms (0.21%)
```

| stage | ms | % of attention | batched over tokens? |
|---|---|---|---|
| **`proj_wo_a`** | **3734.7** | **25.66 %** | **yes** — was invisible in E63's residual |
| `proj_out` (wo_b) | 2798.3 | 19.23 % | yes |
| `proj_wq_b` | 2542.4 | 17.47 % | yes |
| `sparse_core` | 2514.8 | 17.28 % | no — per item, causal KV ring |
| `compressor_indexer` | 2046.1 | 14.06 % | **no — recurrent** |
| `proj_wq_a` | 470.2 | 3.23 % | yes |
| `proj_wkv` | 292.2 | 2.01 % | yes |
| `qnorm` | 92.3 | 0.63 % | — |
| `rope` | 17.8 | 0.12 % | — |
| `kv_assembly` | 12.0 | 0.08 % | — |
| `alloc_free` | 2.2 | 0.02 % | — |

`wo_a` is the **per-group** output projection (`:2826`): `coli_fp8_matmul_batch_ref(group_outputs,
&group_view, group_inputs, batch)` invoked once per output group. It is already batched over tokens
— it was simply never timed.

### Corrected value of the attention lane

**E63 said projections were 6.085 s / 41.8 %. That was wrong — it missed `wo_a`.**
All five projections total **9.838 s = 67.6 % of attention = 22.6 % of the prefill wall.**

| GPU speedup (E59 measured range) | saves | **full prefill** |
|---|---|---|
| 3.26x (worst single shape) | 6.82 s | **1.186x** |
| **6.00x (measured aggregate)** | **8.20 s** | **1.232x** |
| 7.56x (best single shape) | 8.54 s | **1.244x** |

**Even the worst-case shape clears the 1.12x gate.** At the measured aggregate the attention lane is
worth **1.23x on its own** — roughly *five times* the MoE grouped path's measured 1.043x (E61), and
it targets a completely different part of the wall.

### What remains structurally hard
`sparse_core` (17.28 %) is per-item behind the causal KV-ring dependency (`:2694`), and
`compressor_indexer` (14.06 %) is genuinely recurrent (`:2651` carries `kv_state`/`score_state`).
**Together 31.3 % of attention / 10.5 % of the wall is not batchable** by this approach. E59's
concern was justified for that third — but the other two thirds are exactly the shape the GPU wants.

### Priority change
This **overtakes the batched-MoE plan as the highest-value lane**: 1.23x vs the MoE lane's
projected ~1.10-1.15x, on weights that are **resident** (no SSD bound, no lease capacity, no
eviction risk, no wave machinery). The MoE work already landed is kept — it is bit-exact and free —
but the next implementation effort should target the five attention projections.

Regression: default path re-verified `PASS golden md5=5d04890413ff539e802985ce8c727814` with all
12 timing slots compiled in.

---

## E65. **Metal has no fp64, and the attention fp8 path accumulates in `double`** — a blocker found before writing the kernel

E64 made the attention projections the top lane (1.23x). Before writing any Metal kernel I checked
what bit-exactness would actually require, and hit a hard constraint.

### The two paths are NOT symmetric

| path | accumulator | source |
|---|---|---|
| `matmul_mxfp4` — **MoE** | **`float a0,a1,a2,a3`** | `quant.h:1438` |
| `matmul_fp8` — **attention** | **`double a0,a1,a2,a3`** | `quant.h:505` |

`matmul_fp8` keeps a `float` accumulator *within* each 128-column block, then accumulates across
blocks in **`double`**, casting to `float` only at the very end (`quant.h:505-521`):

```c
double a0=0, ...;
for (bi ...) {                       /* 32 blocks for I=4096 */
    float acc0=0, ...;
    for (i ...) acc0 += e4m3_decode(w0[i])*xv;   /* 128 iters, float */
    a0 += (double)acc0*scl0[bi];                 /* double */
}
y[s*O+o] = (float)a0;
```

### Metal cannot do this

```
program_source:3:22: error: 'double' is not supported in Metal
```
Verified by compiling a trivial `device double*` kernel on this device. Apple Silicon GPUs have **no
fp64 at all** — it is a compiler error, not a slow path.

**Therefore a straightforward Metal fp8 kernel cannot be bit-exact with the CPU reference**, and the
project's bit-exactness contract is non-negotiable (K1/K2 in the plan). Had this been discovered
after writing the kernel and host glue, the whole lane would have been wasted work.

### Why it is probably not fatal — the double work is only 1/128th of the arithmetic

| per output element (I=4096) | ops |
|---|---|
| inner loop (stays `float`) | **4096** float MACs |
| outer accumulation (needs `double`) | **32** double MACs |

Emulating just those 32 operations with **double-float arithmetic** (a hi/lo float pair, Dekker
`TwoProduct` + `TwoSum`) costs roughly 10–30 float ops each, i.e. **2.4–7.3 % overhead** on top of a
kernel that already measured **6.0x faster than the CPU** (E59). The expensive inner loop stays
pure `float`.

### The honest catch, and the next experiment
Double-float carries **~48 mantissa bits vs double's 53**. The result is cast to `float` (24 bits)
at the end, so the two will *usually* round identically — but **"usually" is not "bit-exact"**, and
double-rounding can differ in the last ULP.

**This must be settled empirically, not by argument.** The next probe compares, over the real shapes
and many random seeds, `(float)double_accumulate(...)` against `(float)doublefloat_accumulate(...)`
and counts differing outputs. If the count is exactly zero across the real reduction lengths, the
attention lane is viable at full bit-exactness; if not, the lane is capped at whatever
`sparse_core`-free approximation the F-rule would permit — which this project has historically
refused to ship.

### Consequence for priority
The **MoE lane is unaffected** — `matmul_mxfp4` accumulates in `float`, so a Metal kernel can
reproduce it exactly, and E60 already showed the kernel reaching 421.6 GFLOP/s. So the ranking is
now conditional:
- if double-float proves exact -> **attention lane (1.23x) leads**
- if not -> **MoE lane leads** by default, since it has no such obstacle

---

## E66. Double-float **is** bit-exact for this computation — except when the product underflows

E65 identified that Metal has no fp64 while `matmul_fp8` accumulates in `double`, and proposed
double-float (hi/lo float pair) emulation. This tests it instead of assuming.

`validation/probes/df_vs_double.c` — Dekker `TwoProduct`/`TwoSum` (exact via `fmaf` on ARM),
compared against the real `double` accumulation, both cast to `float` at the end, bits compared.

### Random testing gave a FALSE GREEN
| mode | trials | differing |
|---|---|---|
| benign magnitudes | 300 000 | **0** |
| wide dynamic range (2^-60..2^60) | 300 000 | **0** |
| near-cancellation (±1e7 alternating) | 300 000 | **0** |
| **denormal-ish inputs** | 300 000 | **149 825 (49.94 %)** |

Three seeds on wide-range: 0 differing each. Had I stopped at benign random data — the obvious
thing to do — I would have concluded "exact" and been wrong.

### The exact threshold, and why it is not where I expected
Exponent sweep (`df_threshold.c`, 32 blocks = real I=4096):

| input magnitude | differing |
|---|---|
| 8.67e-19 (2^-60) and above | **0 %** |
| 8.47e-22 (2^-70) | 61.2 % |
| 8.27e-25 (2^-80) and below | ~50–62 % |

Failures start around **8.5e-22**, which is **far above** float's min-normal 1.18e-38. The cause is
not denormal *inputs* — it is the **product** underflowing:

```
two_prod(a,b) stores the rounding error of a*b via fma(a,b,-p).
If a*b itself falls below float min-normal 2^-126, that error term is unrepresentable.
inputs ~2^-70  ->  product ~2^-140  ->  UNDERFLOW  ->  double-float breaks
inputs ~2^-60  ->  product ~2^-120  ->  safe
```

### Verdict and the remaining check
**Double-float reproduces the CPU's `double` accumulation bit-exactly for every realistic magnitude
tested, including wide dynamic range and near-cancellation.** The single failure mode is
`|acc x scl| < ~2^-126`.

So a **bit-exact Metal attention kernel is possible**, with one obligation: prove the real model
never produces such a product. `acc` is a 128-term float block sum of `e4m3(w)*x`; `scl` is an F32
weight block scale. Both would have to be around 1e-22 simultaneously. That is implausible — the
fp8 activation QDQ floors its scale at `1e-4` (`:11990`) — **but implausible is not measured**, and
this project has been burned by exactly that gap. The required check is to dump the real
`(acc, scl)` pairs from a prefill run and report `min |acc x scl|`; if the margin is many orders
above 2^-126, the lane is clear, and if not, those blocks fall back to CPU.

**Recorded as: blocker E65 is resolved in principle, conditional on one measurement.**

### E66d. The remaining gate, measured — **margin is 7.9e28x. Blocker fully resolved.**

E66 left one obligation: prove the real model never produces `|acc x scl| < 2^-126`. Instrumented
`matmul_fp8` (`quant.h`, tracker default OFF behind `coli_fp8_minprod_enabled`) and measured it
during a real p064 prefill:

```
fp8_min_abs_acc_x_scl = 9.313226e-10
float_min_normal      = 1.175494e-38
margin                = 7.92e+28x
```

**Nearly 29 orders of magnitude of headroom.** The underflow regime that breaks double-float
(products below ~1e-38) is not remotely approached by this model — the smallest product observed is
~1e-9. The `1e-4` floor in the fp8 activation QDQ (`:11990`) plus realistic weight-block scales keep
it far away.

**Conclusion: E65's blocker is fully resolved. A bit-exact Metal attention kernel is possible**,
using double-float emulation for the block accumulation at an estimated 2.4–7.3 % kernel overhead
(E65) on a path measured 6.0x faster than CPU (E59). Default path re-verified
`PASS golden md5=5d04890413ff539e802985ce8c727814`.

The chain of reasoning that got here is worth noting, because three of the four steps would have
produced a wrong answer if stopped early:
1. E64 said the attention lane is worth 1.23x -> looked like a clear go.
2. E65 found Metal has no fp64 and the fp8 path accumulates in `double` -> looked fatal.
3. E66 random testing said double-float is exact -> **false green**, adversarial testing found a
   49.94 % failure mode.
4. E66c located the true threshold (product underflow at ~1e-22 inputs, not denormal inputs).
5. E66d measured the real margin as 7.9e28x -> **actually fine**.

---

## E67. T0 baseline re-pinned against the current binary (skeptical re-verification pass)

The todo list was challenged for completion, so every T-1 claim was re-verified **against the
current binary rather than trusting earlier runs** — the binary had changed four times since E61
(grouped MoE, `MOE_GROUPED_DUMP`, `ATTN_STATS` x2, the fp8 min-product tracker), which made every
prior verification stale.

### Re-verification vs binary `2e8dbbb5`

| check | result |
|---|---|
| branch `ft-metal-new-arch` @ `38711e7`, tracking `fork/` | ok |
| `validation/moe/test_group.c` (4 subtests, mutation-verified) | **4/4 PASS** |
| golden, grouped **OFF** | `PASS golden md5=5d04890413ff539e802985ce8c727814` |
| golden, grouped **ON** | **same md5 — still bit-exact** |
| K0 gate | overhead **9.70 %** < 16.7 %, `moe_groups=5073`, `moe_wave_fallbacks=0` — **PASS** |

The grouped-MoE result survives every binary change: still bit-exact, still inside the K0 gate,
still executing 5073 real expert groups (not silently falling back).

### T0 — the one task genuinely outstanding
`artifacts/metal_baseline.json` had never been written. Pinned now against the **current** binary
(the todo's `f93118b1` was itself stale):

```json
{ "binary_md5": "2e8dbbb521d9954af5bc970ff8f9d2d3",
  "commit": "38711e79be0772fc7da00a2100b8e473278a183e",
  "p064_median_s": 43.617, "p256_median_s": 111.683,
  "golden_md5": "5d04890413ff539e802985ce8c727814",
  "nonengine_gb": 15.636, "compressor_gb": 0.218 }
```
p064 43.617 s (sd 0.257, n=3), p256 111.683 s (sd 1.073, n=3), golden gated before timing.

### Two notes worth carrying
1. **The new pin differs from the old one** (p064 43.554 -> 43.617, **p256 113.735 -> 111.683,
   -1.80 %**). Different binary, different session — **not** a regression. E57 measured a **29.5 %**
   swing from ambient load alone on identical config, and E56 showed that diffing against a
   stale pin inflated a result **1.73x**. This baseline is for provenance, **never** for computing
   a delta: every A/B must stay interleaved.
2. `pin_baseline.sh` logs `WROTE artifacts/baseline.md` but its preserve/restore logic leaves the
   tracked file unchanged. Defensively correct (no clobbering of a tracked artifact) but the log
   line is misleading; the fresh numbers live in `metal_baseline.json` only.

---

## E68. A1/A2 — **a bit-exact Metal fp8 matmul is PROVEN (0 ULP), and it costs half the speedup**

E66d cleared the last theoretical blocker. This builds the thing and measures it.

`validation/probes/attn_fp8_exact.m`. Two design choices remove whole classes of error:
- the CPU reference **`#include`s `quant.h` and calls the real `matmul_fp8`** — no transcription risk
- the Metal LUT is generated by calling the engine's own **`e4m3_decode(0..255)`** — the shader
  cannot drift from the C table

### The trap: Metal enables fast-math by default

The first GREEN run **failed**, and the failure count was nearly identical to the plain-float RED
case (48467 vs 48691 differing). That similarity was the clue: if the outer accumulator were the
discriminator, double-float should have been dramatically better. It was not — so the divergence
had to be elsewhere.

Two hypotheses were tested and **both were wrong**:
1. *FMA contraction in the inner loop* — isolated with a single 128-element block:
   **0/2000 differing** in both `a += b*c` and explicit `fma()` form.
2. *Multi-block outer accumulation* — a single-block case (O=128, I=128) gave **0/128 differing**.

The real cause: **`MTLCompileOptions.fastMathEnabled` defaults to `YES`**, which algebraically
simplifies Dekker's error term `e = (a-(s-bb))+(b-bb)` to **zero**. Fast math silently deletes the
entire double-float mechanism while leaving code that looks correct.

```objc
MTLCompileOptions *opt = [MTLCompileOptions new];
opt.fastMathEnabled = NO;   /* MANDATORY for Dekker two_sum/two_prod */
```

### A1 result — bit-exact

| shape | outputs | differing | max ULP |
|---|---|---|---|
| `wq_a` I=4096 O=1024 | 65 536 | **0** | 0 |
| `wq_b` I=1024 O=32768 | 2 097 152 | **0** | 0 |
| `wkv` I=4096 O=512 | 32 768 | **0** | 0 |
| `wo_b` I=1024 O=4096 | 262 144 | **0** | 0 |

**2 457 600 outputs, all 0 ULP**, and reproduced at S=64, S=16 and S=1.

A clean 2x2 control shows **both** ingredients are load-bearing:

| | fast-math ON | fast-math OFF |
|---|---|---|
| plain float outer | DIFFERS | **DIFFERS** (137 018) |
| double-float outer | DIFFERS | **0 ULP** |

The plain-float RED case fails by up to **86 928 ULP** — so the `double` accumulation in
`matmul_fp8` is not a rounding nicety, it is load-bearing.

### A2 — bit-exactness costs about half the speed

| shape (S=64) | CPU | GPU | speedup |
|---|---|---|---|
| `wq_a` | 8.23 ms | 3.89 ms | 2.11x |
| `wq_b` | 46.95 ms | 14.39 ms | 3.26x |
| `wkv` | 4.13 ms | 1.00 ms | 4.14x |
| `wo_b` | 6.54 ms | 2.00 ms | 3.28x |
| **aggregate** | **65.85 ms** | **21.28 ms** | **3.09x** |

**E59's 6.0x is superseded.** That figure was measured with an approximate `e4m3` decode, fast-math
enabled, and non-exact accumulation — i.e. it was never a bit-exact number. The honest bit-exact
figure is **3.09x**.

### Corrected lane value

Applying 3.09x to E64's measured 9.838 s of projections in a 43.569 s wall:

| speedup | saves | full prefill |
|---|---|---|
| 6.00x (E59, not bit-exact) | 8.20 s | 1.232x |
| **3.09x (bit-exact)** | **6.65 s** | **1.180x** |
| 2.11x (worst shape) | 5.18 s | 1.135x |

**1.180x — clears the 1.12x gate by 6.0 pp, even at the worst single shape (1.135x).** The lane
survives full bit-exactness. It should also compose with the shipped MoE grouped path (1.043x,
a different part of the wall) for an upper bound of ~1.23x — **to be measured, not assumed**, per
the standing E57 lesson.

---

## E69. A3 step 1 — `-fno-fast-math` on the Metal build (prerequisite, and a latent fragility fixed)

E68 proved a bit-exact Metal fp8 matmul is possible, but **only** with
`MTLCompileOptions.fastMathEnabled = NO`. Integrating that kernel requires the engine's own Metal
build to honour the same flag, so this lands first as a separately-verifiable step.

### What the engine was doing
Both Metal compile paths ran with fast math **enabled**:
- `c/backend_metal_v4.mm:737` — `newLibraryWithSource:source options:nil` (nil = defaults = fast math ON)
- `c/Makefile.deepseek-v4:142` — `$(METALC) -c $< -o $@`, no flag

Production loads the offline metallib (`v4_metal_library kind=metallib`), so the Makefile path is
the one that matters.

### Why this is a fragility, not just a blocker for the new kernel
The existing kernels are the **`ordered`** variants, whose entire purpose is a *deterministic*
two-level accumulation order — that is why they can be bit-exact where a `simd_sum` reduction
cannot. Compiling them with fast math **permits the compiler to reassociate**, which contradicts
their design intent. Their bit-exactness was therefore *incidental* rather than guaranteed, and a
toolchain update could have silently broken it.

### Change
```make
# -fno-fast-math is REQUIRED, not cosmetic: fast math algebraically simplifies Dekker
# two_sum's error term (a-(s-bb))+(b-bb) to zero, which silently destroys any
# double-float accumulation. It also lets the compiler reassociate, which contradicts
# the whole point of the `ordered` kernels' deterministic accumulation order.
METALCFLAGS ?= -fno-fast-math
...
	$(METALC) $(METALCFLAGS) -c $< -o $@
```
Verified the flag is accepted by `xcrun metal` and **materially changes codegen** (the emitted
`.air` differs by md5), so it is not a no-op.

### Verification — on the real production surface
`probe_fused_moe.m` compiles from `.metal` **source** at runtime with its own options, so it does
**not** exercise the metallib production loads. The decisive test is the engine itself:

| check | result |
|---|---|
| engine loads the rebuilt library | `v4_metal_library kind=metallib` |
| Metal genuinely ran (not silently falling back) | `metal_dispatches=550`, `ok=550` |
| **golden, Metal ON** | `PASS golden md5=5d04890413ff539e802985ce8c727814` |
| **golden, Metal OFF** (default path) | `PASS golden md5=5d04890413ff539e802985ce8c727814` |

**Disabling fast math did not change any existing kernel's output.** All 6 shaders rebuilt; engine
binary md5 unchanged (`2e8dbbb5`) since only the metallib moved.

Incidental observation for later: that run showed `ok=550` against `layout=740` rejects — i.e. only
**42.6 %** of expert forwards reached the GPU, consistent with the rows16 layout rejection recorded
as P1 in §12. Not addressed here.

---

## E70. A4 — the attention lane ships **+5–6 %**, and my first A/B was confounded by my own flag choice

### The false negative
The first A/B reported the lane **20 % SLOWER** (p064 `delta=+19.97%`, p256 `+17.12%`, 3/3 pairs
each — consistent, not noise). Against a predicted 1.18x that was a 42 % miss, so it demanded a
measured cause rather than a story.

**The cause was my own experiment design.** I ran
`ab.sh "COLI_V4_METAL=1 COLI_V4_METAL_ATTN=1"`, which also enabled the **MoE expert Metal path**
that §12 had already measured at **1.118x slower**. The diagnostic settles it:

```
metal_dispatches      = 10760      <- the known-bad MoE expert path
metal_fp8_dispatches  =   511      <- my attention path
```

The result was dominated by 10 760 dispatches of a path known to lose, and I had attributed it to
my 511. Enabling a known-bad feature alongside the one under test is exactly the confound this
project keeps re-learning (E57's `--ram`/speculation pairing, E13's stale-baseline inflation).

The lazy Metal init added in A3 is what makes isolation possible — with **only**
`COLI_V4_METAL_ATTN=1`: `metal_dispatches=0`, `metal_fp8_dispatches=256`.

### Corrected result — interleaved, n=3, `COLI_V4_METAL_ATTN=1` alone

| prompt | off | on | delta | speedup | pairs |
|---|---|---|---|---|---|
| p064 | 42.565 s | 40.466 s | **−4.93 %** | **1.052x** | 3/3 |
| p256 | 109.489 s | 103.141 s | **−5.80 %** | **1.062x** | 3/3 |

Bit-exact throughout (`PASS golden md5=5d04890413ff539e802985ce8c727814` with the lane ON and OFF).

**The gain GROWS with prompt length** (4.93 % → 5.80 %) — the opposite of every other feature
measured here (prefill prefetch decayed 6.20 → 2.70 %, grouped MoE decayed 4.15 → 2.09 %). Both of
those decayed because O(n²) attention dilutes them; this lane *is* attention, so length works in
its favour. That makes it the first optimisation in this project that improves on long prompts.

### Why 1.05x and not the projected 1.18x — measured, not guessed

| stage | E64 CPU baseline | with lane ON | speedup |
|---|---|---|---|
| attention total | 14 553.0 ms | **11 610.3 ms** | 1.25x |
| five projections | 9 837.8 ms | **6 942.1 ms** | **1.42x** |

The projections sped up **1.42x**, not the **3.09x** the A2 microbenchmark measured. The per-call
breakdown shows exactly where it went:

```
metal_fp8_ms total=1561.4  memcpy_in=4.6  dispatch_wait=1540.7  memcpy_out=16.1
511 dispatches -> 3.0 ms average, of which dispatch_wait is 98.7%
```

Memory traffic is negligible (21 ms of 1561 ms) and weight upload was only **0.6 MB** — the
resident weights are 16 KB-aligned so the pointer-keyed cache serves them **zero-copy**. The cost
is **one command buffer plus one blocking `waitUntilCompleted` per call**. That is the same
structural failure as the original S=1 MoE seam: the kernel wins, the per-call round trip gives it
back.

### Next lever, identified by this measurement
Amortise the dispatch. The five projections in a layer are issued as five separate blocking round
trips; they could share one command buffer, and the two `wq_*` projections have a genuine
dependency chain but `wkv` does not. Cutting 511 round trips toward ~86 would recover most of the
gap between the measured 1.42x and the kernel's 3.09x. **Projected, therefore to be measured, not
assumed.**

---

## E71. A4b — a silent cache overflow in **my own** code was halving the lane; fixing it gives **1.107x / 1.135x**

E70 shipped the attention lane at +5–6 % and blamed the gap to the kernel's 3.09x on per-call
dispatch overhead. Attributing that properly found a different, larger cause — a bug I had written.

### First, a self-correction: my 77 % claim was arithmetic across runs
E70 inferred the fp8 QDQ was ~77 % of the projection cost by subtracting `6942.1 − 1561.4`. Those
two figures came from **different runs**. Direct instrumentation says QDQ is **1273.5 ms of
7477.8 ms = 17.0 %**, not 77 %. Same invalid-cross-run-arithmetic class as E56's stale baseline
(which inflated a result 1.73x). **Measure the thing; do not subtract across runs.**

Reconciling properly exposed the real gap:

| component | ms | share |
|---|---|---|
| Metal path (measured) | 1861.7 | 24.9 % |
| fp8 QDQ (measured) | 1273.5 | 17.0 % |
| **unaccounted** | **4342.6** | **58.1 %** |

### The bug: my weight cache silently overflowed
`wo_a` is the **only** projection invoked **per output group**, each with a distinct pointer
(`wo_a.data + group * o_rank * group_width`, `:2830`). My pointer-keyed cache therefore needs one
entry per group, not one per projection:

```
43 layers x (4 single projections + o_groups=8 wo_a slices) x (data + scales) = 1032 entries
my cap                                                                        =  512
```

Past 512 the cache **returned nil forever**, permanently falling back to CPU. `wo_a` is registered
last per layer, so it was starved first — which is precisely why it measured as the most expensive
projection (3058 ms, 40.9 %).

**The dispatch count proves it exactly:**

```
metal_fp8_dispatches   511  ->  1031      (exactly double; half the calls had been falling back)
metal_fp8_cache_full_events            0  (after the fix)
```

Fixed by sizing the cache to 4096 **and making overflow observable** — a silent nil is the exact
failure class this project keeps re-learning (§12e's silent CPU fallback, E70's confound).

### Effect

| metric | cache-limited | fixed | vs E64 CPU baseline |
|---|---|---|---|
| `attn_total_ms` | — | **9629.8** | 14553.0 → **1.51x** |
| five projections | 7477.8 | **4987.2** | 9837.8 → **1.97x** |
| p064 TTFT (single run) | 40.557 s | **38.055 s** | ~42.5 s |

### A/B — interleaved, n=3, `COLI_V4_METAL_ATTN=1` alone

| prompt | off | on | delta | speedup | previous |
|---|---|---|---|---|---|
| p064 | 42.420 s | 38.305 s | **−9.70 %** | **1.107x** | 1.052x |
| p256 | 109.443 s | 96.409 s | **−11.91 %** | **1.135x** | 1.062x |

Bit-exact throughout (`PASS golden md5=5d04890413ff539e802985ce8c727814`).

**p256 clears the plan's 1.12x gate** (which requires `delta <= -10.71%`); p064 at −9.70 % sits just
under it. And the lane keeps **strengthening with prompt length** (9.70 % → 11.91 %, up from
4.93 % → 5.80 %) — still the only optimisation measured in this project that improves on longer
prompts, because it *is* the O(n²) term rather than being diluted by it.

### Remaining headroom
Projections now run 1.97x against a kernel measured at 3.09x (A2). `dispatch_wait` is 3586 ms of
the 3625 ms Metal path (98.9 %) across 1031 calls, and memory traffic is negligible (39 ms, upload
1.1 MB — resident weights are 16 KB-aligned and serve zero-copy). So the next lever is still
**amortising the per-call blocking round trip**, now over 1031 calls rather than 511.

---

## E72. Both shipped lanes together — **1.163x / 1.169x, bit-exact, and composition VERIFIED**

E70 and E71 both projected that the two lanes should compose to ~1.23x but explicitly refused to
claim it, because **E57 is the standing lesson**: a composed "+80.7 %" from two separately-measured
deltas turned out to be **−5.4 %** when finally measured end-to-end. So this measures it.

### Correctness first
`COLI_V4_MOE_GROUPED=1 COLI_V4_METAL_ATTN=1` together:
`PASS golden md5=5d04890413ff539e802985ce8c727814` — **bit-exact with both lanes active**.

### Result — interleaved, n=3, sign-correct

| prompt | off | on | delta | **measured** | attn only | MoE only | naive product |
|---|---|---|---|---|---|---|---|
| p064 | 42.303 s | 36.367 s | **−14.03 %** | **1.163x** | 1.107x | 1.043x | 1.155x |
| p256 | 109.483 s | 93.671 s | **−14.44 %** | **1.169x** | 1.135x | 1.021x | 1.159x |

**Composition holds** — the measured result captures **106 %** of the naive product at both lengths
(the small excess is within run-to-run noise). That is the expected outcome *here* because the two
lanes touch genuinely disjoint parts of the wall: grouped MoE reorders expert scheduling, the
attention lane replaces dense projection matmuls. Unlike E57 — where `--ram` and speculation were
measured on **different workloads** and one arm already had the other enabled — there is no shared
term to double-count.

**Both clear the plan's 1.12x gate** (which requires `delta <= -10.71%`), with margin.

### Where the wall went

| | baseline | both lanes | change |
|---|---|---|---|
| p064 TTFT | 42.303 s | **36.367 s** | −5.94 s |
| p256 TTFT | 109.483 s | **93.671 s** | −15.81 s |

### Standing caveats
- n=3 per arm, one host, one model. Directional and interleaved, but not a large sample.
- The gain still **grows with prompt length** (14.03 % → 14.44 %), driven by the attention lane
  being the O(n²) term rather than being diluted by it.
- Headroom remains: projections run **1.97x** against a kernel measured at **3.09x** (A2), with
  `dispatch_wait` at 98.9 % of the Metal path across 1031 calls. Amortising that per-call blocking
  round trip is the next lever, and `wo_a` alone accounts for **8 of every 12 dispatches**
  (one per output group) — batching those into a single dispatch is the concrete target.
