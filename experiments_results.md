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
