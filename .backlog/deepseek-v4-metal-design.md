# VERDICT: DEFER / DO NOT BUILD — closed by measurement 2026-08-12

All three reviewers gated this on profiling (kill <10%, justify >15-20% expert-matmul share).
That profiling is now DONE, via `/usr/bin/sample` on the live engine — no code changes needed,
because the in-tree instrumentation turned out to be a STUB (see below).

Main-thread critical path, DeepSeek-V4, --ram 96, 13.5k samples @1ms per run:

|                     | expert-compute | of which OpenMP join-barrier | truly offloadable |
|---------------------|----------------|------------------------------|-------------------|
| COLD (prompt #1)    | 20.0%          | 8.8%                         | ~11.2%            |
| WARM (mid-run)      | 14.4%          | 6.4%                         | **~8.1%**         |

**8.1% warm is BELOW the reviewers' 10% kill threshold.** Even total expert compute (14.4%)
misses the 15-20% justify band.

Why the offloadable share is smaller than it first appears: most time inside `matmul_mxfp4`
is the main thread blocked at `__kmp_join_barrier` while 12 OpenMP workers do the arithmetic.
That work is ALREADY parallel across cores; Metal would displace working CPU capacity rather
than unlock idle capacity. Amdahl ceiling ~8% before Metal launch latency, activation packing
and synchronisation overhead.

For comparison, measured the same day on the same host: `V4_DRAFT=4 V4_NGRAM=1` yields
**+18.5%** (6/6 paired wins, cold-start excluded) for ZERO lines of code — more than double
this project's theoretical maximum.

Correctness objections from review round 1 stand independently and were never resolved:
- existing Metal MoE pipeline computes a DIFFERENT function than V4's CPU path (BF16 rounding
  boundaries, `swiglu_limit` clamping, route-weight ordering) -> could not be token-exact;
- bit-exactness may be structurally impossible: rows16 CPU accumulates sequentially per row,
  `moe_gemv` tree-reduces across 32 lanes via `simd_sum`;
- `moe_gemv` accepts fmt {1,2,5,6} only; fmt=4 grouped-int4 is explicitly unsupported for
  batched MoE, so even converting weights would not have reused the kernel.

INCIDENTAL FINDING (worth upstreaming): the in-tree profiler is a stub.
`COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE` has 5 call sites and 2 declarations
(`deepseek_v4.c:3550-3551`) but ZERO definitions anywhere, and appears in no Makefile.
Enabling it fails to link — proven: compiling `COLI_V4_UNIT_BLOCK_HYBRID` with the flag gives
undefined `_coli_v4_block_profile_add` / `_coli_v4_block_profile_now`.

Reopen only if: V4 gains its own Metal-native fp4 kernel AND a warm profile shows offloadable
expert compute above ~20%, AND the pipeline-semantics mismatch above is resolved.

---

# DeepSeek-V4 Metal backend — design proposal (PRE-VALIDATION DRAFT)

Author: research synthesis from 4 parallel code-archaeology agents, 2026-08-12.
Status: **AWAITING VALIDATION** (Oracle / Momus / Atlas).
Goal: adapt colibrì's EXISTING Metal kernels to DeepSeek-V4's native fp4 routed experts.
Non-goal: new backend, weight conversion, or any change to model numerics.

## 0. Context established by prior work

- macOS arm64 port of the V4 engine is COMPLETE and validated: builds Mach-O arm64 + libomp,
  `make deepseek-v4-tiny-check` exits 0 (token-exact), full `make test-c` exits 0.
- V4 currently has **zero** Metal: `deepseek_v4.c` has 0 `COLI_METAL` refs (vs colibri.c 70,
  inkling.c 25); `Makefile.deepseek-v4` has 0 METAL mentions.
- Measured on this host, V4 on internal NVMe: ~0.19-0.21 tok/s steady after cold start,
  `--ram 48`. Disk-bound. GPU offload targets the expert matmul, not the disk path.

## 1. Rejected alternative: convert weights to int4

Rejected on four independent grounds:
1. `moe_gemv` accepts only fmt {1,2,5,6}. **fmt=4 (gs64 grouped int4) is explicitly NOT
   supported for batched MoE** (`backend_metal.h:141-145`). Converting to gs64 would still
   not run on the GPU expert path.
2. V4's loader validates experts as I8 weight + F8_E8M0 scale and builds
   `COLI_TENSOR_FP4_NATIVE_BLOCK` views (`deepseek_v4.c:5131-5140, 5185-5202`). It never
   inspects `.qs`. An int4 container would be unreadable without rewriting the loader.
3. fp4 (E2M1) is already 4 bits/weight — conversion yields ~145 GB vs 166.9 GB today, i.e.
   no meaningful saving, and peak disk need ~312 GB against 62 GB free.
4. It would change model numerics, violating the project's stated policy that placement
   changes speed only, never semantics.

## 2. Chosen approach: add fmt=7 (MXFP4) branch to the existing `moe_gemv`

`moe_gemv` is ONE kernel selected by name at pipeline creation
(`backend_metal.mm:637`); `fmt` is a **runtime uniform** at buffer(8)
(`backend_metal.mm:1255`). So fp4 is a new `else if (fmt == 7)` branch, not a new kernel,
not a new pipeline, and requires no public ABI change.

Precedent: fmt=6 (E8/IQ3) already proves the kernel handles **block-local scales**
(`backend_metal.mm:182-202`), which is exactly what UE8M0-per-32-columns needs.

### 2.1 Minimal diff (host + shader)

| # | file:line | change |
|---|---|---|
| 1 | `backend_metal.mm:156-158` | comment: document fmt=7 |
| 2 | `backend_metal.mm:~175` | add `else if (fmt == 7)` MXFP4 branch |
| 3 | `backend_metal.mm:219` | final write: skip `acc*sc[o]` for fmt=7 (scale folded in-loop) |
| 4 | `backend_metal.mm:1203` | allowlist: permit fmt=7 |
| 5 | `backend_metal.mm:1204-1213` | add fmt=7 dimension guards (NOT the fmt=6 FWHT path) |
| 6 | `backend_metal.h:138-145` | docs: `gs/us/ds` semantics per fmt; note UE8M0 bytes for fmt=7 |
| 7 | `Makefile.deepseek-v4` | add `-DCOLI_METAL`, `$(METAL_OBJ)`, `-framework Metal -framework Foundation -lc++` |
| 8 | `deepseek_v4.c` | engine-side adoption (see §4) |
| 9 | `tests/test_backend_metal.mm` | new fmt=7 test vs CPU reference |

Note: `gs/us/ds` are typed `const float *const *` but fmt=6 already passes semantically
unused pointers (`tests/test_backend_metal.mm:337-340`), so passing UE8M0 **byte** pointers
cast to `const float*` needs no ABI change. Type-correctness cleanup (`const void*const*`)
would touch all callers and is deliberately out of scope.

## 3. Bit-exactness specification (the hard requirement)

### 3.1 E2M1 decode — 16-entry LUT
```
{0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6}
```
`quant.h:1354-1362`, `deepseek_v4.c:9995-10000`.
Byte index = column/2. **Even column = LOW nibble (`b & 0xF`), odd = HIGH (`b >> 4`)**.

### 3.2 UE8M0 scale decode — TWO IMPLEMENTATIONS THAT DISAGREE
- `coli_e8m0_decode` (`deepseek_v4.c:9990-9993`): `ldexpf(1, v-127)`; **0xff -> NaN**
- `mx4_scale` (`quant.h:1356-1365`): bit trick `(uint32)s << 23`; **0xff -> +inf, 0x00 -> +0**

They differ at the edges. **OPEN QUESTION Q1 for validators**: which is normative for the
GPU kernel? Proposal: use the bit trick (cheap in MSL: `as_type<float>(uint(s) << 23)`) and
assert at load time that no scale byte is 0x00 or 0xff, matching the code comment that real
checkpoints never contain them.

### 3.3 Accumulation order — TWO CPU PATHS THAT DISAGREE
- `matmul_mxfp4` (`quant.h:1402-1411`): accumulate `ga` over a 32-column block in fp32,
  THEN `a += ga * sc`. Scale applied ONCE per block.
- rows16 fast path (`deepseek_v4.c:10743-10755`, NEON `10702-10716`): per column,
  `sum += (activation * value) * scale`. Scale applied EVERY column. **No FMA** (explicit
  comment at `10659-10662`).

These are not bit-identical in general (different rounding). The routed-expert hot path
uses rows16 (`coli_fp4_dual_matvec_rows16_v10`, `deepseek_v4.c:6186-6198`); the fallback
uses `coli_fp4_matvec_ref` -> `matmul_mxfp4`.

**OPEN QUESTION Q2 for validators**: which is the oracle the GPU must match? Proposal:
match **rows16**, because that is the path actually taken for routed experts, and it
explicitly forbids FMA — a constraint the MSL kernel must honour (use two multiplies then
add, NOT `fma()`).

### 3.4 Activation QDQ — MUST NOT BE FORGOTTEN
`coli_fp4_matvec_ref` quantises the ACTIVATION to E4M3FN with a UE8M0 scale per **128**
columns before the matmul (`deepseek_v4.c:10187-10200`, impl `10095-10119`).

**Proposal**: keep QDQ on the CPU and hand the GPU the already-dequantised fp32 activation
vector. This avoids porting `coli_e4m3fn_encode/decode` to MSL, is bit-exact by
construction, and costs one pass over D floats per token — negligible beside the matmul.
**OPEN QUESTION Q3**: does packing activations per-expert (`xg`) interact with the 128-column
QDQ blocking? The QDQ is over the K dimension of one row, so it should be orthogonal, but
this needs confirmation.

### 3.5 Geometry invariants (already enforced in code)
`columns % 128 == 0`, `block_rows == 1`, `block_columns == 32`,
`data_bytes == rows*columns/2`, `scale_bytes == rows*columns/32`
(`deepseek_v4.c:10171-10186`). rows16 additionally requires `rows % 16 == 0`,
`block_rows == 16` (`deepseek_v4.c:10564-10572`).

## 4. Engine-side adoption (transposed from inkling.c)

### 4.1 THE CRITICAL RISK — stale GPU addresses on eviction
V4's expert store gives **stable slot addresses with unstable contents**. On a miss it
overwrites `slot->slab` in place for a different expert (`deepseek_v4.c:5247-5262`). The
CPU lease model (`references++` / `release`) protects synchronous CPU matmul only.

Metal zero-copy resolves an interior pointer to `gpuAddress + offset`
(`backend_metal.mm:720-726`). If a lease is released before the command buffer COMPLETES
and the slot is refilled, the GPU reads a **different expert's weights** -> silently wrong
output. If refilled mid-flight -> data race.

**Mitigation (mandatory, not optional):**
1. Adopt Inkling's per-layer arena: one 16 KiB-aligned slab per layer, slots carved by
   offset, registered ONCE (`inkling.c:895-919`). Eviction then becomes memcpy into
   already-registered memory — no register/unregister churn.
2. Hold `ColiExpertView` leases until the Metal command buffer COMPLETES, not merely until
   it is encoded. Add `slot->gpu_references`, released from the completion handler.
3. Eviction requires `references == 0 && gpu_references == 0`.
4. Unregister before every `free` (`deepseek_v4.c:5359-5365`, and the duplicate store at
   `9411-9415`).

**OPEN QUESTION Q4**: is per-layer arena rework acceptable scope, or should v1 ship
synchronous-only (`coli_metal_moe_block`, wait for completion before release), accepting
lost CPU/GPU overlap in exchange for a much smaller, safer diff? Proposal: **v1 synchronous**,
async as a follow-up.

### 4.2 Alignment
Current: base store `malloc(record_bytes)` (`5238-5255`); hot store
`posix_memalign(4096, record_bytes + 8192)` (`5878-5888`).
Required: **16384** base alignment, length rounded to a multiple of 16384
(`backend_metal.h:53-63`).

### 4.3 rows16 LAYOUT MISMATCH — newly identified
`moe_gemv` assigns **one SIMD group per output row**, lanes striding columns
(`backend_metal.mm:169-171`). But V4's hot store packs weights **rows16-interleaved**:
`packed[(tile*stride + col) * 16 + lane]`, lane = row % 16 (`deepseek_v4.c:10583-10590`).

These layouts are incompatible. **OPEN QUESTION Q5**: options are
(a) feed the GPU from the BASE (non-rows16) store, which is row-major;
(b) write the fmt=7 branch to consume rows16 directly (16 rows per SIMD group — arguably a
    better fit for Metal's 32-wide SIMD, 2 rows/lane);
(c) unpack rows16 -> row-major before submission (wasteful).
Proposal: **(b)** — it matches the CPU fast path bit-for-bit AND maps naturally onto SIMD
lanes. But it means the fmt=7 branch does NOT look like fmt=2; it is its own addressing
scheme. This is the largest single design decision in the proposal.

## 5. Validation plan (before any perf claim)
1. `make deepseek-v4-tiny-check` must still exit 0 (token-exact) with Metal DISABLED.
2. New `tests/test_backend_metal.mm` fmt=7 case: random E2M1 weights + UE8M0 scales,
   compare GPU vs `coli_fp4_matvec_rows16_v10` — require **bit-identical**, not approximate.
3. Real-model A/B: same prompt, `--temp 0`, `COLI_METAL=0` vs `1`; token streams must be
   IDENTICAL. Any divergence = fail, regardless of speed.
4. Report `[METAL] mode:` marker, GPU blocks, CPU fallbacks, experts-on-GPU — same
   diagnostics as GLM/Inkling (`inkling.c:1698-1704`).

## 6. Honest risk register
| risk | severity | mitigation |
|---|---|---|
| stale gpuAddress after eviction | **CRITICAL — silent wrong output** | §4.1, synchronous v1 |
| rows16 vs row-major layout mismatch | **HIGH — design fork** | Q5 |
| two disagreeing accumulation orders | **HIGH — bit-exactness undefined** | Q2 |
| UE8M0 edge values (0x00/0xff) | MEDIUM | Q1 + load-time assert |
| activation QDQ omitted on GPU path | MEDIUM | §3.4 keep on CPU |
| V4 is disk-bound (~0.2 tok/s) so GPU may not help much | MEDIUM — **expectation management** | measure expert-matmul share of wall time BEFORE building |

## 7. Questions for validators
- Q1: normative UE8M0 decode (ldexpf vs bit-trick) and edge-value policy?
- Q2: which accumulation order is the oracle — `matmul_mxfp4` or rows16?
- Q3: does per-expert activation packing interact with 128-column QDQ blocking?
- Q4: per-layer arena rework now, or synchronous-only v1?
- Q5: rows16-native kernel, base-store row-major, or unpack?
- Q6: **Is this worth building at all?** V4 decode is disk-bound at ~0.2 tok/s. GLM's Metal
  win was 28-33%. If expert matmul is only a small share of V4 wall time, the ceiling is low.
  Should we PROFILE first (expert-matmul vs expert-disk share) before implementing?
