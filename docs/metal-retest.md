# Metal fair re-test — method and verdict

**Verdict: CONFIRMED slower. 2.84x on `expert_forward`, 1.52x end-to-end. Lane closed.**
The batched-dispatch rescue is also ruled out, by measurement rather than argument.

## Why a re-test was warranted
The first Metal measurement (E43) used a **30-token** run taken immediately after a full rebuild.
Warmup/cache-heating was a legitimate objection: the backend has real one-time costs — lazy
`coli_v4_metal_init` (device + metallib compile + probe pipeline), chain pipelines built on the
*first* dispatch, scratch-buffer allocation, and one `MTLBuffer` per slab via
`newBufferWithBytesNoCopy`. A short run charges all of that to Metal.

## Method (every confound controlled)
| control | what was done | why |
|---|---|---|
| measurement vehicle | **one-shot**, not serve | E45: serve is ~40x noisier (10.9 % vs 0.27 % rel sd) and would swamp the effect |
| build | **one binary**, `c/deepseek_v4.metal`, toggled by `COLI_V4_METAL=0/1` | true same-build A/B; verified pure `.cpu` ≡ `.metal` at METAL=0 (md5 `e8c7d8e6…`) |
| amortization | **300 tokens**, not 30 | 10x more work so all one-time costs wash out |
| drift | runs interleaved 0,1,0,1 | cancels thermal/machine drift |
| instrumentation | `COLI_V4_METAL_STATS=1` for timing | `PROFILE` injects 9 commit/waits per dispatch and distorts; used only in a separate attribution pass |
| build trap | `rm -f c/*.o` before each variant | `make clean` does not remove objects and make will not recompile on a `-D` change — this silently produced a wrong conclusion twice |

## Result
**n=3 per configuration, interleaved** (contract met):

| | CPU (METAL=0) | Metal (METAL=1) | ratio |
|---|---|---|---|
| `expert_forward` | 60136.5 ms (sd 6.6 %) | 165187.1 ms (sd 3.3 %) | **2.747x slower** |
| `decode_wall` | 218587.7 ms (sd 2.8 %) | 324436.5 ms (sd 2.9 %) | **1.484x slower** |

Worst-case pairing for Metal (lowest Metal run / highest CPU run) is still **2.51x**; the two
distributions do not overlap, so the verdict is robust to sample size. `hit_rate` matched closely
across configs (90.289 vs 90.333), so cache state was comparable despite the output divergence.
Both configs were internally deterministic (md5 `e8c7d8e6` x3 and `b55c21ec` x3).

Pre-committed criteria were **OVERTURN ≤1.05x / CONFIRM >1.10x** → **CONFIRM.**

**Caveat on strength of evidence (raised in review, partly resolved):**
- ~~n=2~~ **resolved: n=3 per configuration**, distributions non-overlapping.
- At 300 tokens the two configurations **produce different output** (see Correctness below), so
  after the divergence point they generate different tokens, which route to different experts.
  The workloads are therefore not strictly identical over the full window. The 30-token E43 run,
  where outputs *were* identical, gave a consistent 2.84x — which is what keeps the verdict
  credible despite this.
The warmup hypothesis is **falsified**: 30 tokens gave 2.84x/1.41x; 300 tokens gives 2.84x/1.52x —
ten times more work made Metal *worse*.

**Isolated from the hybrid** (46.7 % of experts reach the GPU; 53.3 % rejected to CPU on the rows16
`block_dims` mismatch): `c_gpu = (ef_metal − c_cpu·N_fallback)/N_gpu` = **3.5475 ms/expert** vs
`c_cpu` **0.7185 ms** → **4.94x slower judged on its own merits.**

**Correctness:** at 300 tokens `METAL=1` output **diverges** from CPU (`b55c21ec…` vs `e8c7d8e6…`),
though both are internally deterministic and were identical at 30 tokens — the same slow-accumulation
pattern as E37/E41. Metal is not bit-exact at length.

## The batched-dispatch idea, and why it cannot rescue this
Proposal: stop paying one synchronous `waitUntilCompleted` per expert; enqueue the independent
experts of a (token,layer) — or a whole prefill chunk — into one command buffer. The safety concern
(never recycle a slab with in-flight GPU work) was already handled: the expert store refuses to evict
a slot with live references (`c/deepseek_v4.c:5609-5614`).

`COLI_V4_METAL_PROFILE=1` (forwards=7060, raw log committed at
`.backlog/results/T7b_metal_profile.log`) gives the split. Solving for round-trip `R` and real GPU
compute `C` two ways — **note these are NOT independent**: both derive from the same PROFILE run and
both assume profiling leaves GPU compute unchanged while adding a constant per-commit cost. They
bound the same quantity from one dataset, so they can be wrong in the same direction:

- from the 8 *extra* profile commits: `R = 0.073 ms`, `C = 3.475 ms`
- from `submit_stage / 9`:            `R = 0.221 ms`, `C = 3.327 ms`

**Dispatch overhead is only 2–6 % of the cost.** Weights are already **fully zero-copy**
(`zero_copy_tensors=42360`, `copy_fallback_tensors=0`). A *perfect* 6-way batch would yield
`(0.073 + 6·3.475)/6 = 3.487 ms/expert` — still **4.9x slower than CPU**.

`c_gpu` additionally assumes the layout-rejected CPU fallbacks cost the same as the global average
CPU expert; if rejected experts are systematically cheaper or dearer, the isolated GPU figure shifts.

**Root cause — RETRACTED and corrected by E48.** This document previously claimed the shaders run at
~15 GFLOP/s. That number blended a **first-touch page-mapping/coherency pool** into compute: at 1.98x
the dispatches, `matmul gate`/`down` per-expert cost FELL 2.2x (a fixed pool amortizing) while
`matmul up` — an identical dispatch whose pages were already mapped — held ~0.007 ms =
**2.35 TFLOP/s. The shaders are fine.** True per-dispatch compute is C ~= 0.086 ms against a
~1.53 ms synchronous round-trip, so the batching gate (build if C <= 0.10/0.35) **passes**:
decode batch-6 projects 2.1x FASTER than CPU, prefill batch-64 projects 6.5x — pending one
verification (that the first-touch pool is per-slab-one-time rather than recurring with evictions).
The measured outcome of this document — Metal as currently dispatched is 2.747x slower — still
stands; the *mechanism* and the *remedy* changed. See E48.

---

## 2026-08-26 update — the remedy landed, and the verdict is now conditional

This document's verdict line ("CONFIRMED slower … lane closed") was correct for the kernel the
expert path dispatched at the time, and remains correct for it. It is **not** a statement about the
Metal expert path in general, and should no longer be read as one.

The expert chain selected `coli_v4_matmul_mxfp4_ordered_xcache`, which dispatches **one thread per
output row**. At decode the expert matmul is S=1 with O=2048, so that launches 2048 threads — the
GPU is thread-starved, and the loss recorded here follows from that, not from the shaders.

`coli_v4_matmul_mxfp4_simd_exact` (experiments E95/E96) maps **one simdgroup per output row**, a 32x
occupancy increase, while remaining **bit-identical** to the scalar reference — verified at kernel
level (0 mismatches over ~48k values), at the seam (identical digests at batch 1 and 8), through
`bench/golden.sh` (unchanged md5 `5d04890413ff539e802985ce8c727814`), and by a multi-chunk p256
differential (identical generated-text md5, both arms deterministic).

Measured at p256, N=2, 60 tokens, against `COLI_V4_METAL=1` alone:

| | tok/s | TTFT |
|---|---|---|
| `COLI_V4_METAL=1` (this document's configuration) | 0.9724 | 143.7 s |
| `+ COLI_V4_METAL_VARIANT=simd_exact_cold` | **1.30205** | **119.9 s** |
| | **+33.9%** | **-16.6%** |

Enable with:
```bash
COLI_V4_METAL=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold ./c/deepseek_v4 ...
```
Default is unchanged (`ordered_cold`). **N=2 is below the n>=5 this project requires for the decode
axis, and the CPU arm was not run**, so this is explicitly provisional: it establishes that
`simd_exact` is much faster than the other Metal arm, not yet that Metal now beats the CPU. The
outstanding runs are specified in `.backlog/simd-exact-remaining-measurements.md`.
