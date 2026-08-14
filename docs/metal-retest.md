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
| | CPU (METAL=0) | Metal (METAL=1) | ratio |
|---|---|---|---|
| `expert_forward` | 59132.1 ms | 167856.6 ms | **2.84x slower** |
| `decode_wall` | 218256.8 ms | 331186.9 ms | **1.52x slower** |

Pre-committed criteria were **OVERTURN ≤1.05x / CONFIRM >1.10x** → **CONFIRM.**

**Two caveats on strength of evidence (raised in review):**
- The binding contract called for **n≥3 per configuration**; the table above is n=2. Both pairs
  agree and both ratios sit far above the 1.10x bar, but the contract was not met as written.
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

**Root cause:** 25.2 M MACs in ~3.33 ms = **~15 GFLOP/s on an M3 Max GPU**, roughly two orders of
magnitude below the hardware. The bottleneck is the **mxfp4/UE8M0 shaders**, not the dispatch model,
not buffer copies, not warmup. Making Metal competitive means rewriting those kernels — a far larger
project than batching, against a CPU path already running at 36.5 GMAC/s.
