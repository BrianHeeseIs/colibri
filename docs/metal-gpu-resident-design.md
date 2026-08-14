# GPU-resident expert cache — design + build schedule (PARKED, scheduled for later)

**Status: designed, de-risked on paper, NOT started.** Parked in favour of the prefill I/O lane
(higher ceiling, lower risk). This document is the resume point.

## Why this architecture and no other
Four measured verdicts (E43/E46/E48/E49) established, in order: dispatch sync is 1.53 ms/expert;
the shaders are healthy (**2.35 TFLOP/s**, E48 — the earlier 15 GFLOP/s claim blended first-touch
into compute and was retracted); true per-dispatch compute is **C ≈ 0.086 ms**; and a
**~2.05 ms/dispatch coherency tax RECURS** (E49, marginal-cost method): every cache miss rewrites a
13.37 MB slab CPU-side and the GPU re-pulls the dirty pages. Batching removes only sync → batched-6
is still 3.1x slower than CPU. **The coherency tax is structural to CPU-written shared slabs.**
The only architecture that removes it: experts the CPU never rewrites — a GPU-private tier.

## Architecture
- **Private-storage expert tier**: `MTLStorageModePrivate` buffers holding expert records in
  GPU-optimal layout. CPU never writes them after upload → no coherency pulls, ever.
- **Upload path**: on cache miss, loader thread reads the record from SSD (as today) into a shared
  staging buffer, then a **blit encoder** copies staging → private. 13.37 MB at ~80 GB/s ≈ 0.17 ms,
  queued async on the loader thread — hidden behind compute exactly like today's SSD latency.
- **GPU-side eviction**: same slot/LRU/pin policy as the CPU tier, but slot = private buffer region.
  Active-dispatch protection reuses the existing lease refcounts (`!slots[i].references`).
- **Batched dispatch**: one command buffer per (token, layer) in decode (6 experts, 1 sync), per
  (chunk, layer) in prefill (up to ~93 unique experts, 1 sync). Leases held for the batch.
- **Memory budget**: unified RAM — the private tier COMPETES with the CPU slab cache. Split TBD at
  build time (e.g. 40 GiB GPU tier + reduced CPU cache, or full migration). This is the main open
  design decision.

## Projections (from measured constants R=1.53, C=0.086, c_cpu=0.76)
| | ms/expert | vs CPU |
|---|---|---|
| today (shared slabs, per-expert sync) | 3.66 (marginal, E49) | 4.8x slower |
| private tier, batched-6 (decode) | (R+6C)/6 = **0.341** | **~2.2x faster** |
| private tier, batched-64 (prefill) | (R+64C)/64 = **0.110** | **~6.9x faster** |

Blended: expert_forward is 26.6 % of decode → **~14 % decode wall**. Prefill compute is only ~16 %
of prefill (I/O-bound, E45-era analysis) → prefill gain capped there unless the I/O lane lands too.

## Staged build plan (each stage gated, in order)
- **S0 — de-risk microbench (~1 day, no engine changes).** Extend `c/bench_moe_batch.c`-style
  standalone: (a) our mxfp4/UE8M0 shaders on PRIVATE buffers at real shapes — confirm ~TFLOP/s;
  (b) blit upload throughput for 13.37 MB records; (c) N-experts-per-command-buffer scaling —
  confirm R amortizes. **Gate: all three within 2x of projection, else abandon.**
- **S1 — decode batch-6 on a small fixed private tier** (e.g. pinned experts only), behind
  `COLI_V4_METAL=1`. Gate: same-build A/B decode_wall improves; default md5 golden.
- **S2 — full private tier with eviction + upload pipeline.** Gate: decode_wall −10 % or better.
- **S3 — prefill (chunk, layer) batching.** Gate: prefill wall improves on ≥184-token prompts.

## Known risks (write-downs from the campaign, do not rediscover)
- **Not bit-exact at length**: METAL=1 output diverges from CPU past ~30 tokens (E46, deterministic
  each side). Ship under the opt-in umbrella like `--fast-kernels`; never default.
- **rows16 conflict dissolves**: the CPU rows16 hot-pack layout (43-53 % of dispatches rejected
  today) is irrelevant on a private tier — upload converts layout once. But S1 must not *break*
  CPU rows16 for the fallback path.
- **Memory pressure**: 96 GiB budget already tight; tier split must be measured, not guessed.
- **Build traps**: `rm -f c/*.o` before METAL toggle (make won't recompile on -D change);
  `rm` before `cp` of binaries (vnode signature cache → SIGKILL). Both bit us; see E42/E48.
- **Measurement discipline**: one-shot A/B, interleaved, warm, n≥3, decode_wall not phase numbers
  (E41 cache-contention lesson), STATS not PROFILE for timing.

## Evidence trail
E43 (2.67x slower, 30tok) → E46 (2.747x @ n=3, "shader-bound" — retracted) → E48 (shaders 2.35
TFLOP/s; first-touch found; batching gate passed conditionally) → E49 (coherency tax recurs;
batching dead on shared slabs; this design is the survivor). Raw data: `.backlog/results/
T7_metal_oneshot.csv`, `T7b_metal_profile.log`, `T7c_metal_profile_100.log`, `E49_marginal.csv`.
