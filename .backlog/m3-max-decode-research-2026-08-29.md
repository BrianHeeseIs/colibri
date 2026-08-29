# Maximum decode performance on this MacBook Pro M3 Max

Date: 2026-08-29  
Repository: `simd-apple-metal` at `d5864b00e4a56f9b1f5f9d274b13d387698e3e38`  
Host: MacBook Pro `Mac15,9`, M3 Max, 12P+4E CPU cores, 40-core GPU, 128 GB unified memory  
Scope: evidence through E124, the same-day standalone FP8 microprobe, current source/defaults, exact host state, and Apple-compatible external methods. No model run, long benchmark, power-setting change, commit, or push was performed for this report.

## Technical summary

The repository has changed materially since the 2026-08-28 report. The engine now ships the E114-E119 GPU-prefill champion by default:

```text
COLI_V4_KERNELS=all
COLI_V4_MOE_GROUPED=1
COLI_V4_MOE_BATCHED=1
COLI_V4_MOE_BATCHED_ROWS16=1
COLI_V4_MOE_WHOLE_PROMPT=1
COLI_V4_METAL_ATTN=1
COLI_V4_METAL_VARIANT=simd_exact_cold
COLI_V4_METAL=0
```

`COLI_V4_METAL=0` is deliberate: it keeps single-token decode experts on the CPU, where they are faster on this machine. Prefill attention and batched MoE have independent Metal gates and still run on the GPU. `COLI_V4_BASELINE=1` restores historical defaults for deterministic/reference work.

The champion is a large **prefill and total-wall** improvement, not a decode-speed improvement. At p256 with 40 output-token slots, it measured **-48.19% TTFT**, **-38.49% total wall**, and **-1.75% tok/s** versus CPU + `KERNELS=all`. Current decode remains about **1.67-1.68 tok/s**.

E123 accounts for 99.3% of one 39-token decode run and identifies two nearly equal targets:

- FP8 attention projections: **32.4%** of decode;
- routed expert forward: **32.2%**.

E124 closes the obvious “add NEON to the existing row-major loop” idea. A bit-exact rewrite was neutral/slower because it retained the existing lookup and distant-row access pattern. The x86 fast path is not merely wider arithmetic; it first uses an eight-row interleaved representation.

An unnumbered, standalone microprobe created after E124 now makes the next design substantially more concrete. On the four real attention matrix shapes, a 16-row interleave plus exact E4M3-to-FP16 bit reinterpretation measured **1.61x-2.28x** over scalar at N=21 and reduced the weighted per-layer kernel sum from **4.798 ms to 2.200 ms**. My separate N=7 rerun kept all four shapes positive at **1.28x-2.20x**. The original synthetic 7168x7168 arm measured 2.15x at N=21. Row interleave with the LUT alone was modest, while a heavier arithmetic decoder was slower. This is strong kernel evidence, not engine tok/s evidence: engine build flags, NaNs, load-time cost, whole-model integration, and correctness gates remain unresolved.

The best next single-user decode candidate is therefore:

> **Integrate the measured 16-row interleave plus exact FP8-to-FP16 reinterpretation kernel behind a guarded resident-layout tag, preserving 128-column scale boundaries and scalar accumulation order; then gate it in the engine before spending a resolution-grade model run.**

The E123 Amdahl bounds are useful but are not forecasts. Substituting the N=21 real-shape kernel times into the E123 profile gives a planning scenario of roughly **599 -> 487 ms/token**, or **1.67 -> 2.05 tok/s (+23%)**, if integration transfers without loss. It does not establish that outcome:

- 1.5x faster FP8 projection phase -> at most **1.121x** whole-decode;
- 2x -> at most **1.193x**;
- 4x -> at most **1.321x**;
- eliminating the phase -> at most **1.479x**.

The immediate operational finding is also clear: the host is on its 140 W adapter but currently reports **Low Power Mode on** and **High Power Mode off** for AC. Apple says High Power permits higher fan speeds and may improve sustained intensive work; it does not promise a fixed multiplier. Automatic/High Power therefore needs an operator-approved A/B, not an invented percentage. The old storage emergency has disappeared: the internal volume now has about 112 GiB available.

For MLX, a selective primitive experiment is feasible; a full MLX-LM port is not a drop-in backend and is especially awkward for a 155.43 GiB checkpoint on a 128 GB machine. Benchmark `quantized_matmul`/`gather_qmm` at exact Colibri shapes before integration. A selective bridge has a best-effort planning expectation of **negative to roughly +10% tok/s**, not a defensible positive point estimate. A full port is a multi-month project and may lose Colibri's disk-backed expert-store advantage.

## Fastest and most comfortable configuration today

For one-shot generation, the engine defaults are already the champion. For serve mode, explicitly set `COLI_V4_KERNELS=all`: serve honors an explicit value but does not inherit the one-shot CLI's implicit `all` default.

```bash
COLI_V4_KERNELS=all COLI_V4_METAL=0 \
  ./c/coli serve --model /absolute/path/to/deepseek-v4-flash --ram 64
```

- Use one persistent server instead of repeated fresh processes. This avoids the roughly 35-second load component and preserves a warm expert/session working set. It improves wall time, not steady tok/s.
- Start at 64 GB for a comfortable desktop profile. Raise toward 96 GB only when the machine is quiet and memory pressure remains green. The model payload is larger than RAM, so the expert store and SSD streaming remain necessary.
- Keep one engine active and record memory pressure, compressed memory, and swap growth before interpreting a slow run.
- Never combine `COLI_V4_PREFILL_PREFETCH=1` with the default GPU-prefill stack; E113 observed a 0%-CPU deadlock.
- Use `COLI_V4_BASELINE=1` for exact/reference work. For fastest interactive work, use shipping defaults and evaluate changed output by same-arm determinism, text diff, and capability.
- The current Low Power-on-AC state is inconsistent with a maximum-performance goal. Changing it is an operator choice because High Power can increase fan noise. Apple calls Automatic the default balance and describes High Power as additional cooling that *may* improve intensive workloads ([Apple Power Modes](https://support.apple.com/en-gb/101613)).

## What changed after E108

“Tok/s” below always means generated-token decode. Counters-only and `--max-tokens 1` runs are explicitly not decode evidence.

| experiment | prompt / N | surviving result | current reading |
|---|---|---|---|
| E109 | p064 counters, N=1 | Retracts E101: its MIN_N arms never enabled grouped MoE. Engaged Metal was about 0.38-0.44 ms/row versus roughly 0.69-0.81 CPU. | Correction and engagement proof. |
| E110 | p064, 60 slots, N=2 | About -13 to -16% TTFT direction, decode flat; a 36% CPU outlier invalidates the harness summary. | Direction only. |
| E111 | p064, 60 slots, N=2 | Grouped GPU-prefill without `KERNELS=all`: -13.9% TTFT, -0.05% tok/s, deterministic, taskcheck 5/5. | Clean grouped/capability result. |
| E112 | p256, 40 slots, N=2 | Multi-chunk TTFT -13.9%; decode slowed about 3%; interactive wall still won. | Magnitude superseded by E114. |
| E113 | p256, 40 slots, N=2 | CPU prefetch -3.0% TTFT and -1.39% tok/s; grouped + prefetch deadlocked. | Do not combine. |
| E113b | p256 counters | GPU-prefill engages with `METAL=0`: 52.9% Metal rows, 0.355 ms/row vs 0.703 CPU. | Independent gate proof. |
| E114 | p256, 40 slots, N=5 | GPU-prefill -13.41% TTFT, -1.94% tok/s, -10.43% wall; break-even around 1116 generated tokens. | Resolution-grade grouped result. |
| E115 | p256, 40 slots, N=3/arm | Prefill Metal attention adds -21.01% TTFT; decode +0.62% unresolved; bit-exact between arms. | Shipping default. |
| E116 | p256, 40 slots, N=3/arm | Batched prefill rows16 adds -7.32% TTFT; decode flat; deterministic paraphrase; taskcheck 5/5. | Shipping default; not decode rows16 evidence. |
| E117 | p256 counters | Whole-prompt scope raises mean expert group 4.14 -> 7.97 and gets all 43 layers above mean N=4. | Mechanism only. |
| E118 | p064, 40 slots, N=3/arm | Pre-whole-prompt champion -34.33% TTFT and flat tok/s. Whole-prompt is inert at one chunk. | Short-prompt generalization. |
| E119 | p256, 40 slots, N=3/arm | Whole-prompt adds -17.43% TTFT with identical decode medians; cumulative -48.19% TTFT, -38.49% wall, -1.75% tok/s. | Current champion. |
| E119a | p512 counters | Metal row share 69.8 -> 95.1%; CPU rows 28552 -> 4630. | Counters only. |
| E120 | p512, 40 slots, N=3/arm | Whole-prompt adds -25.89% TTFT and -21.73% wall. One 29.53 s decode outlier means decode is flat, not improved. | Prompt-length scaling. |
| E121 | source/gates | Promotes champion to defaults; splits baseline and shipping golden gates. | Current behavior. |
| E122 | p256 counters/gates | `MOE_TILE=1024` caps deferred buffers near 176 MB; tile 64 and single-tile endpoints reproduce known counters. | Memory bound, not decode speed. |
| E123 | p256, 39 generated, one profile | 23.360 s, 99.3% accounted; FP8 projections 32.4%, expert forward 32.2%. Pins 16 -> 158 leave expert forward flat while wait rises 66%. | Current diagnosis. |
| E124 | p256, 39 generated, profile A/B | Bit-exact row-major NEON: decode 23.235 -> 23.514 s, neutral/slower; reverted. | Simple SIMD widening closed. |

### Post-E124 standalone FP8 microprobe

This is not an `experiments_results.md` entry and generated no model tokens. It isolates the current FP8 GEMV semantics in `.backlog/lab/kbench/fp8bench.c`; tok/s and TTFT therefore do not exist for this result.

| arm / real matrix shape `(I x O)` | durable median, N=21 | independent rerun, N=7 | exactness / reading |
|---|---:|---:|---|
| `wq_b` 1024x32768 | **1.065 -> 0.510 ms, 2.09x** | **1.078 -> 0.510 ms, 2.11x** | bit-exact on generated finite codes |
| `wkv` 4096x512 | **0.113 -> 0.070 ms, 1.61x** | **0.110 -> 0.086 ms, 1.28x** | bit-exact; smallest/most overhead-sensitive |
| `wo_a` 4096x1024, called 8x | **0.317 -> 0.139 ms, 2.28x** | **0.163 -> 0.090 ms, 1.81x** | bit-exact; repeated parallel-region cost remains |
| `wo_b` 8192x4096 | **1.084 -> 0.508 ms, 2.13x** | **1.110 -> 0.504 ms, 2.20x** | bit-exact on generated finite codes |
| weighted per layer | **4.798 -> 2.200 ms, 2.18x** | **3.602 -> 1.820 ms, 1.98x** | `wo_a` counted 8 times |
| synthetic 7168x7168 | **2.15x, 65.87 GB/s** | **2.67x, 82.08 GB/s** | corroborating large-matrix arm |

The supporting bandwidth probe measured about 104.7-115.8 GB/s for a 2 GB CPU read. Scalar FP8 at roughly 13-31 GB/s across the real shapes was therefore not saturating the host's measured CPU read ceiling. The winning kernel maps each non-NaN E4M3 byte into an FP16 bit pattern representing `value / 256`, folds the exact power-of-two factor into the block scale, converts to FP32 in registers, and processes 16 output rows per contiguous byte load. The N=21 kernel sum is within about 6% of the E123 projection-phase time after multiplying across 43 layers, an encouraging cross-check. The benchmark deliberately excludes random NaN codes; engine validation must test all 256 byte values and the actual compiler contraction behavior.

Corrections other agents must retain:

- E101 did not exercise its claimed path.
- E110's baseline outlier contaminates its summary.
- E111's nondeterminism attribution is refined by E114: the `KERNELS=all` + GPU-prefill interaction can produce a minority output variant.
- E112's “prefill-only means decode-free” reasoning was false; initialization/cache state can carry into decode.
- E115/E116 decode differences are unresolved/flat, not gains.
- E117 and E119a generated no continuation tokens.
- E120 does not prove a decode win.
- E123's original “port the AVX2 algorithm” conclusion is corrected by E124 and the standalone microprobe: layout is necessary, but the measured Apple design is rows16 plus cheap FP16 reinterpretation rather than a literal rows8 AVX2 translation.
- E104's earlier “no bit-exact arm64 kernel beats scalar” conclusion is refuted for this isolated kernel. It tested rows4/single-thread assumptions; rows16 decodes one column across 16 independent output rows while preserving each row's ascending-column accumulation order.

## Metrics, baselines, and evidence rules

The project has three useful wall-clock views:

1. **TTFT**: load plus prefill. `bench/ab.sh` requests one token, so decode is excluded.
2. **Decode tok/s**: `.backlog/lab/tokps.sh` computes `(max_tokens - 1) / after_first`.
3. **Interactive wall**: TTFT plus decode duration for the requested continuation.

Do not call a TTFT improvement a model-wide speedup. Current defaults are the counterexample: they nearly halve p256 TTFT while leaving tok/s slightly worse.

Every proposed token-generating experiment must:

- verify `.backlog/lab/coli_usage.snapshot` MD5 `599f3d12e9347ef30541bd6f9ba18bde`;
- build with `METAL=1` and verify the Metal seam before testing a Metal flag;
- confirm only one `deepseek_v4` process exists;
- report prompt, token setting, N, TTFT, decode seconds, tok/s, total wall where relevant, MD5, and determinism;
- use a multi-chunk prompt for chunk/group/batch-conditioned features;
- run same-arm controls before interpreting ON-vs-OFF output differences;
- read/diff changed text and run `.backlog/lab/taskcheck.sh`; MD5 inequality is not capability failure;
- use N>=5 for sub-10% decode claims;
- ask before a long run and stop as soon as the kill criterion is met.

`bench/golden.sh` pins baseline and preserves sacred MD5 `5d04890413ff539e802985ce8c727814`. `bench/golden_default.sh` guards shipping behavior. A remaining defect is that `golden.sh` checks only that its `/tmp` seed exists; newer harnesses verify the durable seed hash.

## Current decode bottleneck

E123 measured 23,360 ms over 39 generated tokens:

```text
23.360 s / 39 = 599.0 ms/token = 1.6695 tok/s
```

| phase | measured share | derived ms/token | calls | 2x phase-local Amdahl speedup |
|---|---:|---:|---:|---:|
| FP8 attention projections | **32.4%** | **194.1** | 3354 combined | **1.193x** |
| routed `expert_forward` | **32.2%** | **192.9** | 10062 | **1.192x** |
| shared expert | 6.9% | 41.3 | 1677 | 1.036x |
| vocabulary head | 6.1% | 36.5 | 39 | 1.031x |
| expert wait | 5.7% | 34.1 | 20124 waits | 1.029x |
| other work | 16.7% | 100.0 | mixed | not one seam |

These are derived from one profile. They rank work; they do not forecast a benchmark. Re-profile after every material win.

### Why E124 was neutral

The current `c/quant.h` FP8 contract is raw row-major E4M3 bytes, one F32 scale per 128x128 block, a 256-entry decode table, and four output rows unrolled in the scalar loop. Each input column loads one activation but four weights from distant rows and four table entries. Vectorizing the four accumulators does not reduce those operations.

The AVX2-only packer at `c/deepseek_v4.c:746-808` changes an eight-row tile to:

```text
[tile][input_column][row_lane_0..7]
```

Its consumer loads eight contiguous bytes. Apple never runs that packer, so its dense views remain `block_rows=128` and use the scalar kernel. FP4 already uses a similar resident interleave/NEON lifecycle, so no on-disk migration is required. The new microprobe shows that rows16 aligns directly with one `vld1q_u8` and outperforms the literal rows8 precedent when paired with FP16 reinterpretation; it must still be implemented as a distinct tagged layout.

### Amdahl opportunity bounds

| accelerated phase | 1.25x local | 1.5x local | 2x local | 4x local | eliminated |
|---|---:|---:|---:|---:|---:|
| FP8 projections, 32.4% | 1.069x | 1.121x | 1.193x | 1.321x | 1.479x |
| routed expert, 32.2% | 1.069x | 1.120x | 1.192x | 1.318x | 1.475x |
| shared expert, 6.9% | 1.014x | 1.024x | 1.036x | 1.055x | 1.074x |
| head, 6.1% | 1.012x | 1.021x | 1.031x | 1.048x | 1.065x |
| expert wait, 5.7% | 1.012x | 1.019x | 1.029x | 1.045x | 1.060x |

This is why head, wait, and pin-count work should not lead the queue.

## M3 Max hardware: usable and unusable acceleration

Apple lists a 40-core GPU and 400 GB/s unified-memory bandwidth for this configuration ([Apple specifications](https://support.apple.com/en-my/117737)). Peak bandwidth is not automatically available to a four-row CPU loop or a tiny GPU grid.

Exact installed-host checks found Apple clang 21, NEON/AdvSIMD, FP16, BF16, DotProd, and **I8MM**; SME/SME2 are absent. Installed BNNS exposes Int4/UInt4/Indexed4 but no public FP8 type. MPS exposes Int4/UInt4 and quantized matrix multiplication but no public Float8 type matching this model.

Consequences:

- I8MM cannot consume E4M3 exactly. FP8-to-int8 needs another scale plus activation quantization, so it is a new lossy format.
- Direct Apple AMX programming is undocumented and unsuitable as a production dependency. Accelerate using private hardware internally is not evidence of a batch-1 FP8 primitive.
- BNNS `Indexed4` is a 4-bit index into a 16-entry table. Colibri uses 8-bit E4M3 codes, a 256-entry table, and 128x128 block scales.
- MPS quantized matmul merits only an isolated Int4 new-format probe. Standalone dequantization risks materializing weights.
- Apple's guidance names tiny compute grids as a low-occupancy cause and warns that high occupancy can still thrash caches, matching Colibri's S=1 Metal losses ([Apple GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy), [GPU counters](https://developer.apple.com/videos/play/wwdc2020/10603/)).

## Ranked candidates

| priority | candidate | expected/possible gain | work | risk | decision |
|---|---|---|---:|---:|---|
| P0 | Packed-row16 FP8 + FP16 reinterpretation NEON | **Measured real-shape microkernel:** 1.61-2.28x; **derived scenario:** about +23% tok/s if it transfers intact. | medium-high | medium | Best tok/s candidate. Integrate behind a layout tag, then gate. |
| P0 | Exact all-256-code and engine-build validation | Synthetic finite-code output is bit-exact; NaN and compiler-contraction behavior remain. | low-medium | medium | Required before any engine claim. |
| P0 operational | `OMP_NUM_THREADS=12` versus current 16 | V4 omits the repo's P-core-aware `omp_tune.h`; current OpenMP default includes four E-cores. Gain unknown. | low | low | Cheap operator-gated A/B; report both axes. |
| P0 operational | Automatic/High Power and scheduler residency control | Unknown; may improve sustained consistency. | low | low; fan tradeoff | Operator-gated A/B. |
| P1 | Six Metal expert chains in one command buffer/one wait | **Planning:** 5-15% expert-phase -> roughly +1.7-5.5% whole decode. | medium | medium-high | Secondary after FP8. |
| P1 | Fused dequant + projection/attention | Bounded by the same 32.4% projection share. | high | medium-high | Only after primitive locality wins. |
| P1 | Speculation snapshot/replay attribution and transactions | Potentially large only if state cost is material and acceptance pays. | medium/high | high | Instrument first. |
| P1 aggregate | Multiple KV slots + continuous decode batching | Aggregate tok/s can rise; interactive TPOT may worsen. | very high | high | Separate serving roadmap. |
| P2 | Selective MLX QMM/GQMM bridge | **Planning:** negative to about +10% end-to-end; per-phase 2x bound +19%. | high | high | Exact-shape probe only. |
| P2 | New Int4/AWQ/GPTQ/T-MAC/I8MM format | Fewer bytes; host gain and quality unknown. | high/very high | high | Follow exact-layout work. |
| P2 | Decode rows16 Metal admission | Historical 45.2% call coverage; no speed A/B and different hot kernel. | low measurement | medium | Secondary operator-gated A/B. |
| reject | More pin slots | Expert compute flat; wait +66%. | — | — | Closed. |
| reject | Simple row-major NEON | E124 neutral/slower. | — | — | Closed. |
| reject | Grouped prefill + prefetch | Deadlock. | — | high | Never combine. |
| reject | Persistent polling Metal kernel | Scheduling/watchdog/responsiveness risk. | very high | very high | Do not prioritize. |

### P0 implementation outline

1. Add a rows16 load-time packer for the resident attention projections, using the existing rows8/FP4 packing lifecycles as precedent.
2. Publish a packed layout only when the matching consumer is compiled.
3. Preserve 128-column scale boundaries.
4. Load 16 contiguous FP8 bytes per input position and preserve each row's product order.
5. Reuse the measured FP16 reinterpretation: construct `E4M3 / 256` exactly, convert lanes to FP32, and fold 256 into the block scale; canonicalize NaNs to match the LUT.
6. Keep row-major Metal dispatch separate; never reinterpret interleaved bytes as row-major.

The resident layout can be memory-neutral after publication, using only a temporary packing tile. Producer/consumer mismatch is the critical corruption hazard.

Before model timing, extend the now-winning real-shape microbenchmark to `O={128,256}`, `I={128,512}` boundary/tail cases; cover all 256 E4M3 codes, build with engine flags, and report exact output, latency, bytes/s, cache-hot/streaming, packing cost, and a path counter. Kill integration if any material shape loses the >=15% threshold or exactness/layout selection is ambiguous.

A separate low-cost lever is OpenMP placement. The attention path enters 12 parallel regions per layer per token: `wq_a`, `wq_b`, `wkv`, eight `wo_a` groups, and `wo_b`. `deepseek_v4.c` does not include the repository's P-core-aware `omp_tune.h`, so the current default uses all 16 logical CPUs, including four efficiency cores. An explicit 12-versus-16-thread engine A/B is justified, but no gain is claimed and it still requires the project's both-axis, N, determinism, and capability rules.

### P1 Metal orchestration

Current decode calls the one-expert Metal seam six times. The safest probe retains six expert leases, encodes six existing chains into one command buffer with disjoint scratch/output slices, submits once, waits once, and combines in current expert-ID order.

This tests submission/wait overhead without argument-buffer redesign, copying six expert banks, changing reductions, or using a persistent worker. A true one-dispatch shader or indirect command buffer follows only if this wins. Record commands/token, encoding, copy, GPU, wait, fallback, and output hashes.

### P1 speculation

The checkpoint already contains MTP/DSpark, Markov, and confidence-head tensors. The blocker is state cost. Each speculative round deep-copies every layer's sliding ring, compressed rows, recurrent compressor state, indexer rows, and indexer state; rejection restores and replays retained inputs.

First instrument snapshot/restore bytes and time, target verification, replay, proposal source, K, acceptance, and rejection position. Gate low-confidence blocks before snapshot creation. If ring copies dominate, use a hybrid overlay/undo log for speculative ring/compressed writes while copying the smaller recurrent arrays.

At 1.68 tok/s one target step costs about 0.595 s. With conditional acceptances `a_i`:

```text
E[L] = 1 + a1 + a1*a2 + ... + product(a1..aK)
```

Speculation wins only if `draft_time + verification_time < 0.595 * E[L]`. Acceptance alone is insufficient.

## MLX and MLX-LM feasibility

MLX core and MLX-LM are separate:

```text
checkpoint + tokenizer
        |
MLX-LM model module
  tensor conversion, MoE routing,
  cache semantics, generation/sampling
        |
MLX core
  lazy arrays, compile, streams,
  quantized_matmul/gather_qmm, Metal kernels
```

`mx.quantized_matmul` consumes MLX-packed weights/scales/biases plus group size, bit width, and mode ([pinned MLX layer](https://github.com/ml-explore/mlx/blob/d1140e61bd3481c321589c270ebb7d90570b0dcf/python/mlx/nn/layers/quantized.py#L270-L280)). MLX-LM routes quantized experts through `gather_qmm` ([pinned switch layer](https://github.com/ml-explore/mlx-lm/blob/77c33b14373ac70d7abd6f82af15962852adadbb/mlx_lm/models/switch_layers.py#L75-L90)). The Metal backend selects QMV/QMM/split-K by exact shape and may allocate an intermediate reduction ([pinned dispatch](https://github.com/ml-explore/mlx/blob/d1140e61bd3481c321589c270ebb7d90570b0dcf/mlx/backend/metal/quantized.cpp#L1804-L1873)).

Issue evidence warns against assumed wins: sorted `gather_qmm` was reported at about 69% of dense QMM for small expert groups on M3 Ultra ([#4246](https://github.com/ml-explore/mlx/issues/4246)); small-M/split-K is shape-dependent ([#3584](https://github.com/ml-explore/mlx/issues/3584)); quantized-KV small-M paths have had correctness defects ([#3480](https://github.com/ml-explore/mlx/issues/3480)). These are counter-signals, not Colibri benchmarks.

Feasible selective bridge:

- retain Colibri's loader, disk-backed experts, router, recurrent/MLA cache, tokenizer, sampler, and server;
- convert one dense projection or expert bank into an MLX-supported persistent format;
- call QMM/GQMM behind a flag with native fallback;
- separate conversion/warmup from steady M=1 execution.

Not drop-in:

- current E4M3 + 128x128 scaling is not an MLX affine/mxfp format;
- MLX owns arrays, streams, and command encoding;
- route ordering, duplicate indices, and sorting thresholds must match;
- V4 compressed/recurrent attention is not an ordinary KV cache;
- the 155.43 GiB checkpoint exceeds physical RAM before runtime/cache/macOS overhead.

Best-effort engineering and performance estimate:

| scope | work estimate | performance expectation |
|---|---:|---|
| exact-shape MLX QMM/GQMM benchmark | 3-7 engineering days | Determines sign; no engine claim. |
| one-layer converter + C++/ObjC++ bridge + differential | 1-3 weeks | Likely negative to modest until packing/copies amortize. |
| production selective backend with fallback/gates | 4-8 weeks | Planning range **negative to about +10% tok/s**. |
| full V4 MLX-LM port/checkpoint conversion | at least 2-4 person-months | No defensible positive estimate; may regress or not fit comfortably. |

Scenario bounds: accelerating one 32% phase 1.5x implies about +12% before bridge overhead; 2x implies +19%. Both 32% phases at 2x give a mathematical scenario near 1.48x total before contention, copies, routing, and paging. Require >=10% bridge-inclusive M=1 advantage before expanding.

## Quantization/layout methods that fit

T-MAC performs low-bit weight multiplication through table lookup on CPUs including ARM/macOS ([paper](https://arxiv.org/abs/2407.00088), [pinned code](https://github.com/microsoft/T-MAC/tree/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee)). It fits bandwidth-bound GEMV conceptually but needs a new 1-4-bit format; its external speedups do not transfer.

At pinned SHA `cc83d7b...`, llama.cpp separates quant-format dot kernels from matrix scheduling, supports multi-row calls, and uses NEON nibble unpack plus integer dot/late scaling ([ARM source](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cpu/arch/arm/quants.c#L702-L724)). Its Metal code keeps batch-1/small-batch matvec distinct from larger matmul and stages a 16-entry MXFP4 LUT per threadgroup ([Metal source](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-metal/kernels/mul_mv.metal#L2892-L2927)).

Transfer the principles, not the block sizes:

- specialize by format and M;
- test rows-per-call 1/2/4/8/16;
- fuse unpack/dequant and accumulation;
- apply scales late where the format permits;
- keep batch-1 matvec separate from prefill matmul;
- preserve route-map -> expert-matmul synchronization.

If exact layout work fails, test lower-bit conversion in this order: expert-/layer-selective W4/W8; one real-shape T-MAC/I8MM representation; AWQ/GPTQ calibration feeding that layout; sparse outlier correction only if quality requires it. BitNet/ternary is a training-time architecture, not a storage tweak.

## Comfortable sustained operation

Current snapshot:

- 140 W adapter connected, battery full;
- Low Power Mode **Yes** on AC; High Power **No**;
- about **112 GiB** internal storage available;
- **155.43 GiB** safetensor payload;
- about **5.54 GiB of 7.17 GiB** swap allocated after 18 days uptime;
- internal display plus 3840x1080 external display at 60 Hz;
- no active `deepseek_v4` process at audit time.

Historical swap allocation is not proof of current pressure; use pressure and swap *growth*. Storage is no longer urgent. Automatic is the control, High Power the sustained-performance arm, and Low Power the quiet arm. External-display impact is plausible but unquantified. Instruments/Metal System Trace, Thread State Trace, Thermal State, and GPU counters are the official measurement surfaces ([Apple Metal analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/)).

| goal | power | RAM | process | note |
|---|---|---:|---|---|
| fastest interactive | Automatic or approved High Power on AC | 64-96 GB | one persistent server | Explicit `KERNELS=all` in serve; observe pressure. |
| quiet/comfortable | Low Power on AC | 64 GB | one persistent server | Accept lower/unmeasured sustained speed for less fan noise. |
| deterministic/reference | fixed recorded mode | 64 GB | isolated one-shot | `BASELINE=1`, frozen seed, quiet host. |
| aggregate research | Automatic/High Power | measured per concurrency | multiple KV slots | Report p50/p95 TTFT/TPOT and aggregate tok/s. |

## Smallest decisive experiment sequence

### A. Finish and freeze the packed-row16 FP8 microbenchmark

- Current result: rows16 + FP16 reinterpretation wins 1.61-2.28x across all four real attention matrix shapes at N=21; an independent N=7 rerun remains positive at 1.28-2.20x. No model tokens were generated.
- Remaining cost: minutes, no model. Add edge/tail blocks, all 256 E4M3 codes, engine compiler flags, pack time, hot/streaming, and path-counter checks.
- Pass: >=15% on every material real shape with exact reference output.
- Kill: no material real-shape gain, any unresolved NaN/scale/accumulation mismatch, or ambiguous layout selection.

### B. Engine FP8 validation

- Ask first. Start p256/40, same-arm control plus A/B, N=2 only for a large expected signal; N>=5 if a sub-10% delta needs resolution.
- Report both axes, total wall, phase/path counters, MD5/determinism, multi-chunk text diff, and taskcheck.
- Kill: no phase reduction, no end-to-end tok/s signal, capability loss, or non-amortized pack cost.

### C. Power/scheduler control

- Operator-approved AC Automatic vs High Power, fixed display/background/software.
- Record tok/s, TTFT, thermal state, P/E residency, pressure/swap, and fan/noise.
- Kill: short-only gain, thermal convergence, or unacceptable comfort tradeoff.

### D. One-command-buffer top-6 Metal

- Measure commands/token, encode/copy/GPU/wait, expert phase, tok/s, and thermal state.
- Kill if expert phase does not improve or CPU remains faster end-to-end.

### E. MLX primitive feasibility

- Real `(M,K,N)`, `M={1,2,4,8}`, relevant bits/groups, sorted/unsorted indices.
- Separate warmup/compile, packing/copy, scratch, steady latency, and numerical error.
- Require >=10% bridge-inclusive M=1 win; otherwise stop.

### F. Speculation attribution

- Instrument snapshot/restore bytes/time, verify/replay, K, source, accepted prefix, and avoided/added target forwards.
- Implement transactions only if a named state-copy bucket is material and round economics are positive.

## Closed or demoted methods

- simple row-major NEON FP8: neutral, reverted;
- more pin slots: compute flat, wait worse;
- current single-token Metal experts: CPU wins;
- grouped prefill + prefetch: deadlock;
- `MOE_TILE` as decode tuning: wrong axis;
- deep I/O queueing: not the profiled compute bottleneck;
- generic n-gram/Markov/MTP defaults: acceptance/replay economics are workload-dependent;
- unfused KV quantization: can reduce tok/s; V4 KV is already compressed;
- direct AMX/private MPS symbols: undocumented;
- persistent Metal polling: scheduling/watchdog risk;
- M4/M5 percentages: not transferable;
- continuous batching as single-user speedup: wrong objective.

## Limitations and open questions

- E123 is one p256 profile; re-profile after any engine win.
- Decode noise is 5-13%; small estimates are expensive to resolve.
- No percentage is transferred from another Apple chip/model/runtime.
- The rows16 FP8 result is a standalone microkernel on synthetic data, not engine tok/s; the 1.28-2.20x real-shape N=7 rerun and the wider square-matrix variation demonstrate shape and host-load sensitivity.
- The derived 2.05 tok/s scenario assumes the isolated phase speedup transfers through load-time packing, engine compiler behavior, 20,124 OpenMP region entries per 39-token generation, and surrounding attention work.
- I8MM exists, but no int8 model format was tested.
- Power state is telemetry, not performance evidence.
- MLX work/gain ranges are planning estimates.
- Newer Metal feature tables do not establish a ready M3 FP8 GEMV API.
- The earlier request referred to a “post,” but no URL/body is present in the available context; no method is attributed to an unidentified post.

Open questions:

1. Does rows16 FP16 reinterpretation remain exact and fast in the engine under its compiler flags and layout lifecycle?
2. Can NaN canonicalization and 128-column scale boundaries be preserved with no hot-loop penalty?
3. What fraction of speculative time is snapshot/restore versus verify/replay?
4. Does current Low Power materially reduce sustained memory throughput?
5. Can six Metal chains share input/submission without extra weight copies?
6. Which MLX branch handles the real M=1 shapes, and does it win after bridge overhead?
7. For aggregate serving, is cross-sequence expert overlap high enough to reuse streamed weights?

## Source map

Local: `AGENTS.md`; `experiments_results.md:E109-E124`; `c/deepseek_v4.c`; `c/quant.h`; `c/backend_metal_v4.mm`; `c/metal/*.metal`; `.backlog/lab/tokps.sh`; `.backlog/lab/taskcheck.sh`.

External:

- [Apple Power Modes](https://support.apple.com/en-gb/101613)
- [Apple M3 Max specifications](https://support.apple.com/en-my/117737)
- [Apple GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy)
- [Apple Metal performance analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/)
- [Apple GPU counters](https://developer.apple.com/videos/play/wwdc2020/10603/)
- [MLX quantized backend, pinned](https://github.com/ml-explore/mlx/blob/d1140e61bd3481c321589c270ebb7d90570b0dcf/mlx/backend/metal/quantized.cpp)
- [MLX-LM switch layers, pinned](https://github.com/ml-explore/mlx-lm/blob/77c33b14373ac70d7abd6f82af15962852adadbb/mlx_lm/models/switch_layers.py)
- [llama.cpp ARM quant kernels, pinned](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cpu/arch/arm/quants.c)
- [llama.cpp Metal matvec, pinned](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-metal/kernels/mul_mv.metal)
- [T-MAC paper](https://arxiv.org/abs/2407.00088) and [pinned code](https://github.com/microsoft/T-MAC/tree/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee)
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec)

## Bottom line

Keep the shipping defaults for interactive use, explicitly add `COLI_V4_KERNELS=all` in serve mode, and do not enable single-token Metal merely because the GPU is present. The champion has largely solved TTFT; decode remains a memory-layout problem.

The no-model gate has now produced a real winner: **packed-row16 FP8 with exact FP16 reinterpretation**. The next engineering step is guarded engine integration plus all-256-code/build-flag validation, not another generic SIMD rewrite. Real-shape kernels win 1.61-2.28x, supporting a derived **about 2.05 tok/s (+23%)** scenario only if the result transfers without packing or integration loss. A 12-versus-16 OpenMP-thread A/B is the cheapest separate host lever. MLX, new quantization, speculative transactions, and deeper Metal orchestration remain behind these gates.
