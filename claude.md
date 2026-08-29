# Colibri DeepSeek-V4 architecture and M3 Max performance guide

Last researched: 2026-08-29

Repository state: `simd-apple-metal` at `d5864b00e4a56f9b1f5f9d274b13d387698e3e38`

Host: MacBook Pro `Mac15,9`, M3 Max, 12P+4E CPU, 40-core GPU, 128 GB unified memory
Full current research: [`.backlog/m3-max-decode-research-2026-08-29.md`](.backlog/m3-max-decode-research-2026-08-29.md)

## Read this first

Colibri does not load the full frontier MoE model into GPU memory. The DeepSeek-V4 checkpoint is about 155.43 GiB, larger than this host's 128 GB unified memory. Colibri keeps dense/head tensors resident, maintains a bounded hot expert cache in RAM, streams cold expert records from safetensors on SSD, routes each token to six experts, and overlaps/reuses expert work where possible.

```text
checkpoint/index
      |
      v
dense tensors + compressed/recurrent attention state
      |
token -> attention -> router/top-6 -> shared expert
                                |
                         expert-store lookup
                         /                 \
                    RAM slab hit       SSD miss
                         \                 /
                    six routed expert forwards
                                |
                    weighted ordered accumulation
                                |
                          residual -> LM head
```

The current performance design is phase-specific:

- **Prefill** uses grouped/batched Metal experts, Metal attention, hot rows16 admission, and whole-prompt expert grouping.
- **Single-token decode** keeps routed experts on the CPU because the current Metal path is slower at S=1.
- `COLI_V4_METAL=0` disables decode Metal only. It does not disable the independently gated prefill GPU paths.

## Shipping defaults

As of E121, these default on:

```text
COLI_V4_KERNELS=all
COLI_V4_MOE_GROUPED=1
COLI_V4_MOE_BATCHED=1
COLI_V4_MOE_BATCHED_ROWS16=1
COLI_V4_MOE_WHOLE_PROMPT=1
COLI_V4_METAL_ATTN=1
COLI_V4_METAL_VARIANT=simd_exact_cold
```

These remain off:

```text
COLI_V4_METAL=0          # single-token decode Metal
COLI_V4_METAL_ROWS16=0   # decode-only hot rows16 admission
```

`COLI_V4_BASELINE=1` restores all historical defaults. Explicit individual environment variables override baseline/default selection.

Important serve mismatch: one-shot CLI makes unset `COLI_V4_KERNELS` behave as `all`, but serve mode does not. Set it explicitly for production serve:

```bash
COLI_V4_KERNELS=all COLI_V4_METAL=0 \
  ./c/coli serve --model /absolute/path/to/deepseek-v4-flash --ram 64
```

Never combine the champion stack with `COLI_V4_PREFILL_PREFETCH=1`; E113 observed a deadlock.

## Current performance truth

At p256, 40 output-token slots, current champion versus CPU + `KERNELS=all`:

- TTFT: **-48.19%**
- total wall: **-38.49%**
- decode: **-1.75% tok/s**

Decode remains about **1.67-1.68 tok/s**. Do not describe the large TTFT improvement as a decode gain.

E123's current p256 decode profile, 39 generated tokens, 23.360 seconds, 99.3% accounted:

| phase | share |
|---|---:|
| FP8 attention projections | **32.4%** |
| routed expert forward | **32.2%** |
| shared expert | 6.9% |
| LM head | 6.1% |
| expert wait | 5.7% |

E124 tested a bit-exact NEON rewrite of the row-major FP8 projection loop. It was neutral/slower and was reverted because it retained the same lookup and distant-row access pattern.

The x86 fast path also changes layout: it packs eight output rows as `[tile][column][lane]`, so eight weights are contiguous. A same-day standalone Apple microprobe has now gone further: rows16 interleave plus exact E4M3-to-FP16 bit reinterpretation measured **1.61x-2.28x** across all four real attention matrix shapes at N=21, with a weighted per-layer kernel sum of **4.798 -> 2.200 ms**. An independent N=7 rerun kept every real shape positive at **1.28x-2.20x**. This is synthetic kernel evidence, not engine tok/s; NaNs, engine flags, packing cost, integration, and model gates remain. If the isolated result transferred intact, the E123 profile gives an unverified scenario near **1.67 -> 2.05 tok/s (+23%)**.

The best next decode candidate is guarded rows16 integration using the measured reinterpretation kernel. Rows8 is useful precedent, not the final Apple design.

## Ground-truth map

| area | source | ownership |
|---|---|---|
| current rules/defaults | `AGENTS.md` | benchmark, correctness, champion stack |
| performance ledger | `experiments_results.md:E109-E124` | measurements, corrections, negative results |
| engine/session loop | `c/deepseek_v4.c` | load, prefill, decode, routing, speculation |
| tensor/FP8 contract | `c/tensor.h`, `c/quant.h` | layouts, scales, E4M3 LUT, CPU kernels |
| expert-store contract | `c/expert_store.h` | leases, lookup/release, concurrency limits |
| Metal ABI/host | `c/backend_metal_v4_seam.h`, `c/backend_metal_v4.mm` | dispatch, buffers, pipelines, waits, counters |
| Metal kernels | `c/metal/coli_v4_moe.metal`, `c/metal/coli_v4_fp8_matmul.metal` | expert and FP8 math |
| CLI/server | `c/coli`, `c/openai_server.py` | process lifetime, serve/API boundary |
| decode harness | `.backlog/lab/tokps.sh` | TTFT + decode + determinism |
| capability gate | `.backlog/lab/taskcheck.sh` | task-level correctness for non-identical output |
| standalone FP8 probe | `.backlog/lab/kbench/fp8bench.c`, `.backlog/ulw-decode-fp8-20260829-142423.md` | real-shape microkernel evidence; not engine tok/s |
| detailed M3 research | `.backlog/m3-max-decode-research-2026-08-29.md` | candidate ranking and experiments |

## Execution structure

### Load and residency

The tensor plan is heterogeneous: FP8-style attention/expert weights, BF16/F32 control tensors, and specialized routing/state tensors. Dense/head residency is planned against the RAM budget; routed expert weights are cached/streamed. A `ColiExpertView` must remain leased until CPU/GPU use completes and must be released once.

### Attention

Single-token decode uses the CPU attention path with recent-window plus compressed/indexed state. Batched prefill has separate Metal FP8 projection/attention gates. This is why `METAL_ATTN=1` can improve TTFT without accelerating decode.

### Routing and experts

Each decode token traverses 43 layers and selects top-6 experts. Shared-expert work can overlap expert loading. Routed outputs are weighted and combined in a defined order.

During prefill, the grouped scheduler sorts routed rows by expert. Batched Metal admits groups meeting `MOE_BATCHED_MIN_N` and supported layouts. Whole-prompt mode defers the MoE finish across prompt chunks, increasing same-expert group sizes while attention remains chunked at 64.

`MOE_TILE=1024` bounds the deferred buffers to roughly 176 MB. It is a prompt-memory/grouping control, not a decode knob.

### Speculation

The checkpoint includes MTP/DSpark, Markov, and confidence-head tensors. Verification snapshots all attention layers. Rejection restores state and replays retained inputs because recurrent compressed state cannot be truncated with a KV cursor. Instrument snapshot/restore/verify/replay before changing draft depth.

## Correct interpretation of recent experiments

- E109 retracts E101: it never enabled grouped MoE.
- E110 contains a 36% baseline outlier; do not quote its harness summary.
- E114 is the resolution-grade grouped result: -13.41% TTFT, -1.94% tok/s, -10.43% wall at p256, N=5.
- E115 Metal attention and E116 prefill rows16 improve TTFT; neither proves a decode gain.
- E119 whole-prompt grouping is decode-flat and is now part of the champion.
- E120's decode column contains an outlier; read it as flat.
- E123 kills higher pin counts: expert compute stays flat while wait rises 66%.
- E124 kills simple row-major NEON widening, not layout-aware NEON.
- The post-E124 microprobe refutes E104's isolated-kernel conclusion: rows16 decodes one column across 16 independent output rows while preserving each row's scalar accumulation order.

## Benchmark and correctness rules

- TTFT and decode are separate axes. `bench/ab.sh` contains no decode measurement.
- Any token-generating experiment must report both TTFT and tok/s.
- Decode noise is 5-13%; use N>=5 to resolve sub-10% deltas.
- Use the durable seed and verify its MD5 before every run.
- Build with `METAL=1` and verify the Metal seam before interpreting Metal flags.
- Run same-arm determinism before ON/OFF comparison.
- A short golden prompt may never execute a chunk/group-conditioned path.
- For multi-chunk differences, read/diff generated text and run `taskcheck`; changed MD5 alone is not failure.
- Ask before any long benchmark.

## Ranked next work

1. **Packed-row16 FP8 + FP16 reinterpretation integration**: highest-value single-user tok/s candidate. Real-shape microkernels win 1.61-2.28x; the derived +23% tok/s scenario remains unverified until engine A/B.
2. **All-256-code and engine-build validation**: verify NaNs, scale boundaries, compiler contraction, edge/tail shapes, and packing cost before any tok/s claim.
3. **12-versus-16 OpenMP threads**: V4 omits the repo's P-core-aware `omp_tune.h`, so its 16-thread default includes four E-cores. This is a cheap separate A/B with no assumed gain.
4. **Power-mode A/B**: current Low Power-on-AC state is a testable host control, with no percentage assumed.
5. **Six experts in one Metal command buffer**: test submission/wait amortization without a persistent kernel or new reduction order.
6. **Speculative-state instrumentation**: only pursue COW/undo state if snapshot/restore is material.
7. **Selective MLX QMM/GQMM probe**: exact M=1 shapes first; full port only for broader compatibility, not presumed speed.
8. **Continuous decode batching**: aggregate-throughput roadmap only; it may worsen one-user latency.

Rejected/demoted: more pins, row-major NEON, current S=1 Metal as default, grouped+prefetch, `MOE_TILE` as decode tuning, undocumented AMX/private MPS APIs, persistent polling Metal kernels, and transferred M4/M5 percentages.

## MLX quick answer

MLX core supplies lazy unified-memory arrays, streams, compilation, quantized matmul, gather-QMM, and Metal kernels. MLX-LM supplies model-specific tensor conversion, routing, attention/cache, generation, tokenizer, and sampling. Safetensors alone does not make a model compatible.

A selective primitive bridge is feasible while Colibri retains its expert store and V4 semantics. A full MLX-LM V4 port requires checkpoint conversion, exact routing, recurrent/compressed cache support, server integration, and a solution for a 155.43 GiB model on 128 GB RAM.

Best-effort planning:

- exact-shape primitive probe: 3-7 engineering days;
- one-layer bridge: 1-3 weeks;
- production selective backend: 4-8 weeks, likely negative to about +10% tok/s until proven;
- full port: at least 2-4 person-months, with no defensible positive performance estimate.

See the full report for sources, Amdahl scenarios, MLX pinned-code evidence, and the experiment gates.

## Host operation

Current state: 140 W AC, Low Power on, High Power off, roughly 112 GiB storage free, 5.54 GiB swap allocated after 18 days, external 3840x1080 display, no engine active at audit.

For fastest interactive work, use Automatic or operator-approved High Power on AC, one persistent server, and 64-96 GB RAM only while pressure stays green. For quiet work, Low Power + 64 GB is reasonable. For reference work, fix the power/background state, set `BASELINE=1`, and use isolated one-shot processes.
