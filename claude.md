# Colibri architecture, Metal performance, and MLX feasibility

Last researched: 2026-08-24  
Local machine: MacBook Pro `Mac15,9`, M3 Max, 16 CPU cores (12P+4E), 40-core GPU, 128 GB unified memory, macOS 26.6.1  
Scope: current `ft-opensourceftw1` worktree; no commit or push was made for this report.

## Read this first

Colibri is not a conventional “load the whole model on the GPU” runtime. For its frontier MoE engines it plans dense tensors as resident or streamed according to the memory budget, keeps a selected hot expert working set in RAM, leaves cold expert records in safetensors shards on SSD, routes each token to a small expert subset, and overlaps expert loading with CPU/GPU work. That is how a model larger than physical memory can run on one machine.

For the current DeepSeek-V4 path, the important split is:

```text
model/config/index
       |
       v
planned dense tensors + compressed/sparse KV state
       |
token -> attention/MLA -> router/top-k -> shared expert
                                  |
                     expert-store lookup/prefetch
                         |                 |
                  RAM slab hit         SSD miss
                         \                 /
                          routed expert compute
                          CPU or Metal batch
                                  |
                         weighted ordered scatter
                                  |
                           residual -> LM head
```

The current Metal gains do not come from “GPU = faster.” The first scalar Metal path was slower. The gains came from creating enough work per dispatch, preserving exact accumulation order, and eliminating many small projection/expert operations:

- FP8 attention projections use bit-exact double-float accumulation on the GPU.
- Eight `wo_a` projection groups are fused into one grouped dispatch.
- Prefill routes are grouped by expert so one loaded expert serves multiple token rows.
- Groups with at least four rows can use one batched Metal MoE dispatch.
- Page-aligned resident expert slabs are wrapped as shared, no-copy `MTLBuffer` objects.
- Unsupported layouts or failed Metal calls fall back to the CPU path.

The strongest current M3 result is TTFT, not decode: the composed grouped-MoE, Metal-attention, and batched-MoE path reduced the p064 harness wall from 42.528 s to 31.975 s, about **1.33x**. The features remain opt-in/default-off. Do not describe that as a universal 33% decode gain.

## Ground-truth map

| Area | Current source of truth | What it owns |
|---|---|---|
| DeepSeek-V4 engine | `c/deepseek_v4.c:9017-9081,9524-9624` | model construction, token/session loop |
| V4 tensor plan | `c/deepseek_v4.c:12504-12618` | tensor names, shapes, FP8/BF16/F32/I64 roles |
| single-token MoE | `c/deepseek_v4.c:4823-5029` | route, async load, shared/routed experts, accumulation |
| grouped/batched MoE | `c/deepseek_v4.c:5067-5485` | route grouping, expert waves, Metal batch gate, fallback |
| expert-store contract | `c/expert_store.h:33-92` | lookup lease, mandatory release, advisory prefetch |
| V4 store implementation | `c/deepseek_v4.c:7022-7485,8257-8384` | safetensors records, RAM slots, SSD reads, hot pins, tier reporting |
| compressed/sparse KV | `c/deepseek_v4.c:1910-2040,2630-2812,6774-6876,11732-11902` | window + selected compressed entries, sparse attention |
| speculation | `c/deepseek_v4.c:10420-10740` | D-Spark draft/target state, KV rollback, exact verification |
| V4 Metal C ABI | `c/backend_metal_v4_seam.h:10-52` | enablement, stats, slab registration, expert/FP8 calls |
| V4 Metal host | `c/backend_metal_v4.mm:180-289,428-640,664-1129` | buffers, pipelines, dispatch, sync, counters, fallback |
| V4 expert shader | `c/metal/coli_v4_moe.metal:48-208` | FP4 decode and routed expert kernels |
| V4 FP8 projection shader | `c/metal/coli_v4_fp8_matmul.metal:3-102` | bit-exact batch/group FP8 matmul |
| server process boundary | `c/openai_server.py:1375-1655,2153-2277` | model detection, persistent child engine, request/stream protocol |
| CLI engine selection | `c/coli:229-315,812-1010` | architecture selection, binary launch, attached chat |
| quantized format registry | `docs/FORMATS.md` | packed layouts, scales, status, collision rules |
| measured Metal history | `experiments_results.md:E66-E84` | hypotheses, corrections, A/Bs, correctness gates |
| M3 research ledger | `performance-boost-research.md` | decode profile, completed/killed/proposed methods |
| M5 system result | `docs/METAL-M5MAX-PERF-REPORT.md` | OMP power trap and PIPE result on a different machine/model |

Do not conflate the DeepSeek-V4 engine with `c/colibri.c`. The latter is the older/general engine and remains useful for shared quant/backend concepts, but V4 has its own store, scheduler, attention, and Metal seam. Likewise, `c/tier.h` is generic/older tier policy; the V4 hot-store policy lives in `deepseek_v4.c`.

## Current DeepSeek-V4 execution

### 1. Load and planning

`coli_v4_config_load` and the safetensors index establish tensor geometry. The runtime computes a memory plan and opens an expert store. The tensor plan is heterogeneous, not one model-wide dtype:

- attention and shared/routed expert projections use FP8-oriented weight plans;
- norms and several projections/router tensors use BF16;
- sinks, compressor/indexer controls, and normal router bias use F32;
- hash-router layers can use an I64 token-to-expert table.

This matters for MLX or converter work: “the model is FP8” is not a sufficient format description.

### 2. Attention and KV

The model uses MLA-like projections plus recent-window and selected compressed KV entries. Runtime code constructs the sparse key/value set, applies attention, and records phase attribution. The standalone cache API at `deepseek_v4.c:11732-11902` mirrors the compressed-window contract, but the active engine paths are the earlier execution functions.

Attention was 38.7% of the measured M3 decode profile snapshot. That number predates later CPU/Metal changes; it is a prioritization signal, not the current exact share.

### 3. Routing and experts

Routing has two live forms:

- hash-router layers obtain selected experts from `ffn.gate.tid2eid`;
- normal layers calculate sigmoid/top-k scores from the router projection and bias.

The engine canonicalizes route IDs/weights, starts expert loading, computes independent shared-expert work, finishes the load, runs the selected expert, adds its weighted output in deterministic order, and releases the expert lease exactly once.

The store reports two V4 residency tiers:

- tier 1: resident RAM slab;
- tier 0: disk-backed safetensors record.

Metal can read registered RAM slabs through unified memory, but that does not create a third VRAM tier.

### 4. Prefill grouping and Metal batch gate

The grouped path routes a token batch, sorts route items into expert groups, loads unique experts in waves, and reuses each lease across its rows. With `COLI_V4_MOE_BATCHED=1`, a group is sent to Metal only when:

- its row count is at least `COLI_V4_MOE_BATCHED_MIN_N` (default 4);
- gate/up/down use the proven rows1 layout;
- the Metal seam accepts the tensors and dimensions.

Rows16 hot slots intentionally stay on CPU because rows16 batch exactness has not been proven. A nonzero Metal return falls through to per-row CPU execution, so no route contribution is dropped.

### 5. Serve/API boundary

`c/openai_server.py` is already a multi-engine gateway. `Engine` starts a persistent child, waits for a binary `READY` sentinel, submits framed `SUBMIT` requests, receives `ACCEPT`/`DATA`/`DONE`/`STAT`, and exposes OpenAI/Anthropic-compatible HTTP routes. This boundary is the least invasive place to add another complete inference runtime such as `mlx-lm`.

## How the current speed changed

All numbers below retain their original scope. `bench/ab.sh` measures TTFT because it requests one generated token and parses `time_to_first_token`; p064/p256 are prompt filenames. Its delta is `100*(on-off)/off`, so faster is negative.

| Stage | Scope | Result | Interpretation |
|---|---|---:|---|
| first scalar V4 Metal expert path | M3, real-model short run | TTFT 2.11x slower; after-first 2.30x slower | S=1 granularity/dispatch/first-touch defeated the GPU |
| controlled scalar Metal retest | M3, 300-token interleaved n=3 | expert 2.747x slower; end-to-end 1.484x slower | reliable negative baseline; later cause attribution corrected |
| CPU paired-row matvec | M3 decode | 1.26x, byte-identical | reuses the activation walk and fuses gate/up work |
| CPU rows4 matvec | M3 decode | 1.38-1.42x cumulative, byte-identical | strongest local decode optimization in the earlier ledger |
| grouped MoE only | M3 TTFT | 1.043x p064; 1.021x p256 | CPU scheduling/reuse win, default off |
| Metal attention after fused `wo_a` | M3 TTFT | 1.133x p064; 1.142x p256 | fewer, larger, bit-exact projection dispatches |
| grouped MoE + Metal attention | M3 TTFT | 1.189x p064; 1.179x p256 | measured composition, not multiplied estimates |
| batched Metal MoE increment | M3 TTFT | 1.121x p064; 1.133x p256 | only groups `N>=4`, rows1, path engaged 1009 times |
| full current opt-in stack | M3 TTFT p064 | 42.528 → 31.975 s, ~1.33x | strongest current M3 Metal-stack claim |
| passive OMP + PIPE | M5 Max, GLM-5.2, warm decode | 2.06 → 2.24 tok/s, +8.5% | separate single-run evidence; do not merge with M3 TTFT |

Why batching changed the sign:

- at production MoE dimensions, the local probe showed GPU/CPU ratios below 1 at S=1 and S=2;
- at S=4, gate/up was 1.68x and down was 2.41x faster on GPU;
- at S=16, both were about 4.4x faster in the isolated probe;
- grouping also reuses a loaded expert and amortizes its weight walk across rows.

Isolated kernel speed remains an upper bound. In this repository, surrounding QDQ, gather, scatter, and synchronization have repeatedly diluted a microbenchmark by about 1.5-2x. E84 happened to beat its projection; that is recorded as an unexplained result, not a new transfer rule.

### Implemented, default-off, killed, and open

Implemented/default-off:

- `COLI_V4_MOE_GROUPED=1`
- `COLI_V4_METAL_ATTN=1`
- `COLI_V4_MOE_BATCHED=1`
- `COLI_V4_MOE_BATCHED_MIN_N` (default 4)
- `COLI_V4_METAL_STATS=1` for engagement/timing evidence

Important killed or negative lines:

- scalar expert Metal at S=1;
- unconditional prewarm of hundreds of experts;
- generic loader-depth increases after cold-cache correction;
- RoPE caching at the measured workload;
- the current speculation retune on coding prompts;
- GPU FP8 QDQ in the already-composed configuration (0.35% wall ceiling);
- the current prefill-prefetch gate (6.2%/2.7%, below its 15% gate);
- cache-policy claims from short runs before the cache filled.

## The referenced post is missing

The request says “the following post,” but no URL, title, author, platform, attachment, or post body is present in the conversation or repository. `performance-boost-research.md` and `docs/experiments/inference-paper-test-matrix-2026-07-28.md` cover many papers and methods; neither identifies one post. Picking a Reddit or Hugging Face article would be fabrication.

The closest transferable method family, without attributing it to an unknown post, is sparse MoE offloading:

1. keep dense layers and a hot expert subset resident;
2. store cold experts in a slower/larger tier;
3. quantize weights to reduce resident and transfer bytes;
4. predict/prefetch selected experts;
5. overlap I/O with independent compute;
6. batch/group tokens by expert to reuse each weight read;
7. fuse GPU work so kernel throughput survives dispatch/sync overhead.

Colibri already implements 1, 2, 3, parts of 4/5, and the current grouped form of 6/7. Once the actual post is supplied, compare its concrete placement policy, predictor, batching dimension, quantization layout, and reported metric against the table below instead of redoing the architecture survey.

## M3 Max-specific optimization roadmap

The machine has 400 GB/s advertised peak unified-memory bandwidth, but neither the CPU cluster nor a small S=1 kernel reaches that peak. The model touches at least about 3.45 GB of expert bytes per decoded token in the profiled configuration before cache-line, activation, scale, and reread overhead. This is why useful work per weight read matters more than theoretical GPU FLOPs.

The profile snapshot gives useful Amdahl bounds:

- making the 38.7% attention phase 2x faster caps whole-token gain near 1.24x;
- making the 36.9% routed expert+wait phase 2x faster caps gain near 1.23x;
- making both 2x faster caps gain near 1.61x on that snapshot;
- later optimizations changed the mix, so reprofile before using these as current forecasts.

| Priority | Method | Fit to current code/machine | Expected gain | Work | Required gate |
|---|---|---|---|---|---|
| P0 | Reprofile the current composed binary | old profile predates rows4 and current Metal lanes | enables honest ranking; no direct gain | S | 98%+ phase accounting with every chosen flag recorded |
| P0 | Passive OMP policy when V4 Metal is active | active spinning can steal shared SoC power/thermal budget; M5 showed a -39% trap | potentially material, unmeasured on this M3 | S | interleaved n>=3, same binary/history, GPU kernel clocks/times, compressor quiet |
| P1 | Full fused V4 attention/MLA Metal path | attention is the largest measured phase; current Metal moves only projections | 2x attention would imply ~1.24x old-profile whole-token bound | L-XL | B=1/4/8 and context 512/2k/8k; exact fallback; counters and end-to-end |
| P1 | Extend grouped/batched MoE carefully | current rows1 prefill path works; rows16 and decode batching remain | already +12-13% incremental TTFT where engaged; extensions unknown | M-L | S histogram, rows16 0-ULP proof, engagement count, end-to-end not probe-only |
| P1 | Continuous decode batching in serve mode | multiple requests create S>1 where single-request decode cannot | aggregate throughput can rise; latency tradeoff unknown | L-XL | concurrency 1/2/4/8, p50/p95 TTFT/TPOT, expert union/cache pressure |
| P1 | Continue CPU activation/weight-walk optimization | CPU rows4 is already the strongest decode evidence | remaining gain unknown after reprofile | M | same prompts, byte/token exact, phase attribution |
| P2 | Prefix/KV reuse and cache quantization | strong for repeated/long contexts; V4 KV is already compressed | TTFT/workload dependent | M | repeated-prefix workloads; bytes copied/allocated; continuation identity |
| P2 | Predictor-driven expert prefetch | literature supports it, but local prefetch and compulsory-miss evidence is weak | only useful if ready-recall avoids visible stalls without extra bytes | M-L | held-out recall, false-prefetch bytes, avoided wait, net tok/s |
| P2 | Speculation-aware grouping/prefetch | drafts can create a batch, but current coding sweep regressed | revisit only after group/replay economics improve | M | acceptance, union size, replay, extra I/O, held-out net speed |
| P2 | Additional format/quantization work | lower bits reduce bytes; packing/scales/quality dominate | potentially large capacity win; speed is not automatic | L | same-model quality, actual bytes, dequant cost, useful tok/s |
| P3 | Argument-buffer expert table / indirect dispatch | supported on Apple9; could reduce CPU binding if many resources are GPU-driven | unknown and likely secondary to current kernel work | M-L | prove binding/CPU encode time is material first |
| P3 | V4 residency set | older backend has one, V4 does not; V4 directly binds a small resource set per call | likely low unless profiling finds residency overhead | S-M | resource/encode attribution before implementation |
| P3 | Private GPU-only hot weights | conflicts with current no-copy streaming and tight memory headroom; copying can erase the benefit | unknown; test only for a small permanent hot set | M | count extra bytes/copies/compression; compare shared no-copy |
| P3 | `MTLHeap` scratch pooling | V4 scratch buffers already persist and grow; allocator churn is not established | probably small | M | allocation trace must show a real wall contribution |
| P3 | pipeline binary archive/function constants | useful mainly for startup and branch specialization | TTFT/startup only unless branches are expensive | M | first-run pipeline time and steady-state branch attribution |
| Do not prioritize | GPU QDQ alone | measured 0.35% wall ceiling in target configuration | below noise floor | — | killed by attribution |
| Do not assume | M5 neural-accelerator speedups from Metal TensorOps | M3 is Apple9; Metal 4 tensor APIs are available, but the per-core neural accelerator is Apple10/M5-class hardware | hardware speedup does not transfer | — | runtime family/capability test plus real kernel A/B |

### Relevant external methods

- **MoE-Infinity:** trace-aware expert cache/prefetch for limited-memory personal machines. Transfer the evaluation method (ready-recall, cache hits, transfer stalls), not its CUDA speedup percentage.
- **PowerInfer:** hot/cold activation placement across GPU/CPU. Colibri already uses a related hot-resident/cold-streamed idea at expert granularity; Apple unified memory changes the PCIe economics.
- **Grouped GEMM/expert batching:** directly relevant and already validated locally. Continue from local S histograms rather than paper averages.
- **Flash/MLA attention:** conceptually relevant because attention is locally dominant, but NVIDIA/H100 percentages do not project to Apple9 Metal.
- **Paged/persistent KV:** useful for multi-request and repeated-prefix workloads; V4’s compressed attention already changes the memory baseline.
- **Continuous batching:** relevant to aggregate server throughput, not necessarily one-user latency.

## How MLX and mlx-lm are structured

MLX and mlx-lm are different layers:

```text
HF/MLX model directory
  config.json + tokenizer assets + *.safetensors
                |
                v
mlx-lm architecture module
  tensor-name sanitization, model layers, MoE/router, generation, KV cache
                |
                v
MLX core arrays/runtime
  unified-memory arrays, lazy graph, mx.compile, CPU/GPU streams,
  quantized primitives, custom Metal kernels, distributed collectives
```

Key properties:

- MLX arrays live in unified memory. Operations select a CPU or GPU stream without an explicit host/device copy.
- Ordinary MLX execution is lazy: operations build a graph until `mx.eval`, materialization, or another synchronization forces work.
- `mx.compile` is separate: it captures/optimizes/compiles a function graph and can improve runtime/memory when shapes/control flow allow reuse.
- MLX supports custom Metal kernels through Python and C++ APIs, but it generates the function signature and owns array/stream metadata. Colibri MSL is not directly ABI-compatible.
- mlx-lm loads Hugging Face-style model directories and converts to an `mlx_model` directory. Safetensors is only the container; architecture, tensor names, shapes, tokenizer, and chat template remain required.
- MLX quantization includes mode, bits, group size, packing, scales/biases, and axes. `mxfp4`, `nvfp4`, `mxfp8`, and affine formats are not interchangeable with Colibri merely because both say “4-bit.”
- mlx-lm owns generation, rotating/batched/quantized KV caches, prompt-cache persistence, and speculative cache trimming.

Current upstream status matters: mlx-lm main has a `deepseek_v3` architecture, but DeepSeek-V4 support is still tracked in open issues/PR work as of this research date. V4 adds CSA/HCA behavior not supplied by the V3 module. A Colibri “MLX engine” can support already-supported mlx-lm models before it supports the current DeepSeek-V4 target.

### 128 GB feasibility for DeepSeek-V4-Flash

The official model card gives 284B total / 13B active parameters and labels the released checkpoint FP4+FP8 mixed. Idealized raw lower bounds, if every parameter were stored at the named bit width, are:

- 4-bit: about 142 GB before scales, metadata, KV/cache, scratch, Python/runtime, and macOS;
- 3-bit: about 106.5 GB before overhead, leaving little safe headroom;
- 2-bit: about 71 GB before overhead, but with a larger quality risk.

Therefore even the impossible best case for a 4-bit fully resident mlx-lm path exceeds this machine's 128 GB before overhead; the actual mixed FP4/FP8 checkpoint is larger. A hypothetical 3-bit conversion is marginal and can enter memory compression; a 2-bit conversion is the plausible resident size class, with a substantial quality-risk gate. Colibri’s disk-backed expert store is a real architectural advantage for this target, not incidental loader complexity.

## MLX support options for Colibri

| Option | Feasibility | Concrete seam | Best-effort work | Performance expectation |
|---|---|---|---|---|
| C. persistent mlx-lm child behind Colibri API/UI | **High for mlx-lm-supported models; recommended first** | implement the existing `READY/SUBMIT/ACCEPT/DATA/DONE/STAT` child contract used by `c/openai_server.py:1447-1655` | prototype 2-5 engineer-days; production parity 3-6 engineer-weeks | wrapper itself adds no speedup over direct mlx-lm; gateway overhead should be measured separately |
| A. convert MLX checkpoints to native Colibri | Medium for one already-supported architecture; low generically | extend converter patterns in `c/tools/convert_*.py`, model detection in `c/coli`/server, format registry/loaders | 1-2 weeks for plain known tensors; 2-6 weeks for quantized mapping; 2-4 months generic | conversion enables execution but has no intrinsic speed gain; requantization can lose quality |
| D. port selected MLX kernel ideas | Medium after profiling; direct binary reuse low | add a C ABI call in `backend_metal_v4_seam.h`, host glue in `.mm`, adapted MSL in `c/metal` | 2-6 weeks per kernel family | only the replaced phase can improve; current dispatch granularity can turn a fast kernel into a regression |
| B. in-process MLX runtime backend | Low today; architectural rewrite | first create a model-runtime ABI covering load/tokenize/submit/stream/cancel/KV/stats/shutdown | 3-6+ engineer-months for production quality | unknown; MLX runtime inclusion alone is not a speedup |

### Recommended Option C work package

1. Add a Python adapter that loads one mlx-lm model once and speaks Colibri’s child protocol.
2. Let mlx-lm own tokenizer, chat template, sampler, and KV cache initially.
3. Add explicit `--engine`/`COLI_ENGINE` selection before automatic model detection.
4. Translate stop strings, finish reasons, usage counts, streaming chunks, cancellation, and errors.
5. Treat grammar/tool calls, prefix slots, telemetry, and batch scheduling as separate parity milestones.
6. Test `/v1/models`, chat/completions streaming and non-streaming, `/health`, cancellation, queue timeout, auth, and malformed child output.
7. Benchmark direct mlx-lm versus wrapped mlx-lm first; only then compare a same model/quantization/prompt against native Colibri.

For DeepSeek-V4 specifically, Option C additionally depends on merged/stable mlx-lm V4 support or maintaining a fork. It still does not supply Colibri-style SSD expert streaming.

### Performance estimate, stated honestly

- **Supported, fully resident smaller model:** MLX may outperform Colibri’s generic CPU path through mature GPU kernels and graph fusion, but no same-model local A/B exists. Expected gain is unknown.
- **Direct mlx-lm versus Colibri-wrapped mlx-lm:** approximately the same model throughput is the design goal; integration is for compatibility/API reuse, not speed.
- **DeepSeek-V4-Flash on this machine:** native Colibri remains the more feasible architecture because it can stream experts beyond RAM. mlx-lm 4-bit does not fit; 3-bit is marginal; 2-bit may fit but changes the quality/capacity comparison.
- **Selective MLX kernel port:** use Amdahl bounds from a current profile and require an end-to-end A/B. Never publish the kernel-only ratio as model gain.

## Validation protocol future agents must preserve

- Build V4 Metal explicitly:

  ```bash
  make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8
  nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch
  ```

  The symbol count must be greater than zero. Plain `make` can silently compile the seam out.

- Preserve the golden output MD5 `5d04890413ff539e802985ce8c727814`; never change it to pass a test.
- Confirm no other `deepseek_v4` engine is running before a benchmark.
- Freeze usage history/seed and verify it exists and hashes correctly.
- Preflight the engine before `bench/ab.sh`; its trap deletes the temporary error log after a nonzero exit.
- Record binary SHA/hash, flags, model, prompt, metric definition, cache state, RSS/compressor state, engagement counters, n, and interleaving.
- Separate TTFT, after-first/decode, aggregate throughput, and kernel time.
- Revisit any file marked experimental/default-off before relying on its comment; other agents may be actively changing it.

## Primary external references

- Apple M3 Max hardware: https://support.apple.com/en-asia/117736
- Apple Metal feature families/limits: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- Apple Metal 4 inline tensor operations and Apple10 neural acceleration: https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4
- Apple GPU storage modes: https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
- Metal residency sets: https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets
- Metal argument buffers: https://developer.apple.com/documentation/metal/managing-groups-of-resources-with-argument-buffers
- MLX documentation: https://ml-explore.github.io/mlx/
- MLX unified memory: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX lazy evaluation: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX compilation: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- mlx-lm conversion/quantization: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/convert.py
- mlx-lm generation/caches: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- mlx-lm DeepSeek-V3 module: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v3.py
- mlx-lm DeepSeek-V4 support status: https://github.com/ml-explore/mlx-lm/issues/1281
- DeepSeek-V4-Flash model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- PowerInfer: https://arxiv.org/abs/2312.12456
- DeepSeek-V2/MLA: https://arxiv.org/abs/2405.04434
- FlashAttention-3: https://arxiv.org/abs/2407.08608

## Research caveats and next update trigger

- The intended post still needs its URL/body. Add a post-specific section only after it is supplied.
- Reprofile after any change to rows4, grouped MoE, Metal attention, batched MoE, OMP policy, or pipeline behavior; the old phase shares become stale.
- Recheck mlx-lm DeepSeek-V4 support before implementation; the upstream issue/PR state can change.
- All speed estimates not tied to a local artifact are bounds or hypotheses, not measured gains.
