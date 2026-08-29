# Ultrawork Notepad — Colibri/Metal throughput and MLX architecture research
Started: 2026-08-24T11:43:00+02:00

## Plan (exhaustively detailed)

- Capture repository topology, shared-worktree state, machine specs, candidate post location, and semantic tools.
- Trace Colibri inference and Metal call paths with exact file/symbol evidence.
- Reconstruct the Metal performance ledger from source, history, and benchmark artifacts.
- Identify the referenced post; never silently substitute an unconfirmed source.
- Validate primary Apple Metal, MLX, and sparse-inference sources.
- Build a machine-specific bottleneck/optimization model for this M3 Max.
- Map MLX support options, effort, risks, and bounded performance expectations.
- Write and QA `claude.md`; recheck concurrent changes; obtain unconditional review.

## Success criteria + QA scenarios

- Deliverable/tier: source-backed `claude.md`; HEAVY because the user requested deep research and architecture review.
- Criterion 1: architecture and Metal-change claims resolve to current file:symbol/line anchors and measured artifacts. QA: read the full document and cross-check every material claim; pure prose has no valid RED seam.
- Criterion 2: post/external analysis cites the actual post or explicitly records that it is missing. QA: inspect prompt/repo candidates and validate primary URLs.
- Criterion 3: recommendations account for the local 16-core CPU, 40-core GPU, 128 GB unified-memory M3 Max, macOS 26.6.1, negative results, and benchmark scope. QA: each method has applicability, confidence, constraints, and a falsifiable experiment.
- Criterion 4: MLX support names concrete Colibri seams, format/runtime mismatches, work estimates, and bounded gain expectations. QA: compare with MLX primary docs/source and current Colibri loader/server/backend APIs.
- Criterion 5: no commit/push by this task and no unrelated edit/revert. QA: final path-scoped diff/status and Git-operation accounting. Concurrent HEAD movement is permitted and must be reported rather than reverted.
- WHEN TO STOP: stop immediately when all criteria pass, the reviewer approves unconditionally, the ledger is current, no task-spawned resource remains, and Git safety is accounted for.

## Now

- Goal achieved; final handoff only.

## Todo

- [x] Finish M3 Max optimization ranking.
- [x] Finish MLX feasibility/effort/performance bounds.
- [x] Draft `claude.md`.
- [x] Run source/citation/read QA and recheck concurrent files.
- [x] Run HEAVY reviewer loop.
- [x] Reconcile plan/ledger and final Git safety.

## Findings

- Hardware: MacBook Pro Mac15,9; M3 Max; 16 CPU cores (12P+4E); 40-core GPU; 128 GB LPDDR5 unified memory; macOS 26.6.1. Apple documents this SKU at 400 GB/s peak unified-memory bandwidth. M3 is Apple GPU family 9.
- Current DeepSeek-V4 flow: config/safetensors index → planned expert store → embedding → per-layer attention/MLA/KV → shared+routed MoE → residual → head. Canonical source: `c/deepseek_v4.c:9017-9081,9524-9624`.
- V4 expert residency is RAM slab versus disk-backed safetensors, not a separate VRAM tier: `c/expert_store.h:33-92`, `c/deepseek_v4.c:7022-7485,8257-8384`.
- Metal v4 registers page-aligned resident slabs as shared no-copy `MTLBuffer` objects and falls back to scratch copies for unresolved pointers: `c/backend_metal_v4.mm:202-289`.
- Current V4 scratch buffers are persistent grow-only shared buffers; the backend binds resources, commits, and waits synchronously in each FP8/experts operation. This makes fewer/fatter operations and full-stage fusion more relevant than generic heap advice.
- Batched expert seam: `c/backend_metal_v4.mm:664-988`, wired from the gathered MoE scheduler in `c/deepseek_v4.c:5161-5485`; exact batch/scalar probe covers S=1,2,6,16.
- Performance progression: scalar Metal was 1.484x slower end-to-end in the controlled 300-token M3 retest and 2.11x/2.30x slower TTFT/after-first in E24. The mechanism was granularity/dispatch/first-touch, not simply a slow shader.
- CPU paired-row/rows4 matvec work measured about 1.26x then 1.38-1.42x cumulative decode speed, byte-identical (`experiments_results.md:933-1002`).
- Grouped MoE + bit-exact Metal attention measured 1.163x/1.169x; fused `wo_a` raised composition to 1.189x/1.179x. Batched Metal MoE then measured 1.121x/1.133x incremental and about 1.33x total p064: 42.528→31.975 s (`experiments_results.md:E72-E84`). These are TTFT harness results, not generic decode.
- M5 Max result is a different engine/model/hardware datum: passive OMP plus PIPE=1/workers=8 measured 2.24 vs 2.06 tok/s (+8.5%), while active spin regressed to 1.25 tok/s. It is single-run warm-cache decode evidence and must not be merged numerically with M3 TTFT.
- Profile snapshot: attention 38.7%; expert forward 28.2%; expert wait 8.7%; shared expert 8.8%; rest 15.6%. The later optimizations mean this is an attribution snapshot, not the current exact phase mix.
- Killed/negative work: scalar expert Metal, unconditional prewarm, generic loader-depth, RoPE cache, current coding-workload speculation retune, GPU QDQ in the composed configuration, and current prefill-prefetch gate. Preserve these to avoid repeating experiments.
- The referenced post is not present in the prompt or repository: `performance-boost-research.md` and the paper matrix name no singular post URL/title/author. Do not present a guessed Reddit/Hugging Face candidate as the source.
- Primary external facts: MLX separates lazy graph construction from `mx.compile`; arrays use unified memory; custom Metal kernels exist in Python/C++; mlx-lm conversion uses HF-style model directories/safetensors plus architecture/config/tokenizer; quantized formats carry mode/group/packing semantics.
- Recommended MLX sequence: (C) persistent mlx-lm child adapter behind `c/openai_server.py:1447-1655`; (A) per-architecture converter only when native execution is required; (D) selective algorithm/kernel ports after profiling; (B) in-process runtime backend last.
- MLX work estimates (best effort): child adapter prototype 2-5 engineer-days, production parity 3-6 engineer-weeks; one known native conversion 1-6 weeks depending on quantization; generic multi-architecture import 2-4 months; in-process runtime backend 3-6+ engineer-months; selective kernel family 2-6 weeks.
- MLX performance: integration itself provides no speedup over direct mlx-lm. On this frontier disk-streamed V4 workload, mlx-lm lacks Colibri's expert-store/offload policy, so a native MLX run may not fit. Any gain on a smaller fully resident compatible model requires same-model A/B; no defensible local percentage exists.
- Apple Metal source validation: M3 is Apple9; residency sets, indirect commands, argument buffers, shared/private buffers are available. Metal 4 neural-accelerator paths documented for Apple10+ must not be projected onto M3.
- Machine-specific priority: (1) passive OMP/power-policy A/B when V4 Metal is enabled, (2) full attention/MLA fusion and fewer sync boundaries, (3) extend grouped/batched MoE to actual decode batching/rows16 only after telemetry, (4) continuous batching for service throughput, (5) CPU matvec/attention optimization, (6) predictor/cache research only where avoidable misses are measured. Private-weight duplication, heaps, argument buffers, residency sets, GPU QDQ, and Metal4 TensorOps are gated/low priority here.
- Concurrent Git movement occurred during research; never require final HEAD to equal an earlier SHA. The task itself has not invoked commit or push.

## Learnings

- Do not infer benchmark scope from p064/p256 names: `bench/ab.sh` is TTFT, and faster deltas are negative.
- Never combine kernel microbenchmarks with application speed. Local dilution has repeatedly been 1.5-2x.
- Zero-copy shared expert slabs are architectural, not an accidental default; private buffers trade possible GPU locality against extra copy/residency pressure.
- The durable report must distinguish implemented/default-off, measured, proposed, killed, and unconfirmed states.

## Draft and QA evidence — 2026-08-24

- Wrote repository-root `claude.md` (335 lines before the final source clarification; 336 after) without modifying runtime source.
- Read the entire report after drafting. Markdown structure has four balanced code-fence markers and no conflict markers. The two trailing-space hits are intentional hard line breaks in the title metadata.
- Cross-checked current code symbols and gates: `coli_v4_config_load`, hash routing, `COLI_V4_MOE_GROUPED`, `COLI_V4_MOE_BATCHED_MIN_N`, rows1/rows16 gate, nonzero Metal fallback, shared no-copy slabs, persistent shared scratch, synchronous command-buffer waits, and the `READY/SUBMIT/ACCEPT/DATA/DONE/STAT` server seam.
- Cross-checked benchmark semantics from `bench/ab.sh` and `AGENTS.md`: one generated token, TTFT parser, p064/p256 are filenames, faster delta is negative, `METAL ?= 0`, and the golden MD5 remains `5d04890413ff539e802985ce8c727814`.
- Cross-checked all headline speed values against `experiments_results.md`, the M3 research ledger, and the separate M5 report. No M3 TTFT number is presented as decode, and the M5 result is isolated as different hardware/model/metric evidence.
- Validated primary Apple, MLX, mlx-lm, Hugging Face, MoE-Infinity, PowerInfer, DeepSeek-V2, and FlashAttention-3 URLs via live web access on 2026-08-24. Verified that mlx-lm issue #1281 is open and reports CSA/HCA missing; verified the model card's 284B/13B mixed FP4+FP8 specification.
- Clarified that M3/Apple9 supports Metal 4 tensor APIs but lacks the Apple10/M5 per-core neural-accelerator hardware. Clarified 4/3/2-bit size math as idealized lower bounds, not the released mixed-format checkpoint size.
- Pure-prose documentation has no meaningful compiler/LSP/test target. Validation is source-anchor, arithmetic, citation, full-read, and Git-safety QA; no runtime build or model benchmark was needed or run.

## Final gate and safety evidence — 2026-08-24

- Mandatory HEAVY gate reviewer `/root/heavy_report_review` returned unconditional `APPROVED`: all five criteria passed, with no blocker, scope drift, or unsupported success claim.
- Final `claude.md`: 336 lines, SHA-256 `7cbe3b2443bbfcff86a9773e3e3816b551ae84a3c5eca65b7bf60340e8548e39`.
- Final Git state at audit: branch `ft-opensourceftw1`, HEAD `7f93efeb5a92d9f3c24dfef28b5b42d695baa8db`, zero commits ahead of upstream. Task-scoped status is exactly two untracked files: `claude.md` and this ledger.
- Three unrelated/pre-existing untracked gate logs remain untouched: `.backlog/lab/gate_20260815-154737.log`, `.backlog/lab/gate_20260816-025008.log`, and `.backlog/lab/gate_20260824-114605.log`.
- This task invoked neither `git commit` nor `git push` and made no tracked source edit. No model/server runtime was launched. Process enumeration itself was unavailable because this environment's `pgrep` could not access `sysmond`; this does not leave a task-owned process because none was created.
- All research and reviewer agents are terminal/completed; no child task remains running.
