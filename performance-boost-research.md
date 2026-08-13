# DeepSeek-V4-flash on M3 Max — Performance-Boost Research

Status: **v4** — v3 PASSED review; v4 adds M11's measured profile, the §0.1 retraction, M1/M2 kill records, M16, and data-driven resequencing (2026-08-13 22:10).
Oracle: PASS — "none of prior 4 blockers survives … technically careful, hypothesis-driven
plan"; one standing caution: the 100–200 GB/s CPU-attainable-bandwidth figure is an estimate,
to be pinned by M11/M15 measurement, and is framed as such.
Momus: OKAY — "Executability blocker no longer survives. Plan now has runnable standing QA,
per-method counters, concrete gates, and valid references."
Experiments authorized. Order: Wave 0 → M1 `ft-spec_keep` (+ M11 next), per §2.

Machine: Apple M3 Max — 40 GPU cores, 16 CPU cores, 128 GiB unified LPDDR5 @ 400 GB/s (Apple
spec), internal APFS SSD (~5.6–7.2 GB/s seq read, M3-era public benchmarks).
Model: deepseek-v4-flash — 43 layers, hidden 4096, 256 routed experts (top-6) + 1 shared/layer,
moe_intermediate 2048, MXFP4+UE8M0 (~162 GB on disk), FP8-E4M3 activations, MTP heads.
Engine: `c/deepseek_v4.c` — OpenMP CPU path, env-gated Metal seam (measured net-negative at S=1).

## Standing QA contract (inherited by EVERY method's QA below)
```
# baseline pair (run per arm; restore history between arms; Metal stays OFF unless the method is Metal):
cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
COLI_V4_SAVE_USAGE=0 <METHOD_ENVS> ./c/deepseek_v4 models/deepseek-v4-flash \
    "The capital of France is" --max-tokens 8 --memory-gb 96 2>&1 \
  | grep -E "generated_text|timing|v4_tokens|v4_dspark|metal_dispatches"
# gates, all mandatory:
#  - compressor < 20 GB throughout (vm_stat), non-engine RSS <= ~31 GB BEFORE launch
#  - method-engaged counter > 0 on the ON arm (each method names its counter)
#  - generated_text identical OFF vs ON, unless the method declares bounded divergence
#    (then divergence must be stated WITH the reduction length it was measured at)
#  - regression: cd c && python3 tests/test_deepseek_v4_tiny.py --binary ./deepseek_v4 \
#      --fixture ./deepseek_v4_tiny   -> exit 0; default build stays 26 objects
# PREFILL variant of the pair: replace prompt with --prompt-file /tmp/longprompt.txt
#   (python3 -c "...700 words..." generator, 799 tok) and read time_to_first_token.
```

---

## 0. Empirical foundation (measured on THIS machine, paired, frozen history)

| fact | value | source |
|---|---|---|
| warm decode, ram96 | 0.82 s/token (`after_first` 6.56 s / 8 tok) | E24 |
| harness tok/s incl. cold effects | 0.16–0.21 (ram64), ~0.33 (ram96) | §4b/§13 |
| prefill | 708 s / 799 tok = **0.89 s/token** | E25 |
| CPU **expert-forward chain** | 2.81–2.92 ms each; **258/token** (43×6) → 725–753 ms/token serial | E24 |
| expert record | 13.37 MB (3×4.19 weights + 0.79 scales) | §13c |
| hit rate | 57.7–67.7 %; miss = 13.37 MB SSD read | §13b |
| Metal per-expert S=1 seam | 1.118× slower decode, 1.063× slower prefill | §12 |
| Metal matmul isolated | ordered+xc: 1.10×/2.20× (S=1), **7.26× (S=8)**; simd 5.98–8.32× | E15 |
| n-gram speculation | **+18.5 %** measured, cfg `V4_DRAFT=4 V4_NGRAM=1` | RESULTS §2 |
| full MTP | 70–89 % acceptance, but only +3.3–9.4 % net (rejection-replay cost) | RESULTS §2 |
| observed failure | ONE partial n-gram rejection → `spec_disabled=1` for the whole generation | :8077-8081 |
| PREWARM (688 experts eager) | killed −29.7 %; +9.98 pp hit rate; net +3.7 GB I/O | §13 |

### 0.1 Where a decoded token's time goes — MEASURED (M11 profiler, ram96, 98% accounted)

| phase | ms/token | share |
|---|---|---|
| **attention** | 374.4 | **38.7 %** |
| **expert_forward** (1806 calls, 1.06 ms each) | 273.3 | **28.2 %** |
| shared_expert | 84.8 | 8.8 % |
| expert_wait (loader stall) | 84.3 | 8.7 % |
| head | 34.4 | 3.6 % |
| router | 33.2 | 3.4 % |
| hc_norm | 28.3 | 2.9 % |
| indexer + compressor | 35.7 | 3.7 % |
| embed + rope + other | 19.7 | 2.1 % |

**RETRACTION (v3→v4):** earlier versions claimed the expert-forward chain was "~88 % of warm
decode" from 258 × 2.81 ms. That 2.81 ms figure came from dividing TOTAL wall time by dispatch
count — it silently attributed attention, router, shared expert and head to the experts. The
profiler shows expert_forward is **1.06 ms/call**; the routed-expert chain (forward + wait) is
**36.9 %**, and **attention is co-dominant at 38.7 % (8.7 ms/call)** — a cost center v3 treated
as minor. Every downstream estimate inherited this error; §1 statuses below are corrected.

### 0.2 The walls, stated precisely (revised per Oracle blocker #1)
1. **Weight-traffic floor:** a decoded token touches ≥ 258 × 13.37 MB ≈ 3.45 GB of *model bytes*
   if nothing is reused across tokens. Against the 400 GB/s *machine peak* that floors at
   ~8.6 ms/token — but that comparison only rules out "ideal single-pass streaming at peak" as
   the limiter. **It does NOT prove the CPU kernel is compute-bound**: real traffic exceeds model
   bytes (activation re-reads per row, scales, cache-line waste, TLB), and CPU-cluster attainable
   bandwidth on M3 Max is well below 400 GB/s (order 100-200 GB/s, and a single kernel rarely
   attains even that). The honest statement: effective model-byte throughput is ~4.5 GB/s per
   forward; the gap to CPU-attainable bandwidth is the *upper bound* of kernel-level headroom,
   and **how much of that gap is closable is an open question M11+M15 must answer**, not a
   settled 100× claim.
2. **Granularity wall (killed Metal v1):** at S=1 the GPU pays launch+sync per 12.6 MB matvec
   with zero weight reuse and no unified-memory bandwidth edge. Upstream: llama.cpp Metal
   `MUL_MAT_ID` switches to mat-mat only at ≥32 rows/expert (`ggml-metal-ops.cpp:2589-2680`);
   yet llama.cpp GPU decode beats CPU on Apple Silicon (7B Q4_0: 66-71 vs ~29 t/s). The GPU can
   win — above the granularity threshold. **Consequence: S-batching gains are weight-REUSE
   gains** (walk 12.6 MB once for S tokens), valuable on CPU and GPU alike.

### 0.3 The engine's decisive structural fact
Both ends of the "missing middle" already exist:
- Above MoE: prefill + spec-verify are token-batched — `target_batch` (:7272-7295) chunks ≤64
  positions through `coli_v4_block_window_batch_ref` (:3889-3950) with batched attention
  (:2222-2382) — then drop to a serial per-item `moe_token_pipeline` at :3941-3944. Routing is
  separable (:3639-3660); all layer-L MoE inputs exist before any layer-L expert runs.
- Below MoE: `coli_fp4_matmul_batch_ref` (S=batch FP4 matmul, :10660) exists and is never called
  by the expert path.
Missing middle = batched router storage → group tokens by expert → one lease per unique expert →
S=group matmul → scatter.

---

## 1. Candidate methods
(Gain = expected effect; Work = S/M/L; each QA block runs under the standing contract above.)

### M1 · `spec_keep` — KILLED 2026-08-13 (pre-registered kill-test fired)
**Result (ft-spec_keep, 20 runs, ram96, 64 tok, paired, token-exact everywhere, gates ok):**
base 0.4474 tok/s; d4k0 −2.1 %; **d4k1 (the hypothesis) −8.1 %**; d2k1 −4.3 %; d8k1 −7.9 %.
Acceptance was 0 % on 3 of 4 prompts (only p3 drafted well, 34-75 %, and still didn't beat
baseline wall time). KILL condition was "no (D,KEEP) beats baseline ≥5 %" — nothing beat it at
all. **The self-disable heuristic is protective in this regime, not a bug.** Secondary finding:
even the previously-recommended `V4_DRAFT=4 V4_NGRAM=1` is −2.1 % here — the historical +18.5 %
came from a 24-token Q&A regime on an older binary and does NOT transfer to 64-token coding
prompts. Evidence: `.backlog/spec_keep_sweep.csv`. Consequence: M12 (MTP economics) drops in
priority — its premise (speculation pays if replay is cheap) now needs the M4 batch first.
<details><summary>original method text</summary>

### M1 (original) — stop speculation self-disabling; retune draft depth  [Work: S]
**Mechanism.** n-gram speculation measured **+18.5 %** here, but ONE partial rejection sets
`spec_disabled=1` for the rest of the generation (:8077-8081) — observed live (`attempts=1
drafted=4 accepted=0 → adaptive_disabled=1`). `V4_NGRAM_PARTIAL_KEEP=1` exists. Sweep
`V4_DRAFT ∈ {2,4,8}` × KEEP; if KEEP wins, make it default. Compounds with M4: accepted-draft
rows ARE the decode-time batch (upstream: spec+cache composed +51 % on GLM, llama.cpp d#24528).
**Gain:** +10-25 % decode on draftable text. **Risk:** KEEP loses on adversarial text;
acceptance decay on long generations.
**QA.** 4-prompt paired sweep, 64 tok each:
```
for KEEP in 0 1; do for D in 2 4 8; do
  cp /tmp/coli_usage.snapshot models/deepseek-v4-flash/.coli_usage
  V4_DRAFT=$D V4_NGRAM=1 V4_NGRAM_PARTIAL_KEEP=$KEEP COLI_V4_SAVE_USAGE=0 \
    ./c/deepseek_v4 models/deepseek-v4-flash "<P1..P4>" --max-tokens 64 --memory-gb 96 2>&1 \
    | grep -E "v4_dspark|timing|generated_text"; done; done
```
Engaged-counter: `v4_dspark attempts>0` on every arm. PASS = best (D,KEEP) beats `V4_DRAFT=0`
baseline by ≥10 % paired-mean tok/s with identical `generated_text`. KILL = no (D,KEEP) beats
baseline ≥5 %.

</details>

### M15 · `matvec_kernels` — **DONE, −20.4 % decode, bit-exact** (ft-matvec_kernels)
Rescoped by M16's profile from "expert kernel" (28.7 %) to BOTH matvec families (76.3 %).
Shipped: tail-branch hoist + paired-row activation walk + gate/up dual-matvec fusion, on MXFP4
and FP8-rows8. Real-model paired: decode 6684→5322 ms (**1.26×**), attn_out −32.9 %, attn_qkv
−36.0 %, expert_forward −25.7 %, shared_expert −37.6 %, `generated_text` byte-identical.
Savings landed 117 % inside the predicted phases. See E30/E32.
<details><summary>original method text</summary>

### M15 (original) — speed the B=1 CPU expert forward  [Work: M]
**Mechanism.** §0.1 says warm decode ≈ 88 % expert-forward chain, yet v2 had no method for it.
Candidates inside `matmul_mxfp4` scalar path (quant.h:1401-1412) / `coli_fp4_dual_matvec_ref`:
NEON/vectorized nibble decode (the AVX2 branch shows the shape — arm64 got none), multi-row
processing per pass (walk activation once for 2-4 output rows), scale-decode hoisting,
gate+up true fusion (one weight walk feeds both, dual_matvec today runs two S=1 calls
:10461-10464), prefaulted/interleaved layout. Bounded above by CPU-attainable bandwidth (§0.2).
**Gain:** unknown until M11; even 1.3× on the 88 % slice = ~1.25× decode. **Risk:** bit-exactness
demands the two-level accumulation order be preserved per row (proven preservable: rows are
independent).
**QA.** Microbench first, then engine pair:
```
cd c && cc -O3 -mcpu=native tests/bench variant vs baseline (new bench_mxfp4.c) # mean±sd n=20
# engine: standing 8-tok pair, envs none (kernel change), PASS = tiny-oracle token-exact AND
# paired after_first improves >=10%. KILL = <5% on both microbench and engine.
```

</details>

### M2 · `rope_cache` — KILLED 2026-08-13 by its own pre-registered gate
**M11 measured rope = 1.242 ms of a 6775 ms decode = 0.02 %** (gate required ≥3 %). The rebuild
loops are real but cost nothing at this scale. No branch, no build — the gate did its job.
<details><summary>original method text</summary>

### M2 (original) — cache RoPE tables on the DECODE path  [Work: S–M]
**Mechanism.** DECODE attention rebuilds the RoPE table to `position+1` every token·layer
(:1647-1664, dup :4847-4864; compressor/indexer sites :2846-2930). **Prefill batched attention
already precomputes once per chunk/layer (:2233-2313) — v2's prefill claim was WRONG and is
withdrawn.** Scope: decode + compressor/indexer only. Cache grow-only per-position rows.
**Gain:** unattributed until M11; O(P·43) transcendentals/token suggests it grows with context;
treat as small until measured. **Risk:** low (cache stores exactly the loop's values).
**QA.** Gated on M11 showing RoPE ≥3 % of decode. Then standing 8-tok pair at position≥512
(long-context session), PASS = token-exact + measurable win consistent with M11's attribution.

</details>

### M3 · `resident_head_embed` — stop per-token head/embedding re-reads  [Work: S]
**Mechanism.** `final_hidden` reloads `hc_head_fn/base/scale/norm.weight` every token
(:6802-6842); embedding row buffered-pread per token (:6781-6799). Cache head tensors; cache/mmap
hot embed rows.
**Gain (revised by M11 data):** embed measured 0.06 % → that half is DEAD. head phase is 3.6 % total and includes the argmax compute, so the reload slice is a fraction of that. Ceiling ≈1-2 %; DEPRIORITIZED below M7/M10/M15.
**QA.** `sudo fs_usage -w -f filesys <pid>` (or ktrace) during 8 warm tokens: PASS = per-token
reads of those tensors drop to 0 AND standing pair token-exact, non-negative delta.

### M4 · `moe_batch` — token-batched MoE for prefill + spec-verify (CPU)  [Work: M–L]
**Mechanism.** New `moe_batch_pipeline` in `coli_v4_block_window_batch_ref`: FFN-norm all items,
route all items, group `expert→rows`, ONE lease per unique expert, `coli_fp4_matmul_batch_ref`
S=group, scatter. Per-row accumulation order unchanged ⇒ bit-exact per token.
**Batch math (Oracle-verified):** batch 64 → 384 selections; E[unique]=256(1−(250/256)^64)≈200 →
mean S≈1.92 under uniform routing; measured skew (top-10 % experts ≈80 % of hits, ll.cpp #24524)
concentrates hot-expert S higher. S≈2 halves weight-walk per selection. Lock ops drop
768→~400 per layer-batch (lookup+release per selection today, :5250/:5318).
**Gain:** prefill 1.5-2× (weight-amortization, first-order); spec-verify rides free. Decode B=1
untouched except via M1's accepted rows. **Risk:** HC state ordering; route-weight per-token
application; scatter correctness.
**QA.**
```
# 1. correctness: cd c && python3 tests/test_deepseek_v4_tiny.py ... -> exit 0 (token-exact)
# 2. prefill pair: standing contract with --prompt-file /tmp/longprompt.txt (799 tok), 4 tok out
#    engaged-counter: new stderr line moe_batch groups=<n> mean_S=<x> (must appear, mean_S>1)
# PASS = time_to_first_token improves >=25% paired, token-exact. KILL = <10%.
# 3. telemetry for M5: per-group S histogram dumped to CSV.
```

### M5 · `gpu_gather_moe` — gathered expert matmul on Metal  [Work: L, HARD-GATED on M4 telemetry]
**Mechanism.** Hand M4's groups with S≥S_min to the proven Metal kernels (7.26× at S=8,
bit-exact at I=4096) or an MLX-gather_qmm-style kernel (one dispatch per gathered block; MLX's
own bench shape is 256 experts). Zero-copy weights already work. Sub-option (reinstated T8):
one command buffer per token over its 6 experts to amortize submit/wait even at S=1.
**Gain (revised per Oracle):** UNKNOWN until M4's S histogram exists — the 7.26× is an isolated
S=8 kernel number and mean S≈1.9; end-to-end projection is withdrawn. Plausible only if hot-tail
mass sits ≥8. **Risk:** dispatch overhead at small S (the §12 failure mode again).
**QA.** Gate: histogram from M4 shows ≥30 % of selections in groups S≥8, else DO NOT BUILD.
Then standing prefill pair, engaged-counter `metal_dispatches>0`, PASS = ttft improves ≥15 %
over M4-alone with identical tokens (ordered) or stated bounded divergence (simd).

### M6 · `expert_lookahead` — NOVEL heuristic: prefetch L+1 experts from pre-MoE hidden  [Work: M]
**(Relabeled per Oracle: this is NOT the literature's predictor.)** Papers (ST-MoE 85 % acc,
SpecPrefetch 91.7 % ReadyRecall) train predictors; our cheap variant — run layer-L+1's router on
layer-L's PRE-MoE hidden — is a **different, uncalibrated approximation** (the true L+1 input
includes the MoE output we haven't computed). Treat recall as unknown; risk HIGH until measured.
Plumbing exists: `posix_fadvise(WILLNEED)` prefetch (:5325-5377), loader threads (:3675-3755
currently intra-layer only).
**Gain:** only at miss-heavy budgets (ram64's 42 % miss). **QA — measurement FIRST, build second:**
```
# phase 1 (no fetch code): log predicted-vs-actual top-6 overlap for 200 tokens, 43 layers.
# PASS gate to build phase 2: mean overlap >= 4/6. Else KILL (record recall number).
# phase 2: standing ram64 pair; PASS = hit_rate +>=3pp AND tok/s +>=8%.
```

### M15b · `matvec_rows4` — **DONE, cumulative 1.38-1.42x** (ft-matvec_rows4)
Row-block 2->4. attn_out -47.1 %, attn_qkv -51.6 %, shared_expert -52.0 %, expert_forward
-34.3 % vs pre-M15; decode 6684 -> 4693-4838 ms, byte-identical. **Shifted the bottleneck**:
expert_wait 6.2 % -> 17.9 %, promoting M6/M7 to the next lane. See E33.

### M7 · `cache_policy` — frequency+recency eviction instead of LRU  [Work: S–M]  **<- NEXT LANE (expert_wait 17.9 %)**
**Mechanism.** Eviction is pure-LRU (:5261-5297); skew is extreme; literature (Least-Stale,
SpecMD) shows LRU collision-misses. Add decayed hit-count to victim score.
**Gain:** +hit-rate at fixed RAM; each avoided miss saves a 13.37 MB read. **Risk:** low.
**QA.** Standing ram64 pair (4 prompts × 128 tok, the §13 harness): PASS = hit_rate +≥3 pp AND
paired tok/s +≥5 %, token-exact. KILL = hit_rate flat.

### M8 · `warm_break_even` — prewarm at the break-even count  [Work: S]
**Mechanism.** §13d prescription: sweep warm-count {64,128,256} vs 688 (which lost −29.7 %).
Existing code + env; needs a count knob if none exists (small patch).
**QA.** Cold-start prompt #1 only, paired per count vs PREWARM=0:
PASS = some count improves cold p1 ≥8 % with gate ok. KILL = none do.

### M9 · `mmap_load` — mmap+madvise load path; parallel dense load  [Work: M]
**Mechanism.** All loading is serial buffered pread (headers st.h:449-574; dense :520-554;
repeated index opens :943-1017). llama.cpp/macOS uses mmap+WILLNEED (llama-mmap.cpp:445-466);
MLX uses a 4-thread pread pool (load.cpp:337-395). Keep expert-miss reads pread/direct.
**Gain:** startup/TTFR only, plausibly 2-4× of the load phase. **Risk:** pager thrash if madvise
is sloppy near the ~100 GB ceiling.
**QA.** `sudo purge` then time exec→READY and cold ttft, 3 runs each arm:
PASS = READY time −≥40 %, no compressor growth. (purge makes runs comparable.)

### M10 · `omp_gaps` — parallelize scalar non-expert loops  [Work: S–M]
**Mechanism.** Sparse attention scalar (:2953-3006), HC/RMSNorm scalar (:1178-1251), indexer
mostly scalar (:2846-2930). Element-parallel only (NO sum-reorder) to preserve token-exactness.
**Gain:** part of the ~95 ms/token non-expert slice (M11 will size it). **QA.** Standing 8-tok
pair; PASS = token-exact + after_first −≥5 %. KILL if M11 shows the slice <8 % of decode.

### M11 · `phase_profiler` — DONE 2026-08-13 (branch ft-phase_profiler)
Runtime `COLI_V4_PROFILE=1`; 98-99 % accounted (tiny + flash); zero lines when unset; token-exact with profiler ON; 26-object default build. Its first real-model profile forced the §0.1 retraction, killed M2, gutted M3, and exposed attention as co-dominant.
<details><summary>original method text</summary>

### M11 (original) — [Work: S]
**Mechanism.** `COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE` hooks exist without glue (:3553-3587).
Wire + add startup-phase timers + RoPE/head/embed counters. Converts every "est." above into a
measurement; Oracle's re-sequencing depends on it.
**QA.** `COLI_V4_PROFILE=1` run emits per-phase ms summing to within 10 % of wall `after_first`;
zero overhead when env unset (paired check).

</details>

### M12 · `mtp_economics` — make 70-89 % MTP acceptance pay  [Work: S sweeps; rides M4]
**Mechanism.** MTP loses on rejection-replay only. Sweep `V4_MTP_PARTIAL_KEEP=1`,
`V4_MTP_CONF ∈ {0.55,0.7,0.85}` (higher = fewer, surer drafts = fewer replays); M4 later makes
replay itself batched-cheap.
**QA.** `V4_MTP=1 V4_NGRAM=0 V4_DRAFT=4 V4_MTP_LOG=1` sweep vs n-gram best from M1:
PASS = beats n-gram config ≥5 % paired. Else record and keep n-gram.

### M13 · `wire_hot` — mlock the hottest few GiB  [Work: S]
**Mechanism.** Wire dense-resident 6.27 GiB + shared experts (≤8 GiB total, RLIMIT permitting)
so ambient pressure can't compress them (the §11/GATEFAIL mode).
**QA.** Deliberate 40 GB ambient load (dd to tmpfs or stress tool), standing pair:
PASS = gap_gb stays ≤1 and compressor stays flat on the wired arm where the unwired arm degrades.

### M14 · `cpu_gpu_split` — heterogeneous co-execution  [Work: L, PARKED]
Revisit only if M11 shows the non-expert track ≥20 % of decode. Ceiling ≤ min(track).

### M16 · `attention_dissect` — sub-attribute then attack the 38.7 % attention phase  [Work: S then ?]
**(Added v4 — the profile exposed a 374 ms/token blob with no dedicated method.)** Attention is
8.7 ms/call and internally unattributed: FP8 matvecs wq_a/wq_b/wkv/wo (already OpenMP :10377),
scalar sparse attention (:2953-3006), `all_kv` assembly copies (:1687-1726), KV ring writes.
Step 1 (S): extend the M11 profiler with attention sub-phases. Step 2: attack whatever dominates
(scalar sparse-attn → M10-style OpenMP; copies → buffer reuse; matvecs → M15-style kernel work).
**QA.** Sub-profile first (same standing contract); the sub-phase table IS the gate for step 2.
PASS for step 2 = paired after_first −≥8 % with token-exact output. Absorbs M10's
sparse-attention slice; M10 keeps hc_norm/indexer (~5 % combined).

### Rejected / falsified (do not re-tread)
Per-expert S=1 GPU offload (−1.118×, §12) · bulk eager prewarm-688 (−29.7 %, §13) · Apple
Tensor API/MPP on M3 (M5-generation only, ll.cpp PR#20962) · F_NOCACHE for repeated sessions ·
wiring 162 GB into 128 GiB.

## 2. Sequencing (v4 — reordered by M11 measurement)

```
DONE:    M11 profiler (98% attribution)   M1 spec_keep (KILLED)   M2 rope_cache (KILLED by gate)
Wave 1:  M16 attention_dissect step-1 (S, same enabler lane)  ──► pick attention attack from data
Wave 2:  M15 cpu_expert_kernel (28.2% target, microbench-gated)
         M7 cache_policy + M6-phase-1 recall log (expert_wait 8.7% + miss economics)
Wave 3:  M4 moe_batch (prefill lane) ──gate──► M5 gpu_gather_moe
Startup lane (independent): M9 mmap_load, M8 warm_break_even
Deprioritized by data: M3 (≤1-2%), M12 (premise weakened by M1), M13 (robustness only)
Parked: M14, M10-residual
```
Rationale: decode = attention 38.7 % + experts 36.9 % + everything-else 24 %. The two dominant
blocks get the next two lanes; M16 step-1 is the least-work next action and directly picks the
Wave-1 attack.

## 3. Sources
(unchanged from v2 — llama.cpp MUL_MAT_ID/mmap/PRs #24524 #25294 #20962 d#12985 d#4167 d#24528;
MLX gather_qmm + load.cpp; arXiv 2603.19289, 2606.15453, 2607.24787, 2602.03921; ACM Diff-MoE;
Apple specs/manpages; local RESULTS §2-§13, experiments E13-E27; line refs = c/deepseek_v4.c.)
