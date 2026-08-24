# Benchmarks to run

Planned, not yet executed. Written 2026-08-15 after E57 retracted the modelled +80.7 %
"stock -> recommended" figure by measuring it and finding the **opposite sign**.

Nothing here is a result. Results live in `experiments_results.md` (E-numbers) and
`.backlog/deepseek-v4-RESULTS.md` (sections). When a benchmark below is run, write the outcome
there and link it back to the ID used here.

---

## 0. Read this before running anything

Every rule below was bought with a wrong answer that got published internally first.

### 0.1 Universal protocol (applies to EVERY benchmark on this list)

1. **Fresh engine process per arm.** The runtime plan is fixed at engine open
   (`build_runtime_plan()`); you cannot change `--ram` on a live process.
2. **Identical workload, token limit, sampling, thread count, prompt set** across arms.
3. **Declare cold or warm. Never mix them in one mean.**
4. **Gate the compressor before AND after every prompt.** A row that started clean can spend its
   whole life under compression — that flaw was real and is fixed in `coldwarm2.sh`.
5. **Discard any row whose gate failed.** It measured memory pressure, not your variable.
6. **Interleave. Never all-A-then-all-B.** ABBA or randomised. See §0.2 for why this is not
   optional.
7. **Freeze `.coli_usage`**: restore the frozen snapshot before each arm and run with
   `COLI_V4_SAVE_USAGE=0`. Verify by md5 **before and after** each arm. Without this a later arm
   starts from a larger history and part of your delta is "more history".
8. **Assert which binary is under test.** Print the resolved engine path + mtime + md5 at startup.
9. **Report `phys_footprint`, RSS, the gap, compressor start/end, and non-engine RSS with every
   throughput number.** The budget headroom is part of the measurement.
10. **Change ONE variable per arm.** E57 violated this deliberately (it asked a product question,
    "is the recommended config faster") and consequently cannot attribute its own result.

### 0.2 Traps that have already cost this project a wrong conclusion

| trap | what happened | defence |
|---|---|---|
| **Ambient load** | identical config measured **+29.5 % cold** between a busy and a quiet desktop (E57). §11 had it at 22 %. | interleave; log non-engine RSS per row |
| **Stale baseline** | diffing against a baseline pinned at another time inflated a prefill number **1.73x** (E56) | pair every A with its own B in-session |
| **Stale binary** | `libexec/` held a 2-day-old engine while `c/` was current; every `bin/coli` harness measured old code (§10) | assert md5 at startup |
| **`ps` RSS** | excludes compressed pages; produced two wrong memory conclusions in a row (§8) | always pair with `phys_footprint` |
| **Unfrozen history** | a later arm seeds from a bigger `.coli_usage` (`prewarm_ab.sh:17-19`) | snapshot + `SAVE_USAGE=0` + md5 assert |
| **Failed measurement read as data** | `fp_gb()` returned `0` on failure, indistinguishable from a real 0.0 GB gap | emit `NA`, never `0` |
| **Composing across workloads** | +18.5 % (chat) x +52.5 % (coding) = "+80.7 %"; measured **−5.4 %** (E57) | never multiply deltas from different workloads |
| **Partial arms** | one row of a 4-row arm showed **+14.1 %** where the finished arm showed **−29.7 %** (§13e) | never report an unfinished arm |
| **Profile mode != production** | per-stage profile claimed a 25.7x gap on identical shapes (§12e) | validate that stages sum to the wall |

### 0.3 Host constraints (137 GB Apple Silicon, 16 KB pages)

- **Working rule: `engine + non-engine <= ~100 GB`.** Confirmed again in E57 — `--ram 96` at
  ~122 GB total failed 7/8 rows with the compressor at **73.1 GB**; the same config at ~107 GB
  total ran **8/8 clean**.
- `--ram 96` needs the desktop genuinely quiet (non-engine <= ~13 GB). `--ram 64` is the largest
  budget that survives a normal working desktop.
- `--ram 110+` computes to ~120 GB total and **will** compress. Do not bother.
- `vm_stat` pages are **16 KB** on this host. Using 4096 understates memory 4x — I made exactly
  this error once during E57 setup.

---

## 1. B1 — 2x2 factorial: split the E57 confound  **[HIGHEST PRIORITY]**

### Question
E57 moved **two** variables at once (`--ram 48->96` **and** speculation `off->on`) and measured the
recommended config **−12.5 % cold / −5.4 % warm**, 8/8 paired losses. Which variable caused the loss?

### Why it matters
This is currently unanswerable, and two headline recommendations depend on it:
- `--ram 96` is documented at **+33.8 % cold / +52.5 % warm** (§4c) — but that was measured
  **with speculation on in both arms**, non-interleaved, on another day.
- n-gram is documented at **+18.5 %** (§2) — but on a **chat** workload, not coding.

If speculation is the culprit, `V4_DRAFT=4 V4_NGRAM=1` must stop being a blanket recommendation.
If RAM is the culprit, §4c is wrong and the cache-size story needs rewriting.

### Design
Full 2x2, all four cells **in one session**, single frozen history, ABCD then DCBA (8 arms) so cell
position is balanced against drift.

| cell | `--ram` | `V4_DRAFT` | note |
|---|---|---|---|
| **A** | 48 | 0 | stock (E57 measured 0.4814 cold / 0.4959 warm) |
| **B** | 96 | 4 (+`V4_NGRAM=1`) | reco (E57 measured 0.4213 / 0.4691) |
| **C** | 96 | 0 | **new** — isolates RAM |
| **D** | 48 | 4 (+`V4_NGRAM=1`) | **new** — isolates speculation |

Do **not** reuse E57's A2/B2 numbers as two of the cells: they were taken under different ambient
load (27.9 GB vs 9.7 GB non-engine), and §0.2 row 1 is exactly why that is invalid.

### Attribution
- RAM effect, no spec: `C − A`
- RAM effect, with spec: `B − D`
- spec effect at ram48: `D − A`
- spec effect at ram96: `B − C`
- interaction: `(B − D) − (C − A)`

### Command
```bash
NTOK=64 ./validation/dsv4/ab_2x2.sh        # to be written, model on ab_stock_vs_reco.sh
```

### Success criteria
All 4 cells x 2 repeats gate-clean (>= 6/8 rows per cell). Report all four contrasts with paired
per-prompt deltas and win counts, never means alone.

### Cost / risk
~3 h. `--ram 96` cells require a quiet desktop (<= ~13 GB non-engine) or they will gate-fail like
E57's B1 did.

---

## 2. B2 — Is n-gram's +18.5 % workload-specific?  **[HIGH]**

### Question
§2 measured n-gram at **+18.5 %** on **10 fixed prompts on one topic, 24 tokens** — maximally
repetitive, the best possible case for prompt-lookup drafting. E57's coding prompts suggest it may
be **negative** there. Is the +18.5 % a general win or a chat artefact?

### Why it matters
`V4_DRAFT=4 V4_NGRAM=1` is the single most-repeated recommendation in RESULTS.md ("+18.5 % for zero
code"). If it only holds on repetitive text, that recommendation is actively harmful for coding
agents — the primary use case.

### Design
2x2: speculation {off, on} x workload {chat-10 @24 tok, coding-4 @64 tok}, `--ram 64` fixed
(largest budget that survives a normal desktop, removing memory as a variable). Interleaved,
frozen history.

### Mechanism to capture
Log `v4_dspark`/draft acceptance per arm. §2 attributes the MTP loss to **rejection cost**
(recurrent-attention replay), not acceptance rate. Capture acceptance **and** wall time so the same
distinction can be made for n-gram.

### Success / kill
If n-gram is negative on coding at >= 5 % with >= 6/8 paired losses, **the blanket recommendation is
withdrawn** and becomes workload-conditional.

### Cost
~1.5 h.

---

## 3. B3 — Attention attribution in prefill  **[HIGH, different kind of work]**

### Question
E54 established attention is **33.18 % of prefill wall** (14.4 s of 43.5 s, 168 ms per layer-chunk
call x 86) and it has been touched by **no lane in any campaign so far**. What is inside it?

### Why it matters
It is the largest untouched block in prefill and it is **O(n^2)** in context, which is why the
prefill-prefetch gain decays **6.20 % -> 2.70 %** as prompts grow (E55). Every other lever measured
so far attacks expert I/O, whose share **shrinks** as prompts lengthen. For long prompts attention
is the only component that scales the wrong way, so it is where the remaining headroom is.

### Design
Not a throughput A/B — an **attribution run**, same discipline as E54:
- extend `COLI_V4_PREFILL_TRACE` with per-stage attention timers (QK^T, softmax, AV, RoPE, KV
  assembly, any recurrent/sparse path)
- single `--max-tokens 1` run at p064 and p256 so the O(n^2) term is visible across two lengths
- **stages must sum to the measured attention block** with a stated residual, exactly as E54 did
  (it closed to 1.55 % residual). A profile that does not sum is not evidence — §12e.

### Success criteria
A table where attention sub-stages sum to the attention block with < 5 % residual, at two prompt
lengths, with the n^2 term identified.

### Explicit rule for this lane
**Attribute before proposing a mechanism.** Four consecutive prefill hypotheses died because a
mechanism was proposed before the engine was measured (E48/E52/E53/E54). Do not skip to a fix.

### Cost
~30 min of runs, plus instrumentation.

---

## 4. B4 — Where does prefill prefetch reach zero?  **[MEDIUM]**

### Question
`COLI_V4_PREFILL_PREFETCH=1` measured **−6.20 % at p064 (70 tok)** and **−2.70 % at p256 (184 tok)**
(E55). Extrapolating, it crosses zero somewhere past p512. Where?

### Why it matters
The feature ships **default OFF** having failed GATE B (>= 15 %). Knowing the crossover tells us
whether it is worth enabling for short prompts specifically, or should be deleted. It also gives a
second, independent estimate of attention's growth rate — cross-checking B3.

### Design
`bench/ab.sh` extended with p512 and p1024 fixtures; interleaved OFF/ON, n>=3 per length, cold.
Plot delta vs prompt tokens.

### Success criteria
A monotonic decay curve with the zero-crossing bracketed. If the gain is already <= 1 % at p512,
recommend deleting the feature rather than carrying a dead flag.

### Cost
~40 min (p1024 prefill is slow — p256 already costs ~111 s per run).

---

## 5. B5 — Does `--fast-kernels` help or hurt DECODE?  **[MEDIUM]**

### Question
`--fast-kernels` is measured only on **prefill TTFT** (42.780 -> 38.942 s alone; 36.392 s stacked
with prefetch, −14.9 %). Its effect on **decode throughput** has never been measured.

### Why it matters
It is being recommended on the strength of a prefill number. If it costs decode, the recommendation
is wrong for long generations. E50 established output quality is unchanged (10/10 both configs), so
only speed is open.

### Design
`--fast-kernels` {off, on} at fixed `--ram 64`, `V4_DRAFT=0` (isolate the kernel change from
speculation), coding prompts, 128 tok to give decode room. Interleaved, frozen history.

### Cost
~40 min.

---

## 6. B6 — Re-derive the `--ram` curve without speculation  **[MEDIUM — may be absorbed by B1]**

### Question
The whole `--ram` curve (48 / 64 / 96 at **+8.3 % / +33.8 %**) was measured with
`V4_DRAFT=4 V4_NGRAM=1` **on**. If B2 shows speculation is negative on coding, the curve was
measured through a distorting lens.

### Why it matters
"More RAM helps, and helps more the longer the session runs" (§4c) is a headline conclusion, and
§4c is separately caveated for being **non-interleaved** (arms 90 min apart, protocol step 5
violated). It has never been re-run cleanly.

### Design
`--ram` {48, 64, 96} x speculation off, interleaved, one session. B1 supplies the 48/96 no-spec
cells; this adds 64 and the interleaving. **Run B1 first** and only do this if B1 shows the RAM
effect survives without speculation.

### Cost
~1.5 h (or ~40 min if B1's cells are reused within the same session).

---

## 7. B7 — Metal: batch the token dimension (S=1 -> S=N)  **[LARGE, implementation-gated]**

### Not a benchmark yet — a prerequisite implementation

§12 built a Metal backend that is **correct** (bit-exact end-to-end at production reduction
lengths, byte-identical output) and **1.118x slower**. The cause is structural, not tuning:

- the seam is `coli_v4_metal_expert_forward(..., const float *input, ...)` — **one token, always**
- prefill does not help: 799 tok x 43 layers x 6 top-k = 206,142 ~ the 206,916 observed
  `expert_requests`; prefill is merely **more batch-1 calls**
- unified memory gives the GPU no bandwidth edge on a bandwidth-bound workload
- isolated matmuls **did** hit **7.26x at batch-8** — real, but unreachable through this seam

### What must change first
Gather all tokens routed to expert E within a layer and issue **one S=N matmul**. That is a change
to the MoE scheduler (`moe_token_pipeline`, `deepseek_v4.c:3579`), **not** to the backend — and it
is exactly the shape the 7.26x microbenchmark already measured.

### Only then benchmark
Metal {off, on} after batching, paired, token-exact check every run.

### Kill criterion
If, after batching, Metal is still not **>= 1.15x faster** end-to-end, close the GPU lane
permanently and record it. Three architectural fixes have already moved it 2.30x -> 1.17x -> 1.118x
slower; diminishing returns are established, so the batching change is the last honest attempt.

### RESOLVED — implemented and measured (E84, commit `9224329`). Kill criterion CLEARED.
The prerequisite implementation described above was built as `COLI_V4_MOE_BATCHED` (default OFF,
with `COLI_V4_MOE_BATCHED_MIN_N` default 4). It does exactly what this section asked for: gather all
rows routed to expert E within a layer-chunk and issue ONE S=N dispatch.

Measured against the criterion **as written** (Metal {off, on} end-to-end, ab.sh TTFT):
**42.528 -> 31.975 s on p064 = 1.330x**, and 109.575 -> 83.221 s on p256 = 1.317x. **>= 1.15x, so the
GPU lane stays open.**

Do not confuse that with the *incremental* of batching alone on top of the already-Metal attention
lane, which is **1.121x / 1.133x** (p064 / p256). Both numbers are real; they answer different
questions, and this file's criterion asks the first one.

Corrections to this section's premises, now that it has been measured:
- "unified memory gives the GPU no bandwidth edge on a bandwidth-bound workload" — the workload is
  NOT bandwidth-bound at the MoE matmul: E81 showed cache size is irrelevant (96 -> 64 GB moved
  nothing) and MoE runs at 2.28 GB/s against a 6.83 GB/s disk, i.e. compute-bound.
- "isolated matmuls hit 7.26x at batch-8 ... unreachable through this seam" — reachable after all,
  but the realised in-situ figure is far lower. Measured CPU/GPU at production dims: S=1 **0.40x**,
  S=2 **0.73x**, S=4 **1.68-2.41x**, S=16 **4.4x**. The GPU LOSES below S=4, which is why the
  implementation gates on a minimum group size.
- Independent confirmation: upstream PR #763 lifted a Metal attention `S<=4` cap for prefill on the
  older engine and reported prefill attention 35.9s -> 9.0s. Same lesson, arrived at separately.

---

## 8. Deferred, with reasons

| item | why deferred |
|---|---|
| **Prewarm with a smaller warm count** | §13 killed `COLI_V4_PREWARM` at −29.7 %: it reads ~9.2 GB eagerly to avoid ~5.5 GB of misses — net **+3.69 GB more I/O**. §13d notes the fix is warming **far fewer** experts (sweep 64/128/256 vs 688) so saved misses exceed bytes read. Worth doing only if someone needs cold-start latency specifically. |
| **MTP re-measurement** | measured **+3.3 %** (3/6 wins) against n-gram's +18.5 %; §2 attributes the ceiling to **rejection cost**, not acceptance (which reached 70-89 % here). Nothing suggests the economics changed. |
| **`COLI_V4_EXPERT_PREFETCH` (decode)** | conclusively **null**: −0.6 % cold / +0.2 % warm over 16 rows, 0 gate failures. Settled. |
| **Deep-queue / QD expert reads** | **GATE A: STOP.** QD8/QD1 = **1.34x** against a >= 2x bar; the device saturates at QD4 (7.028 GB/s). No design survives that. |
| **`--ram 110+`** | ~120 GB total on a 137 GB host; guaranteed compression per §0.3. |
| **Upstream PR #1024 (FUSED3 expert matmul)** | #1024 (FUSED3, "40% less matmul time") is the tempting one — but it's `#if defined(__AVX2__)` with no NEON variant. x86 only. |
| **Upstream PR #934 (ARM NEON matmul_e8)** | #934 is ARM NEON but only `fmt=6 E8/IQ3`, not our MXFP4. |
| **Upstream PR #1017 (`coli_v4_route_bf16` in `moe_token`)** | #1017 helps `moe_token` — the single-token decode path, which our TTFT metric excludes by construction. |

---

## 9. Suggested order

1. **B1** (2x2) — unblocks B6 and decides whether two headline recommendations survive
2. **B2** (n-gram workload sensitivity) — directly tests E57's proposed cause
3. **B3** (attention attribution) — the only lane pointing at remaining headroom
4. **B4**, **B5** — cheap, close open questions
5. **B6** — only if B1 says the RAM effect is real without speculation
6. **B7** — large; schedule deliberately, not opportunistically

B1 and B2 together decide whether `V4_DRAFT=4 V4_NGRAM=1 --ram 96` remains the recommended config
at all. Until they run, **the honest recommendation for a coding workload is the one E57 actually
measured fastest: `--ram 48`, speculation off** — with the caveat that E57 could not attribute why.

---

## 10. Upstream re-survey (2026-08-24) — new work since the first survey

Verified against `origin` directly, not taken from the PR description.

### 10.1 ADOPT — beneficial, applies to arm64 + Metal
| PR | commit | what it does | why us |
|---|---|---|---|
| #1097 | `31debb7` | DeepSeek-V4 loader pool default **3 → 9 lanes**, claims **1.41x decode** | our TTFT metric excludes decode, so verify on TTFT before believing it helps *us* |
| #1109 | `93271fd` | ARM64 dotprod probe by compile, `armv8.2-a` first | Apple Silicon build path |
| #1211 | `c6b951a` | build shard index once per engine open | TTFT includes load, so this lands directly on our metric |
| #1164 | `74a5270` | resident expert lookup O(1) | high-RAM box, hot path |
| #1193 | `1d371230` | index expert slots | same hot path |
| #1162 | `c27a203` | coalesce duplicate concurrent expert loads | expert-store contention |
| #1121 | `909f2b4` | LRU victim selection respects lowered `ecap` | fixes a cache-thrash loop |
| #1092 | `b00688b` | second-chance expert eviction | same cache-pressure lane |

### 10.2 ISSUE #900 IS FIXED UPSTREAM — supersedes the W5/W6 plan
`e36a1c7` (merge of PR #1023) packs hot pinned experts **outside `state->mutex`**. Confirmed present
upstream; confirmed ABSENT here (`hot_pack_slot_locked` x3, zero `hot_pack_slot_prepare`/`pack_mutex`).

This changes the plan: W5 was "measure PACK_SHARE, then decide whether hand-porting #983 is worth
the risk". The fix is now a cherry-pick, so the adoption cost collapses and the decision bar drops
with it. **Still measure** — adopting churn that buys nothing is its own failure — but the
measurement is now validation, not a go/no-go gate.

### 10.3 OVERLAP — upstream is building the same thing we already shipped
| PR | commit | overlaps |
|---|---|---|
| #941 | `4b9b47d` | expert-major prefill, each distinct expert read once per chunk = our `COLI_V4_MOE_GROUPED` |
| #1166 | `d2bb447` | batch FP4 expert matmuls during prefill = our `COLI_V4_MOE_BATCHED`, different engine |
| #1180 | `5129f836` | reuse expert cache + batch shared prefill |
| #1197 | `44d1970` | batch shared experts during prefill |
| #1200 | `3d0bab1` | batch dense + shared prefill |
| #1213 | `ee9bc1f` | double-buffered expert bank prefetch for GPU prefill — latency hiding, not S |
| #1202/#1203 | `929e99b`/`6896e5f` | hybrid GPU/CPU split on decode misses, DMA overlap — latency hiding, not S |

**Read this honestly:** convergent evolution, and upstream is now moving faster on this lane than we
are. Our differentiator is not "batching MoE" — they have that too. It is that ours runs **on Metal
and is bit-exact**, which upstream has no equivalent of for deepseek_v4.

### 10.4 IGNORE
#1177 (RTX 3090 / Direct I/O / MTP), #1171/#1178/#1155 (CUDA/CPU quant, attention, KV cache),
#1160/#1212 (CI noise). All x86/CUDA or non-perf.

### 10.5 Apple Silicon datapoints for calibration (decode tok/s)
M4 Max 128 GB `#1210` **0.45** (TB5 SSD, iobench 12.49 GB/s) | M1 Max 64 GB `#1030` **0.61**
(Metal ~2.5x CPU, thermally sensitive) | M3 Max 128 GB `#47` **0.35–0.45**.
Ours on M3 Max measures **~1.2–1.5 tok/s** decode — 2-3x the published figures for comparable
hardware. Treat with suspicion until compared like-for-like (prompt, topp, RAM_GB all differ).

---

## 11. B8 — MIN_N threshold sweep for `COLI_V4_MOE_BATCHED`  **[next experiment]**

### The question
Metal loses badly at S=1 and only becomes worthwhile around S>=4. The scalar wrapper invokes the
batch kernel with `batch_rows=1` (`c/backend_metal_v4.mm:979`), and the grouped scheduler therefore
offloads only groups of >= `COLI_V4_MOE_BATCHED_MIN_N` (default 4).

**Raising MIN_N 4 → 8 is NOT automatically faster.** It makes each accepted dispatch larger but sends
FEWER groups to Metal. And the down projection measured faster at S=4 (2.413x) than S=8 (1.933x) in
`validation/probes/mxfp4_s_scaling.m`. That inversion is non-monotonic in a way that smells like
noise, and it was measured while a stray process was stealing a core — **re-measure it on the clean
host before treating it as a kernel property.**

### Design — interleaved sweep of MIN_N ∈ {4, 5, 6, 8, 12, 16}
Record per threshold:
- TTFT **and** decode separately (`time_to_first_token=` / `after_first=` — already emitted)
- number of offloaded groups
- rows handled by Metal
- group-size histogram
- Metal dispatch/wait time
- CPU fallback time

Instrumentation for the middle four did not exist and has been added to
`coli_v4_moe_grouped_stats_emit`:
```
moe_batched min_n=N groups=N rows=N ms=N rejects=N | cpu_rows=N cpu_ms=N | metal_row_share=N%
moe_group_hist 1:N 2:N 3:N 4:N ...
```
The histogram counts **every** group regardless of path, so the capture rate of any candidate
threshold is a property of the measured distribution rather than something re-derived per run. That
is what makes a losing threshold interpretable: it distinguishes "few groups reach 8" from "the ones
that do cannot offset the CPU absorbing 4-7".

### Beyond the threshold — the real ways to raise S
The threshold only *selects* from the group sizes routing happens to produce. To actually raise S:
1. **Continuous batching across concurrent decode requests** — needs a scheduler; biggest structural win.
2. **Grouping accepted speculative tokens before expert execution** — caution: upstream #689 reports
   deep speculative verify batches are not token-exact on CUDA; our bit-exactness bar is absolute.
3. **Grouping routes across a larger prefill window**, bounded by expert-cache capacity.
4. **Fusing multiple expert ops into fewer command buffers** — we already did this once for `wo_a`
   (A6, 8 dispatches → 1); the same shape should apply here.
5. **A small hot-expert set in private GPU-optimised buffers** — the only idea that could rescue S=1,
   which is where the GPU currently loses 0.40x.
