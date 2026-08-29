# Ultrawork Notepad — solidify GPU-prefill, then #7 / #8 / #10
Started: 2026-08-29 (autonomous, ~8-9h, questions banked until operator greeting)

## Goal
Maximise real-world speed/comfort of DeepSeek-V4 on this M3 Max. Current champion:
`COLI_V4_KERNELS=all` + Metal OFF = 1.67 tok/s p064. Candidate: + GPU prefill
(`COLI_V4_MOE_GROUPED=1 COLI_V4_MOE_BATCHED=1 COLI_V4_METAL_VARIANT=simd_exact_cold`)
= -13.9% TTFT / +3.4% decode / -10.5% wall@40tok, N=2 only.

## Scenarios (the contract)
S1 decode-regression resolved  : N>=5 p256, 3 arms. PASS = decode delta has non-overlapping
                                 ranges or is provably inside noise. Artifact: tokps log.
S2 determinism resolved        : same N>=5 run reports deterministic/NONDET per arm with md5 list.
                                 PASS = a stated verdict backed by >=5 samples.
S3 #7 rows16 prefill           : engagement counter >0, then TTFT+tok/s both axes, text diff,
                                 taskcheck. PASS/FAIL either way, recorded.
S4 #8 phase-dependent attention: determine feasibility from source FIRST (is phase routing even
                                 expressible?). Only then measure.
S5 #10 whole-prompt MoE        : route-shape validation before any implementation (report says
                                 repeat E62 union on 2 multi-chunk prompts).
S6 no regression               : golden PASS 5d04890413ff539e802985ce8c727814 at default flags
                                 after every code change.

## Rules binding this run
- ONE engine at a time. pgrep guard before every engine task.
- Both axes whenever tokens generated. TTFT-only allowed only with --max-tokens 1, stated.
- Never judge on md5 alone: diff the TEXT, then taskcheck.
- Engagement counter REQUIRED before believing any flag result (E101 lesson).
- Name arms for the BACKEND they exercise, not the scheduling change (E113b lesson).
- Seed: cp .backlog/lab/coli_usage.snapshot before every run; restore model .coli_usage after.

## Findings
- (append)

## Learnings
- (append)

## Findings (recon, 2026-08-29 ~02:20)

### #8 phase-dependent Metal attention: ALREADY TRUE — no code needed
`coli_v4_metal_fp8_enabled()` (reads `COLI_V4_METAL_ATTN`, backend_metal_v4.mm:558-560) has exactly
TWO call sites, both on the BATCH path: deepseek_v4.c:2838 (`coli_v4_attention_window_batch_ref`)
and :13736 (`coli_fp8_matmul_batch_ref`). Normal decode uses a SEPARATE function,
`coli_v4_attention_window_token_ref` (:2129) / `attention_token_impl` (:1830) — verified by grep to
contain ZERO Metal references across :1830-2200.
Speculation defaults off (`V4_DRAFT` unset -> 0, :10768), and speculative decode is the only other
`target_batch` consumer. => `COLI_V4_METAL_ATTN=1` IS already "Metal prefill attention, CPU decode
attention". The report scoped #8 as small/medium implementation work; it is a MEASUREMENT.
Corollary: E87's "decode trended -5.25%" cannot have been decode attention moving to Metal. It must
be indirect (Metal context init / memory pressure), same class as E112's +3.4% decode cost.

### #10 whole-prompt MoE: FEASIBLE, and it does NOT need the batch-64 contract lifted
- Chunk N+1 attention depends on `attention[layer_id]` KV/compressed/indexer state written DURING
  attention (:2707 kv from `inputs`; :2755-2762 KV ring write; :2780-2787 sparse reads state->kv),
  NOT on MoE output. `state`/`next` swap happens only after ALL chunks of the layer (:10013).
  => MoE is deferrable across chunks within a layer.
- `coli_v4_moe_grouped_batch` (:5220) groups by expert over all `routes`, reads
  `inputs + item*dimension`, writes `outputs[item*dimension+...]` (:5587-5591). Explorer confirms
  no chunk/start_position assumption inside it; the ONLY 64 cap is on the caller
  (`coli_v4_block_window_batch_ref` :5728) and on attention.
  => Keep attention chunked at 64; call MoE once with batch = whole prompt. The E108 contract is
  untouched.
- Whole-prompt buffers already exist: `session->state`/`session->next` are `max_prompt_tokens * hd`
  (:10504-10508). Extra buffer needed is prompt_tokens * hidden floats (~13 MB at 799 tokens).
- Combine point is `coli_v4_hc_post` (:1480-1493, `value += post[dest] * branch[col]`), called per
  item at :5953-5964 in the grouped path — so deferral must preserve per-item post/comb state.

### Order of work
#8 and #7 are measurements (no code). #10 is the only implementation. Do measurements first.

### #7 MOE_BATCHED_ROWS16: ALSO a measurement, not implementation
`COLI_V4_MOE_BATCHED_ROWS16` is an existing env flag (deepseek_v4.c:5100). `coli_v4_moe_layout_batchable`
(:5195) returns 1 for block_rows==1 always, and for block_rows==16 ONLY when the flag is set; otherwise
those groups are rejected from the Metal batch and fall to CPU. rows16 == the HOT-PINNED experts
(:8442, `COLI_V4_PIN_SLOTS`), so the flag decides whether hot pins can reach the GPU at all.
The comment at :5497 says rows16 "has no such proof [of bit-exactness], so it keeps the CPU path" -
that is the E97 revert reasoning, which AGENTS.md records as WRONG (one token in sixty, taskcheck 5/5).
Engagement counter is free: `metal_row_share=%` in the `moe_batched` stats line (:5162), which also
prints `rows16=%d`. Interaction to watch: champion uses METAL_VARIANT=simd_exact_COLD while rows16
are the HOT pins, so the two may target disjoint expert sets.
=> All three report items (#7, #8, #10): only #10 needs code.

## Progress 2026-08-29 03:25
DONE: E114 (N=5 resolves decode regression +1.94% real, determinism = interaction not either flag),
E115 (#8 METAL_ATTN=1: -21.0% TTFT, BIT-EXACT, no code needed), E116 (#7 ROWS16=1: -7.3% TTFT,
row share 52.9->71.0%, taskcheck 5/5, no code needed).
CHAMPION: METAL=0 KERNELS=all MOE_GROUPED=1 MOE_BATCHED=1 METAL_VARIANT=simd_exact_cold
          METAL_ATTN=1 MOE_BATCHED_ROWS16=1
          => vs cpu_only: TTFT -37.3%, net wall@40 -29.7%, tok/s -1.75%, break-even ~3400 tokens.
REMAINING: #10 whole-prompt MoE (only item needing code).
NEXT STEP CHOSEN: instrumentation-only patch to measure the whole-prompt group histogram BEFORE
restructuring. Reason: the restructure touches the hottest path; the report's own precondition was
to validate the route-union shape (E62 saw 4.14 -> 7.99 mean group size). Measure, then decide.

## #10 design recon (03:50-04:15) — findings BEFORE implementing

1. **Loop nesting is already layer-major, chunk-minor** (`target_batch` :9989-10018). The chunk loop is
   INSIDE the layer loop, `state`/`next` are whole-prompt buffers, and the swap happens only after all
   chunks of that layer. So #10 needs NO re-nesting — it is a contained change, much smaller than the
   report implied.

2. **MoE is already ONE call per chunk** (`coli_v4_moe_grouped_batch`, :5945), sandwiched between two
   per-item loops: (A) hc_post + ffn norm -> `ffn_normalized_batch`, (B) the MoE, (C) hc_post combine
   into `outputs_hc`. #10 = hoist (B) out of the chunk loop and run (C) over the whole prompt.

3. **`coli_v4_moe_grouped_batch` has no batch cap** — the `batch<=64` contract lives on the CALLER
   (:5729) and on attention/matmul (4 sites, E108). Keeping attention chunked at 64 and widening only
   the MoE therefore leaves that contract untouched. This is the key enabling fact.

4. **The route-cache/loader coupling I feared DOES NOT EXIST in the champion config.** The block at
   :5973 is not a next-chunk prefetch, it is per-chunk teardown (`loader_batch_finish` +
   `route_cache_destroy`) of an async expert loader the MoE consumes. But:
   - the whole route_cache path is gated on ROUTEAHEAD, which only runs under
     `COLI_V4_PREFILL_PREFETCH` — unset here, and E113 proved it DEADLOCKS with GPU prefill anyway.
     Confirmed empirically: no `v4_prefill_routeahead active` line in a champion-stack run.
   - even if enabled, the loader starts only when `n_unique <= capacity`, and measured
     `pin_slots_per_layer=16` vs per-chunk `n_unique` of 170-188. The gate can never pass at prefill;
     it would take the `v4_prefill_loader fallback` branch.
   => Two independent reasons deferring the MoE cannot starve a loader. Risk retired.

5. Buffers needed at whole-prompt scope (184 tok, d=7168, hc=4, hd=28672): states 21 MB,
   ffn_normalized 5.3 MB, ffn_branch 5.3 MB, ffn_post 2.9 KB, ffn_comb 11.8 KB. At 2048 tokens states
   grows to ~235 MB — must be documented, and argues for allocating per-layer rather than per-call.

STATUS: Oracle (bg_b79439b7) consulted on semantic safety, aliasing, numerical identity, dense layers,
and buffer ownership. Implementation BLOCKED until it returns (own rule: never ship a decision Oracle
was asked to make).

## Oracle verdict on #10 (bg_b79439b7, 4m24s) — design APPROVED with guards
1. Semantically safe. No same-layer later chunk reads `next`; swap is after all chunks (:10011-10015).
2. **Biggest real risk is ALIASING, not scheduling.** `coli_v4_hc_post` (:1486-1494) is NOT in-place
   safe: it reads all `residual[source,*]` while overwriting `output[destination,*]`. If deferred
   `states` aliases `state` or `next`, the finish pass reads clobbered residual rows. Add hard guards:
   fail if `state == next`, fail if deferred scratch aliases either.
3. Routeahead: **skip it entirely in defer mode.** Oracle independently reached my empirical
   conclusion - it is same-chunk machinery (`routeahead_build` recomputes from the current chunk at
   :4655-4676; `moe_token_pipeline` consumes it for the same item/layer at :4829-4858), NOT a
   next-chunk dependency. In the grouped branch cached routes are not even fed to
   `coli_v4_moe_grouped_batch`. Skipping changes no semantics.
4. Ownership: `target_batch` owns ONE whole-prompt scratch bundle, heap, allocated once and reused
   every layer, freed in its own cleanup. NOT the engine (breaks the `engine==NULL` callers).
   `block_window_batch_ref` keeps cleaning only its chunk-local temps.
5. Bit-exactness: expect NO. Pre/post HC buffers stay exact if the split is right; divergence enters
   at the grouped MoE output because groups cross the CPU->Metal threshold and dispatch shape changes.
   So a diff in HC buffers = real bug; a diff only in final text = reassociation.
6. Preserve rounding points exactly: `state` round at :5919, `normalized_hc_pre` internal rounds
   :3681-3685.
7. Effort: medium, 1-2 days for production quality.

DECISION: attempt behind `COLI_V4_MOE_WHOLE_PROMPT` (default OFF) so the existing path stays
bit-for-bit and a non-converging attempt costs nothing. Timeboxed; if it does not converge, leave the
flag off, tree green, and hand over the design + this verdict.

## RUN COMPLETE 2026-08-29 ~07:00 — all three report items delivered
E114 N=5 (decode regression real at +1.94%, determinism = flag interaction)
E115 #8 METAL_ATTN=1  -> -21.0% TTFT, BIT-EXACT, no code needed
E116 #7 ROWS16=1      -> -7.3% TTFT, taskcheck 5/5, no code needed (E97's revert was wrong)
E117 #10 precondition -> group size 4.14 -> 7.97 (reproduces E62 exactly), no code needed
E118 champion holds at p064 (-34.3% TTFT)
E119 #10 IMPLEMENTED  -> -17.4% TTFT, decode free, taskcheck 5/5, default-OFF bit-identical
E119a mechanism scales: row share 69.8->95.1% at p512 (counter-only)

CHAMPION: COLI_V4_METAL=0 KERNELS=all MOE_GROUPED=1 MOE_BATCHED=1
          METAL_VARIANT=simd_exact_cold METAL_ATTN=1 MOE_BATCHED_ROWS16=1 MOE_WHOLE_PROMPT=1
  vs cpu_only @p256: TTFT -48.2%, net wall@40 -38.5%, tok/s -1.75%
REGRESSION GATE: bench/golden.sh PASS md5=5d04890413ff539e802985ce8c727814 (default path intact).

OPEN / BACKLOG (need operator go-ahead, all are LONG timing sweeps):
- p512 + p1024 timing sweep of MOE_WHOLE_PROMPT (mechanism says the win grows; only counters so far).
- Memory: `states` is ~235 MB at 2048 tokens. If it bites, tile the deferred FINISH pass only —
  never the attention contract.
- Decode axis is untouched by all of tonight's work: every win is prefill. tok/s is still ~1.68.
  The remaining decode lever list is in .backlog/simd-exact-remaining-measurements.md.
