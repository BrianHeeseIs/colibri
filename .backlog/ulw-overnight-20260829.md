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
