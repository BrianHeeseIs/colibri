# Ultrawork Notepad — colibri M4 moe_batch (prefill lane) + startup lane
Started: 2026-08-14T01:26 Europe/Amsterdam

## Mandate
User: "keep working until you can't continue at all without my input."
Save ALL questions for tomorrow's "good morning"/"good afternoon" -> ask via question tool then.
=> DO NOT block on questions. Log them under ## Questions For User and keep going.

## Findings (carried from this session)
- ft-deepmetal @197583b. Merged+pushed: M15(2row), M15b(4row), E37 attn_sparse opt-in.
- Steady state (220 tok): decode_wall 134619ms w/ --fast-sparse-attn, 151426ms default. 1.446->1.627 tok/s.
- Leaf ranking steady: expert_forward 26.6%, attn_out 18.1%, attn_sparse 15.0%(now 3.3%), attn_qkv 8.9%,
  expert_wait 6.0%, shared_expert 5.8%, head 4.9%, router 4.8%, hc_norm 4.0%.
- c/deepseek_v4.c:4141-4162 prefill loop: attention batched over 64-token chunks
  (coli_v4_attention_window_batch_ref) BUT MoE is per-token: moe_token_pipeline() at :4154
  called once per item. No dedup of expert leases across tokens in a chunk.
- target_batch() :7528, chunk size 64 hardcoded at :7547 (offset += 64).
- expert_requests == tokens*layers*6 exactly (61662 ~ 240*43*6) => confirms no grouping today.
- MEASUREMENT PROBLEM: ttft is dominated by model LOAD (~40s), not prefill. Must isolate.

## Questions For User (ASK TOMORROW, DO NOT BLOCK)
1. (pending — none yet beyond scope confirmations)

## Now
Isolate true prefill cost from model-load cost; establish grouping factor S before building M4.

## Todo
see todowrite list

## Learnings
- 8-token gates are too short (3x this session). Use >=60 tokens for correctness gates.
- Always warm up after rebuild; cold first-run inflated numbers twice.
- generated_text is MULTI-LINE; grep '^generated_text=' compares only first line. Use awk block extract.

## Durability
Relocated from /tmp to repo (.backlog/ULW-NOTEPAD.md) 2026-08-14 ~01:30 per user:
/tmp is wiped on crash and this machine crashes. This file is COMMITTED and is the
authoritative resume point. Append only.

## Permissions granted by user
- Commit freely whenever work is verified + has results (no need to ask).
- Work autonomously; collect questions and ask them at "good morning"/"good afternoon".

## Findings 2026-08-14 01:29 (pre-plan measurements, M4)
- Fixed floor: prompt="Hi" (5 tok), max-tokens 1 -> ttft=5.278s, wall real=5.78s.
  Dense weights are resident/mmap (v4_dense_resident 6.267GiB), so ENGINE LOAD IS CHEAP.
  Earlier assumption that ttft was ~40s of model load was WRONG.
- Prefill is COLD-EXPERT-I/O heavy: 5 prompt tokens -> expert_requests=1290 (=5*43*6),
  hits=443 misses=847, hit_rate=34.3%, bytes=11.32 GB. 847 misses * 13.37MB = 11.3GB. checks out.
- IMPORTANT DESIGN CORRECTION for M4: the expert cache ALREADY dedups repeated reads
  (2nd token using expert 7 is a cache HIT, not a re-read). So grouping tokens by expert
  does NOT reduce bytes read. M4's win must come from ARITHMETIC INTENSITY:
  S=1 matvec reads 13.37MB of expert weights to do ~13.37M MACs (~1 MAC/byte);
  S=g matmul does g x the MACs for the same weight traffic. It is a bandwidth win.
- ACHIEVABLE GROUP SIZE IS THE CRUX (must measure, not assume):
  64-token chunk -> 64*6 = 384 selections over 256 experts/layer.
  Uniform-routing estimate: unique ~ 256*(1-(1-1/256)^384) ~ 199 => mean group S ~ 1.93.
  To reach S~8 you need ~341 tokens/chunk. Chunk size is HARDCODED 64 at c/deepseek_v4.c:7547
  (offset += 64) inside target_batch() :7528.
  => M4 may require RAISING the chunk size, which costs buffer memory. Real routing is skewed
  so measure actual unique-expert counts before committing to a design.

## Questions For User (ASK AT "good morning"/"good afternoon")
Q1. Prefill chunk size is hardcoded 64 (:7547). Raising it (256-512) is likely REQUIRED for M4
    to reach useful group sizes, but multiplies prefill buffer memory. Is raising it acceptable,
    and is there a memory ceiling I should respect beyond --memory-gb?
Q2. E37 shipped --fast-sparse-attn as opt-in (not bit-exact). If M4 grouping also changes FP
    combine order (likely - summing expert contributions in a different order), do you want the
    same opt-in treatment, or is prefill-only reordering acceptable by default since it affects
    the prompt pass rather than sampled output? (I will default to opt-in/bit-exact unless told.)
