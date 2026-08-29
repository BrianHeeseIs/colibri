# fp8 rows16 layout — design verdict (blocking task T2)

## The conflict (found by the plan agent; I had missed it)
`wq_a/wq_b/wkv/wo_a/wo_b` weight bytes are read by BOTH:
- `coli_fp8_matvec_ref` (DECODE, c/deepseek_v4.c:13546) — the function I want to accelerate, and
- `coli_fp8_matmul_batch_ref` (PREFILL, :13794) — plus Metal prefill attention, `deepseek_v4_dspark.inc`
  (`v4_ds_pack_rows8` :113/:165) and a SECOND attention implementation at :2346-2508.
An IN-PLACE rows16 repack at load would mutate the bytes every one of those readers consumes.

## What actually happens if we repack in place — verified, not assumed
`coli_fp8_matmul_batch_ref:13801` validates:
    (weight->block_rows != 128 && weight->block_rows != 8) || ... return -1;
So `block_rows == 16` makes PREFILL RETURN -1 → hard engine failure. It fails LOUDLY rather than
corrupting silently, which is a mercy, but prefill breaks outright.

## How x86 resolves this today (the precedent, cited)
`packed_rows8` sets block_rows=8, which the batch path EXPLICITLY accepts (:13801) and has its own
AVX2 rows8 branch for (:13830-13831). So the codebase's existing answer is design (a): make EVERY
consumer layout-aware. On ARM that surface also includes the Metal prefill attention upload, which is
now default-on (METAL_ATTN=1, E115). That is a large and risky surface for a decode-only win.

## VERDICT: design (b), a DECODE-ONLY SHADOW BUFFER
Keep `weight->data` byte-for-byte untouched. Build a SEPARATE rows16-interleaved copy, consulted only
by `coli_fp8_matvec_ref` on aarch64. Every other reader — prefill batch, Metal, dspark, the second
attention implementation — continues to read the original bytes and is provably unaffected.

Consequence: scenario S4 ("prefill unchanged") holds EXACTLY and by construction, rather than needing
to be argued. Prefill output must be bit-identical, and that is now a cheap assertion instead of a
wide refactor.

### Why this is affordable here (verified)
`coli_v4_layer_load` (:841-874) is RESIDENT: `engine->dense_resident.ready[layer]` gates a one-time
load, and subsequent calls do `*weights = engine->dense_resident.layers[layer]`, so DATA POINTERS ARE
STABLE for the whole run. `coli_v4_layer_free` (:876-885) detects a resident layer and only memsets
the caller's copy. Therefore the shadow is built ONCE per tensor and amortised over every token —
not rebuilt per token, which is what would have killed this design.

### Cost
Attention fp8 per layer: wq_b 33.55 + wo_a 33.55 + wo_b 33.55 + wq_a ~4.19 + wkv 2.10 MB ~= 107 MB.
x43 resident layers ~= 4.6 GB additional resident memory on a 128 GB host (engine RSS ~84 GB,
free+inactive ~40 GB). Affordable, but it MUST be measured and reported, and capped by an env knob.

### Rejected
(a) layout-aware everywhere — correct and idiomatic, but drags in the Metal prefill upload path and
    the dspark packer for a decode-only benefit. Higher risk, more surface, no extra speed.
(c) decode-only tensor subset — the census shows the same specs feed both paths, so the subset is
    empty. Not viable.

## Rules carried into implementation
- DO NOT touch `matmul_fp8` in c/quant.h: shared by colibri.c, inkling.c, kimi_k3.c, olmoe.c,
  deepseek_v4.c (E103). The kernel lives in `coli_fp8_matvec_ref` only.
- The shadow is keyed by (data pointer, rows, columns); `wo_a`'s 8 group views have distinct offset
  pointers and therefore get 8 distinct entries, which is correct and costs the same total.
- Engagement counter required (E101): 0 with the flag off, non-zero with it on.
- Bit-exactness must be RE-VERIFIED in-engine; the microbench's FMA contraction may differ from the
  engine's build flags.
