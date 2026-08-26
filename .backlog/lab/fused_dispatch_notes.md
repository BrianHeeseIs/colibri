# Decode Fused MoE Dispatch

## ABI

`coli_v4_metal_expert_forward_fused(float *out, const ColiExpertView *experts,
const float *route_weights, int expert_count, const float *input, float
swiglu_limit)` accepts one sorted, heterogeneous expert-view array. `expert_count`
is limited to six. Return zero means exactly one Metal dispatch wrote the weighted
routed sum. Any nonzero return leaves caller on existing scalar-Metal/CPU chain.

Shader argument buffer has 36 resources: six slots each for gate q4, gate scale,
up q4, up scale, down q4, and down scale. Every resource is whole registered slab
at binding offset zero. `ColiV4MoeFusedParams` carries six per-tensor byte-offset
tables, route weights, and dimensions. This supports independent expert slabs
without requiring cross-expert contiguity or 256-byte alignment for each tensor
offset.

## Dispatch And Arithmetic

`coli_v4_moe_experts_fp4_ordered_fused` has one 1024-thread threadgroup and is
the only dispatch in fused seam. Static threadgroup memory is exactly 32768 bytes:
4096 input-QDQ floats plus 4096 gate/up-or-down-input floats. Pipeline creation
rejects devices below 1024 threads or above either 32768 bytes or device maximum.

Input QDQ is retained across expert slots. Expert slots run serially in caller's
ascending-ID order. Each output row has one thread, which adds each BF16-rounded
down contribution in slot order. Matvecs use `coli_v4_moe_ordered_matvec`: serial
within each 32-column group, then `accumulator += group_accumulator * scale` once
per group. Gate/up BF16 rounding, asymmetric SwiGLU clamp and stable sigmoid,
weighted BF16 rounding, down-input FP8 QDQ, and final down BF16 rounding match
existing batch-chain stages.

## Guards And Fallback

`COLI_V4_MOE_FUSED` defaults off. Decode only attempts fusion when Metal is on,
six-or-fewer sorted selections exist, and no prefill loader owns the route. It
leases all selected experts, calls fused seam once, then releases all leases.
Rows16 layouts, bad tensor layouts, unsupported dimensions, missing pipeline
capacity, or any tensor pointer outside a registered slab reject before dispatch.
The decode caller then runs existing individual scalar-Metal-or-CPU expert path
for that same token.

No feasibility blocker found. `c/metal/coli_v4_moe.metal` already established
that one complete cold expert chain fits in 32 KiB; retaining input-QDQ while
reusing gate/up storage for down input makes serial six-expert execution fit the
same limit.

## Build Evidence

Built with `make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`.
No engine or benchmark was run.
