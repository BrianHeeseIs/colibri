#ifndef COLI_BACKEND_METAL_V4_SEAM_H
#define COLI_BACKEND_METAL_V4_SEAM_H

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_metal_enabled(void);
int coli_v4_metal_variant(void);
unsigned long coli_v4_metal_dispatches(void);
void coli_v4_metal_profile_report(void);
void coli_v4_metal_register_slab(void *base, size_t length);
void coli_v4_metal_unregister_slab(void *base);
/* Zero means out was produced; any non-zero result requires CPU fallback. */
int coli_v4_metal_expert_forward(float *out, const ColiExpertView *expert,
                                 const float *input, float route_weight,
                                 float swiglu_limit);
int coli_v4_metal_expert_forward_batch(float *outs,
                                       const ColiExpertView *expert,
                                       const float *inputs_gathered,
                                       const float *route_weights, int batch,
                                       float swiglu_limit);

/* ---- Bit-exact batched fp8 matmul (attention projections) ----------------------------
 * Mirrors quant.h matmul_fp8 exactly: float accumulation within each 128-column block,
 * double accumulation across blocks (emulated with double-float on the GPU, which has no
 * fp64). Proven 0 ULP over 2,457,600 outputs on the four real attention shapes (E68).
 * REQUIRES the metallib to be built with -fno-fast-math (see Makefile METALCFLAGS).
 *
 * The 256-entry E4M3 decode table is registered from the engine's own e4m3_decode() so the
 * shader can never drift from the C table. Registration is idempotent and cheap.
 * Returns 0 when outputs were produced; any non-zero result requires CPU fallback. */
void coli_v4_metal_fp8_register_lut(const float *lut256);
int  coli_v4_metal_fp8_enabled(void);
int  coli_v4_metal_fp8_matmul_batch(float *outputs,
                                    const void *weight_data,
                                    const float *weight_scales,
                                    const float *inputs,
                                    int batch, int rows, int columns);

/* A6: fused grouped variant - one dispatch over all output groups instead of one per group.
 * Layout is fully contiguous per group: weights g*rows*columns bytes, scales
 * g*ceil(rows/128)*nblkI floats, inputs g*batch*columns floats, outputs g*batch*rows floats.
 * Per-output arithmetic is identical to the ungrouped kernel, so results are bit-exact.
 * Returns 0 on success; any non-zero result requires the caller's CPU fallback. */
int  coli_v4_metal_fp8_matmul_grouped(float *outputs,
                                      const void *weight_data,
                                      const float *weight_scales,
                                      const float *inputs,
                                      int batch, int rows, int columns, int groups);

#define COLI_V4_MOE_FUSED_MAX_EXPERTS 6
int coli_v4_metal_expert_forward_fused(
    float *out, const ColiExpertView *experts, const float *route_weights,
    int expert_count, const float *input, float swiglu_limit);

#ifdef __cplusplus
}
#endif

#endif
