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

#ifdef __cplusplus
}
#endif

#endif
