// probe_expert_variant -- seam-level differential for the expert matmul variant.
//
// WHY THIS EXISTS, given probe_simd_parity already proves the kernel exact:
// probe_simd_parity dispatches the kernel ITSELF. It says nothing about whether the SEAM
// dispatches it correctly. Buffer indices, the ColiV4MatmulDims fields, the weight/scale
// offsets and the new grid shape are all untested by it, and every one of them can produce a
// plausible wrong answer rather than a crash. This probe calls the real exported entry point
// coli_v4_metal_expert_forward_batch() and hashes what comes out.
//
// COLI_V4_METAL_VARIANT is read once in a __attribute__((constructor)), so a single process
// cannot compare two variants. The probe therefore prints a digest of the outputs and the
// variant it actually saw; run it twice under different env and diff the digests:
//
//   clang -O2 -fobjc-arc -I c -o validation/metal/probe_expert_variant \
//         validation/metal/probe_expert_variant.m c/backend_metal_v4.o \
//         -framework Metal -framework Foundation -lc++
//   A=$(COLI_V4_METALLIB=c/build/metal-v4/deepseek_v4.metallib \
//       ./validation/metal/probe_expert_variant)
//   B=$(COLI_V4_METAL_VARIANT=simd_exact_cold COLI_V4_METALLIB=... \
//       ./validation/metal/probe_expert_variant)
//   [ "$A" = "$B" ] && echo IDENTICAL || echo DIVERGED
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensor.h"
#include "expert_store.h"
#include "backend_metal_v4_seam.h"

#define HID 4096      // production hidden
#define INT_DIM 2048  // production intermediate

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void fill(uint8_t *q4, uint8_t *e8, size_t nq, size_t ns, unsigned seed) {
    srandom(seed);
    for (size_t i = 0; i < nq; i++) q4[i] = (uint8_t)(random() & 0xFF);
    for (size_t i = 0; i < ns; i++) e8[i] = (uint8_t)(120 + (random() % 15));
}

static int run_batch(int batch) {
    const size_t gate_rb = (HID + 1) / 2, gate_ng = (HID + 31) / 32;
    const size_t down_rb = (INT_DIM + 1) / 2, down_ng = (INT_DIM + 31) / 32;

    uint8_t *gq = malloc((size_t)INT_DIM * gate_rb), *gs = malloc((size_t)INT_DIM * gate_ng);
    uint8_t *uq = malloc((size_t)INT_DIM * gate_rb), *us = malloc((size_t)INT_DIM * gate_ng);
    uint8_t *dq = malloc((size_t)HID * down_rb),     *ds = malloc((size_t)HID * down_ng);
    fill(gq, gs, (size_t)INT_DIM * gate_rb, (size_t)INT_DIM * gate_ng, 11);
    fill(uq, us, (size_t)INT_DIM * gate_rb, (size_t)INT_DIM * gate_ng, 22);
    fill(dq, ds, (size_t)HID * down_rb,     (size_t)HID * down_ng,     33);

    float *in  = malloc((size_t)batch * HID * 4);
    float *out = calloc((size_t)batch * HID, 4);
    float *rw  = malloc((size_t)batch * 4);
    srandom(7);
    for (int i = 0; i < batch * HID; i++) in[i] = ((float)(random() % 2001) - 1000) / 673.0f;
    for (int i = 0; i < batch; i++) rw[i] = 0.25f + (float)(random() % 100) / 400.0f;

    ColiExpertView e; memset(&e, 0, sizeof(e));
    e.gate.format = COLI_TENSOR_FP4_NATIVE_BLOCK; e.gate.scale_format = COLI_SCALE_UE8M0;
    e.gate.data = gq; e.gate.scales = gs;
    e.gate.data_bytes = (size_t)INT_DIM * gate_rb; e.gate.scale_bytes = (size_t)INT_DIM * gate_ng;
    e.gate.rows = INT_DIM; e.gate.columns = HID; e.gate.block_rows = 1; e.gate.block_columns = 32;
    e.up = e.gate; e.up.data = uq; e.up.scales = us;
    e.down.format = COLI_TENSOR_FP4_NATIVE_BLOCK; e.down.scale_format = COLI_SCALE_UE8M0;
    e.down.data = dq; e.down.scales = ds;
    e.down.data_bytes = (size_t)HID * down_rb; e.down.scale_bytes = (size_t)HID * down_ng;
    e.down.rows = HID; e.down.columns = INT_DIM; e.down.block_rows = 1; e.down.block_columns = 32;

    int rc = coli_v4_metal_expert_forward_batch(out, &e, in, rw, batch, 7.0f);
    if (rc != 0) {
        printf("  batch=%-2d SEAM REJECTED rc=%d\n", batch, rc);
    } else {
        printf("  batch=%-2d digest=%016llx  out[0]=%.9g out[%d]=%.9g\n", batch,
               (unsigned long long)fnv1a(out, (size_t)batch * HID * 4),
               out[0], batch * HID - 1, out[batch * HID - 1]);
    }
    free(gq); free(gs); free(uq); free(us); free(dq); free(ds);
    free(in); free(out); free(rw);
    return rc;
}

int main(void) { @autoreleasepool {
    const char *v = getenv("COLI_V4_METAL_VARIANT");
    int parsed = coli_v4_metal_variant();
    printf("  probe_expert_variant  COLI_V4_METAL_VARIANT=%s -> coli_v4_metal_variant()=%d\n",
           v ? v : "(unset)", parsed);
    /* SILENT-FALLBACK GUARD. coli_v4_metal_read_environment maps any UNRECOGNISED variant
     * string to 0 (= ordered_cold, the production baseline). So a typo, or a variant that has
     * not been implemented yet, does not fail -- it quietly measures the baseline and reports
     * "identical", which would look exactly like a successful bit-exactness result. Every
     * downstream A/B depends on this not happening, so refuse to run in that state. */
    if (v && *v && parsed == 0 && strcmp(v, "ordered_cold") != 0) {
        printf("  *** REFUSING TO RUN: variant '%s' is not recognised and silently parsed to 0\n", v);
        printf("  *** (any comparison from here would be baseline-vs-baseline)\n");
        return 2;
    }
    int bad = 0;
    bad |= run_batch(1) != 0;    // decode shape
    bad |= run_batch(8) != 0;    // prefill-ish shape
    printf("  %s\n", bad ? "SEAM REJECTED (rc != 0)" : "seam produced outputs");
    return bad ? 1 : 0;
} }
