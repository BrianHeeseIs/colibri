/* head_ilp.h -- ILP restructuring of the LM head dot product. Header-only, like c/quant.h.
 *
 * WHY THIS EXISTS
 * The vocabulary head is 6.7% of decode (1399.6 ms of 20977.5 at p256/39 tokens) and its loop in
 * `head_argmax` already runs one OpenMP team over all 129280 rows. The cost is NOT bandwidth: each
 * row is a 4096-deep SERIAL dependency chain of multiply-accumulates. At ~4-cycle FP latency that
 * is ~16.4k cycles per row; 129280 rows over 16 threads is ~132M cycles, i.e. ~35 ms/token at
 * ~3.8 GHz. Measured is 1399.6/39 = 35.9 ms/token. The chain latency IS the wall.
 *
 * THE FIX, AND WHY IT IS BIT-EXACT
 * Process four vocabulary rows in one pass over `hidden`, each keeping its OWN accumulator. Four
 * independent chains hide the latency. Every row still accumulates in ascending column order using
 * the IDENTICAL expression it uses today, so whatever the compiler does -- fuse into FMLA or not,
 * vectorise or not -- it does the same thing to each chain. Exactness follows STRUCTURALLY rather
 * than from matching a disassembly, which matters because `head_bf16_dot` is static+inlined and
 * built with -flto, so its contraction cannot be read authoritatively from the shipped binary
 * (see .backlog/lab/NOTES-head-contraction.md).
 *
 * This is the `o += 4` idiom already used by matmul_fp8 (c/quant.h:506).
 * Ceiling note: 4x ILP would demand ~118 GB/s, above this host's measured ~105 GB/s read ceiling,
 * so expect ~3x on the phase, not 4x.
 */
#ifndef COLI_V4_HEAD_ILP_H
#define COLI_V4_HEAD_ILP_H

#include <stdint.h>
#include <string.h>

/* Identical to coli_bf16_decode (c/deepseek_v4.c:13388): bits << 16, reinterpreted. */
static inline float coli_v4_head_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

/* Reference: one row, exactly the expression head_bf16_dot uses today. */
static inline float coli_v4_head_dot1(const uint16_t *weight, const float *hidden, int dimension) {
    float sum = 0.0f;
    for (int column = 0; column < dimension; column++)
        sum += coli_v4_head_bf16(weight[column]) * hidden[column];
    return sum;
}

/* Four rows in flight. Each lane's accumulation order and expression match coli_v4_head_dot1. */
static inline void coli_v4_head_dot4(const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const float *hidden, int dimension, float *out4) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int column = 0; column < dimension; column++) {
        const float h = hidden[column];
        s0 += coli_v4_head_bf16(w0[column]) * h;
        s1 += coli_v4_head_bf16(w1[column]) * h;
        s2 += coli_v4_head_bf16(w2[column]) * h;
        s3 += coli_v4_head_bf16(w3[column]) * h;
    }
    out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = s3;
}

#endif /* COLI_V4_HEAD_ILP_H */
