/*
 * DeepSeek-V4-Flash routed-expert kernel gap gate.
 *
 * This standalone benchmark faithfully copies the measured engine code from:
 *   c/quant.h:1422-1549              fallback MXFP4 kernels
 *   c/deepseek_v4.c:11355-11484      activation QDQ and scalar decoders
 *   c/deepseek_v4.c:11938-12374      rows16 pack and arm64 NEON kernels
 *   c/deepseek_v4.c:7102-7133        full expert operation
 *
 * Build (Apple Silicon, intentionally no OpenMP and no -ffast-math):
 *   cc -O3 -mcpu=native -pthread bench/kernel_gap.c -o bench/kernel_gap -lm
 * Run:
 *   ./bench/kernel_gap [iterations]  # default 30, minimum 20
 */

#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif

#include <arm_neon.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(__aarch64__)
#error "kernel_gap measures the DeepSeek-V4 arm64 rows16 kernel"
#endif

enum {
    HIDDEN_SIZE = 4096,
    INTERMEDIATE_SIZE = 2048,
    BLOCK_COLUMNS = 32,
    ALIGNMENT = 16 * 1024,
    DEFAULT_ITERATIONS = 30,
    WARMUP_ITERATIONS = 5,
};

static const size_t MATRIX_DATA_BYTES =
    (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
static const size_t MATRIX_SCALE_BYTES =
    (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE / BLOCK_COLUMNS;
static const size_t MATRIX_BYTES =
    (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 +
    (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE / BLOCK_COLUMNS;
static const size_t RECORD_BYTES = UINT64_C(13369344);
static const double MACS_PER_EXPERT = 25.2e6;
static const float ROUTE_WEIGHT = 0.25f;
static const float SWIGLU_LIMIT = 10.0f;
static volatile float output_guard;

typedef struct {
    const unsigned char *data;
    const unsigned char *scales;
    size_t data_bytes;
    size_t scale_bytes;
    int rows;
    int columns;
    int block_rows;
} TensorView;

typedef struct {
    TensorView gate;
    TensorView down;
    TensorView up;
} ExpertView;

typedef struct {
    double mean_ms;
    double stddev_ms;
} Measurement;

static void die(const char *message) {
    fprintf(stderr, "kernel_gap: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *aligned_alloc_or_die(size_t alignment, size_t bytes) {
    void *memory = NULL;
    int result = posix_memalign(&memory, alignment, bytes);
    if (result != 0) {
        errno = result;
        perror("kernel_gap: posix_memalign");
        exit(EXIT_FAILURE);
    }
    return memory;
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        perror("kernel_gap: clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint64_t random_state = UINT64_C(0x243f6a8885a308d3);

static uint32_t random_u32(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return (uint32_t)(random_state >> 32);
}

static float e8m0_decode(uint8_t value) {
    if (value == 0xff) return NAN;
    return ldexpf(1.0f, (int)value - 127);
}

static float e2m1_decode(uint8_t nibble) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 15];
}

static float e4m3fn_decode(uint8_t value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number = !exponent
        ? ldexpf((float)mantissa, -9)
        : ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

static uint8_t e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f)
        return (uint8_t)((negative ? 0x80 : 0) | 0x7e);
    uint8_t best;
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {
        uint32_t bits;
        memcpy(&bits, &magnitude, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu);
        uint32_t rounded = significand >> 20;
        uint32_t remainder = significand & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) {
            rounded = 8u;
            exponent++;
        }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uint8_t)(best | (negative ? 0x80 : 0));
}

static float bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void bf16_round_array(float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = bf16_round(values[index]);
}

static int ceil_log2_positive(float value) {
    int exponent;
    float fraction = frexpf(value, &exponent);
    return fraction == 0.5f ? exponent - 1 : exponent;
}

static void fp8_activation_qdq(float *output, uint8_t *scales,
                               const float *input, size_t length) {
    const size_t block_size = 128;
    for (size_t base = 0; base < length; base += block_size) {
        size_t count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (size_t index = 0; index < count; index++)
            maximum = fmaxf(maximum, fabsf(input[base + index]));
        maximum = fmaxf(maximum, 1e-4f);
        int scale_exponent = ceil_log2_positive(maximum / 448.0f);
        if (scale_exponent < -127) scale_exponent = -127;
        if (scale_exponent > 127) scale_exponent = 127;
        uint8_t encoded_scale = (uint8_t)(scale_exponent + 127);
        float scale = e8m0_decode(encoded_scale);
        scales[base / block_size] = encoded_scale;
        for (size_t index = 0; index < count; index++) {
            float normalized = fmaxf(
                -448.0f, fminf(448.0f, input[base + index] / scale));
            output[base + index] =
                e4m3fn_decode(e4m3fn_encode(normalized)) * scale;
        }
    }
}

static const float mx4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float mx4_scale(uint8_t scale) {
    union {
        uint32_t bits;
        float value;
    } decoded;
    decoded.bits = (uint32_t)scale << 23;
    return decoded.value;
}

static void matmul_mxfp4(float *y, const float *x, const uint8_t *q4,
                         const uint8_t *e8s, int S, int I, int O) {
    int rb = (I + 1) / 2, ng = (I + 31) / 32;
    for (int o = 0; o < O; o += 4) {
        int o1 = o + 1 < O ? o + 1 : o;
        int o2 = o + 2 < O ? o + 2 : o;
        int o3 = o + 3 < O ? o + 3 : o;
        const uint8_t *w0 = q4 + (int64_t)o * rb;
        const uint8_t *w1 = q4 + (int64_t)o1 * rb;
        const uint8_t *w2 = q4 + (int64_t)o2 * rb;
        const uint8_t *w3 = q4 + (int64_t)o3 * rb;
        const uint8_t *s0 = e8s + (int64_t)o * ng;
        const uint8_t *s1 = e8s + (int64_t)o1 * ng;
        const uint8_t *s2 = e8s + (int64_t)o2 * ng;
        const uint8_t *s3 = e8s + (int64_t)o3 * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int g = 0; g < ng; g++) {
                int base = g * 32, glen = 32;
                if (base + glen > I) glen = I - base;
                float c0 = mx4_scale(s0[g]), c1 = mx4_scale(s1[g]);
                float c2 = mx4_scale(s2[g]), c3 = mx4_scale(s3[g]);
                float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
                if (glen == 32) {
                    const uint8_t *p0 = w0 + (base >> 1);
                    const uint8_t *p1 = w1 + (base >> 1);
                    const uint8_t *p2 = w2 + (base >> 1);
                    const uint8_t *p3 = w3 + (base >> 1);
                    for (int k = 0; k < 16; k++) {
                        float xa = xs[base + 2 * k], xb = xs[base + 2 * k + 1];
                        uint8_t b0 = p0[k], b1 = p1[k], b2 = p2[k], b3 = p3[k];
                        g0 += xa * mx4_lut[b0 & 0xF];
                        g0 += xb * mx4_lut[b0 >> 4];
                        g1 += xa * mx4_lut[b1 & 0xF];
                        g1 += xb * mx4_lut[b1 >> 4];
                        g2 += xa * mx4_lut[b2 & 0xF];
                        g2 += xb * mx4_lut[b2 >> 4];
                        g3 += xa * mx4_lut[b3 & 0xF];
                        g3 += xb * mx4_lut[b3 >> 4];
                    }
                } else {
                    for (int i = base; i < base + glen; i += 2) {
                        uint8_t b0 = w0[i >> 1], b1 = w1[i >> 1];
                        uint8_t b2 = w2[i >> 1], b3 = w3[i >> 1];
                        float xa = xs[i];
                        g0 += xa * mx4_lut[b0 & 0xF];
                        g1 += xa * mx4_lut[b1 & 0xF];
                        g2 += xa * mx4_lut[b2 & 0xF];
                        g3 += xa * mx4_lut[b3 & 0xF];
                        if (i + 1 < base + glen) {
                            float xb = xs[i + 1];
                            g0 += xb * mx4_lut[b0 >> 4];
                            g1 += xb * mx4_lut[b1 >> 4];
                            g2 += xb * mx4_lut[b2 >> 4];
                            g3 += xb * mx4_lut[b3 >> 4];
                        }
                    }
                }
                a0 += g0 * c0;
                a1 += g1 * c1;
                a2 += g2 * c2;
                a3 += g3 * c3;
            }
            y[(int64_t)s * O + o] = a0;
            if (o1 != o) y[(int64_t)s * O + o1] = a1;
            if (o2 != o) y[(int64_t)s * O + o2] = a2;
            if (o3 != o) y[(int64_t)s * O + o3] = a3;
        }
    }
}

static void matmul_mxfp4_dual(float *y0, float *y1, const float *x,
                              const uint8_t *q0, const uint8_t *e0,
                              const uint8_t *q1, const uint8_t *e1,
                              int S, int I, int O) {
    int rb = (I + 1) / 2, ng = (I + 31) / 32;
    for (int o = 0; o < O; o += 4) {
        int o1 = o + 1 < O ? o + 1 : o;
        int o2 = o + 2 < O ? o + 2 : o;
        int o3 = o + 3 < O ? o + 3 : o;
        const uint8_t *wa0 = q0 + (int64_t)o * rb, *wa1 = q0 + (int64_t)o1 * rb;
        const uint8_t *wa2 = q0 + (int64_t)o2 * rb, *wa3 = q0 + (int64_t)o3 * rb;
        const uint8_t *wb0 = q1 + (int64_t)o * rb, *wb1 = q1 + (int64_t)o1 * rb;
        const uint8_t *wb2 = q1 + (int64_t)o2 * rb, *wb3 = q1 + (int64_t)o3 * rb;
        const uint8_t *sa0 = e0 + (int64_t)o * ng, *sa1 = e0 + (int64_t)o1 * ng;
        const uint8_t *sa2 = e0 + (int64_t)o2 * ng, *sa3 = e0 + (int64_t)o3 * ng;
        const uint8_t *sb0 = e1 + (int64_t)o * ng, *sb1 = e1 + (int64_t)o1 * ng;
        const uint8_t *sb2 = e1 + (int64_t)o2 * ng, *sb3 = e1 + (int64_t)o3 * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float aa0 = 0, aa1 = 0, aa2 = 0, aa3 = 0;
            float ab0 = 0, ab1 = 0, ab2 = 0, ab3 = 0;
            for (int g = 0; g < ng; g++) {
                int base = g * 32, glen = 32;
                if (base + glen > I) glen = I - base;
                float ca0 = mx4_scale(sa0[g]), ca1 = mx4_scale(sa1[g]);
                float ca2 = mx4_scale(sa2[g]), ca3 = mx4_scale(sa3[g]);
                float cb0 = mx4_scale(sb0[g]), cb1 = mx4_scale(sb1[g]);
                float cb2 = mx4_scale(sb2[g]), cb3 = mx4_scale(sb3[g]);
                float ga0 = 0, ga1 = 0, ga2 = 0, ga3 = 0;
                float gb0 = 0, gb1 = 0, gb2 = 0, gb3 = 0;
                if (glen == 32) {
                    const uint8_t *pa0 = wa0 + (base >> 1), *pa1 = wa1 + (base >> 1);
                    const uint8_t *pa2 = wa2 + (base >> 1), *pa3 = wa3 + (base >> 1);
                    const uint8_t *pb0 = wb0 + (base >> 1), *pb1 = wb1 + (base >> 1);
                    const uint8_t *pb2 = wb2 + (base >> 1), *pb3 = wb3 + (base >> 1);
                    for (int k = 0; k < 16; k++) {
                        float xa = xs[base + 2 * k], xb = xs[base + 2 * k + 1];
                        uint8_t a0 = pa0[k], a1 = pa1[k], a2 = pa2[k], a3 = pa3[k];
                        uint8_t b0 = pb0[k], b1 = pb1[k], b2 = pb2[k], b3 = pb3[k];
                        ga0 += xa * mx4_lut[a0 & 15]; ga0 += xb * mx4_lut[a0 >> 4];
                        ga1 += xa * mx4_lut[a1 & 15]; ga1 += xb * mx4_lut[a1 >> 4];
                        ga2 += xa * mx4_lut[a2 & 15]; ga2 += xb * mx4_lut[a2 >> 4];
                        ga3 += xa * mx4_lut[a3 & 15]; ga3 += xb * mx4_lut[a3 >> 4];
                        gb0 += xa * mx4_lut[b0 & 15]; gb0 += xb * mx4_lut[b0 >> 4];
                        gb1 += xa * mx4_lut[b1 & 15]; gb1 += xb * mx4_lut[b1 >> 4];
                        gb2 += xa * mx4_lut[b2 & 15]; gb2 += xb * mx4_lut[b2 >> 4];
                        gb3 += xa * mx4_lut[b3 & 15]; gb3 += xb * mx4_lut[b3 >> 4];
                    }
                } else {
                    for (int i = base; i < base + glen; i += 2) {
                        uint8_t a0 = wa0[i >> 1], a1 = wa1[i >> 1];
                        uint8_t a2 = wa2[i >> 1], a3 = wa3[i >> 1];
                        uint8_t b0 = wb0[i >> 1], b1 = wb1[i >> 1];
                        uint8_t b2 = wb2[i >> 1], b3 = wb3[i >> 1];
                        float xa = xs[i];
                        ga0 += xa * mx4_lut[a0 & 15]; ga1 += xa * mx4_lut[a1 & 15];
                        ga2 += xa * mx4_lut[a2 & 15]; ga3 += xa * mx4_lut[a3 & 15];
                        gb0 += xa * mx4_lut[b0 & 15]; gb1 += xa * mx4_lut[b1 & 15];
                        gb2 += xa * mx4_lut[b2 & 15]; gb3 += xa * mx4_lut[b3 & 15];
                        if (i + 1 < base + glen) {
                            float xb = xs[i + 1];
                            ga0 += xb * mx4_lut[a0 >> 4]; ga1 += xb * mx4_lut[a1 >> 4];
                            ga2 += xb * mx4_lut[a2 >> 4]; ga3 += xb * mx4_lut[a3 >> 4];
                            gb0 += xb * mx4_lut[b0 >> 4]; gb1 += xb * mx4_lut[b1 >> 4];
                            gb2 += xb * mx4_lut[b2 >> 4]; gb3 += xb * mx4_lut[b3 >> 4];
                        }
                    }
                }
                aa0 += ga0 * ca0; aa1 += ga1 * ca1;
                aa2 += ga2 * ca2; aa3 += ga3 * ca3;
                ab0 += gb0 * cb0; ab1 += gb1 * cb1;
                ab2 += gb2 * cb2; ab3 += gb3 * cb3;
            }
            y0[(int64_t)s * O + o] = aa0; y1[(int64_t)s * O + o] = ab0;
            if (o1 != o) {
                y0[(int64_t)s * O + o1] = aa1; y1[(int64_t)s * O + o1] = ab1;
            }
            if (o2 != o) {
                y0[(int64_t)s * O + o2] = aa2; y1[(int64_t)s * O + o2] = ab2;
            }
            if (o3 != o) {
                y0[(int64_t)s * O + o3] = aa3; y1[(int64_t)s * O + o3] = ab3;
            }
        }
    }
}

__attribute__((noinline))
static void fallback_matvec(float *output, const TensorView *weight,
                            const float *input) {
    float *activation = malloc((size_t)weight->columns * sizeof(*activation));
    uint8_t *activation_scales = malloc((size_t)weight->columns / 128);
    if (!activation || !activation_scales) die("fallback matvec allocation failed");
    fp8_activation_qdq(activation, activation_scales, input,
                       (size_t)weight->columns);
    matmul_mxfp4(output, activation, weight->data, weight->scales,
                 1, weight->columns, weight->rows);
    free(activation_scales);
    free(activation);
}

__attribute__((noinline))
static void fallback_dual_matvec(float *output_a, float *output_b,
                                 const TensorView *a, const TensorView *b,
                                 const float *input) {
    float *activation = malloc((size_t)a->columns * sizeof(*activation));
    uint8_t *activation_scales = malloc((size_t)a->columns / 128);
    if (!activation || !activation_scales)
        die("fallback dual matvec allocation failed");
    fp8_activation_qdq(activation, activation_scales, input, (size_t)a->columns);
    matmul_mxfp4_dual(output_a, output_b, activation,
                      a->data, a->scales, b->data, b->scales,
                      1, a->columns, a->rows);
    free(activation_scales);
    free(activation);
}

typedef struct {
    uint8x16x4_t fp4_bytes;
    uint8x16_t spread[4];
    uint8x16_t byte_offsets;
    float e8[256];
} NeonRows16Tables;

static void neon_rows16_tables(NeonRows16Tables *tables) {
    float fp4[16];
    for (int index = 0; index < 16; index++) fp4[index] = e2m1_decode((uint8_t)index);
    for (int index = 0; index < 256; index++)
        tables->e8[index] = e8m0_decode((uint8_t)index);
    const unsigned char *bytes = (const unsigned char *)fp4;
    for (int group = 0; group < 4; group++)
        tables->fp4_bytes.val[group] = vld1q_u8(bytes + 16 * group);
    for (int group = 0; group < 4; group++) {
        uint8_t lanes[16];
        for (int byte = 0; byte < 16; byte++)
            lanes[byte] = (uint8_t)(4 * group + byte / 4);
        tables->spread[group] = vld1q_u8(lanes);
    }
    uint8_t offsets[16];
    for (int byte = 0; byte < 16; byte++) offsets[byte] = (uint8_t)(byte % 4);
    tables->byte_offsets = vld1q_u8(offsets);
}

static inline void neon_rows16_block_scales(float32x4_t scales[4],
                                             const unsigned char *codes,
                                             const NeonRows16Tables *tables) {
    float decoded[16];
    for (int lane = 0; lane < 16; lane++) decoded[lane] = tables->e8[codes[lane]];
    for (int group = 0; group < 4; group++)
        scales[group] = vld1q_f32(decoded + 4 * group);
}

static inline void neon_rows16_accumulate(float32x4_t sums[4],
                                           uint8x16_t codes, float activation,
                                           const float32x4_t scales[4],
                                           const NeonRows16Tables *tables) {
    uint8x16_t byte_base = vshlq_n_u8(codes, 2);
    float32x4_t input = vdupq_n_f32(activation);
    for (int group = 0; group < 4; group++) {
        uint8x16_t gather = vaddq_u8(
            vqtbl1q_u8(byte_base, tables->spread[group]), tables->byte_offsets);
        float32x4_t values =
            vreinterpretq_f32_u8(vqtbl4q_u8(tables->fp4_bytes, gather));
        sums[group] = vaddq_f32(
            sums[group], vmulq_f32(vmulq_f32(input, values), scales[group]));
    }
}

__attribute__((noinline))
static void rows16_matvec(float *output, const TensorView *weight,
                          const float *input) {
    size_t columns = (size_t)weight->columns;
    size_t data_stride = columns / 2;
    size_t scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) die("rows16 matvec allocation failed");
    fp8_activation_qdq(activation, activation_scales, input, columns);
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    for (int64_t tile = 0; tile < weight->rows / 16; tile++) {
        float32x4_t sums[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        };
        for (size_t base = 0; base < columns; base += 32) {
            float32x4_t block_scales[4];
            neon_rows16_block_scales(
                block_scales,
                weight->scales + ((size_t)tile * scale_stride + base / 32) * 16,
                &tables);
            for (size_t offset = 0; offset < 32; offset += 2) {
                const unsigned char *codes = weight->data +
                    ((size_t)tile * data_stride + (base + offset) / 2) * 16;
                uint8x16_t bytes = vld1q_u8(codes);
                neon_rows16_accumulate(
                    sums, vandq_u8(bytes, vdupq_n_u8(15)),
                    activation[base + offset], block_scales, &tables);
                neon_rows16_accumulate(
                    sums, vshrq_n_u8(bytes, 4),
                    activation[base + offset + 1], block_scales, &tables);
            }
        }
        for (int group = 0; group < 4; group++)
            vst1q_f32(output + (size_t)tile * 16 + 4 * group, sums[group]);
    }
    free(activation_scales);
    free(activation);
}

__attribute__((noinline))
static void rows16_dual_matvec(float *output_a, float *output_b,
                               const TensorView *a, const TensorView *b,
                               const float *input) {
    size_t columns = (size_t)a->columns;
    size_t data_stride = columns / 2;
    size_t scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales)
        die("rows16 dual matvec allocation failed");
    fp8_activation_qdq(activation, activation_scales, input, columns);
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    for (int64_t tile = 0; tile < a->rows / 16; tile++) {
        float32x4_t sums_a[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        };
        float32x4_t sums_b[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        };
        for (size_t base = 0; base < columns; base += 32) {
            size_t scale_offset =
                ((size_t)tile * scale_stride + base / 32) * 16;
            float32x4_t block_scales_a[4], block_scales_b[4];
            neon_rows16_block_scales(
                block_scales_a, a->scales + scale_offset, &tables);
            neon_rows16_block_scales(
                block_scales_b, b->scales + scale_offset, &tables);
            for (size_t offset = 0; offset < 32; offset += 2) {
                size_t packed_offset =
                    ((size_t)tile * data_stride + (base + offset) / 2) * 16;
                uint8x16_t bytes_a = vld1q_u8(a->data + packed_offset);
                uint8x16_t bytes_b = vld1q_u8(b->data + packed_offset);
                neon_rows16_accumulate(
                    sums_a, vandq_u8(bytes_a, vdupq_n_u8(15)),
                    activation[base + offset], block_scales_a, &tables);
                neon_rows16_accumulate(
                    sums_b, vandq_u8(bytes_b, vdupq_n_u8(15)),
                    activation[base + offset], block_scales_b, &tables);
                neon_rows16_accumulate(
                    sums_a, vshrq_n_u8(bytes_a, 4),
                    activation[base + offset + 1], block_scales_a, &tables);
                neon_rows16_accumulate(
                    sums_b, vshrq_n_u8(bytes_b, 4),
                    activation[base + offset + 1], block_scales_b, &tables);
            }
        }
        for (int group = 0; group < 4; group++) {
            vst1q_f32(output_a + (size_t)tile * 16 + 4 * group, sums_a[group]);
            vst1q_f32(output_b + (size_t)tile * 16 + 4 * group, sums_b[group]);
        }
    }
    free(activation_scales);
    free(activation);
}

static float sigmoid_stable(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

static void swiglu(float *output, const float *gate, const float *up) {
    for (int index = 0; index < INTERMEDIATE_SIZE; index++) {
        float gate_value = fminf(gate[index], SWIGLU_LIMIT);
        float up_value = fmaxf(-SWIGLU_LIMIT, fminf(up[index], SWIGLU_LIMIT));
        output[index] = gate_value * sigmoid_stable(gate_value) * up_value;
    }
}

__attribute__((noinline))
static void fallback_expert_forward(float *output, const ExpertView *expert,
                                    const float *input) {
    float *gate = malloc(INTERMEDIATE_SIZE * sizeof(*gate));
    float *up = malloc(INTERMEDIATE_SIZE * sizeof(*up));
    float *activated = malloc(INTERMEDIATE_SIZE * sizeof(*activated));
    if (!gate || !up || !activated) die("fallback expert allocation failed");
    fallback_dual_matvec(gate, up, &expert->gate, &expert->up, input);
    bf16_round_array(gate, INTERMEDIATE_SIZE);
    bf16_round_array(up, INTERMEDIATE_SIZE);
    swiglu(activated, gate, up);
    for (size_t index = 0; index < INTERMEDIATE_SIZE; index++)
        activated[index] = bf16_round(activated[index] * ROUTE_WEIGHT);
    fallback_matvec(output, &expert->down, activated);
    bf16_round_array(output, HIDDEN_SIZE);
    free(activated);
    free(up);
    free(gate);
}

__attribute__((noinline))
static void rows16_expert_forward(float *output, const ExpertView *expert,
                                  const float *input) {
    float *gate = malloc(INTERMEDIATE_SIZE * sizeof(*gate));
    float *up = malloc(INTERMEDIATE_SIZE * sizeof(*up));
    float *activated = malloc(INTERMEDIATE_SIZE * sizeof(*activated));
    if (!gate || !up || !activated) die("rows16 expert allocation failed");
    rows16_dual_matvec(gate, up, &expert->gate, &expert->up, input);
    bf16_round_array(gate, INTERMEDIATE_SIZE);
    bf16_round_array(up, INTERMEDIATE_SIZE);
    swiglu(activated, gate, up);
    for (size_t index = 0; index < INTERMEDIATE_SIZE; index++)
        activated[index] = bf16_round(activated[index] * ROUTE_WEIGHT);
    rows16_matvec(output, &expert->down, activated);
    bf16_round_array(output, HIDDEN_SIZE);
    free(activated);
    free(up);
    free(gate);
}

static ExpertView make_expert_view(unsigned char *slab, int block_rows) {
    unsigned char *gate_scales = slab;
    unsigned char *down_scales = gate_scales + MATRIX_SCALE_BYTES;
    unsigned char *up_scales = down_scales + MATRIX_SCALE_BYTES;
    unsigned char *gate_data = up_scales + MATRIX_SCALE_BYTES;
    unsigned char *down_data = gate_data + MATRIX_DATA_BYTES;
    unsigned char *up_data = down_data + MATRIX_DATA_BYTES;
    TensorView gate = {
        gate_data, gate_scales, MATRIX_DATA_BYTES, MATRIX_SCALE_BYTES,
        INTERMEDIATE_SIZE, HIDDEN_SIZE, block_rows,
    };
    TensorView down = {
        down_data, down_scales, MATRIX_DATA_BYTES, MATRIX_SCALE_BYTES,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, block_rows,
    };
    TensorView up = {
        up_data, up_scales, MATRIX_DATA_BYTES, MATRIX_SCALE_BYTES,
        INTERMEDIATE_SIZE, HIDDEN_SIZE, block_rows,
    };
    return (ExpertView){gate, down, up};
}

static void fill_expert(ExpertView *expert) {
    TensorView *matrices[3] = {&expert->gate, &expert->down, &expert->up};
    for (int matrix = 0; matrix < 3; matrix++) {
        unsigned char *data = (unsigned char *)matrices[matrix]->data;
        unsigned char *scales = (unsigned char *)matrices[matrix]->scales;
        for (size_t index = 0; index < matrices[matrix]->data_bytes; index++)
            data[index] = (uint8_t)random_u32();
        for (size_t index = 0; index < matrices[matrix]->scale_bytes; index++)
            scales[index] = (uint8_t)(120 + random_u32() % 15);
    }
}

static void pack_rows16(unsigned char *packed_data,
                        unsigned char *packed_scales,
                        const TensorView *source) {
    size_t rows = (size_t)source->rows;
    size_t data_stride = (size_t)source->columns / 2;
    size_t scale_stride = (size_t)source->columns / 32;
    for (size_t row = 0; row < rows; row++) {
        size_t tile = row / 16;
        size_t lane = row % 16;
        for (size_t column = 0; column < data_stride; column++)
            packed_data[(tile * data_stride + column) * 16 + lane] =
                source->data[row * data_stride + column];
        for (size_t column = 0; column < scale_stride; column++)
            packed_scales[(tile * scale_stride + column) * 16 + lane] =
                source->scales[row * scale_stride + column];
    }
}

static void pack_matrix_in_place(TensorView *matrix, unsigned char *scratch) {
    unsigned char *packed_scales = scratch + matrix->data_bytes;
    pack_rows16(scratch, packed_scales, matrix);
    memcpy((void *)matrix->data, scratch, matrix->data_bytes);
    memcpy((void *)matrix->scales, packed_scales, matrix->scale_bytes);
}

static void pack_expert_in_place(ExpertView *expert) {
    const TensorView *matrices[3] = {&expert->gate, &expert->down, &expert->up};
    size_t scratch_size = 0;
    for (int matrix = 0; matrix < 3; matrix++) {
        size_t needed = matrices[matrix]->data_bytes + matrices[matrix]->scale_bytes;
        if (needed > scratch_size) scratch_size = needed;
    }
    unsigned char *scratch = malloc(scratch_size);
    if (!scratch) die("pack scratch allocation failed");
    pack_matrix_in_place(&expert->gate, scratch);
    pack_matrix_in_place(&expert->down, scratch);
    pack_matrix_in_place(&expert->up, scratch);
    free(scratch);
}

static Measurement summarize(const double *samples_ms, int count) {
    double sum = 0.0;
    for (int index = 0; index < count; index++) sum += samples_ms[index];
    double mean = sum / count;
    double squared_deviation = 0.0;
    for (int index = 0; index < count; index++) {
        double deviation = samples_ms[index] - mean;
        squared_deviation += deviation * deviation;
    }
    return (Measurement){mean, sqrt(squared_deviation / count)};
}

static void measure_kernels(Measurement *fallback, Measurement *rows16,
                            int iterations, const ExpertView *flat,
                            const ExpertView *packed, const float *input,
                            float *fallback_output, float *rows16_output) {
    for (int warmup = 0; warmup < WARMUP_ITERATIONS; warmup++) {
        fallback_expert_forward(fallback_output, flat, input);
        rows16_expert_forward(rows16_output, packed, input);
    }
    double *fallback_samples = malloc((size_t)iterations * sizeof(*fallback_samples));
    double *rows16_samples = malloc((size_t)iterations * sizeof(*rows16_samples));
    if (!fallback_samples || !rows16_samples) die("sample allocation failed");
    for (int iteration = 0; iteration < iterations; iteration++) {
        if ((iteration & 1) == 0) {
            double started = now_seconds();
            fallback_expert_forward(fallback_output, flat, input);
            fallback_samples[iteration] = (now_seconds() - started) * 1e3;
            started = now_seconds();
            rows16_expert_forward(rows16_output, packed, input);
            rows16_samples[iteration] = (now_seconds() - started) * 1e3;
        } else {
            double started = now_seconds();
            rows16_expert_forward(rows16_output, packed, input);
            rows16_samples[iteration] = (now_seconds() - started) * 1e3;
            started = now_seconds();
            fallback_expert_forward(fallback_output, flat, input);
            fallback_samples[iteration] = (now_seconds() - started) * 1e3;
        }
        output_guard += fallback_output[iteration % HIDDEN_SIZE];
        output_guard += rows16_output[iteration % HIDDEN_SIZE];
    }
    *fallback = summarize(fallback_samples, iterations);
    *rows16 = summarize(rows16_samples, iterations);
    free(rows16_samples);
    free(fallback_samples);
}

static Measurement measure_pack(int iterations, const unsigned char *source_slab,
                                unsigned char *work_slab) {
    double *samples = malloc((size_t)iterations * sizeof(*samples));
    if (!samples) die("pack sample allocation failed");
    for (int iteration = -WARMUP_ITERATIONS; iteration < iterations; iteration++) {
        memcpy(work_slab, source_slab, RECORD_BYTES);
        ExpertView work = make_expert_view(work_slab, 1);
        double started = now_seconds();
        pack_expert_in_place(&work);
        double elapsed_ms = (now_seconds() - started) * 1e3;
        output_guard += work.gate.data[iteration < 0 ? 0 : (size_t)iteration];
        if (iteration >= 0) samples[iteration] = elapsed_ms;
    }
    Measurement result = summarize(samples, iterations);
    free(samples);
    return result;
}

static int parse_iterations(int argc, char **argv) {
    if (argc > 2) die("usage: ./bench/kernel_gap [iterations>=20]");
    if (argc == 1) return DEFAULT_ITERATIONS;
    char *end = NULL;
    errno = 0;
    long value = strtol(argv[1], &end, 10);
    if (errno || !end || *end || value < 20 || value > 100000)
        die("iterations must be an integer from 20 to 100000");
    return (int)value;
}

int main(int argc, char **argv) {
    int iterations = parse_iterations(argc, argv);
    if (3 * MATRIX_BYTES != RECORD_BYTES) die("record byte model mismatch");

    unsigned char *source_slab = aligned_alloc_or_die(ALIGNMENT, RECORD_BYTES);
    unsigned char *packed_slab = aligned_alloc_or_die(ALIGNMENT, RECORD_BYTES);
    unsigned char *work_slab = aligned_alloc_or_die(ALIGNMENT, RECORD_BYTES);
    float *input = aligned_alloc_or_die(64, HIDDEN_SIZE * sizeof(*input));
    float *fallback_output = aligned_alloc_or_die(64, HIDDEN_SIZE * sizeof(*fallback_output));
    float *rows16_output = aligned_alloc_or_die(64, HIDDEN_SIZE * sizeof(*rows16_output));
    ExpertView flat = make_expert_view(source_slab, 1);
    fill_expert(&flat);
    for (int index = 0; index < HIDDEN_SIZE; index++)
        input[index] = ((float)(random_u32() % 2001) - 1000.0f) / 337.0f;
    memcpy(packed_slab, source_slab, RECORD_BYTES);
    ExpertView packed_source = make_expert_view(packed_slab, 1);
    pack_expert_in_place(&packed_source);
    ExpertView packed = make_expert_view(packed_slab, 16);

    fallback_expert_forward(fallback_output, &flat, input);
    rows16_expert_forward(rows16_output, &packed, input);
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t different = 0;
    for (int index = 0; index < HIDDEN_SIZE; index++) {
        double absolute = fabs((double)fallback_output[index] - rows16_output[index]);
        double denominator = fmax(
            fmax(fabs((double)fallback_output[index]), fabs((double)rows16_output[index])),
            1e-30);
        if (absolute > max_abs) max_abs = absolute;
        if (absolute / denominator > max_rel) max_rel = absolute / denominator;
        uint32_t fallback_bits, rows16_bits;
        memcpy(&fallback_bits, &fallback_output[index], sizeof(fallback_bits));
        memcpy(&rows16_bits, &rows16_output[index], sizeof(rows16_bits));
        if (fallback_bits != rows16_bits) different++;
    }

    Measurement fallback_measurement, rows16_measurement;
    measure_kernels(&fallback_measurement, &rows16_measurement, iterations,
                    &flat, &packed, input, fallback_output, rows16_output);
    Measurement pack_measurement = measure_pack(iterations, source_slab, work_slab);

    double ratio = fallback_measurement.mean_ms / rows16_measurement.mean_ms;
    double fallback_gflops =
        2.0 * MACS_PER_EXPERT / (fallback_measurement.mean_ms * 1e6);
    double rows16_gflops =
        2.0 * MACS_PER_EXPERT / (rows16_measurement.mean_ms * 1e6);
    double traffic_bytes = 4.0 * (double)RECORD_BYTES;
    double pack_gbps = traffic_bytes / (pack_measurement.mean_ms * 1e6);
    double saving_ms = fallback_measurement.mean_ms - rows16_measurement.mean_ms;
    double breakeven = saving_ms > 0.0
        ? pack_measurement.mean_ms / saving_ms : INFINITY;
    bool kernel_passes = ratio >= 1.5;
    bool packing_pays = breakeven <= 4.4;

    printf("KERNEL_GAP method=replicated_arm64_engine_code "
           "iterations=%d warmups=%d record_bytes=%zu route_weight=%.2f "
           "swiglu_limit=%.1f macs_per_expert=%.0f\n",
           iterations, WARMUP_ITERATIONS, RECORD_BYTES,
           ROUTE_WEIGHT, SWIGLU_LIMIT, MACS_PER_EXPERT);
    puts("S1 path       mean_ms      stddev_ms    GFLOP/s");
    printf("S1 fallback   %.6f     %.6f     %.6f\n",
           fallback_measurement.mean_ms, fallback_measurement.stddev_ms,
           fallback_gflops);
    printf("S1 rows16     %.6f     %.6f     %.6f\n",
           rows16_measurement.mean_ms, rows16_measurement.stddev_ms,
           rows16_gflops);
    printf("S1 ratio=%.6f\n", ratio);
    printf("S1B max_abs=%.9g max_rel=%.9g bit_identical=%s different=%zu total=%d\n",
           max_abs, max_rel, different == 0 ? "true" : "false",
           different, HIDDEN_SIZE);
    printf("S2 pack_mean_ms=%.6f pack_stddev_ms=%.6f effective_GBps=%.6f "
           "traffic_bytes=%.0f N_breakeven=%.6f uses_per_chunk=4.4\n",
           pack_measurement.mean_ms, pack_measurement.stddev_ms,
           pack_gbps, traffic_bytes, breakeven);
    printf("VERDICT kernel_ratio_ge_1_5=%s packing_pays_at_4_4_uses=%s\n",
           kernel_passes ? "true" : "false", packing_pays ? "true" : "false");
    printf("GUARD %.9g\n", output_guard);

    free(rows16_output);
    free(fallback_output);
    free(input);
    free(work_slab);
    free(packed_slab);
    free(source_slab);
    return EXIT_SUCCESS;
}
