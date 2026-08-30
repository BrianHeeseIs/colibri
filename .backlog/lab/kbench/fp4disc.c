/* fp4disc -- distinguish MXFP4 accumulator recurrence from TBL pressure.
 *
 * This is a standalone matrix-vector microbenchmark for both real DeepSeek-V4
 * expert shapes. It copies matmul_mxfp4's scalar semantics and the aarch64
 * rows16 decoder from deepseek_v4.c. Synthetic input is already the activation
 * consumed by those kernels; activation QDQ is intentionally outside scope.
 *
 * EXPERIMENTAL INVARIANT -- DO NOT WEAKEN:
 * Within each family, every sibling is generated from the same source and
 * differs ONLY in CHAIN_COUNT. Scalar siblings use the identical per-element
 * MX4_LUT expression. Rows16 siblings call the identical accumulate helper,
 * preserving every vqtbl1q_u8/vqtbl4q_u8 operation per tile-element. Changing
 * decode, layout, scale placement, arithmetic, or memory traffic would make
 * this benchmark answer a different question.
 *
 * The CEILING arm is deliberately exempt: it hoists block scales, uses FMA,
 * permits reassociation, and exists only as an upper bound.
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(__aarch64__)
#error "fp4disc requires aarch64 NEON"
#endif

#define FP4_GROUP 32
#define ROWS16 16

static const float MX4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float mx4_scale(uint8_t scale) {
    union { uint32_t bits; float value; } decoded;
    decoded.bits = (uint32_t)scale << 23;
    return decoded.value;
}

static double now(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return stamp.tv_sec + 1e-9 * stamp.tv_nsec;
}

/* Exact S=1 transcription of c/quant.h:1440. */
__attribute__((noinline))
static void cold_reference(float *output, const float *input,
                           const uint8_t *data, const uint8_t *scales,
                           int columns, int rows) {
    int row_bytes = (columns + 1) / 2;
    int groups = (columns + 31) / 32;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row += 4) {
        int row1 = row + 1 < rows ? row + 1 : row;
        int row2 = row + 2 < rows ? row + 2 : row;
        int row3 = row + 3 < rows ? row + 3 : row;
        const uint8_t *weight0 = data + (int64_t)row * row_bytes;
        const uint8_t *weight1 = data + (int64_t)row1 * row_bytes;
        const uint8_t *weight2 = data + (int64_t)row2 * row_bytes;
        const uint8_t *weight3 = data + (int64_t)row3 * row_bytes;
        const uint8_t *scale0 = scales + (int64_t)row * groups;
        const uint8_t *scale1 = scales + (int64_t)row1 * groups;
        const uint8_t *scale2 = scales + (int64_t)row2 * groups;
        const uint8_t *scale3 = scales + (int64_t)row3 * groups;
        float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
        for (int group = 0; group < groups; group++) {
            int base = group * FP4_GROUP;
            float block0 = 0.0f, block1 = 0.0f, block2 = 0.0f, block3 = 0.0f;
            const uint8_t *packed0 = weight0 + (base >> 1);
            const uint8_t *packed1 = weight1 + (base >> 1);
            const uint8_t *packed2 = weight2 + (base >> 1);
            const uint8_t *packed3 = weight3 + (base >> 1);
            for (int packed_column = 0; packed_column < 16; packed_column++) {
                float even = input[base + 2 * packed_column];
                float odd = input[base + 2 * packed_column + 1];
                uint8_t byte0 = packed0[packed_column];
                uint8_t byte1 = packed1[packed_column];
                uint8_t byte2 = packed2[packed_column];
                uint8_t byte3 = packed3[packed_column];
                block0 += even * MX4_LUT[byte0 & 0x0f];
                block0 += odd * MX4_LUT[byte0 >> 4];
                block1 += even * MX4_LUT[byte1 & 0x0f];
                block1 += odd * MX4_LUT[byte1 >> 4];
                block2 += even * MX4_LUT[byte2 & 0x0f];
                block2 += odd * MX4_LUT[byte2 >> 4];
                block3 += even * MX4_LUT[byte3 & 0x0f];
                block3 += odd * MX4_LUT[byte3 >> 4];
            }
            total0 += block0 * mx4_scale(scale0[group]);
            total1 += block1 * mx4_scale(scale1[group]);
            total2 += block2 * mx4_scale(scale2[group]);
            total3 += block3 * mx4_scale(scale3[group]);
        }
        output[row] = total0;
        if (row1 != row) output[row1] = total1;
        if (row2 != row) output[row2] = total2;
        if (row3 != row) output[row3] = total3;
    }
}

#define DEFINE_COLD_CHAINS(NAME, CHAIN_COUNT)                                      \
__attribute__((noinline))                                                         \
static void NAME(float *output, const float *input, const uint8_t *data,           \
                 const uint8_t *scales, int columns, int rows) {                   \
    const int row_bytes = (columns + 1) / 2;                                      \
    const int groups = (columns + 31) / 32;                                       \
    _Pragma("omp parallel for schedule(static)")                                  \
    for (int row_base = 0; row_base < rows; row_base += CHAIN_COUNT) {             \
        const uint8_t *weight_rows[CHAIN_COUNT];                                  \
        const uint8_t *scale_rows[CHAIN_COUNT];                                   \
        float totals[CHAIN_COUNT] = {0};                                           \
        for (int chain = 0; chain < CHAIN_COUNT; chain++) {                        \
            const int row = row_base + chain < rows ? row_base + chain : row_base;\
            weight_rows[chain] = data + (int64_t)row * row_bytes;                 \
            scale_rows[chain] = scales + (int64_t)row * groups;                   \
        }                                                                          \
        for (int group = 0; group < groups; group++) {                             \
            const int base = group * FP4_GROUP;                                   \
            const uint8_t *packed_rows[CHAIN_COUNT];                              \
            float block_scales[CHAIN_COUNT];                                      \
            float block_sums[CHAIN_COUNT] = {0};                                  \
            for (int chain = 0; chain < CHAIN_COUNT; chain++) {                   \
                packed_rows[chain] = weight_rows[chain] + (base >> 1);            \
                block_scales[chain] = mx4_scale(scale_rows[chain][group]);         \
            }                                                                      \
            for (int packed_column = 0; packed_column < 16; packed_column++) {     \
                const float even = input[base + 2 * packed_column];                \
                const float odd = input[base + 2 * packed_column + 1];             \
                for (int chain = 0; chain < CHAIN_COUNT; chain++) {                \
                    const uint8_t packed = packed_rows[chain][packed_column];      \
                    block_sums[chain] += even * MX4_LUT[packed & 0x0f];            \
                    block_sums[chain] += odd * MX4_LUT[packed >> 4];               \
                }                                                                  \
            }                                                                      \
            for (int chain = 0; chain < CHAIN_COUNT; chain++)                     \
                totals[chain] += block_sums[chain] * block_scales[chain];          \
        }                                                                          \
        for (int chain = 0; chain < CHAIN_COUNT; chain++)                          \
            if (row_base + chain < rows) output[row_base + chain] = totals[chain];\
    }                                                                              \
}

DEFINE_COLD_CHAINS(cold_chains4, 4)
DEFINE_COLD_CHAINS(cold_chains8, 8)
DEFINE_COLD_CHAINS(cold_chains12, 12)
DEFINE_COLD_CHAINS(cold_chains16, 16)

typedef struct {
    uint8x16x4_t fp4_bytes;
    uint8x16_t spread[4];
    uint8x16_t byte_offsets;
    float e8[256];
} NeonRows16Tables;

static void neon_rows16_tables(NeonRows16Tables *tables) {
    const unsigned char *bytes = (const unsigned char *)MX4_LUT;
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
    for (int code = 0; code < 256; code++) tables->e8[code] = mx4_scale((uint8_t)code);
}

static inline void neon_rows16_block_scales(float32x4_t decoded_scales[4],
                                             const uint8_t *codes,
                                             const NeonRows16Tables *tables) {
    float decoded[16];
    for (int lane = 0; lane < 16; lane++) decoded[lane] = tables->e8[codes[lane]];
    for (int group = 0; group < 4; group++)
        decoded_scales[group] = vld1q_f32(decoded + 4 * group);
}

/* Keep byte-for-byte aligned with c/deepseek_v4.c:14600. */
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
static void rows16_reference(float *output, const float *input,
                             const uint8_t *data, const uint8_t *scales,
                             int columns, int rows) {
    size_t data_stride = (size_t)columns / 2;
    size_t scale_stride = (size_t)columns / 32;
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < rows / ROWS16; tile++) {
        float32x4_t sums[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        };
        for (size_t base = 0; base < (size_t)columns; base += FP4_GROUP) {
            float32x4_t block_scales[4];
            neon_rows16_block_scales(
                block_scales,
                scales + ((size_t)tile * scale_stride + base / FP4_GROUP) * ROWS16,
                &tables);
            for (size_t offset = 0; offset < FP4_GROUP; offset += 2) {
                const uint8_t *codes = data +
                    ((size_t)tile * data_stride + (base + offset) / 2) * ROWS16;
                uint8x16_t packed = vld1q_u8(codes);
                neon_rows16_accumulate(
                    sums, vandq_u8(packed, vdupq_n_u8(15)),
                    input[base + offset], block_scales, &tables);
                neon_rows16_accumulate(
                    sums, vshrq_n_u8(packed, 4),
                    input[base + offset + 1], block_scales, &tables);
            }
        }
        for (int group = 0; group < 4; group++)
            vst1q_f32(output + (size_t)tile * ROWS16 + 4 * group, sums[group]);
    }
}

#define DEFINE_ROWS16_CHAINS(NAME, CHAIN_COUNT)                                    \
__attribute__((noinline))                                                         \
static void NAME(float *output, const float *input, const uint8_t *data,           \
                 const uint8_t *scales, int columns, int rows) {                   \
    const size_t data_stride = (size_t)columns / 2;                               \
    const size_t scale_stride = (size_t)columns / FP4_GROUP;                      \
    const int tiles_in_flight = CHAIN_COUNT / 4;                                  \
    NeonRows16Tables tables;                                                       \
    neon_rows16_tables(&tables);                                                   \
    _Pragma("omp parallel for schedule(static)")                                  \
    for (int tile_base = 0; tile_base < rows / ROWS16;                            \
         tile_base += tiles_in_flight) {                                           \
        float32x4_t sums[CHAIN_COUNT];                                             \
        for (int chain = 0; chain < CHAIN_COUNT; chain++)                         \
            sums[chain] = vdupq_n_f32(0.0f);                                      \
        for (size_t base = 0; base < (size_t)columns; base += FP4_GROUP) {         \
            float32x4_t block_scales[CHAIN_COUNT];                                \
            for (int tile_offset = 0; tile_offset < tiles_in_flight;              \
                 tile_offset++)                                                    \
                neon_rows16_block_scales(                                          \
                    block_scales + 4 * tile_offset,                               \
                    scales + ((size_t)(tile_base + tile_offset) * scale_stride +  \
                              base / FP4_GROUP) * ROWS16,                          \
                    &tables);                                                      \
            for (size_t offset = 0; offset < FP4_GROUP; offset += 2) {            \
                for (int tile_offset = 0; tile_offset < tiles_in_flight;          \
                     tile_offset++) {                                              \
                    const uint8_t *codes = data +                                 \
                        ((size_t)(tile_base + tile_offset) * data_stride +         \
                         (base + offset) / 2) * ROWS16;                            \
                    uint8x16_t packed = vld1q_u8(codes);                           \
                    neon_rows16_accumulate(                                       \
                        sums + 4 * tile_offset,                                    \
                        vandq_u8(packed, vdupq_n_u8(15)),                          \
                        input[base + offset],                                      \
                        block_scales + 4 * tile_offset, &tables);                  \
                    neon_rows16_accumulate(                                       \
                        sums + 4 * tile_offset, vshrq_n_u8(packed, 4),             \
                        input[base + offset + 1],                                  \
                        block_scales + 4 * tile_offset, &tables);                  \
                }                                                                  \
            }                                                                      \
        }                                                                          \
        for (int tile_offset = 0; tile_offset < tiles_in_flight; tile_offset++)    \
            for (int group = 0; group < 4; group++)                               \
                vst1q_f32(output + (size_t)(tile_base + tile_offset) * ROWS16 +   \
                                4 * group,                                        \
                           sums[4 * tile_offset + group]);                         \
    }                                                                              \
}

DEFINE_ROWS16_CHAINS(rows16_chains4, 4)
DEFINE_ROWS16_CHAINS(rows16_chains8, 8)
DEFINE_ROWS16_CHAINS(rows16_chains16, 16)

static inline void neon_rows16_accumulate_ceiling(
    float32x4_t sums[4], uint8x16_t codes, float activation,
    const NeonRows16Tables *tables) {
    uint8x16_t byte_base = vshlq_n_u8(codes, 2);
    float32x4_t input = vdupq_n_f32(activation);
    for (int group = 0; group < 4; group++) {
        uint8x16_t gather = vaddq_u8(
            vqtbl1q_u8(byte_base, tables->spread[group]), tables->byte_offsets);
        float32x4_t values =
            vreinterpretq_f32_u8(vqtbl4q_u8(tables->fp4_bytes, gather));
        sums[group] = vfmaq_f32(sums[group], input, values);
    }
}

__attribute__((noinline))
static void rows16_ceiling(float *output, const float *input,
                           const uint8_t *data, const uint8_t *scales,
                           int columns, int rows) {
    const size_t data_stride = (size_t)columns / 2;
    const size_t scale_stride = (size_t)columns / FP4_GROUP;
    const int tiles_in_flight = 4;
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    #pragma omp parallel for schedule(static)
    for (int tile_base = 0; tile_base < rows / ROWS16;
         tile_base += tiles_in_flight) {
        float32x4_t sums[16];
        for (int chain = 0; chain < 16; chain++) sums[chain] = vdupq_n_f32(0.0f);
        for (size_t base = 0; base < (size_t)columns; base += FP4_GROUP) {
            float32x4_t block_scales[16];
            float32x4_t block_sums[16];
            for (int chain = 0; chain < 16; chain++)
                block_sums[chain] = vdupq_n_f32(0.0f);
            for (int tile_offset = 0; tile_offset < tiles_in_flight; tile_offset++)
                neon_rows16_block_scales(
                    block_scales + 4 * tile_offset,
                    scales + ((size_t)(tile_base + tile_offset) * scale_stride +
                              base / FP4_GROUP) * ROWS16,
                    &tables);
            for (size_t offset = 0; offset < FP4_GROUP; offset += 2) {
                for (int tile_offset = 0; tile_offset < tiles_in_flight;
                     tile_offset++) {
                    const uint8_t *codes = data +
                        ((size_t)(tile_base + tile_offset) * data_stride +
                         (base + offset) / 2) * ROWS16;
                    uint8x16_t packed = vld1q_u8(codes);
                    neon_rows16_accumulate_ceiling(
                        block_sums + 4 * tile_offset,
                        vandq_u8(packed, vdupq_n_u8(15)), input[base + offset],
                        &tables);
                    neon_rows16_accumulate_ceiling(
                        block_sums + 4 * tile_offset, vshrq_n_u8(packed, 4),
                        input[base + offset + 1], &tables);
                }
            }
            for (int chain = 0; chain < 16; chain++)
                sums[chain] = vfmaq_f32(
                    sums[chain], block_sums[chain], block_scales[chain]);
        }
        for (int tile_offset = 0; tile_offset < tiles_in_flight; tile_offset++)
            for (int group = 0; group < 4; group++)
                vst1q_f32(output + (size_t)(tile_base + tile_offset) * ROWS16 +
                                  4 * group,
                           sums[4 * tile_offset + group]);
    }
}

static void pack_rows16(uint8_t *packed_data, uint8_t *packed_scales,
                        const uint8_t *data, const uint8_t *scales,
                        int columns, int rows) {
    size_t data_stride = (size_t)columns / 2;
    size_t scale_stride = (size_t)columns / FP4_GROUP;
    for (int row = 0; row < rows; row++) {
        size_t tile = (size_t)row / ROWS16;
        size_t lane = (size_t)row % ROWS16;
        for (size_t column = 0; column < data_stride; column++)
            packed_data[(tile * data_stride + column) * ROWS16 + lane] =
                data[(size_t)row * data_stride + column];
        for (size_t group = 0; group < scale_stride; group++)
            packed_scales[(tile * scale_stride + group) * ROWS16 + lane] =
                scales[(size_t)row * scale_stride + group];
    }
}

typedef void (*Kernel)(float *, const float *, const uint8_t *, const uint8_t *,
                       int, int);

typedef struct {
    double minimum;
    double median;
    double maximum;
} Timing;

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return a < b ? -1 : a > b;
}

static Timing benchmark(Kernel kernel, float *output, const float *input,
                        const uint8_t *data, const uint8_t *scales,
                        int columns, int rows, int repetitions) {
    double *samples = malloc((size_t)repetitions * sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "timing allocation failed\n");
        exit(1);
    }
    kernel(output, input, data, scales, columns, rows);
    for (int repetition = 0; repetition < repetitions; repetition++) {
        double start = now();
        kernel(output, input, data, scales, columns, rows);
        samples[repetition] = now() - start;
    }
    qsort(samples, (size_t)repetitions, sizeof(*samples), compare_double);
    Timing timing = {
        .minimum = samples[0],
        .median = samples[repetitions / 2],
        .maximum = samples[repetitions - 1],
    };
    free(samples);
    return timing;
}

static int bitexact(const float *reference, const float *candidate, int rows) {
    return memcmp(reference, candidate, (size_t)rows * sizeof(*reference)) == 0;
}

static void print_result(const char *family, const char *arm, int chains,
                         Timing timing, double bytes, double four_chain_median,
                         const char *check) {
    double spread = 100.0 * (timing.maximum - timing.minimum) / timing.median;
    printf("%-8s %-25s %6d %10.3f %10.3f %9.2f %9.2f %17.3f  %s\n",
           family, arm, chains, timing.median * 1e3, timing.minimum * 1e3,
           spread, bytes / timing.median / 1e9,
           four_chain_median / timing.median, check);
}

static int run_shape(const char *name, int columns, int rows, int repetitions) {
    size_t data_bytes = (size_t)rows * (size_t)columns / 2;
    size_t scale_bytes = (size_t)rows * (size_t)columns / FP4_GROUP;
    uint8_t *data = malloc(data_bytes);
    uint8_t *scales = malloc(scale_bytes);
    uint8_t *packed_data = malloc(data_bytes);
    uint8_t *packed_scales = malloc(scale_bytes);
    float *input = malloc((size_t)columns * sizeof(*input));
    float *reference = malloc((size_t)rows * sizeof(*reference));
    float *candidate = malloc((size_t)rows * sizeof(*candidate));
    if (!data || !scales || !packed_data || !packed_scales || !input ||
        !reference || !candidate) {
        fprintf(stderr, "%s: allocation failed\n", name);
        free(candidate); free(reference); free(input); free(packed_scales);
        free(packed_data); free(scales); free(data);
        return 1;
    }

    uint32_t state = 2246822519u;
    for (size_t index = 0; index < data_bytes; index++) {
        state = state * 1103515245u + 12345u;
        data[index] = (uint8_t)(state >> 16);
    }
    for (size_t index = 0; index < scale_bytes; index++) {
        state = state * 1103515245u + 12345u;
        scales[index] = (uint8_t)(120 + (state >> 24) % 16);
    }
    for (int column = 0; column < columns; column++) {
        state = state * 1103515245u + 12345u;
        input[column] = ((float)((int)(state >> 20) - 2048)) / 2048.0f;
    }
    pack_rows16(packed_data, packed_scales, data, scales, columns, rows);

    Timing cold_ref = benchmark(cold_reference, reference, input, data, scales,
                                columns, rows, repetitions);
    Timing cold4 = benchmark(cold_chains4, candidate, input, data, scales,
                             columns, rows, repetitions);
    int cold4_exact = bitexact(reference, candidate, rows);
    Timing cold8 = benchmark(cold_chains8, candidate, input, data, scales,
                             columns, rows, repetitions);
    int cold8_exact = bitexact(reference, candidate, rows);
    Timing cold12 = benchmark(cold_chains12, candidate, input, data, scales,
                              columns, rows, repetitions);
    int cold12_exact = bitexact(reference, candidate, rows);
    Timing cold16 = benchmark(cold_chains16, candidate, input, data, scales,
                              columns, rows, repetitions);
    int cold16_exact = bitexact(reference, candidate, rows);

    Timing rows_ref = benchmark(rows16_reference, reference, input, packed_data,
                                packed_scales, columns, rows, repetitions);
    Timing rows4 = benchmark(rows16_chains4, candidate, input, packed_data,
                             packed_scales, columns, rows, repetitions);
    int rows4_exact = bitexact(reference, candidate, rows);
    Timing rows8 = benchmark(rows16_chains8, candidate, input, packed_data,
                             packed_scales, columns, rows, repetitions);
    int rows8_exact = bitexact(reference, candidate, rows);
    Timing rows16 = benchmark(rows16_chains16, candidate, input, packed_data,
                              packed_scales, columns, rows, repetitions);
    int rows16_exact = bitexact(reference, candidate, rows);
    Timing ceiling = benchmark(rows16_ceiling, candidate, input, packed_data,
                               packed_scales, columns, rows, repetitions);

    printf("\nshape %-8s I=%d O=%d packed_weights=%.3f MB\n",
           name, columns, rows, (double)data_bytes / 1e6);
    printf("family   arm                       chains  median_ms     min_ms  spread%%"
           "      GB/s  ratio_vs_4chains  check\n");
    print_result("COLD", "reference-current", 4, cold_ref, (double)data_bytes,
                 cold4.median, "reference");
    print_result("COLD", "SELF-CHECK-param", 4, cold4, (double)data_bytes,
                 cold4.median, cold4_exact ? "bit-exact" : "MISMATCH");
    print_result("COLD", "parameterized", 8, cold8, (double)data_bytes,
                 cold4.median, cold8_exact ? "bit-exact" : "MISMATCH");
    print_result("COLD", "parameterized", 12, cold12, (double)data_bytes,
                 cold4.median, cold12_exact ? "bit-exact" : "MISMATCH");
    print_result("COLD", "parameterized", 16, cold16, (double)data_bytes,
                 cold4.median, cold16_exact ? "bit-exact" : "MISMATCH");
    print_result("ROWS16", "reference-current", 4, rows_ref, (double)data_bytes,
                 rows4.median, "reference");
    print_result("ROWS16", "SELF-CHECK-param", 4, rows4, (double)data_bytes,
                 rows4.median, rows4_exact ? "bit-exact" : "MISMATCH");
    print_result("ROWS16", "parameterized", 8, rows8, (double)data_bytes,
                 rows4.median, rows8_exact ? "bit-exact" : "MISMATCH");
    print_result("ROWS16", "parameterized", 16, rows16, (double)data_bytes,
                 rows4.median, rows16_exact ? "bit-exact" : "MISMATCH");
    print_result("CEILING", "scale-hoist+FMA", 16, ceiling, (double)data_bytes,
                 rows4.median, "upper-bound");

    double cold_delta = fabs(cold4.median / cold_ref.median - 1.0);
    double rows_delta = fabs(rows4.median / rows_ref.median - 1.0);
    int semantics_ok = cold4_exact && cold8_exact && cold12_exact && cold16_exact &&
                       rows4_exact && rows8_exact && rows16_exact;
    int harness_void = cold_delta > 0.02 || rows_delta > 0.02 || !semantics_ok;
    if (harness_void) {
        printf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("HARNESS VOID: COLD self-check delta=%.2f%%, ROWS16 delta=%.2f%%, "
               "semantics=%s\n",
               100.0 * cold_delta, 100.0 * rows_delta,
               semantics_ok ? "exact" : "MISMATCH");
        printf("DO NOT INTERPRET THESE TIMINGS\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    free(candidate); free(reference); free(input); free(packed_scales);
    free(packed_data); free(scales); free(data);
    return harness_void ? 3 : 0;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s [reps] | %s I O reps\n", program, program);
}

int main(int argc, char **argv) {
    int repetitions = 15;
    int custom_columns = 0;
    int custom_rows = 0;
    if (argc == 2) {
        repetitions = atoi(argv[1]);
    } else if (argc == 4) {
        custom_columns = atoi(argv[1]);
        custom_rows = atoi(argv[2]);
        repetitions = atoi(argv[3]);
    } else if (argc != 1) {
        usage(argv[0]);
        return 2;
    }
    if (repetitions < 9) {
        fprintf(stderr, "reps must be >= 9\n");
        return 2;
    }
    if (custom_columns &&
        (custom_columns % 128 || custom_rows % 64 || custom_rows < 64)) {
        fprintf(stderr, "I must be a multiple of 128 and O a multiple of 64\n");
        return 2;
    }

    printf("MXFP4 recurrence discriminator, median of %d, threads=%d\n",
           repetitions,
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
    );
    printf("spread=(max-min)/median; GB/s counts packed weight bytes only\n");

    if (custom_columns)
        return run_shape("custom", custom_columns, custom_rows, repetitions);

    int status = 0;
    status |= run_shape("gate/up", 4096, 2048, repetitions);
    status |= run_shape("down", 2048, 4096, repetitions);
    return status;
}
