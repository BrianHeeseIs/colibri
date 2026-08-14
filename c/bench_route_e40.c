#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { EXPERTS = 256, DIMENSION = 4096, TOPK = 6, ROUNDS = 7 };

typedef int (*route_fn)(float *, int *, const float *, const uint16_t *,
                        const float *, int, int, int);

static volatile float benchmark_sink;

static float bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static uint16_t bf16_encode(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float softplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

static float dot_current(const uint16_t *row, const float *hidden, int count) {
    float sum = 0.0f;
    for (int column = 0; column < count; column++)
        sum += bf16_decode(row[column]) * hidden[column];
    return sum;
}

static float dot_ordered(const uint16_t *row, const float *hidden, int count) {
    float sum = 0.0f;
    int column = 0;
    for (; column + 7 < count; column += 8) {
        uint16x8_t packed = vld1q_u16(row + column);
        float32x4_t low = vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(packed), 16));
        float32x4_t high = vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(packed), 16));
        float32x4_t low_products = vmulq_f32(
            low, vld1q_f32(hidden + column));
        float32x4_t high_products = vmulq_f32(
            high, vld1q_f32(hidden + column + 4));
        sum += vgetq_lane_f32(low_products, 0);
        sum += vgetq_lane_f32(low_products, 1);
        sum += vgetq_lane_f32(low_products, 2);
        sum += vgetq_lane_f32(low_products, 3);
        sum += vgetq_lane_f32(high_products, 0);
        sum += vgetq_lane_f32(high_products, 1);
        sum += vgetq_lane_f32(high_products, 2);
        sum += vgetq_lane_f32(high_products, 3);
    }
    for (; column < count; column++)
        sum += bf16_decode(row[column]) * hidden[column];
    return sum;
}

static float dot_fast(const uint16_t *row, const float *hidden, int count) {
    float32x4_t sums[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
    };
    int column = 0;
    for (; column + 15 < count; column += 16) {
        uint16x8_t packed0 = vld1q_u16(row + column);
        uint16x8_t packed1 = vld1q_u16(row + column + 8);
        float32x4_t values0 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(packed0), 16));
        float32x4_t values1 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(packed0), 16));
        float32x4_t values2 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(packed1), 16));
        float32x4_t values3 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(packed1), 16));
        sums[0] = vfmaq_f32(sums[0], values0,
                            vld1q_f32(hidden + column));
        sums[1] = vfmaq_f32(sums[1], values1,
                            vld1q_f32(hidden + column + 4));
        sums[2] = vfmaq_f32(sums[2], values2,
                            vld1q_f32(hidden + column + 8));
        sums[3] = vfmaq_f32(sums[3], values3,
                            vld1q_f32(hidden + column + 12));
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(sums[0], sums[1]),
                                vaddq_f32(sums[2], sums[3]));
    float result = vaddvq_f32(sum);
    for (; column < count; column++)
        result += bf16_decode(row[column]) * hidden[column];
    return result;
}

static int select_routes(float *weights, int *indices, const float *scores,
                         float *selection, unsigned char *selected, int experts,
                         int topk) {
    memset(selected, 0, (size_t)experts);
    for (int rank = 0; rank < topk; rank++) {
        int best = -1;
        for (int expert = 0; expert < experts; expert++)
            if (!selected[expert] &&
                (best < 0 || selection[expert] > selection[best]))
                best = expert;
        indices[rank] = best;
        selected[best] = 1;
    }
    float total = 0.0f;
    for (int rank = 0; rank < topk; rank++) total += scores[indices[rank]];
    if (!(total > 0.0f)) return -1;
    for (int rank = 0; rank < topk; rank++)
        weights[rank] = scores[indices[rank]] / total * 2.5f;
    return 0;
}

__attribute__((noinline))
static int route_current(float *weights, int *indices, const float *hidden,
                         const uint16_t *gate, const float *bias, int experts,
                         int dimension, int topk) {
    float *scores = malloc((size_t)experts * sizeof(*scores));
    float *selection = malloc((size_t)experts * sizeof(*selection));
    unsigned char *selected = calloc((size_t)experts, 1);
    if (!scores || !selection || !selected) abort();
    for (int expert = 0; expert < experts; expert++) {
        float sum = dot_current(gate + (size_t)expert * dimension, hidden,
                                dimension);
        scores[expert] = sqrtf(softplus(sum));
        selection[expert] = scores[expert] + bias[expert];
    }
    int result = select_routes(weights, indices, scores, selection, selected,
                               experts, topk);
    free(selected);
    free(selection);
    free(scores);
    return result;
}

#define DEFINE_STACK_ROUTE(name, dot)                                            \
    __attribute__((noinline))                                                    \
    static int name(float *weights, int *indices, const float *hidden,           \
                    const uint16_t *gate, const float *bias, int experts,        \
                    int dimension, int topk) {                                   \
        float scores[EXPERTS];                                                   \
        float selection[EXPERTS];                                                \
        unsigned char selected[EXPERTS];                                         \
        for (int expert = 0; expert < experts; expert++) {                       \
            float sum = dot(gate + (size_t)expert * dimension, hidden,          \
                            dimension);                                          \
            scores[expert] = sqrtf(softplus(sum));                               \
            selection[expert] = scores[expert] + bias[expert];                   \
        }                                                                        \
        return select_routes(weights, indices, scores, selection, selected,     \
                             experts, topk);                                     \
    }

DEFINE_STACK_ROUTE(route_ordered, dot_ordered)
DEFINE_STACK_ROUTE(route_fast, dot_fast)

static double now_seconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double run_iterations(route_fn function, const float *hidden,
                             const uint16_t *gate, const float *bias,
                             int iterations) {
    float weights[TOPK];
    int indices[TOPK];
    double began = now_seconds();
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (function(weights, indices, hidden, gate, bias,
                     EXPERTS, DIMENSION, TOPK)) abort();
        benchmark_sink += weights[iteration % TOPK];
    }
    return now_seconds() - began;
}

static int calibrate(route_fn function, const float *hidden,
                     const uint16_t *gate, const float *bias) {
    int iterations = 1;
    while (run_iterations(function, hidden, gate, bias, iterations) < 0.2)
        iterations *= 2;
    return iterations;
}

static void benchmark(const char *name, route_fn function, const float *hidden,
                      const uint16_t *gate, const float *bias) {
    run_iterations(function, hidden, gate, bias, 32);
    int iterations = calibrate(function, hidden, gate, bias);
    double samples[ROUNDS];
    double mean = 0.0;
    for (int round = 0; round < ROUNDS; round++) {
        samples[round] = run_iterations(function, hidden, gate, bias, iterations)
                       * 1000.0 / iterations;
        mean += samples[round];
    }
    mean /= ROUNDS;
    double squared_difference = 0.0;
    for (int round = 0; round < ROUNDS; round++) {
        double difference = samples[round] - mean;
        squared_difference += difference * difference;
    }
    double standard_deviation = sqrt(squared_difference / (ROUNDS - 1));
    printf("%-8s %.6f +/- %.6f ms  iterations=%d\n",
           name, mean, standard_deviation, iterations);
}

static uint32_t random_state = 0x7142ac19u;

static float random_float(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return ((float)(random_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

int main(void) {
    float *hidden = malloc(DIMENSION * sizeof(*hidden));
    uint16_t *gate = malloc((size_t)EXPERTS * DIMENSION * sizeof(*gate));
    float *bias = malloc(EXPERTS * sizeof(*bias));
    if (!hidden || !gate || !bias) return 1;
    for (int column = 0; column < DIMENSION; column++)
        hidden[column] = random_float();
    for (size_t index = 0; index < (size_t)EXPERTS * DIMENSION; index++)
        gate[index] = bf16_encode(random_float() * 0.05f);
    for (int expert = 0; expert < EXPERTS; expert++)
        bias[expert] = random_float() * 0.1f;

    float current_weights[TOPK], ordered_weights[TOPK];
    int current_indices[TOPK], ordered_indices[TOPK];
    route_current(current_weights, current_indices, hidden, gate, bias,
                  EXPERTS, DIMENSION, TOPK);
    route_ordered(ordered_weights, ordered_indices, hidden, gate, bias,
                  EXPERTS, DIMENSION, TOPK);
    int exact = !memcmp(current_weights, ordered_weights, sizeof(current_weights))
             && !memcmp(current_indices, ordered_indices, sizeof(current_indices));
    printf("tier1_bit_exact=%s\n", exact ? "yes" : "no");
    if (!exact) return 2;

    benchmark("current", route_current, hidden, gate, bias);
    benchmark("tier1", route_ordered, hidden, gate, bias);
    benchmark("tier2", route_fast, hidden, gate, bias);
    printf("sink=%f\n", benchmark_sink);
    free(bias);
    free(gate);
    free(hidden);
    return 0;
}
