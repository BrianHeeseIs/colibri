#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "native_quant_batch.h"

enum {
    HIDDEN_SIZE = 4096,
    MOE_INTERMEDIATE_SIZE = 2048,
    MAX_BATCH = 64,
};

typedef int (*BatchMatmul)(float *, const ColiTensorView *, const float *, int);

typedef struct {
    ColiTensorView gate;
    ColiTensorView up;
    ColiTensorView down;
    void *allocations[6];
    size_t record_bytes;
} ExpertWeights;

typedef struct {
    double mean_ms;
    double stddev_ms;
    int iterations;
} Measurement;

enum { MEASUREMENT_ROUNDS = 5 };

static volatile float output_guard;

static double now_seconds(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1e-9;
}

static uint64_t random_state = UINT64_C(0x243f6a8885a308d3);

static uint32_t random_u32(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return (uint32_t)(random_state >> 32);
}

static void fill_inputs(float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = ((float)(random_u32() % 2001) - 1000.0f) / 337.0f;
}

static int make_fp4_tensor(ColiTensorView *view, int rows, int columns,
                           void **data_allocation, void **scale_allocation) {
    size_t data_bytes = (size_t)rows * (size_t)columns / 2;
    size_t scale_bytes = (size_t)rows * (size_t)columns / 32;
    uint8_t *data = malloc(data_bytes);
    uint8_t *scales = malloc(scale_bytes);
    if (!data || !scales) {
        free(data);
        free(scales);
        return -1;
    }
    for (size_t index = 0; index < data_bytes; index++)
        data[index] = (uint8_t)random_u32();
    for (size_t index = 0; index < scale_bytes; index++)
        scales[index] = (uint8_t)(120 + random_u32() % 15);
    *view = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        data, scales, data_bytes, scale_bytes, rows, columns, 1, 32,
    };
    *data_allocation = data;
    *scale_allocation = scales;
    return 0;
}

static int make_fp8_tensor(ColiTensorView *view, int rows, int columns,
                           void **data_allocation, void **scale_allocation) {
    size_t data_bytes = (size_t)rows * (size_t)columns;
    size_t scale_count = (size_t)((rows + 127) / 128) * (size_t)(columns / 128);
    size_t scale_bytes = scale_count * sizeof(float);
    uint8_t *data = malloc(data_bytes);
    float *scales = malloc(scale_bytes);
    if (!data || !scales) {
        free(data);
        free(scales);
        return -1;
    }
    for (size_t index = 0; index < data_bytes; index++) {
        uint8_t code = (uint8_t)random_u32();
        data[index] = (code & 0x7f) == 0x7f ? 0x38 : code;
    }
    for (size_t index = 0; index < scale_count; index++)
        scales[index] = ldexpf(1.0f, (int)(random_u32() % 15) - 7);
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        data, scales, data_bytes, scale_bytes, rows, columns, 128, 128,
    };
    *data_allocation = data;
    *scale_allocation = scales;
    return 0;
}

static int make_expert(ExpertWeights *expert, int fp4) {
    memset(expert, 0, sizeof(*expert));
    int (*make_tensor)(ColiTensorView *, int, int, void **, void **) =
        fp4 ? make_fp4_tensor : make_fp8_tensor;
    if (make_tensor(&expert->gate, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
                    &expert->allocations[0], &expert->allocations[1]) ||
        make_tensor(&expert->up, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
                    &expert->allocations[2], &expert->allocations[3]) ||
        make_tensor(&expert->down, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE,
                    &expert->allocations[4], &expert->allocations[5]))
        return -1;
    expert->record_bytes = expert->gate.data_bytes + expert->gate.scale_bytes +
                           expert->up.data_bytes + expert->up.scale_bytes +
                           expert->down.data_bytes + expert->down.scale_bytes;
    return 0;
}

static void free_expert(ExpertWeights *expert) {
    for (size_t index = 0; index < 6; index++) free(expert->allocations[index]);
}

static int run_expert(BatchMatmul matmul, const ExpertWeights *expert,
                      const float *hidden_inputs, const float *intermediate_inputs,
                      float *intermediate_outputs, float *hidden_outputs, int batch) {
    size_t intermediate_count = (size_t)batch * MOE_INTERMEDIATE_SIZE;
    if (matmul(intermediate_outputs, &expert->gate, hidden_inputs, batch) ||
        matmul(intermediate_outputs + intermediate_count, &expert->up,
               hidden_inputs, batch) ||
        matmul(hidden_outputs, &expert->down, intermediate_inputs, batch))
        return -1;
    output_guard += intermediate_outputs[0] +
                    intermediate_outputs[intermediate_count] + hidden_outputs[0];
    return 0;
}

static int measure(Measurement *measurement, BatchMatmul matmul,
                   const ExpertWeights *expert, const float *hidden_inputs,
                   const float *intermediate_inputs, float *intermediate_outputs,
                   float *hidden_outputs, int batch) {
    for (int warmup = 0; warmup < 3; warmup++)
        if (run_expert(matmul, expert, hidden_inputs, intermediate_inputs,
                       intermediate_outputs, hidden_outputs, batch))
            return -1;

    double calibration_start = now_seconds();
    if (run_expert(matmul, expert, hidden_inputs, intermediate_inputs,
                   intermediate_outputs, hidden_outputs, batch))
        return -1;
    double calibration = now_seconds() - calibration_start;
    int iterations = calibration > 0.0 ? (int)ceil(0.20 / calibration) : 1;
    if (iterations < 1) iterations = 1;
    if (iterations > 500) iterations = 500;

    double samples[MEASUREMENT_ROUNDS];
    double sum = 0.0;
    for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
        double started = now_seconds();
        int result = 0;
        for (int iteration = 0; !result && iteration < iterations; iteration++)
            result = run_expert(matmul, expert, hidden_inputs, intermediate_inputs,
                                intermediate_outputs, hidden_outputs, batch);
        double elapsed = now_seconds() - started;
        if (result) return -1;
        samples[round] = elapsed / iterations;
        sum += samples[round];
    }
    double mean = sum / MEASUREMENT_ROUNDS;
    double squared_deviation = 0.0;
    for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
        double deviation = samples[round] - mean;
        squared_deviation += deviation * deviation;
    }
    measurement->mean_ms = mean * 1e3;
    measurement->stddev_ms = sqrt(squared_deviation / MEASUREMENT_ROUNDS) * 1e3;
    measurement->iterations = iterations * MEASUREMENT_ROUNDS;
    return 0;
}

static int benchmark_format(const char *name, BatchMatmul matmul,
                            const ExpertWeights *expert, const float *hidden_inputs,
                            const float *intermediate_inputs, float *intermediate_outputs,
                            float *hidden_outputs) {
    static const int batches[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    Measurement measurements[sizeof(batches) / sizeof(batches[0])];
    printf("\n%s expert record: %zu bytes (%.3f MB decimal)\n", name,
           expert->record_bytes, (double)expert->record_bytes / 1e6);
    printf("S | f(S) ms +/- stddev | speedup_op | effective GB/s | N\n");
    for (size_t index = 0; index < sizeof(batches) / sizeof(batches[0]); index++) {
        int batch = batches[index];
        if (measure(&measurements[index], matmul, expert, hidden_inputs,
                    intermediate_inputs, intermediate_outputs, hidden_outputs,
                    batch))
            return -1;
        double speedup = (double)batch * measurements[0].mean_ms /
                         measurements[index].mean_ms;
        double effective_gbps = (double)batch * (double)expert->record_bytes /
                                (measurements[index].mean_ms * 1e6);
        printf("%2d | %9.3f +/- %-7.3f | %10.3f | %14.3f | %d\n",
               batch, measurements[index].mean_ms, measurements[index].stddev_ms,
               speedup, effective_gbps, measurements[index].iterations);
        fflush(stdout);
    }
    return 0;
}

int main(void) {
    ExpertWeights fp4_expert, fp8_expert;
    float *hidden_inputs = malloc((size_t)MAX_BATCH * HIDDEN_SIZE * sizeof(float));
    float *intermediate_inputs = malloc(
        (size_t)MAX_BATCH * MOE_INTERMEDIATE_SIZE * sizeof(float));
    float *intermediate_outputs = malloc(
        (size_t)2 * MAX_BATCH * MOE_INTERMEDIATE_SIZE * sizeof(float));
    float *hidden_outputs = malloc((size_t)MAX_BATCH * HIDDEN_SIZE * sizeof(float));
    if (!hidden_inputs || !intermediate_inputs || !intermediate_outputs ||
        !hidden_outputs || make_expert(&fp4_expert, 1) ||
        make_expert(&fp8_expert, 0)) {
        fprintf(stderr, "benchmark allocation failed\n");
        return 1;
    }
    fill_inputs(hidden_inputs, (size_t)MAX_BATCH * HIDDEN_SIZE);
    fill_inputs(intermediate_inputs,
                (size_t)MAX_BATCH * MOE_INTERMEDIATE_SIZE);

    printf("DeepSeek-V4 MoE batch benchmark\n");
    printf("gate/up=%dx%d down=%dx%d max_batch=%d\n",
           MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
           HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, MAX_BATCH);
#ifdef _OPENMP
    printf("OpenMP max_threads=%d\n", omp_get_max_threads());
#else
    printf("OpenMP disabled\n");
#endif
    int result = benchmark_format(
        "FP4 routed", coli_fp4_matmul_batch_ref, &fp4_expert,
        hidden_inputs, intermediate_inputs, intermediate_outputs, hidden_outputs) ||
        benchmark_format(
            "FP8 shared", coli_fp8_matmul_batch_ref, &fp8_expert,
            hidden_inputs, intermediate_inputs, intermediate_outputs, hidden_outputs);
    printf("output_guard=%g\n", (double)output_guard);

    free_expert(&fp8_expert);
    free_expert(&fp4_expert);
    free(hidden_outputs);
    free(intermediate_outputs);
    free(intermediate_inputs);
    free(hidden_inputs);
    return result ? 1 : 0;
}
