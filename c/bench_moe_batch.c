#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "native_quant.h"
#include "native_quant_batch.h"

enum {
    HIDDEN_SIZE = 4096,
    MOE_INTERMEDIATE_SIZE = 2048,
    MAX_BATCH = 64,
    FP4_BLOCK_SIZE = 32,
    FP4_ROW_TILE = 4,
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

static void matmul_fp4_hoisted(float *outputs, const float *inputs,
                               const uint8_t *packed, const uint8_t *scale_codes,
                               int batch, int columns, int rows) {
    static const float fp4_values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    int packed_stride = (columns + 1) / 2;
    int group_count = (columns + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row += FP4_ROW_TILE) {
        int row_indices[FP4_ROW_TILE];
        const uint8_t *row_weights[FP4_ROW_TILE];
        const uint8_t *row_scales[FP4_ROW_TILE];
        float accumulators[MAX_BATCH][FP4_ROW_TILE] = {{0}};
        for (int lane = 0; lane < FP4_ROW_TILE; lane++) {
            row_indices[lane] = row + lane < rows ? row + lane : row;
            row_weights[lane] = packed + (int64_t)row_indices[lane] * packed_stride;
            row_scales[lane] = scale_codes + (int64_t)row_indices[lane] * group_count;
        }
        for (int group = 0; group < group_count; group++) {
            int base = group * FP4_BLOCK_SIZE;
            int group_length = columns - base < FP4_BLOCK_SIZE
                                   ? columns - base : FP4_BLOCK_SIZE;
            float weights[FP4_ROW_TILE][FP4_BLOCK_SIZE];
            float scales[FP4_ROW_TILE];
            for (int lane = 0; lane < FP4_ROW_TILE; lane++) {
                union { uint32_t bits; float value; } scale;
                scale.bits = (uint32_t)row_scales[lane][group] << 23;
                scales[lane] = scale.value;
                const uint8_t *block = row_weights[lane] + (base >> 1);
                for (int offset = 0; offset < group_length; offset += 2) {
                    uint8_t byte = block[offset >> 1];
                    weights[lane][offset] = fp4_values[byte & 15];
                    if (offset + 1 < group_length)
                        weights[lane][offset + 1] = fp4_values[byte >> 4];
                }
            }
            for (int item = 0; item < batch; item++) {
                const float *input = inputs + (int64_t)item * columns + base;
                float group_sums[FP4_ROW_TILE] = {0};
                for (int offset = 0; offset < group_length; offset++) {
                    float activation = input[offset];
                    for (int lane = 0; lane < FP4_ROW_TILE; lane++)
                        group_sums[lane] += activation * weights[lane][offset];
                }
                for (int lane = 0; lane < FP4_ROW_TILE; lane++)
                    accumulators[item][lane] += group_sums[lane] * scales[lane];
            }
        }
        for (int item = 0; item < batch; item++)
            for (int lane = 0; lane < FP4_ROW_TILE; lane++)
                if (row_indices[lane] != row || lane == 0)
                    outputs[(int64_t)item * rows + row_indices[lane]] =
                        accumulators[item][lane];
    }
}

static int fp4_matmul_batch_hoisted(float *outputs, const ColiTensorView *weight,
                                    const float *inputs, int batch) {
    if (!outputs || !weight || !inputs || batch < 1 || batch > MAX_BATCH ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 || !weight->data ||
        !weight->scales || weight->rows < 1 || weight->columns < 1 ||
        weight->columns % 128 || weight->block_rows != 1 ||
        weight->block_columns != FP4_BLOCK_SIZE)
        return -1;
    size_t rows = (size_t)weight->rows;
    size_t columns = (size_t)weight->columns;
    if (weight->data_bytes != rows * columns / 2 ||
        weight->scale_bytes != rows * columns / FP4_BLOCK_SIZE)
        return -1;
    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * columns / 128);
    if (!activations || !activation_scales) {
        free(activation_scales);
        free(activations);
        return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * columns / 128,
                inputs + (size_t)item * columns, columns, 128)) {
            free(activation_scales);
            free(activations);
            return -1;
        }
    matmul_fp4_hoisted(outputs, activations, weight->data, weight->scales,
                       batch, (int)columns, (int)rows);
    free(activation_scales);
    free(activations);
    return 0;
}

static int check_equivalence_case(const char *shape, const ColiTensorView *weight,
                                  const float *inputs, float *reference,
                                  float *hoisted, int batch) {
    if (coli_fp4_matmul_batch_ref(reference, weight, inputs, batch) ||
        fp4_matmul_batch_hoisted(hoisted, weight, inputs, batch))
        return -1;
    size_t count = (size_t)batch * (size_t)weight->rows;
    size_t different = 0;
    double max_absolute = 0.0;
    double max_relative = 0.0;
    for (size_t index = 0; index < count; index++) {
        if (memcmp(reference + index, hoisted + index, sizeof(float))) different++;
        double absolute = fabs((double)reference[index] - (double)hoisted[index]);
        double denominator = fabs((double)reference[index]);
        double relative = denominator > 0.0 ? absolute / denominator
                                             : (absolute > 0.0 ? INFINITY : 0.0);
        if (absolute > max_absolute) max_absolute = absolute;
        if (relative > max_relative) max_relative = relative;
    }
    printf("equivalence %-9s S=%2d outputs=%zu bitwise_different=%zu "
           "max_abs=%.9g max_rel=%.9g\n",
           shape, batch, count, different, max_absolute, max_relative);
    return different ? 1 : 0;
}

static int check_fp4_equivalence(const ExpertWeights *expert,
                                 const float *hidden_inputs,
                                 const float *intermediate_inputs) {
    static const int batches[] = {1, 2, 4, 8, 64};
    float *reference = malloc((size_t)MAX_BATCH * HIDDEN_SIZE * sizeof(float));
    float *hoisted = malloc((size_t)MAX_BATCH * HIDDEN_SIZE * sizeof(float));
    if (!reference || !hoisted) {
        free(hoisted);
        free(reference);
        return -1;
    }
    int result = 0;
    for (size_t index = 0; index < sizeof(batches) / sizeof(batches[0]); index++) {
        int batch = batches[index];
        int gate_result = check_equivalence_case(
            "gate/up", &expert->gate, hidden_inputs, reference, hoisted, batch);
        int down_result = check_equivalence_case(
            "down", &expert->down, intermediate_inputs, reference, hoisted, batch);
        if (gate_result || down_result) result = gate_result < 0 || down_result < 0 ? -1 : 1;
    }
    free(hoisted);
    free(reference);
    return result;
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

    int iterations = 1;
    for (;;) {
        double calibration_start = now_seconds();
        for (int iteration = 0; iteration < iterations; iteration++)
            if (run_expert(matmul, expert, hidden_inputs, intermediate_inputs,
                           intermediate_outputs, hidden_outputs, batch))
                return -1;
        double calibration = now_seconds() - calibration_start;
        if (calibration >= 0.25) break;
        int calibrated_iterations = calibration > 0.0
            ? (int)ceil((double)iterations * 0.30 / calibration)
            : iterations * 2;
        iterations = calibrated_iterations > iterations
            ? calibrated_iterations : iterations + 1;
    }

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

static double interpolate_measurement(const int *batches,
                                      const Measurement *measurements,
                                      size_t count, double batch) {
    for (size_t upper = 1; upper < count; upper++) {
        if (batch <= batches[upper]) {
            size_t lower = upper - 1;
            double fraction = (batch - batches[lower]) /
                              (batches[upper] - batches[lower]);
            return measurements[lower].mean_ms + fraction *
                (measurements[upper].mean_ms - measurements[lower].mean_ms);
        }
    }
    return measurements[count - 1].mean_ms;
}

static double projected_prefill_speedup(const int *batches,
                                        const Measurement *current,
                                        const Measurement *hoisted,
                                        size_t count) {
    static const struct {
        double batch;
        int groups;
    } distribution[] = {
        {1.0, 4555}, {2.5, 3894}, {5.5, 2266}, {11.0, 853}, {27.1, 418},
    };
    double unbatched = 0.0;
    double batched = 0.0;
    for (size_t index = 0;
         index < sizeof(distribution) / sizeof(distribution[0]); index++) {
        unbatched += distribution[index].groups * distribution[index].batch *
                     current[0].mean_ms;
        batched += distribution[index].groups * interpolate_measurement(
            batches, hoisted, count, distribution[index].batch);
    }
    return unbatched / batched;
}

static int benchmark_fp4_comparison(
        const ExpertWeights *expert, const float *hidden_inputs,
        const float *intermediate_inputs, float *intermediate_outputs,
        float *hidden_outputs) {
    static const int batches[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    enum { BATCH_COUNT = sizeof(batches) / sizeof(batches[0]) };
    Measurement current[BATCH_COUNT];
    Measurement hoisted[BATCH_COUNT];
    printf("\nFP4 routed expert record: %zu bytes (%.3f MB decimal)\n",
           expert->record_bytes, (double)expert->record_bytes / 1e6);
    printf("scratch: %zu-byte decoded weights + %zu-byte scales + "
           "%zu-byte max-batch accumulators per OpenMP worker\n",
           sizeof(float) * FP4_ROW_TILE * FP4_BLOCK_SIZE,
           sizeof(float) * FP4_ROW_TILE,
           sizeof(float) * MAX_BATCH * FP4_ROW_TILE);
    printf("S | current f(S) ms +/- sd | hoisted f(S) ms +/- sd | "
           "vs current | hoisted speedup_op | N current/hoisted\n");
    for (size_t index = 0; index < BATCH_COUNT; index++) {
        int batch = batches[index];
        if (measure(&current[index], coli_fp4_matmul_batch_ref, expert,
                    hidden_inputs, intermediate_inputs, intermediate_outputs,
                    hidden_outputs, batch) ||
            measure(&hoisted[index], fp4_matmul_batch_hoisted, expert,
                    hidden_inputs, intermediate_inputs, intermediate_outputs,
                    hidden_outputs, batch))
            return -1;
        double versus_current = current[index].mean_ms / hoisted[index].mean_ms;
        double operation_speedup = (double)batch * hoisted[0].mean_ms /
                                   hoisted[index].mean_ms;
        printf("%2d | %9.3f +/- %-7.3f | %9.3f +/- %-7.3f | %10.3f | "
               "%18.3f | %d/%d\n",
               batch, current[index].mean_ms, current[index].stddev_ms,
               hoisted[index].mean_ms, hoisted[index].stddev_ms,
               versus_current, operation_speedup, current[index].iterations,
               hoisted[index].iterations);
        fflush(stdout);
    }
    double projection = projected_prefill_speedup(
        batches, current, hoisted, BATCH_COUNT);
    printf("projected prefill speedup: %.4fx (%s vs 1.6000x decision bar)\n",
           projection, projection >= 1.6 ? "GO" : "NO-GO");
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
    printf("FP4 scratch block=%d weights (%zu bytes/row)\n",
           FP4_BLOCK_SIZE, sizeof(float) * FP4_BLOCK_SIZE);
    int equivalence = check_fp4_equivalence(
        &fp4_expert, hidden_inputs, intermediate_inputs);
    if (equivalence > 0)
        fprintf(stderr, "FP4 hoisted equivalence FAILED\n");
    int result = equivalence || benchmark_fp4_comparison(
        &fp4_expert, hidden_inputs, intermediate_inputs,
        intermediate_outputs, hidden_outputs);
    printf("output_guard=%g\n", (double)output_guard);

    free_expert(&fp8_expert);
    free_expert(&fp4_expert);
    free(hidden_outputs);
    free(intermediate_outputs);
    free(intermediate_inputs);
    free(hidden_inputs);
    return result ? 1 : 0;
}
