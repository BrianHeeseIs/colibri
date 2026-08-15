#define COLI_METAL 1
#include "../../c/backend_metal_v4.h"
#include "../../c/backend_metal_v4_seam.h"
#include "../../c/deepseek_v4_internal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int coli_v4_metal_init(const char *metallib_path);
void coli_v4_metal_shutdown(void);
int coli_v4_metal_available(void);

enum {
    HIDDEN = 4096,
    INTERMEDIATE = 2048,
    FP4_GROUP = 32,
    EXPERT_BLOCK_ROWS = 16,
    MAX_BATCH = 16,
    SLAB_ALIGNMENT = 16384,
};

typedef struct {
    ColiExpertView view;
    ColiExpertView cold_view;
    void *slab;
    size_t slab_bytes;
} FixtureExpert;

typedef struct {
    uint64_t state;
} FixtureRng;

static uint32_t rng_next(FixtureRng *rng) {
    rng->state = rng->state * UINT64_C(6364136223846793005) +
                 UINT64_C(1442695040888963407);
    return (uint32_t)(rng->state >> 32);
}

static void fill_layouts(FixtureRng *rng, uint8_t *rows16, uint8_t *cold,
                         int rows, size_t row_bytes, uint32_t minimum,
                         uint32_t range) {
    for (int row = 0; row < rows; row++) {
        size_t tile = (size_t)row / EXPERT_BLOCK_ROWS;
        size_t lane = (size_t)row % EXPERT_BLOCK_ROWS;
        for (size_t column = 0; column < row_bytes; column++) {
            uint8_t value = (uint8_t)(minimum + rng_next(rng) % range);
            cold[(size_t)row * row_bytes + column] = value;
            rows16[(tile * row_bytes + column) * EXPERT_BLOCK_ROWS + lane] = value;
        }
    }
}

static int init_expert(FixtureExpert *fixture) {
    const size_t gate_data_bytes = (size_t)INTERMEDIATE * HIDDEN / 2;
    const size_t gate_scale_bytes = (size_t)INTERMEDIATE * HIDDEN / FP4_GROUP;
    const size_t down_data_bytes = (size_t)HIDDEN * INTERMEDIATE / 2;
    const size_t down_scale_bytes = (size_t)HIDDEN * INTERMEDIATE / FP4_GROUP;
    const size_t layout_bytes = gate_data_bytes + gate_scale_bytes +
                                gate_data_bytes + gate_scale_bytes +
                                down_data_bytes + down_scale_bytes;
    const size_t slab_bytes = 2 * layout_bytes;
    memset(fixture, 0, sizeof(*fixture));
    if (posix_memalign(&fixture->slab, SLAB_ALIGNMENT, slab_bytes) != 0)
        return -1;
    fixture->slab_bytes = slab_bytes;

    uint8_t *cursor = fixture->slab;
    uint8_t *gate_data = cursor;
    cursor += gate_data_bytes;
    uint8_t *gate_scales = cursor;
    cursor += gate_scale_bytes;
    uint8_t *up_data = cursor;
    cursor += gate_data_bytes;
    uint8_t *up_scales = cursor;
    cursor += gate_scale_bytes;
    uint8_t *down_data = cursor;
    cursor += down_data_bytes;
    uint8_t *down_scales = cursor;
    cursor += down_scale_bytes;
    uint8_t *cold_gate_data = cursor;
    cursor += gate_data_bytes;
    uint8_t *cold_gate_scales = cursor;
    cursor += gate_scale_bytes;
    uint8_t *cold_up_data = cursor;
    cursor += gate_data_bytes;
    uint8_t *cold_up_scales = cursor;
    cursor += gate_scale_bytes;
    uint8_t *cold_down_data = cursor;
    cursor += down_data_bytes;
    uint8_t *cold_down_scales = cursor;

    FixtureRng rng = {UINT64_C(0xd1b54a32d192ed03)};
    fill_layouts(&rng, gate_data, cold_gate_data, INTERMEDIATE,
                 HIDDEN / 2, 0, 256);
    fill_layouts(&rng, gate_scales, cold_gate_scales, INTERMEDIATE,
                 HIDDEN / FP4_GROUP, 124, 7);
    fill_layouts(&rng, up_data, cold_up_data, INTERMEDIATE,
                 HIDDEN / 2, 0, 256);
    fill_layouts(&rng, up_scales, cold_up_scales, INTERMEDIATE,
                 HIDDEN / FP4_GROUP, 124, 7);
    fill_layouts(&rng, down_data, cold_down_data, HIDDEN,
                 INTERMEDIATE / 2, 0, 256);
    fill_layouts(&rng, down_scales, cold_down_scales, HIDDEN,
                 INTERMEDIATE / FP4_GROUP, 124, 7);

    fixture->view.gate = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        gate_data, gate_scales, gate_data_bytes, gate_scale_bytes,
        INTERMEDIATE, HIDDEN, EXPERT_BLOCK_ROWS, FP4_GROUP,
    };
    fixture->view.up = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        up_data, up_scales, gate_data_bytes, gate_scale_bytes,
        INTERMEDIATE, HIDDEN, EXPERT_BLOCK_ROWS, FP4_GROUP,
    };
    fixture->view.down = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        down_data, down_scales, down_data_bytes, down_scale_bytes,
        HIDDEN, INTERMEDIATE, EXPERT_BLOCK_ROWS, FP4_GROUP,
    };
    fixture->cold_view.gate = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        cold_gate_data, cold_gate_scales, gate_data_bytes, gate_scale_bytes,
        INTERMEDIATE, HIDDEN, 1, FP4_GROUP,
    };
    fixture->cold_view.up = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        cold_up_data, cold_up_scales, gate_data_bytes, gate_scale_bytes,
        INTERMEDIATE, HIDDEN, 1, FP4_GROUP,
    };
    fixture->cold_view.down = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        cold_down_data, cold_down_scales, down_data_bytes, down_scale_bytes,
        HIDDEN, INTERMEDIATE, 1, FP4_GROUP,
    };
    coli_v4_metal_register_slab(fixture->slab, fixture->slab_bytes);
    return 0;
}

static void destroy_expert(FixtureExpert *fixture) {
    if (fixture->slab)
        coli_v4_metal_unregister_slab(fixture->slab);
    free(fixture->slab);
    memset(fixture, 0, sizeof(*fixture));
}

static void fill_inputs(float *inputs, float *route_weights) {
    static const float magnitudes[] = {
        0.01f, 0.03125f, 0.125f, 0.5f, 2.0f, 8.0f, 32.0f, 100.0f,
    };
    FixtureRng rng = {UINT64_C(0x94d049bb133111eb)};
    for (int row = 0; row < MAX_BATCH; row++) {
        route_weights[row] = 0.125f + 0.046875f * (float)row;
        for (int column = 0; column < HIDDEN; column++) {
            uint32_t bits = rng_next(&rng);
            float magnitude = magnitudes[bits %
                (sizeof(magnitudes) / sizeof(magnitudes[0]))];
            float modulation = 0.5f + (float)((bits >> 8) & 255u) / 510.0f;
            inputs[(size_t)row * HIDDEN + column] =
                (bits & 0x80000000u ? -1.0f : 1.0f) * magnitude * modulation;
        }
    }
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t ordered_float_bits(uint32_t bits) {
    return bits & UINT32_C(0x80000000) ? ~bits : bits | UINT32_C(0x80000000);
}

static uint32_t ulp_distance(uint32_t left, uint32_t right) {
    uint32_t ordered_left = ordered_float_bits(left);
    uint32_t ordered_right = ordered_float_bits(right);
    return ordered_left > ordered_right ? ordered_left - ordered_right
                                        : ordered_right - ordered_left;
}

static int run_case(const FixtureExpert *fixture, const float *inputs,
                    const float *route_weights, int batch) {
    size_t count = (size_t)batch * HIDDEN;
    float *batched = malloc(count * sizeof(*batched));
    float *cpu = malloc(count * sizeof(*cpu));
    if (!batched || !cpu) {
        free(cpu);
        free(batched);
        return 1;
    }

    int failed = coli_v4_metal_expert_forward_batch(
        batched, &fixture->view, inputs, route_weights, batch, 10.0f) != 0;
    for (int row = 0; row < batch && !failed; row++)
        failed = coli_v4_expert_forward_ref(
            cpu + (size_t)row * HIDDEN, &fixture->view,
            inputs + (size_t)row * HIDDEN, route_weights[row], 10.0f) != 0;
    if (failed) {
        fprintf(stderr, "ERROR N=%d expert forward call failed\n", batch);
        free(cpu);
        free(batched);
        return 1;
    }

    size_t differences = 0;
    uint32_t max_ulp = 0;
    size_t first_difference = 0;
    for (size_t index = 0; index < count; index++) {
        uint32_t batch_bits = float_bits(batched[index]);
        uint32_t cpu_bits = float_bits(cpu[index]);
        if (batch_bits == cpu_bits) continue;
        uint32_t ulp = ulp_distance(batch_bits, cpu_bits);
        if (!differences) first_difference = index;
        if (ulp > max_ulp) max_ulp = ulp;
        differences++;
    }
    printf("RESULT N=%d max_ulp=%" PRIu32 " differing_floats=%zu/%zu\n",
           batch, max_ulp, differences, count);
    if (differences) {
        size_t index = first_difference;
        fprintf(stderr,
                "FIRST_MISMATCH N=%d index=%zu row=%zu column=%zu "
                "batch=0x%08" PRIx32 " cpu=0x%08" PRIx32 "\n",
                batch, index, index / HIDDEN, index % HIDDEN,
                float_bits(batched[index]), float_bits(cpu[index]));
    }
    free(cpu);
    free(batched);
    return differences != 0;
}

int main(void) {
    @autoreleasepool {
        FixtureExpert fixture;
        float *inputs = malloc((size_t)MAX_BATCH * HIDDEN * sizeof(*inputs));
        float route_weights[MAX_BATCH];
        if (!inputs || init_expert(&fixture) != 0) {
            fprintf(stderr, "fixture allocation failed\n");
            free(inputs);
            return 2;
        }
        fill_inputs(inputs, route_weights);
        if (!coli_v4_metal_init(NULL) || !coli_v4_metal_available()) {
            fprintf(stderr, "Metal initialization failed\n");
            destroy_expert(&fixture);
            free(inputs);
            return 2;
        }

        static const int batches[] = {1, 2, 4, 6, 16};
        int failed = 0;
        for (size_t index = 0; index < sizeof(batches) / sizeof(batches[0]); index++)
            failed |= run_case(&fixture, inputs, route_weights, batches[index]);

        coli_v4_metal_shutdown();
        destroy_expert(&fixture);
        free(inputs);
        printf("SUMMARY %s\n", failed ? "FAIL" : "PASS BIT-IDENTICAL");
        return failed;
    }
}
