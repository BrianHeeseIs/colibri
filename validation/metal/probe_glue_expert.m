// Single-expert V4 host-glue parity probe. Synthetic fixtures only.
#define COLI_V4_UNIT_MATH 1
#define COLI_V4_UNIT_NATIVE_QUANT 1
#define COLI_V4_UNIT_EXPERT 1
#include "../../c/deepseek_v4.c"

#define COLI_METAL 1
#include "../../c/backend_metal_v4.mm"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FP4_GROUP = 32 };

typedef struct {
    uint64_t state;
} FixtureRng;

typedef struct {
    ColiExpertView view;
    uint8_t *gate_data;
    uint8_t *gate_scales;
    uint8_t *up_data;
    uint8_t *up_scales;
    uint8_t *down_data;
    uint8_t *down_scales;
} FixtureExpert;

static uint32_t rng_next(FixtureRng *rng) {
    rng->state = rng->state * UINT64_C(6364136223846793005) +
                 UINT64_C(1442695040888963407);
    return (uint32_t)(rng->state >> 32);
}

static void fill_bytes(FixtureRng *rng, uint8_t *bytes, size_t count,
                       uint32_t minimum, uint32_t range) {
    for (size_t index = 0; index < count; index++)
        bytes[index] = (uint8_t)(minimum + rng_next(rng) % range);
}

static void free_expert(FixtureExpert *expert) {
    free(expert->down_scales);
    free(expert->down_data);
    free(expert->up_scales);
    free(expert->up_data);
    free(expert->gate_scales);
    free(expert->gate_data);
    memset(expert, 0, sizeof(*expert));
}

static int init_expert(FixtureExpert *expert, int hidden, int intermediate,
                       uint64_t seed) {
    const size_t gate_row_bytes = ((size_t)hidden + 1) / 2;
    const size_t gate_groups = ((size_t)hidden + FP4_GROUP - 1) / FP4_GROUP;
    const size_t down_row_bytes = ((size_t)intermediate + 1) / 2;
    const size_t down_groups =
        ((size_t)intermediate + FP4_GROUP - 1) / FP4_GROUP;
    const size_t gate_data_bytes = (size_t)intermediate * gate_row_bytes;
    const size_t gate_scale_bytes = (size_t)intermediate * gate_groups;
    const size_t down_data_bytes = (size_t)hidden * down_row_bytes;
    const size_t down_scale_bytes = (size_t)hidden * down_groups;
    memset(expert, 0, sizeof(*expert));
    expert->gate_data = malloc(gate_data_bytes);
    expert->gate_scales = malloc(gate_scale_bytes);
    expert->up_data = malloc(gate_data_bytes);
    expert->up_scales = malloc(gate_scale_bytes);
    expert->down_data = malloc(down_data_bytes);
    expert->down_scales = malloc(down_scale_bytes);
    if (!expert->gate_data || !expert->gate_scales || !expert->up_data ||
        !expert->up_scales || !expert->down_data || !expert->down_scales) {
        free_expert(expert);
        return -1;
    }
    FixtureRng rng = {seed};
    fill_bytes(&rng, expert->gate_data, gate_data_bytes, 0, 256);
    fill_bytes(&rng, expert->gate_scales, gate_scale_bytes, 112, 22);
    fill_bytes(&rng, expert->up_data, gate_data_bytes, 0, 256);
    fill_bytes(&rng, expert->up_scales, gate_scale_bytes, 112, 22);
    fill_bytes(&rng, expert->down_data, down_data_bytes, 0, 256);
    fill_bytes(&rng, expert->down_scales, down_scale_bytes, 112, 22);
    expert->view.gate = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        expert->gate_data, expert->gate_scales, gate_data_bytes,
        gate_scale_bytes, intermediate, hidden, 1, FP4_GROUP,
    };
    expert->view.up = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        expert->up_data, expert->up_scales, gate_data_bytes, gate_scale_bytes,
        intermediate, hidden, 1, FP4_GROUP,
    };
    expert->view.down = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        expert->down_data, expert->down_scales, down_data_bytes,
        down_scale_bytes, hidden, intermediate, 1, FP4_GROUP,
    };
    return 0;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int compare_exact(const char *label, const float *cpu, const float *gpu,
                         int hidden, int intermediate) {
    int differences = 0;
    for (int index = 0; index < hidden; index++) {
        if (float_bits(cpu[index]) == float_bits(gpu[index])) continue;
        if (differences < 8)
            printf("MISMATCH label=%s index=%d cpu=0x%08" PRIx32
                   " gpu=0x%08" PRIx32 "\n",
                   label, index, float_bits(cpu[index]), float_bits(gpu[index]));
        differences++;
    }
    printf("RESULT label=%s reductions=gate/up:%d,down:%d %s diff=%d/%d\n",
           label, hidden, intermediate,
           differences ? "MISMATCH" : "BIT-EXACT", differences, hidden);
    return differences != 0;
}

static int run_case(const char *label, int hidden, int intermediate,
                    uint64_t seed) {
    FixtureExpert expert;
    float *input = calloc((size_t)hidden, sizeof(*input));
    float *cpu = malloc((size_t)hidden * sizeof(*cpu));
    float *gpu = malloc((size_t)hidden * sizeof(*gpu));
    if (!input || !cpu || !gpu ||
        init_expert(&expert, hidden, intermediate, seed) != 0) {
        free(gpu); free(cpu); free(input); free_expert(&expert);
        return 1;
    }
    FixtureRng rng = {seed ^ UINT64_C(0x94d049bb133111eb)};
    for (int index = 0; index < hidden; index++)
        input[index] = ((float)(rng_next(&rng) % 4000) - 2000.0f) / 311.0f;
    const float route_weight = 0.625f;
    const float swiglu_limit = 10.0f;
    int failed = coli_v4_expert_forward_ref(cpu, &expert.view, input,
                                             route_weight, swiglu_limit) != 0;
    int glue_result = failed ? -1 : coli_v4_metal_expert_forward(
        gpu, &expert.view, input, route_weight, swiglu_limit);
    printf("CASE label=%s hidden=%d intermediate=%d gate_row_bytes=%zu "
           "gate_groups=%zu down_row_bytes=%zu down_groups=%zu\n",
           label, hidden, intermediate,
           expert.view.gate.data_bytes / (size_t)intermediate,
           expert.view.gate.scale_bytes / (size_t)intermediate,
           expert.view.down.data_bytes / (size_t)hidden,
           expert.view.down.scale_bytes / (size_t)hidden);
    if (glue_result != 0) {
        fprintf(stderr, "glue forward failed label=%s result=%d\n", label,
                glue_result);
        failed = 1;
    } else {
        failed |= compare_exact(label, cpu, gpu, hidden, intermediate);
    }
    free(gpu); free(cpu); free(input); free_expert(&expert);
    return failed;
}

static int check_cap_fallback(void) {
    FixtureExpert expert;
    float input[128] = {0};
    float output[128];
    for (int index = 0; index < 128; index++) output[index] = 123.0f;
    if (init_expert(&expert, 128, 128, UINT64_C(77)) != 0) return 1;
    expert.view.gate.rows = 2049;
    expert.view.up.rows = 2049;
    expert.view.down.columns = 2049;
    int result = coli_v4_metal_expert_forward(output, &expert.view, input,
                                               0.5f, 10.0f);
    int changed = 0;
    for (int index = 0; index < 128; index++) changed |= output[index] != 123.0f;
    free_expert(&expert);
    printf("CAP_FALLBACK result=%d output_untouched=%s\n", result,
           changed ? "FAIL" : "PASS");
    return result == 0 || changed;
}

int main(void) {
    @autoreleasepool {
        if (!coli_v4_metal_init(NULL) || !coli_v4_metal_available()) {
            fprintf(stderr, "Metal initialization failed\n");
            return 2;
        }
        int failed = 0;
        failed |= run_case("small", 128, 128, UINT64_C(24601));
        failed |= run_case("production", 4096, 2048, UINT64_C(1));
        failed |= check_cap_fallback();
        coli_v4_metal_shutdown();
        printf("SUMMARY %s small=128/128 production=4096/2048\n",
               failed ? "FAIL" : "PASS BIT-EXACT");
        return failed;
    }
}
