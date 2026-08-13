#define COLI_V4_UNIT_MATH 1
#define COLI_V4_UNIT_NATIVE_QUANT 1
#define COLI_V4_UNIT_EXPERT 1
#include "../../c/deepseek_v4.c"
#include "../../c/quant.h"
#include "synth_tensors.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SYNTH_V4_STEPS = 7 };

typedef struct {
    const char *name;
    uint64_t hash;
    uint64_t count;
    uint32_t first;
    uint32_t last;
} GoldenSummary;

static const char *const k_golden_names[SYNTH_V4_STEPS] = {
    "fp8_activation_qdq", "gate_up_matmul", "gate_up_bf16", "swiglu",
    "route_weight_bf16", "down_matmul", "output_bf16"
};

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int write_golden_hex(FILE *output, const SynthV4Fixture *fixture) {
    for (int token = 0; token < SYNTH_V4_TOKENS; token++)
        for (int rank = 0; rank < SYNTH_V4_TOPK; rank++) {
            const int expert_index = fixture->selected[token][rank];
            const SynthV4Expert *expert = &fixture->experts[expert_index];
            const float *input = fixture->inputs[token];
            const float route_weight = fixture->route_weights[token][rank];
            float qdq[SYNTH_V4_HIDDEN];
            uint8_t qdq_scales[1];
            float gate[SYNTH_V4_INTERMEDIATE], up[SYNTH_V4_INTERMEDIATE];
            float gate_up[2 * SYNTH_V4_INTERMEDIATE];
            float activated[SYNTH_V4_INTERMEDIATE];
            float weighted[SYNTH_V4_INTERMEDIATE];
            float down_input[SYNTH_V4_INTERMEDIATE], down[SYNTH_V4_HIDDEN];
            uint8_t down_scales[1];
            const float *steps[SYNTH_V4_STEPS];
            size_t counts[SYNTH_V4_STEPS];
            if (coli_fp8_activation_qdq_ref(qdq, qdq_scales, input,
                    SYNTH_V4_HIDDEN, SYNTH_V4_FP8_GROUP) != 0) return -1;
            matmul_mxfp4(gate, qdq, expert->gate.cold.data,
                         expert->gate.cold.scales, 1, SYNTH_V4_HIDDEN,
                         SYNTH_V4_INTERMEDIATE);
            matmul_mxfp4(up, qdq, expert->up.cold.data,
                         expert->up.cold.scales, 1, SYNTH_V4_HIDDEN,
                         SYNTH_V4_INTERMEDIATE);
            memcpy(gate_up, gate, sizeof(gate));
            memcpy(gate_up + SYNTH_V4_INTERMEDIATE, up, sizeof(up));
            steps[0] = qdq; counts[0] = SYNTH_V4_HIDDEN;
            steps[1] = gate_up; counts[1] = 2 * SYNTH_V4_INTERMEDIATE;
            fprintf(output,
                    "GOLDEN step=1 name=%s token=%d rank=%d expert=%d count=%zu hex=",
                    k_golden_names[0], token, rank, expert_index, counts[0]);
            for (size_t index = 0; index < counts[0]; index++)
                fprintf(output, "%s%08" PRIx32, index ? "," : "",
                        float_bits(steps[0][index]));
            fputc('\n', output);
            fprintf(output,
                    "GOLDEN step=2 name=%s token=%d rank=%d expert=%d count=%zu hex=",
                    k_golden_names[1], token, rank, expert_index, counts[1]);
            for (size_t index = 0; index < counts[1]; index++)
                fprintf(output, "%s%08" PRIx32, index ? "," : "",
                        float_bits(steps[1][index]));
            fputc('\n', output);
            coli_bf16_round_array(gate, SYNTH_V4_INTERMEDIATE);
            coli_bf16_round_array(up, SYNTH_V4_INTERMEDIATE);
            memcpy(gate_up, gate, sizeof(gate));
            memcpy(gate_up + SYNTH_V4_INTERMEDIATE, up, sizeof(up));
            if (coli_v4_swiglu(activated, gate, up,
                               SYNTH_V4_INTERMEDIATE, 3.0f) != 0) return -1;
            for (int index = 0; index < SYNTH_V4_INTERMEDIATE; index++)
                weighted[index] = coli_bf16_round(
                    activated[index] * route_weight);
            if (coli_fp8_activation_qdq_ref(down_input, down_scales, weighted,
                    SYNTH_V4_INTERMEDIATE, SYNTH_V4_FP8_GROUP) != 0) return -1;
            matmul_mxfp4(down, down_input, expert->down.cold.data,
                         expert->down.cold.scales, 1,
                         SYNTH_V4_INTERMEDIATE, SYNTH_V4_HIDDEN);
            steps[2] = gate_up; counts[2] = 2 * SYNTH_V4_INTERMEDIATE;
            steps[3] = activated; counts[3] = SYNTH_V4_INTERMEDIATE;
            steps[4] = weighted; counts[4] = SYNTH_V4_INTERMEDIATE;
            steps[5] = down; counts[5] = SYNTH_V4_HIDDEN;
            for (int step = 2; step < 6; step++) {
                fprintf(output,
                        "GOLDEN step=%d name=%s token=%d rank=%d expert=%d count=%zu hex=",
                        step + 1, k_golden_names[step], token, rank,
                        expert_index, counts[step]);
                for (size_t index = 0; index < counts[step]; index++)
                    fprintf(output, "%s%08" PRIx32, index ? "," : "",
                            float_bits(steps[step][index]));
                fputc('\n', output);
            }
            coli_bf16_round_array(down, SYNTH_V4_HIDDEN);
            fprintf(output,
                    "GOLDEN step=7 name=%s token=%d rank=%d expert=%d count=%d hex=",
                    k_golden_names[6], token, rank, expert_index,
                    SYNTH_V4_HIDDEN);
            for (int index = 0; index < SYNTH_V4_HIDDEN; index++)
                fprintf(output, "%s%08" PRIx32, index ? "," : "",
                        float_bits(down[index]));
            fputc('\n', output);
        }
    return ferror(output) ? -1 : 0;
}

static void golden_update(GoldenSummary *summary,
                          const float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        uint32_t bits = float_bits(values[index]);
        if (summary->count == 0) summary->first = bits;
        summary->last = bits;
        for (int byte = 0; byte < 4; byte++) {
            summary->hash ^= (uint8_t)(bits >> (8 * byte));
            summary->hash *= UINT64_C(1099511628211);
        }
        summary->count++;
    }
}

static void golden_dump(int full, int step, const char *name,
                        int token, int rank, int expert,
                        const float *values, size_t count,
                        GoldenSummary *summary) {
    golden_update(summary, values, count);
    if (!full) return;
    printf("GOLDEN step=%d name=%s token=%d rank=%d expert=%d count=%zu hex=",
           step, name, token, rank, expert, count);
    for (size_t index = 0; index < count; index++)
        printf("%s%08" PRIx32, index ? "," : "", float_bits(values[index]));
    putchar('\n');
}

static int run_case(const SynthV4Fixture *fixture, int token, int rank,
                    int full, GoldenSummary summaries[SYNTH_V4_STEPS]) {
    const int expert_index = fixture->selected[token][rank];
    const SynthV4Expert *expert = &fixture->experts[expert_index];
    const float *input = fixture->inputs[token];
    const float route_weight = fixture->route_weights[token][rank];
    float qdq[SYNTH_V4_HIDDEN];
    uint8_t qdq_scales[(SYNTH_V4_HIDDEN + SYNTH_V4_FP8_GROUP - 1) /
                       SYNTH_V4_FP8_GROUP];
    float gate[SYNTH_V4_INTERMEDIATE];
    float up[SYNTH_V4_INTERMEDIATE];
    float gate_up[2 * SYNTH_V4_INTERMEDIATE];
    float activated[SYNTH_V4_INTERMEDIATE];
    float weighted[SYNTH_V4_INTERMEDIATE];
    float down_input[SYNTH_V4_INTERMEDIATE];
    uint8_t down_scales[SYNTH_V4_INTERMEDIATE / SYNTH_V4_FP8_GROUP];
    float down[SYNTH_V4_HIDDEN];
    float reference[SYNTH_V4_STORAGE_HIDDEN];
    float reference_input[SYNTH_V4_STORAGE_HIDDEN] = {0};

    if (coli_fp8_activation_qdq_ref(qdq, qdq_scales, input,
                                    SYNTH_V4_HIDDEN,
                                    SYNTH_V4_FP8_GROUP) != 0) return -1;
    golden_dump(full, 1, "fp8_activation_qdq", token, rank, expert_index,
                qdq, SYNTH_V4_HIDDEN, &summaries[0]);

    matmul_mxfp4(gate, qdq, expert->gate.cold.data,
                 expert->gate.cold.scales, 1, SYNTH_V4_HIDDEN,
                 SYNTH_V4_INTERMEDIATE);
    matmul_mxfp4(up, qdq, expert->up.cold.data,
                 expert->up.cold.scales, 1, SYNTH_V4_HIDDEN,
                 SYNTH_V4_INTERMEDIATE);
    memcpy(gate_up, gate, sizeof(gate));
    memcpy(gate_up + SYNTH_V4_INTERMEDIATE, up, sizeof(up));
    golden_dump(full, 2, "gate_up_matmul", token, rank, expert_index,
                gate_up, 2 * SYNTH_V4_INTERMEDIATE, &summaries[1]);

    coli_bf16_round_array(gate, SYNTH_V4_INTERMEDIATE);
    coli_bf16_round_array(up, SYNTH_V4_INTERMEDIATE);
    memcpy(gate_up, gate, sizeof(gate));
    memcpy(gate_up + SYNTH_V4_INTERMEDIATE, up, sizeof(up));
    golden_dump(full, 3, "gate_up_bf16", token, rank, expert_index,
                gate_up, 2 * SYNTH_V4_INTERMEDIATE, &summaries[2]);

    if (coli_v4_swiglu(activated, gate, up,
                       SYNTH_V4_INTERMEDIATE, 3.0f) != 0) return -1;
    golden_dump(full, 4, "swiglu", token, rank, expert_index,
                activated, SYNTH_V4_INTERMEDIATE, &summaries[3]);

    for (int index = 0; index < SYNTH_V4_INTERMEDIATE; index++)
        weighted[index] = coli_bf16_round(activated[index] * route_weight);
    golden_dump(full, 5, "route_weight_bf16", token, rank, expert_index,
                weighted, SYNTH_V4_INTERMEDIATE, &summaries[4]);

    if (coli_fp8_activation_qdq_ref(down_input, down_scales, weighted,
                                    SYNTH_V4_INTERMEDIATE,
                                    SYNTH_V4_FP8_GROUP) != 0) return -1;
    matmul_mxfp4(down, down_input, expert->down.cold.data,
                 expert->down.cold.scales, 1, SYNTH_V4_INTERMEDIATE,
                 SYNTH_V4_HIDDEN);
    golden_dump(full, 6, "down_matmul", token, rank, expert_index,
                down, SYNTH_V4_HIDDEN, &summaries[5]);

    coli_bf16_round_array(down, SYNTH_V4_HIDDEN);
    golden_dump(full, 7, "output_bf16", token, rank, expert_index,
                down, SYNTH_V4_HIDDEN, &summaries[6]);

    memcpy(reference_input, input, SYNTH_V4_HIDDEN * sizeof(*input));
    if (coli_v4_expert_forward_ref(reference, &expert->reference,
                                   reference_input,
                                   route_weight, 3.0f) != 0 ||
        memcmp(reference, down, sizeof(down)) != 0) return -1;
    return 0;
}

int main(int argc, char **argv) {
    int full = argc == 2 && strcmp(argv[1], "--full") == 0;
    const char *golden_path =
        argc == 3 && strcmp(argv[1], "--golden") == 0 ? argv[2] : NULL;
    if (argc > 3 || (argc == 2 && !full) || (argc == 3 && !golden_path)) {
        fprintf(stderr, "usage: %s [--full | --golden FILE]\n", argv[0]);
        return 2;
    }
    SynthV4Fixture *fixture = calloc(1, sizeof(*fixture));
    if (!fixture) return 1;
    synth_v4_fixture_init(fixture);
    if (!synth_v4_fixture_roundtrip(fixture)) {
        fprintf(stderr, "rows16 round-trip mismatch\n");
        free(fixture);
        return 1;
    }
    if (golden_path) {
        FILE *output = fopen(golden_path, "wb");
        if (!output || write_golden_hex(output, fixture) != 0 ||
            fclose(output) != 0) {
            fprintf(stderr, "failed to write golden file: %s\n", golden_path);
            free(fixture);
            return 1;
        }
    }
    GoldenSummary summaries[SYNTH_V4_STEPS] = {
        {"fp8_activation_qdq", UINT64_C(14695981039346656037), 0, 0, 0},
        {"gate_up_matmul", UINT64_C(14695981039346656037), 0, 0, 0},
        {"gate_up_bf16", UINT64_C(14695981039346656037), 0, 0, 0},
        {"swiglu", UINT64_C(14695981039346656037), 0, 0, 0},
        {"route_weight_bf16", UINT64_C(14695981039346656037), 0, 0, 0},
        {"down_matmul", UINT64_C(14695981039346656037), 0, 0, 0},
        {"output_bf16", UINT64_C(14695981039346656037), 0, 0, 0}
    };
    for (int token = 0; token < SYNTH_V4_TOKENS; token++)
        for (int rank = 0; rank < SYNTH_V4_TOPK; rank++)
            if (run_case(fixture, token, rank, full, summaries) != 0) {
                fprintf(stderr, "CPU reference mismatch token=%d rank=%d\n",
                        token, rank);
                free(fixture);
                return 1;
            }
    puts("LAYOUT cold_rows16_roundtrip=PASS matrices=12");
    for (int step = 0; step < SYNTH_V4_STEPS; step++)
        printf("GOLDEN_SUMMARY step=%d name=%s count=%" PRIu64
               " fnv64=0x%016" PRIx64 " first=0x%08" PRIx32
               " last=0x%08" PRIx32 "\n",
               step + 1, summaries[step].name, summaries[step].count,
               summaries[step].hash, summaries[step].first,
               summaries[step].last);
    puts("PARITY_V4 PASS tokens=2 topk=2 experts=4 hidden=64 "
         "intermediate=128 fp4_block=32 fp8_group=128");
    free(fixture);
    return 0;
}
