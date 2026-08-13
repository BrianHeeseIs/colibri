#ifndef COLIBRI_VALIDATION_METAL_SYNTH_TENSORS_H
#define COLIBRI_VALIDATION_METAL_SYNTH_TENSORS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../../c/deepseek_v4_internal.h"

enum {
    SYNTH_V4_HIDDEN = 64,
    SYNTH_V4_STORAGE_HIDDEN = 128,
    SYNTH_V4_INTERMEDIATE = 128,
    SYNTH_V4_EXPERTS = 4,
    SYNTH_V4_TOPK = 2,
    SYNTH_V4_TOKENS = 2,
    SYNTH_V4_FP4_BLOCK = 32,
    SYNTH_V4_FP8_GROUP = 128,
    SYNTH_V4_ROWS16 = 16,
    SYNTH_V4_WEIGHT_BYTES =
        SYNTH_V4_INTERMEDIATE * SYNTH_V4_STORAGE_HIDDEN / 2,
    SYNTH_V4_SCALE_BYTES =
        SYNTH_V4_INTERMEDIATE * SYNTH_V4_STORAGE_HIDDEN / SYNTH_V4_FP4_BLOCK
};

typedef struct {
    uint8_t cold_data[SYNTH_V4_WEIGHT_BYTES];
    uint8_t cold_scales[SYNTH_V4_SCALE_BYTES];
    uint8_t hot_data[SYNTH_V4_WEIGHT_BYTES];
    uint8_t hot_scales[SYNTH_V4_SCALE_BYTES];
    ColiTensorView cold;
    ColiTensorView hot;
} SynthV4Matrix;

typedef struct {
    SynthV4Matrix gate;
    SynthV4Matrix down;
    SynthV4Matrix up;
    SynthV4Matrix reference_gate;
    SynthV4Matrix reference_down;
    SynthV4Matrix reference_up;
    ColiExpertView cold;
    ColiExpertView hot;
    ColiExpertView reference;
} SynthV4Expert;

typedef struct {
    float inputs[SYNTH_V4_TOKENS][SYNTH_V4_STORAGE_HIDDEN];
    int selected[SYNTH_V4_TOKENS][SYNTH_V4_TOPK];
    float route_weights[SYNTH_V4_TOKENS][SYNTH_V4_TOPK];
    SynthV4Expert experts[SYNTH_V4_EXPERTS];
} SynthV4Fixture;

static inline uint32_t synth_v4_mix(uint32_t value) {
    value ^= value >> 16;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15;
    value *= UINT32_C(0x846ca68b);
    return value ^ (value >> 16);
}

static inline void synth_v4_rows16_pack(uint8_t *packed,
                                        const uint8_t *data,
                                        size_t rows, size_t stride) {
    for (size_t row = 0; row < rows; row++) {
        size_t tile = row / SYNTH_V4_ROWS16;
        size_t lane = row % SYNTH_V4_ROWS16;
        for (size_t column = 0; column < stride; column++)
            packed[(tile * stride + column) * SYNTH_V4_ROWS16 + lane] =
                data[row * stride + column];
    }
}

static inline void synth_v4_rows16_unpack(uint8_t *data,
                                          const uint8_t *packed,
                                          size_t rows, size_t stride) {
    for (size_t row = 0; row < rows; row++) {
        size_t tile = row / SYNTH_V4_ROWS16;
        size_t lane = row % SYNTH_V4_ROWS16;
        for (size_t column = 0; column < stride; column++)
            data[row * stride + column] =
                packed[(tile * stride + column) * SYNTH_V4_ROWS16 + lane];
    }
}

static inline uint8_t synth_v4_weight_code(int expert, int matrix,
                                            int row, int column) {
    uint32_t key = UINT32_C(0x6d2b79f5) ^ (uint32_t)(expert + 1) * 977u;
    key ^= (uint32_t)(matrix + 1) * 1319u;
    key ^= (uint32_t)(row + 1) * 3571u;
    key ^= (uint32_t)(column + 1) * 7411u;
    return (uint8_t)(synth_v4_mix(key) & 15u);
}

static inline uint8_t synth_v4_scale_code(int expert, int matrix,
                                           int row, int block) {
    uint32_t key = UINT32_C(0xa511e9b3) ^ (uint32_t)(expert + 1) * 2017u;
    key ^= (uint32_t)(matrix + 1) * 2539u;
    key ^= (uint32_t)(row + 1) * 4001u;
    key ^= (uint32_t)(block + 1) * 4421u;
    return (uint8_t)(124u + synth_v4_mix(key) % 5u);
}

static inline void synth_v4_matrix_init(SynthV4Matrix *matrix,
                                        int expert, int matrix_index,
                                        int rows, int columns) {
    const int data_stride = columns / 2;
    const int scale_stride = columns / SYNTH_V4_FP4_BLOCK;
    memset(matrix, 0, sizeof(*matrix));
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column += 2) {
            uint8_t low = synth_v4_weight_code(
                expert, matrix_index, row, column);
            uint8_t high = synth_v4_weight_code(
                expert, matrix_index, row, column + 1);
            matrix->cold_data[(size_t)row * data_stride + column / 2] =
                (uint8_t)(low | (uint8_t)(high << 4));
        }
        for (int block = 0; block < scale_stride; block++)
            matrix->cold_scales[(size_t)row * scale_stride + block] =
                synth_v4_scale_code(expert, matrix_index, row, block);
    }
    synth_v4_rows16_pack(matrix->hot_data, matrix->cold_data,
                         rows, data_stride);
    synth_v4_rows16_pack(matrix->hot_scales, matrix->cold_scales,
                         rows, scale_stride);
    matrix->cold = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        matrix->cold_data, matrix->cold_scales,
        (size_t)rows * data_stride, (size_t)rows * scale_stride,
        rows, columns, 1, SYNTH_V4_FP4_BLOCK
    };
    matrix->hot = matrix->cold;
    matrix->hot.data = matrix->hot_data;
    matrix->hot.scales = matrix->hot_scales;
    matrix->hot.block_rows = SYNTH_V4_ROWS16;
}

static inline void synth_v4_fixture_init(SynthV4Fixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    for (int token = 0; token < SYNTH_V4_TOKENS; token++)
        for (int column = 0; column < SYNTH_V4_HIDDEN; column++) {
            uint32_t key = synth_v4_mix(UINT32_C(0x9e3779b9) ^
                (uint32_t)(token + 1) * 7919u ^
                (uint32_t)(column + 1) * 104729u);
            fixture->inputs[token][column] =
                (float)((int)(key % 97u) - 48) * (1.0f / 32.0f);
        }
    fixture->selected[0][0] = 0;
    fixture->selected[0][1] = 2;
    fixture->selected[1][0] = 1;
    fixture->selected[1][1] = 3;
    fixture->route_weights[0][0] = 0.625f;
    fixture->route_weights[0][1] = 0.375f;
    fixture->route_weights[1][0] = 0.5625f;
    fixture->route_weights[1][1] = 0.4375f;
    for (int expert = 0; expert < SYNTH_V4_EXPERTS; expert++) {
        SynthV4Expert *item = &fixture->experts[expert];
        synth_v4_matrix_init(&item->gate, expert, 0,
                             SYNTH_V4_INTERMEDIATE, SYNTH_V4_HIDDEN);
        synth_v4_matrix_init(&item->down, expert, 1,
                             SYNTH_V4_HIDDEN, SYNTH_V4_INTERMEDIATE);
        synth_v4_matrix_init(&item->up, expert, 2,
                             SYNTH_V4_INTERMEDIATE, SYNTH_V4_HIDDEN);
        synth_v4_matrix_init(&item->reference_gate, expert, 0,
                             SYNTH_V4_INTERMEDIATE,
                             SYNTH_V4_STORAGE_HIDDEN);
        synth_v4_matrix_init(&item->reference_down, expert, 1,
                             SYNTH_V4_STORAGE_HIDDEN,
                             SYNTH_V4_INTERMEDIATE);
        synth_v4_matrix_init(&item->reference_up, expert, 2,
                             SYNTH_V4_INTERMEDIATE,
                             SYNTH_V4_STORAGE_HIDDEN);
        item->cold.key.expert = expert;
        item->cold.gate = item->gate.cold;
        item->cold.down = item->down.cold;
        item->cold.up = item->up.cold;
        item->hot = item->cold;
        item->hot.gate = item->gate.hot;
        item->hot.down = item->down.hot;
        item->hot.up = item->up.hot;
        item->reference = item->cold;
        item->reference.gate = item->reference_gate.cold;
        item->reference.down = item->reference_down.cold;
        item->reference.up = item->reference_up.cold;
    }
}

static inline int synth_v4_matrix_roundtrip(const SynthV4Matrix *matrix) {
    uint8_t data[SYNTH_V4_WEIGHT_BYTES];
    uint8_t scales[SYNTH_V4_SCALE_BYTES];
    size_t rows = (size_t)matrix->cold.rows;
    size_t data_stride = (size_t)matrix->cold.columns / 2;
    size_t scale_stride =
        (size_t)matrix->cold.columns / SYNTH_V4_FP4_BLOCK;
    synth_v4_rows16_unpack(data, matrix->hot_data,
                           rows, data_stride);
    synth_v4_rows16_unpack(scales, matrix->hot_scales,
                           rows, scale_stride);
    return memcmp(data, matrix->cold_data, matrix->cold.data_bytes) == 0 &&
           memcmp(scales, matrix->cold_scales, matrix->cold.scale_bytes) == 0;
}

static inline int synth_v4_fixture_roundtrip(const SynthV4Fixture *fixture) {
    for (int expert = 0; expert < SYNTH_V4_EXPERTS; expert++) {
        const SynthV4Expert *item = &fixture->experts[expert];
        if (!synth_v4_matrix_roundtrip(&item->gate) ||
            !synth_v4_matrix_roundtrip(&item->down) ||
            !synth_v4_matrix_roundtrip(&item->up)) return 0;
    }
    return 1;
}

#endif
