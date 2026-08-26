// coli_v4_moe.metal -- fused bit-exact DeepSeek-V4 cold routed-expert forward.
// One threadgroup performs input FP8 QDQ, ordered gate/up MXFP4 matvecs, bf16
// rounding, SwiGLU, weighted FP8 QDQ, ordered down MXFP4 matvec, and final bf16.
#include <metal_stdlib>
using namespace metal;

#ifndef COLI_V4_MX4_DEFINED
#define COLI_V4_MX4_DEFINED
constant float coli_v4_mx4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
inline float coli_v4_mx4_scale(uchar scale) {
    return as_type<float>((uint)scale << 23);
}
#endif // COLI_V4_MX4_DEFINED

#ifndef COLI_V4_SIGMOID_DEFINED
#define COLI_V4_SIGMOID_DEFINED
inline float coli_v4_sigmoid_stable(float value) {
    if (value >= 0.0f) {
        float decay = precise::exp(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = precise::exp(value);
    return growth / (1.0f + growth);
}
#endif // COLI_V4_SIGMOID_DEFINED

#ifndef COLI_V4_MOE_HELPERS_DEFINED
#define COLI_V4_MOE_HELPERS_DEFINED
inline float coli_v4_moe_bf16_round(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    return as_type<float>(bits);
}

inline int coli_v4_moe_ceil_log2_positive(float value) {
    int exponent;
    float fraction = frexp(value, exponent);
    return fraction == 0.5f ? exponent - 1 : exponent;
}

inline float coli_v4_moe_e8m0_decode(uchar value) {
    return ldexp(1.0f, (int)value - 127);
}

inline float coli_v4_moe_e4m3fn_decode(uchar value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7)
        return as_type<float>(0x7fc00000u);
    float number;
    if (exponent == 0)
        number = ldexp((float)mantissa, -9);
    else
        number = ldexp(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

inline uchar coli_v4_moe_e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabs(value);
    if (magnitude == 0.0f) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f)
        return (uchar)((negative ? 0x80 : 0) | 0x7e);

    uchar best = 0;
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uchar rounded = (uchar)scaled;
        float fraction = scaled - (float)rounded;
        if (fraction > 0.5f ||
            (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {
        uint bits = as_type<uint>(magnitude);
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint significand = 0x800000u | (bits & 0x7fffffu);
        uint rounded = significand >> 20;
        uint remainder = significand & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) {
            rounded = 8u;
            exponent++;
        }
        best = (uchar)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uchar)(best | (negative ? 0x80 : 0));
}

inline float coli_v4_moe_ordered_matvec(
        threadgroup const float *input,
        device const uchar *q4,
        device const uchar *e8s,
        int input_size,
        int row_bytes,
        int groups,
        int row) {
    device const uchar *weights = q4 + (long)row * row_bytes;
    device const uchar *scales = e8s + (long)row * groups;
    float accumulator = 0.0f;
    for (int group = 0; group < groups; ++group) {
        int base = group * 32;
        int group_length = 32;
        if (base + group_length > input_size)
            group_length = input_size - base;
        float scale = coli_v4_mx4_scale(scales[group]);
        float group_accumulator = 0.0f;
        for (int column = base;
             column < base + group_length;
             column += 2) {
            uchar packed = weights[column >> 1];
            group_accumulator +=
                input[column] * coli_v4_mx4_lut[packed & 0x0f];
            if (column + 1 < base + group_length)
                group_accumulator +=
                    input[column + 1] * coli_v4_mx4_lut[packed >> 4];
        }
        accumulator += group_accumulator * scale;
    }
    return accumulator;
}

inline void coli_v4_moe_fp8_qdq_block(
        threadgroup float *output,
        threadgroup const float *input,
        int base,
        int count) {
    float maximum = 0.0f;
    for (int index = 0; index < count; ++index)
        maximum = fmax(maximum, fabs(input[base + index]));
    maximum = fmax(maximum, 1e-4f);
    int scale_exponent =
        coli_v4_moe_ceil_log2_positive(maximum / 448.0f);
    scale_exponent = clamp(scale_exponent, -127, 127);
    uchar encoded_scale = (uchar)(scale_exponent + 127);
    float scale = coli_v4_moe_e8m0_decode(encoded_scale);
    for (int index = 0; index < count; ++index) {
        float normalized = fmax(
            -448.0f, fmin(448.0f, input[base + index] / scale));
        output[base + index] = coli_v4_moe_e4m3fn_decode(
            coli_v4_moe_e4m3fn_encode(normalized)) * scale;
    }
}
#endif // COLI_V4_MOE_HELPERS_DEFINED

struct ColiV4MoeParams {
    int hidden;
    int intermediate;
    int gate_row_bytes;
    int gate_groups;
    int down_row_bytes;
    int down_groups;
    int fp8_block;
    float route_weight;
    float swiglu_limit;
};

#define COLI_V4_MOE_MAX_HIDDEN 4096
#define COLI_V4_MOE_MAX_INTERMEDIATE 2048

kernel void coli_v4_moe_expert_fp4_ordered_cold(
        device const float *input          [[buffer(0)]],
        device const uchar *gate_q4        [[buffer(1)]],
        device const uchar *gate_scales    [[buffer(2)]],
        device const uchar *up_q4          [[buffer(3)]],
        device const uchar *up_scales      [[buffer(4)]],
        device const uchar *down_q4        [[buffer(5)]],
        device const uchar *down_scales    [[buffer(6)]],
        device float *output               [[buffer(7)]],
        constant ColiV4MoeParams &params   [[buffer(8)]],
        uint local_id [[thread_position_in_threadgroup]],
        uint threadgroup_size [[threads_per_threadgroup]]) {
    // Production dimensions consume the full 32 KB threadgroup budget:
    // input_qdq is 4096 floats (16 KB), while gate_up is 2 x 2048 floats
    // (16 KB). Phase barriers let input_qdq later become weighted/down_input.
    threadgroup float input_qdq[COLI_V4_MOE_MAX_HIDDEN];
    threadgroup float gate_up[2 * COLI_V4_MOE_MAX_INTERMEDIATE];

    int input_blocks =
        (params.hidden + params.fp8_block - 1) / params.fp8_block;
    for (int block = (int)local_id;
         block < input_blocks;
         block += (int)threadgroup_size) {
        int base = block * params.fp8_block;
        int count = params.hidden - base < params.fp8_block
            ? params.hidden - base : params.fp8_block;
        float maximum = 0.0f;
        for (int index = 0; index < count; ++index)
            maximum = fmax(maximum, fabs(input[base + index]));
        maximum = fmax(maximum, 1e-4f);
        int scale_exponent =
            coli_v4_moe_ceil_log2_positive(maximum / 448.0f);
        scale_exponent = clamp(scale_exponent, -127, 127);
        uchar encoded_scale = (uchar)(scale_exponent + 127);
        float scale = coli_v4_moe_e8m0_decode(encoded_scale);
        for (int index = 0; index < count; ++index) {
            float normalized = fmax(
                -448.0f, fmin(448.0f, input[base + index] / scale));
            input_qdq[base + index] = coli_v4_moe_e4m3fn_decode(
                coli_v4_moe_e4m3fn_encode(normalized)) * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = (int)local_id;
         row < params.intermediate;
         row += (int)threadgroup_size) {
        float gate = coli_v4_moe_ordered_matvec(
            input_qdq, gate_q4, gate_scales, params.hidden,
            params.gate_row_bytes, params.gate_groups, row);
        float up = coli_v4_moe_ordered_matvec(
            input_qdq, up_q4, up_scales, params.hidden,
            params.gate_row_bytes, params.gate_groups, row);
        gate_up[2 * row] = coli_v4_moe_bf16_round(gate);
        gate_up[2 * row + 1] = coli_v4_moe_bf16_round(up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int down_input_blocks =
        (params.intermediate + params.fp8_block - 1) / params.fp8_block;
    for (int block = (int)local_id;
         block < down_input_blocks;
         block += (int)threadgroup_size) {
        int base = block * params.fp8_block;
        int count = params.intermediate - base < params.fp8_block
            ? params.intermediate - base : params.fp8_block;
        for (int index = 0; index < count; ++index) {
            int offset = base + index;
            float gate = gate_up[2 * offset];
            float up = gate_up[2 * offset + 1];
            if (params.swiglu_limit > 0.0f) {
                gate = fmin(gate, params.swiglu_limit);
                up = fmax(-params.swiglu_limit,
                          fmin(up, params.swiglu_limit));
            }
            float activated =
                (gate * coli_v4_sigmoid_stable(gate)) * up;
            input_qdq[offset] = coli_v4_moe_bf16_round(
                activated * params.route_weight);
        }
        coli_v4_moe_fp8_qdq_block(input_qdq, input_qdq, base, count);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = (int)local_id;
         row < params.hidden;
         row += (int)threadgroup_size) {
        float down = coli_v4_moe_ordered_matvec(
            input_qdq, down_q4, down_scales, params.intermediate,
            params.down_row_bytes, params.down_groups, row);
        output[row] = coli_v4_moe_bf16_round(down);
    }
}

struct ColiV4MoeFusedArgs {
    device const uchar *gate_q4_0 [[id(0)]];
    device const uchar *gate_q4_1 [[id(1)]];
    device const uchar *gate_q4_2 [[id(2)]];
    device const uchar *gate_q4_3 [[id(3)]];
    device const uchar *gate_q4_4 [[id(4)]];
    device const uchar *gate_q4_5 [[id(5)]];
    device const uchar *gate_scales_0 [[id(6)]];
    device const uchar *gate_scales_1 [[id(7)]];
    device const uchar *gate_scales_2 [[id(8)]];
    device const uchar *gate_scales_3 [[id(9)]];
    device const uchar *gate_scales_4 [[id(10)]];
    device const uchar *gate_scales_5 [[id(11)]];
    device const uchar *up_q4_0 [[id(12)]];
    device const uchar *up_q4_1 [[id(13)]];
    device const uchar *up_q4_2 [[id(14)]];
    device const uchar *up_q4_3 [[id(15)]];
    device const uchar *up_q4_4 [[id(16)]];
    device const uchar *up_q4_5 [[id(17)]];
    device const uchar *up_scales_0 [[id(18)]];
    device const uchar *up_scales_1 [[id(19)]];
    device const uchar *up_scales_2 [[id(20)]];
    device const uchar *up_scales_3 [[id(21)]];
    device const uchar *up_scales_4 [[id(22)]];
    device const uchar *up_scales_5 [[id(23)]];
    device const uchar *down_q4_0 [[id(24)]];
    device const uchar *down_q4_1 [[id(25)]];
    device const uchar *down_q4_2 [[id(26)]];
    device const uchar *down_q4_3 [[id(27)]];
    device const uchar *down_q4_4 [[id(28)]];
    device const uchar *down_q4_5 [[id(29)]];
    device const uchar *down_scales_0 [[id(30)]];
    device const uchar *down_scales_1 [[id(31)]];
    device const uchar *down_scales_2 [[id(32)]];
    device const uchar *down_scales_3 [[id(33)]];
    device const uchar *down_scales_4 [[id(34)]];
    device const uchar *down_scales_5 [[id(35)]];
};

struct ColiV4MoeFusedParams {
    int expert_count;
    int hidden;
    int intermediate;
    int gate_row_bytes;
    int gate_groups;
    int down_row_bytes;
    int down_groups;
    int fp8_block;
    float swiglu_limit;
    float route_weights[6];
    ulong gate_q4_offsets[6];
    ulong gate_scales_offsets[6];
    ulong up_q4_offsets[6];
    ulong up_scales_offsets[6];
    ulong down_q4_offsets[6];
    ulong down_scales_offsets[6];
};

inline device const uchar *coli_v4_moe_fused_select(
        constant ColiV4MoeFusedArgs &args, int expert, int tensor) {
#define COLI_V4_FUSED_SELECT(field) \
    switch (expert) { \
    case 0: return args.field##_0; case 1: return args.field##_1; \
    case 2: return args.field##_2; case 3: return args.field##_3; \
    case 4: return args.field##_4; default: return args.field##_5; \
    }
    switch (tensor) {
    case 0: COLI_V4_FUSED_SELECT(gate_q4)
    case 1: COLI_V4_FUSED_SELECT(gate_scales)
    case 2: COLI_V4_FUSED_SELECT(up_q4)
    case 3: COLI_V4_FUSED_SELECT(up_scales)
    case 4: COLI_V4_FUSED_SELECT(down_q4)
    default: COLI_V4_FUSED_SELECT(down_scales)
    }
#undef COLI_V4_FUSED_SELECT
}

kernel void coli_v4_moe_experts_fp4_ordered_fused(
        device const float *input [[buffer(0)]],
        constant ColiV4MoeFusedArgs &args [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant ColiV4MoeFusedParams &params [[buffer(3)]],
        uint local_id [[thread_position_in_threadgroup]],
        uint threadgroup_size [[threads_per_threadgroup]]) {
    threadgroup float input_qdq[COLI_V4_MOE_MAX_HIDDEN];
    threadgroup float gate_up_or_down_input[2 * COLI_V4_MOE_MAX_INTERMEDIATE];

    for (int row = (int)local_id; row < params.hidden;
         row += (int)threadgroup_size)
        output[row] = 0.0f;
    for (int index = (int)local_id; index < params.hidden;
         index += (int)threadgroup_size)
        input_qdq[index] = input[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int input_blocks = (params.hidden + params.fp8_block - 1) / params.fp8_block;
    for (int block = (int)local_id; block < input_blocks;
         block += (int)threadgroup_size) {
        int base = block * params.fp8_block;
        int count = params.hidden - base < params.fp8_block
            ? params.hidden - base : params.fp8_block;
        coli_v4_moe_fp8_qdq_block(input_qdq, input_qdq, base, count);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int expert = 0; expert < params.expert_count; expert++) {
        device const uchar *gate_q4 = coli_v4_moe_fused_select(args, expert, 0) +
            params.gate_q4_offsets[expert];
        device const uchar *gate_scales = coli_v4_moe_fused_select(args, expert, 1) +
            params.gate_scales_offsets[expert];
        device const uchar *up_q4 = coli_v4_moe_fused_select(args, expert, 2) +
            params.up_q4_offsets[expert];
        device const uchar *up_scales = coli_v4_moe_fused_select(args, expert, 3) +
            params.up_scales_offsets[expert];
        device const uchar *down_q4 = coli_v4_moe_fused_select(args, expert, 4) +
            params.down_q4_offsets[expert];
        device const uchar *down_scales = coli_v4_moe_fused_select(args, expert, 5) +
            params.down_scales_offsets[expert];
        for (int row = (int)local_id; row < params.intermediate;
             row += (int)threadgroup_size) {
            float gate = coli_v4_moe_ordered_matvec(
                input_qdq, gate_q4, gate_scales, params.hidden,
                params.gate_row_bytes, params.gate_groups, row);
            float up = coli_v4_moe_ordered_matvec(
                input_qdq, up_q4, up_scales, params.hidden,
                params.gate_row_bytes, params.gate_groups, row);
            gate_up_or_down_input[2 * row] = coli_v4_moe_bf16_round(gate);
            gate_up_or_down_input[2 * row + 1] = coli_v4_moe_bf16_round(up);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int down_input_blocks =
            (params.intermediate + params.fp8_block - 1) / params.fp8_block;
        for (int block = (int)local_id; block < down_input_blocks;
             block += (int)threadgroup_size) {
            int base = block * params.fp8_block;
            int count = params.intermediate - base < params.fp8_block
                ? params.intermediate - base : params.fp8_block;
            for (int index = 0; index < count; index++) {
                int offset = base + index;
                float gate = gate_up_or_down_input[2 * offset];
                float up = gate_up_or_down_input[2 * offset + 1];
                if (params.swiglu_limit > 0.0f) {
                    gate = fmin(gate, params.swiglu_limit);
                    up = fmax(-params.swiglu_limit, fmin(up, params.swiglu_limit));
                }
                float activated =
                    (gate * coli_v4_sigmoid_stable(gate)) * up;
                gate_up_or_down_input[offset] = coli_v4_moe_bf16_round(
                    activated * params.route_weights[expert]);
            }
            coli_v4_moe_fp8_qdq_block(
                gate_up_or_down_input, gate_up_or_down_input, base, count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int row = (int)local_id; row < params.hidden;
             row += (int)threadgroup_size) {
            float down = coli_v4_moe_ordered_matvec(
                gate_up_or_down_input, down_q4, down_scales, params.intermediate,
                params.down_row_bytes, params.down_groups, row);
            output[row] += coli_v4_moe_bf16_round(down);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
