// coli_v4_fp8qdq.metal -- GPU port of coli_fp8_activation_qdq_ref (c/deepseek_v4.c:10160).
// Steps 1 and 5 of the MoE expert chain: quantize activations to FP8 E4M3 in blocks of
// block_size (128 in V4), with a per-block UE8M0 scale, then dequantize.
//
// SUBTLETY: the per-block scale uses coli_e8m0_decode = ldexp(1, s-127), which is NOT the
// mx4_scale bit-trick. They diverge at s=0 (2^-127 vs +0). Do not substitute one for the other.
// One thread owns one block so the max-reduction runs in exact CPU order.
#include <metal_stdlib>
using namespace metal;

inline int coli_v4_ceil_log2_positive(float value) {
    int e; float fr = frexp(value, e);
    return fr == 0.5f ? e - 1 : e;
}
inline float coli_v4_e8m0_decode(uchar v) {          // NaN case unreachable for scales here
    return ldexp(1.0f, (int)v - 127);
}
inline float coli_v4_e4m3fn_decode_local(uchar value) {
    int sign = value >> 7, exponent = (value >> 3) & 15, mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return as_type<float>(0x7fc00000u);
    float number;
    if (exponent == 0) number = ldexp((float)mantissa, -9);
    else               number = ldexp(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}
inline uchar coli_v4_e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int   negative  = signbit(value) != 0;
    float magnitude = fabs(value);
    if (magnitude == 0.0f) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uchar)((negative ? 0x80 : 0) | 0x7e);
    uchar best = 0;
    if (magnitude < 0.015625f) {                     // subnormal E4M3 region
        float scaled = magnitude * 512.0f;
        uchar rounded = (uchar)scaled;
        float fraction = scaled - (float)rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {                                         // normal region, round-to-nearest-even
        uint bits = as_type<uint>(magnitude);
        int  exponent    = (int)((bits >> 23) & 0xff) - 127;
        uint significand = 0x800000u | (bits & 0x7fffffu);
        uint rounded     = significand >> 20;
        uint remainder   = significand & 0xfffffu;
        if (remainder > 0x80000u || (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) { rounded = 8u; exponent++; }
        best = (uchar)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uchar)(best | (negative ? 0x80 : 0));
}

struct ColiV4QdqParams { int length; int block_size; };

kernel void coli_v4_fp8_qdq(device float *output [[buffer(0)]],
                             device uchar *scales [[buffer(1)]],
                             device const float *input [[buffer(2)]],
                             constant ColiV4QdqParams &P [[buffer(3)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint blk [[threadgroup_position_in_grid]]) {
    int base = (int)blk * P.block_size;
    if (base >= P.length) return;
    int count = (P.length - base < P.block_size) ? (P.length - base) : P.block_size;
    threadgroup float maxima[128];
    maxima[tid] = tid < (uint)count ? fabs(input[base + tid]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride; stride >>= 1) {
        if (tid < stride) maxima[tid] = fmax(maxima[tid], maxima[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = fmax(maxima[0], 1e-4f);
    int se = coli_v4_ceil_log2_positive(maximum / 448.0f);
    se = clamp(se, -127, 127);
    uchar enc = (uchar)(se + 127);
    float scale = coli_v4_e8m0_decode(enc);
    if (tid == 0) scales[base / P.block_size] = enc;
    if (tid < (uint)count) {
        int i = (int)tid;
        float nrm = fmax(-448.0f, fmin(448.0f, input[base + i] / scale));
        output[base + i] = coli_v4_e4m3fn_decode_local(coli_v4_e4m3fn_encode(nrm)) * scale;
    }
}
