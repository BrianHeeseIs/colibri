// coli_v4_decode.metal -- bit-exact MSL ports of DeepSeek-V4 CPU math primitives.
// Sources of truth: c/quant.h (mx4_lut, mx4_scale)
//                   c/deepseek_v4.c (coli_bf16_round, coli_e4m3fn_decode, sigmoidf_stable)
// Every function here must be bit-identical to its CPU original.
#include <metal_stdlib>
using namespace metal;

// MXFP4 (E2M1) value table. Index 8 is NEGATIVE ZERO (0x80000000), not +0 --
// it must stay distinguishable from index 0 or sign information is lost.
#ifndef COLI_V4_MX4_DEFINED
#define COLI_V4_MX4_DEFINED
constant float coli_v4_mx4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// UE8M0 scale. Bit trick, NOT ldexp: they differ at s=0.
// s=0 -> +0 (2^-127 flushed), s=255 -> +inf. Real checkpoints contain neither.
inline float coli_v4_mx4_scale(uchar s) {
    return as_type<float>((uint)s << 23);
}

// bfloat16 round-to-nearest-EVEN, preserving NaN/Inf untouched.
#endif // COLI_V4_MX4_DEFINED

inline float coli_v4_bf16_round(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    return as_type<float>(bits);
}

// FP8 E4M3FN decode (finite variant: no infinities; 0x7f/0xff are NaN).
inline float coli_v4_e4m3fn_decode(uchar value) {
    int sign     = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return as_type<float>(0x7fc00000u);
    float number;
    if (exponent == 0) number = ldexp((float)mantissa, -9);
    else               number = ldexp(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

// Numerically stable logistic sigmoid, sign-branched exactly as the CPU does.
// precise:: is required -- the fast variants are not correctly rounded.
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
#endif

// Probe entry. Index ranges mirror validation/metal/probe_primitives.m exactly.
kernel void coli_v4_probe_primitives(device const uint *in  [[buffer(0)]],
                                     device float      *out [[buffer(1)]],
                                     uint tid [[thread_position_in_grid]]) {
    const uint NL = 16u, NS = 256u, NE = 256u, NB = 32u, NG = 32u;
    const uint OFF_S = NL, OFF_E = OFF_S + NS, OFF_B = OFF_E + NE, OFF_G = OFF_B + NB;
    if (tid >= OFF_G + NG) return;
    uint v = in[tid];
    float r;
    if      (tid < OFF_S) r = coli_v4_mx4_lut[v & 15u];
    else if (tid < OFF_E) r = coli_v4_mx4_scale((uchar)(v & 255u));
    else if (tid < OFF_B) r = coli_v4_e4m3fn_decode((uchar)(v & 255u));
    else if (tid < OFF_G) r = coli_v4_bf16_round(as_type<float>(v));
    else                  r = coli_v4_sigmoid_stable(as_type<float>(v));
    out[tid] = r;
}

// Elementwise bf16 round over an array. Wraps the proven-exact coli_v4_bf16_round.
kernel void coli_v4_bf16_round_array(device float *data [[buffer(0)]],
                                     constant uint &n   [[buffer(1)]],
                                     uint tid [[thread_position_in_grid]]) {
    if (tid < n) data[tid] = coli_v4_bf16_round(data[tid]);
}
