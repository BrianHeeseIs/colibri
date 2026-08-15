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
// coli_v4_matmul.metal -- GPU port of matmul_mxfp4.
// Reference: c/quant.h:1401-1412 SCALAR path. That is the branch arm64 takes
// (the AVX2 branch above it is #ifdef __AVX2__ and never compiles on Apple silicon).
//
// The accumulation is TWO-LEVEL and the order is load-bearing:
//   per 32-column group: ga accumulates serially over columns
//   then:                a += ga * sc      <- scale applied ONCE per group, AFTER it closes
// A flat reduction over I rounds differently and is NOT equivalent.
#include <metal_stdlib>
using namespace metal;

#ifndef COLI_V4_MX4_DEFINED
#define COLI_V4_MX4_DEFINED
constant float coli_v4_mx4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
inline float coli_v4_mx4_scale(uchar s) { return as_type<float>((uint)s << 23); }
#endif // COLI_V4_MX4_DEFINED

struct ColiV4MatmulDims { int S, I, O, rb, ng; };

// ORDERED: one thread per (s,o). Reproduces the CPU loop nest exactly, including
// nibble order (low = even column, high = odd) and the guarded odd-column tail.
kernel void coli_v4_matmul_mxfp4_ordered(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],
        device const uchar          *e8s [[buffer(2)]],
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= D.S * D.O) return;
    int s = (int)tid / D.O;
    int o = (int)tid % D.O;
    device const uchar *w   = q4  + (long)o * D.rb;
    device const uchar *scl = e8s + (long)o * D.ng;
    device const float *xs  = x   + (long)s * D.I;

    float a = 0.0f;
    for (int g = 0; g < D.ng; ++g) {
        int base = g * 32, glen = 32;
        if (base + glen > D.I) glen = D.I - base;
        float sc = coli_v4_mx4_scale(scl[g]);
        float ga = 0.0f;
        for (int i = base; i < base + glen; i += 2) {
            uchar byte = w[i >> 1];
            ga += xs[i] * coli_v4_mx4_lut[byte & 0xF];
            if (i + 1 < base + glen) ga += xs[i + 1] * coli_v4_mx4_lut[byte >> 4];
        }
        a += ga * sc;
    }
    y[(long)s * D.O + o] = a;
}

// SIMD: one simdgroup per (s,o); lane L owns column base+L.
// V4 block_columns == 32 == threadExecutionWidth, so one FP4 group maps to one simdgroup.
// Group-level scale structure is preserved; only the intra-group summation order changes.
kernel void coli_v4_matmul_mxfp4_simd(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],
        device const uchar          *e8s [[buffer(2)]],
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    int lane = (int)gid.x;
    int idx  = (int)gid.y;
    if (idx >= D.S * D.O) return;
    int s = idx / D.O;
    int o = idx % D.O;
    device const uchar *w   = q4  + (long)o * D.rb;
    device const uchar *scl = e8s + (long)o * D.ng;
    device const float *xs  = x   + (long)s * D.I;

    float a = 0.0f;
    for (int g = 0; g < D.ng; ++g) {
        int base = g * 32, glen = 32;
        if (base + glen > D.I) glen = D.I - base;
        int col = base + lane;
        float v = 0.0f;
        if (lane < glen) {
            uchar byte = w[col >> 1];
            uint  nib  = (col & 1) ? (uint)(byte >> 4) : (uint)(byte & 0xF);
            v = xs[col] * coli_v4_mx4_lut[nib];
        }
        float ga = simd_sum(v);              // uniform: every lane participates
        a += ga * coli_v4_mx4_scale(scl[g]);
    }
    if (lane == 0) y[(long)s * D.O + o] = a;
}
// coli_v4_swiglu.metal -- GPU port of coli_v4_swiglu (c/deepseek_v4.c:1383-1396).
// NOTE the asymmetric clamp, which is deliberate in the CPU source and must be preserved:
//   gate is clamped ONLY FROM ABOVE      -> fmin(gate, limit)
//   up   is clamped on BOTH sides        -> fmax(-limit, fmin(up, limit))
// Multiplication associates left-to-right: (gate * sigmoid(gate)) * up.
#include <metal_stdlib>
using namespace metal;

#ifndef COLI_V4_SIGMOID_DEFINED
#define COLI_V4_SIGMOID_DEFINED
inline float coli_v4_sigmoid_stable(float value) {
    if (value >= 0.0f) { float decay = precise::exp(-value); return 1.0f / (1.0f + decay); }
    float growth = precise::exp(value); return growth / (1.0f + growth);
}
#endif // COLI_V4_SIGMOID_DEFINED

struct ColiV4SwigluParams { int dimension; float limit; };

kernel void coli_v4_swiglu(device float                 *output [[buffer(0)]],
                           device const float           *gate   [[buffer(1)]],
                           device const float           *up     [[buffer(2)]],
                           constant ColiV4SwigluParams  &P      [[buffer(3)]],
                           uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= P.dimension) return;
    float gate_value = gate[tid];
    float up_value   = up[tid];
    if (P.limit > 0.0f) {
        gate_value = fmin(gate_value, P.limit);
        up_value   = fmax(-P.limit, fmin(up_value, P.limit));
    }
    output[tid] = gate_value * coli_v4_sigmoid_stable(gate_value) * up_value;
}
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

kernel void coli_v4_fp8_qdq(device float               *output [[buffer(0)]],
                            device uchar               *scales [[buffer(1)]],
                            device const float         *input  [[buffer(2)]],
                            constant ColiV4QdqParams   &P      [[buffer(3)]],
                            uint blk [[thread_position_in_grid]]) {
    int base = (int)blk * P.block_size;
    if (base >= P.length) return;
    int count = (P.length - base < P.block_size) ? (P.length - base) : P.block_size;

    float maximum = 0.0f;
    for (int i = 0; i < count; ++i) maximum = fmax(maximum, fabs(input[base + i]));
    maximum = fmax(maximum, 1e-4f);
    int se = coli_v4_ceil_log2_positive(maximum / 448.0f);
    se = clamp(se, -127, 127);
    uchar enc = (uchar)(se + 127);
    float scale = coli_v4_e8m0_decode(enc);
    scales[base / P.block_size] = enc;

    for (int i = 0; i < count; ++i) {
        float nrm = fmax(-448.0f, fmin(448.0f, input[base + i] / scale));
        output[base + i] = coli_v4_e4m3fn_decode_local(coli_v4_e4m3fn_encode(nrm)) * scale;
    }
}
