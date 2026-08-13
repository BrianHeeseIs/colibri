#include <metal_stdlib>
using namespace metal;

#ifndef COLI_V4_WEIGHTED_BF16_DEFINED
#define COLI_V4_WEIGHTED_BF16_DEFINED
inline float coli_v4_weighted_bf16_round(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    return as_type<float>(bits & 0xffff0000u);
}

struct ColiV4WeightedBf16Params { int n; float w; };

kernel void coli_v4_weighted_bf16(
        device float *out [[buffer(0)]],
        device const float *in [[buffer(1)]],
        constant ColiV4WeightedBf16Params &P [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if ((int)tid < P.n) out[tid] = coli_v4_weighted_bf16_round(in[tid] * P.w);
}
#endif // COLI_V4_WEIGHTED_BF16_DEFINED
