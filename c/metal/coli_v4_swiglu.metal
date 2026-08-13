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
