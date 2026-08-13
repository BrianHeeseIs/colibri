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

// ORDERED, rows16 HOT layout. Weights and scales are repacked so that 16 consecutive
// output rows are interleaved in the innermost dimension (lane = row % 16), giving coalesced
// reads for a 16-row tile. Reference: synth_v4_rows16_pack --
//   packed[(tile*stride + col)*16 + lane] = data[row*stride + col], tile=row/16, lane=row%16.
// Math and accumulation order are IDENTICAL to coli_v4_matmul_mxfp4_ordered; only the weight
// and scale addressing changes. Output must be bit-identical to the cold ordered kernel.
kernel void coli_v4_matmul_mxfp4_ordered_hot(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],   // rows16-packed weights
        device const uchar          *e8s [[buffer(2)]],   // rows16-packed scales
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= D.S * D.O) return;
    int s = (int)tid / D.O;
    int o = (int)tid % D.O;
    int tile = o / 16, lane = o % 16;
    device const float *xs = x + (long)s * D.I;

    float a = 0.0f;
    for (int g = 0; g < D.ng; ++g) {
        int base = g * 32, glen = 32;
        if (base + glen > D.I) glen = D.I - base;
        uchar sc_code = e8s[((long)(tile * D.ng) + g) * 16 + lane];
        float sc = coli_v4_mx4_scale(sc_code);
        float ga = 0.0f;
        for (int i = base; i < base + glen; i += 2) {
            int col = i >> 1;
            uchar byte = q4[((long)(tile * D.rb) + col) * 16 + lane];
            ga += xs[i] * coli_v4_mx4_lut[byte & 0xF];
            if (i + 1 < base + glen) ga += xs[i + 1] * coli_v4_mx4_lut[byte >> 4];
        }
        a += ga * sc;
    }
    y[(long)s * D.O + o] = a;
}

// ---------------------------------------------------------------------------
// D1 FIX: stage x in THREADGROUP memory.
// Problem measured at batch-1 (I=4096, O=2048): every thread reads the ENTIRE input vector
// from device memory. With O=2048 threads that is 2048 x 16 KB = ~32 MB of redundant traffic
// against a weight matrix of only ~4 MB -- x traffic is 8x the weights. That is why the GPU
// lost to the CPU on effective bandwidth despite far more compute.
// Staging x once per threadgroup cuts that to a single 16 KB read per threadgroup.
//
// BIT-EXACTNESS IS PRESERVED EXACTLY: same values, same two-level accumulation order
// (serial within a 32-col group, then a += ga*sc serially across groups). This is purely a
// data-placement change. It cannot alter rounding.
//
// Grid MUST be 2D: (O, S) with threadgroup (TG,1,1), so every thread in a threadgroup shares
// one token s and therefore one x vector.
// CONSTRAINT: I <= COLI_V4_XCACHE_MAX (4096 floats = 16 KB of the 32 KB budget). V4 uses
// I=4096 for gate/up and I=2048 for down, so this covers the model. Host must check.
#define COLI_V4_XCACHE_MAX 4096

kernel void coli_v4_matmul_mxfp4_ordered_xcache(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],
        device const uchar          *e8s [[buffer(2)]],
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint2 gid  [[thread_position_in_grid]],
        uint2 ltid [[thread_position_in_threadgroup]],
        uint2 tgsz [[threads_per_threadgroup]]) {
    threadgroup float xcache[COLI_V4_XCACHE_MAX];
    int s = (int)gid.y, o = (int)gid.x;
    device const float *xs = x + (long)s * D.I;
    for (int i = (int)ltid.x; i < D.I; i += (int)tgsz.x) xcache[i] = xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= D.O || s >= D.S) return;

    device const uchar *w   = q4  + (long)o * D.rb;
    device const uchar *scl = e8s + (long)o * D.ng;
    float a = 0.0f;
    for (int g = 0; g < D.ng; ++g) {
        int base = g * 32, glen = 32;
        if (base + glen > D.I) glen = D.I - base;
        float sc = coli_v4_mx4_scale(scl[g]);
        float ga = 0.0f;
        for (int i = base; i < base + glen; i += 2) {
            uchar byte = w[i >> 1];
            ga += xcache[i] * coli_v4_mx4_lut[byte & 0xF];
            if (i + 1 < base + glen) ga += xcache[i + 1] * coli_v4_mx4_lut[byte >> 4];
        }
        a += ga * sc;
    }
    y[(long)s * D.O + o] = a;
}

// Same staging, rows16 HOT addressing (coalesced weight reads + cached x = both fixes together).
kernel void coli_v4_matmul_mxfp4_ordered_hot_xcache(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],
        device const uchar          *e8s [[buffer(2)]],
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint2 gid  [[thread_position_in_grid]],
        uint2 ltid [[thread_position_in_threadgroup]],
        uint2 tgsz [[threads_per_threadgroup]]) {
    threadgroup float xcache[COLI_V4_XCACHE_MAX];
    int s = (int)gid.y, o = (int)gid.x;
    device const float *xs = x + (long)s * D.I;
    for (int i = (int)ltid.x; i < D.I; i += (int)tgsz.x) xcache[i] = xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= D.O || s >= D.S) return;

    int tile = o / 16, lane = o % 16;
    float a = 0.0f;
    for (int g = 0; g < D.ng; ++g) {
        int base = g * 32, glen = 32;
        if (base + glen > D.I) glen = D.I - base;
        float sc = coli_v4_mx4_scale(e8s[((long)(tile * D.ng) + g) * 16 + lane]);
        float ga = 0.0f;
        for (int i = base; i < base + glen; i += 2) {
            uchar byte = q4[((long)(tile * D.rb) + (i >> 1)) * 16 + lane];
            ga += xcache[i] * coli_v4_mx4_lut[byte & 0xF];
            if (i + 1 < base + glen) ga += xcache[i + 1] * coli_v4_mx4_lut[byte >> 4];
        }
        a += ga * sc;
    }
    y[(long)s * D.O + o] = a;
}
