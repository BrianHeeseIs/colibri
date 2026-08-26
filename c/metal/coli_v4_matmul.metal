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
//
// NOT BIT-EXACT. simd_sum() is a tree reduction; the CPU reference sums the 32 columns
// serially. Measured against the scalar reference by validation/metal/probe_simd_parity:
// max relative error ~1e-3 on production shapes (the outer `a += ga*sc` accumulation over
// 128 groups whose UE8M0 scales span many exponents amplifies the ~1e-7 inner difference).
// Use only behind a task-level capability gate. For a BIT-EXACT simdgroup kernel with the
// same occupancy, see coli_v4_matmul_mxfp4_simd_exact below.
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

// SIMD-EXACT: bit-identical to coli_v4_matmul_mxfp4_ordered, with the simdgroup spread over
// GROUPS rather than over columns.
//
// The reference accumulation is two-level and BOTH levels are strictly serial:
//     inner: ga = 0; ga = fma(x[base+0], w0, ga); fma(x[base+1], w1, ga); ...   (32 columns)
//     outer: a  = 0; a  = fma(ga_0, sc_0, a);     fma(ga_1, sc_1, a);  ...      (ng groups)
// (each `+=` of a product is contracted into one fma, so the product is never separately
// rounded -- reproducing that is what bit-exactness turns on).
//
// A previous cut of this kernel put lane L on COLUMN base+L and replayed the inner sum with
// per-column shuffles. That is bit-exact but was measured at 1.374 ms vs 0.379 ms for
// ordered_xcache on gate|up S=1 -- 3.6x SLOWER. It keeps the full 4096-deep dependent FMA chain
// AND adds two shuffles per link, with 31 of 32 lanes doing redundant work. Recorded as dead.
//
// This version instead puts lane L on GROUP gb+L. Each lane runs ONE group's 32-FMA inner chain
// privately -- serial, therefore exact -- and all 32 groups run concurrently. Only the outer
// accumulation needs shuffles, and it needs ng of them instead of I. The dependent chain falls
// from ~4096 links to ~(32 + ng). Both factors are shuffled so the outer link stays a single
// fma, matching the reference rounding.
//
// Memory behaviour is also better than the column split: lane L reads w[16*(gb+L) .. +16), so
// consecutive lanes read consecutive 16-byte spans -- one contiguous 512-byte burst per round.
kernel void coli_v4_matmul_mxfp4_simd_exact(
        device const float          *x   [[buffer(0)]],
        device const uchar          *q4  [[buffer(1)]],
        device const uchar          *e8s [[buffer(2)]],
        device float                *y   [[buffer(3)]],
        constant ColiV4MatmulDims   &D   [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    int lane = (int)gid.x;
    int idx  = (int)gid.y;
    if (idx >= D.S * D.O) return;          // uniform: whole simdgroup shares idx
    int s = idx / D.O;
    int o = idx % D.O;
    device const uchar *w   = q4  + (long)o * D.rb;
    device const uchar *scl = e8s + (long)o * D.ng;
    device const float *xs  = x   + (long)s * D.I;

    float a = 0.0f;
    for (int gb = 0; gb < D.ng; gb += 32) {
        int g  = gb + lane;
        float ga = 0.0f, sc = 0.0f;
        if (g < D.ng) {
            int base = g * 32, glen = 32;
            if (base + glen > D.I) glen = D.I - base;
            for (int i = base; i < base + glen; i += 2) {
                uchar byte = w[i >> 1];
                ga = fma(xs[i], coli_v4_mx4_lut[byte & 0xF], ga);
                if (i + 1 < base + glen)
                    ga = fma(xs[i + 1], coli_v4_mx4_lut[byte >> 4], ga);
            }
            sc = coli_v4_mx4_scale(scl[g]);
        }
        int n = D.ng - gb; if (n > 32) n = 32;   // uniform across the simdgroup
        for (int l = 0; l < n; ++l)
            a = fma(simd_shuffle(ga, (ushort)l), simd_shuffle(sc, (ushort)l), a);
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
        /* Scale is applied PER COLUMN, not factored out per 32-group.  This kernel must
         * match coli_fp4_dual_matvec_rows16_v10's NEON inner loop, which does
         * sums += (x * w) * scale for every column.  Both forms are individually exact
         * (UE8M0 scales are powers of two) but they build DIFFERENT summation trees:
         * factoring the scale out sums 32 unscaled products before rounding, while the
         * CPU rounds each scaled product into the running total.  Measured divergence
         * between the two orders is 96.25% of dot products, which is why the rows16 path
         * failed golden while the cold kernel - whose per-group form matches the scalar
         * matmul_mxfp4 reference - passes.  Do not "optimise" this back into a per-group
         * multiply; it is bit-exactness, not arithmetic waste. */
        for (int i = base; i < base + glen; i += 2) {
            uchar byte = q4[((long)(tile * D.rb) + (i >> 1)) * 16 + lane];
            a += (xcache[i] * coli_v4_mx4_lut[byte & 0xF]) * sc;
            if (i + 1 < base + glen) a += (xcache[i + 1] * coli_v4_mx4_lut[byte >> 4]) * sc;
        }
    }
    y[(long)s * D.O + o] = a;
}
