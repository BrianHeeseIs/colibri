/* Bit-exact batched fp8 (E4M3 + F32 128x128 block scales) matmul, mirroring quant.h matmul_fp8.
 *
 * CONTRACT (proven 0 ULP over 2,457,600 outputs across the four real attention shapes, E68):
 *   per output (o,s):
 *     for each 128-column block bi:
 *         float acc = 0;  for i in block:  acc += LUT[w[o][i]] * x[s][i];   <- FLOAT, in order
 *         a += (double)acc * scale[o/128][bi];                             <- DOUBLE on CPU
 *     y[s][o] = (float)a;
 *
 * Metal has NO fp64, so the outer accumulation is emulated with double-float (Dekker hi/lo).
 * THIS REQUIRES -fno-fast-math: fast math algebraically simplifies two_sum's error term
 * (a-(s-bb))+(b-bb) to ZERO and silently destroys the emulation. See Makefile METALCFLAGS.
 *
 * The E4M3 decode table is uploaded by the host from the engine's own e4m3_decode(0..255)
 * so the shader can never drift from the C table.
 */
#include <metal_stdlib>
using namespace metal;

struct coli_df { float hi; float lo; };

static inline coli_df coli_two_sum(float a, float b) {
    float s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);
    return coli_df{ s, e };
}
static inline coli_df coli_two_prod(float a, float b) {
    float p = a * b, e = fma(a, b, -p);
    return coli_df{ p, e };
}
static inline coli_df coli_df_add_prod(coli_df acc, float a, float b) {
    coli_df p = coli_two_prod(a, b);
    coli_df s = coli_two_sum(acc.hi, p.hi);
    float lo = s.lo + (acc.lo + p.lo);
    return coli_two_sum(s.hi, lo);
}

struct ColiV4Fp8Dims { uint O; uint I; uint S; uint nblkI; };

kernel void coli_v4_fp8_matmul_batch(
        device const uchar         *weights [[buffer(0)]],   /* [O][I] e4m3 bytes        */
        device const float         *scales  [[buffer(1)]],   /* [ceil(O/128)][nblkI] f32 */
        device const float         *inputs  [[buffer(2)]],   /* [S][I] activations       */
        device float               *outputs [[buffer(3)]],   /* [S][O]                   */
        device const float         *lut     [[buffer(4)]],   /* 256-entry e4m3 decode    */
        constant ColiV4Fp8Dims     &dims    [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint O = dims.O, I = dims.I, S = dims.S, NB = dims.nblkI;
    uint o = gid.x, s = gid.y;
    if (o >= O || s >= S) return;

    device const uchar *w   = weights + (ulong)o * I;
    device const float *x   = inputs  + (ulong)s * I;
    device const float *scl = scales  + (ulong)(o / 128u) * NB;

    coli_df a = coli_df{ 0.0f, 0.0f };
    for (uint bi = 0; bi < NB; ++bi) {
        uint base = bi * 128u;
        uint blen = (base + 128u > I) ? (I - base) : 128u;
        float acc = 0.0f;
        for (uint i = 0; i < blen; ++i)
            acc += lut[w[base + i]] * x[base + i];
        a = coli_df_add_prod(a, acc, scl[bi]);
    }
    outputs[(ulong)s * O + o] = a.hi + a.lo;
}
