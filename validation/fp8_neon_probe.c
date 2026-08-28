/* fp8_neon_probe -- can matmul_fp8 be given an arm64 kernel, bit-exactly?
 *
 * WHY: c/quant.h:502 matmul_fp8 is a scalar loop with a 256-entry LUT gather per element and no
 * NEON. Its only SIMD path (coli_fp8_matvec_ref, #ifdef __AVX2__) is dead on aarch64. It serves
 * attn_out (17.9% of decode), attn_qkv (9.0%) and the indexer (2.3%) -- at least 29% of decode on
 * the configuration measured fastest (experiments E102/E103). quant.h already has hand-written
 * NEON for the int4/int8 matmuls, and the MXFP4 expert path has a full NEON rows16 kernel; the
 * FP8 path is the one that never got one.
 *
 * THE BIT-EXACTNESS CONTRACT (from the reference below, and stated in backend_metal_v4_seam.h):
 *   per output row o, per 128-column block:  acc = 0.f; acc += decode(w[i])*x[i]  serially
 *   then across blocks:                      a  += (double)acc * scale[block]
 * float within a block, double across blocks, ascending order. Any candidate must reproduce both.
 *
 * CANDIDATES
 *   A  lane-per-row: one float32x4 holds 4 output rows, each lane accumulating ITS OWN row
 *      serially over i. Same order as the reference => bit-exact. Mirrors the 4-row unroll the
 *      scalar code already does. Weights are row-major so the 4 bytes for column i come from 4
 *      pointers strided by I -- four scalar loads per step.
 *   B  rows4-packed + arithmetic decode: the same lane-per-row maths, but the weights are
 *      PRE-PACKED so the 4 rows' byte i are contiguous (one 4-byte load), and E4M3 is decoded
 *      with bit arithmetic on a vector instead of four LUT gathers. Packing is exactly what the
 *      FP4 expert path already does (coli_fp4_pack_rows16_v10). Still bit-exact: identical
 *      values, identical order.
 *
 * Build:  clang -O3 -march=native -fopenmp ... (see run_fp8_neon_probe.sh)
 * This program touches NO engine code and needs no model.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define FP8_BLOCK 128
static float E4M3[256];

/* reference decode, mirrors quant.h e4m3_decode */
static float e4m3_decode_ref(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return NAN;
    float n = e ? ldexpf(1.0f + (float)m / 8.0f, e - 7) : ldexpf((float)m, -9);
    return s ? -n : n;
}
static void build_lut(void) { for (int i = 0; i < 256; i++) E4M3[i] = e4m3_decode_ref((uint8_t)i); }

static int64_t nblk(int I) { return (I + FP8_BLOCK - 1) / FP8_BLOCK; }

/* ---- REFERENCE: quant.h:502 matmul_fp8, S=1, verbatim structure ---- */
static void ref_matvec(float *y, const float *x, const uint8_t *q8, const float *bscale,
                       int I, int O) {
    int64_t nb = nblk(I);
    for (int o = 0; o < O; o += 4) {
        int o1 = o+1<O?o+1:o, o2 = o+2<O?o+2:o, o3 = o+3<O?o+3:o;
        const uint8_t *w0=q8+(int64_t)o*I, *w1=q8+(int64_t)o1*I;
        const uint8_t *w2=q8+(int64_t)o2*I, *w3=q8+(int64_t)o3*I;
        const float *s0=bscale+(o/FP8_BLOCK)*nb,  *s1=bscale+(o1/FP8_BLOCK)*nb;
        const float *s2=bscale+(o2/FP8_BLOCK)*nb, *s3=bscale+(o3/FP8_BLOCK)*nb;
        double a0=0,a1=0,a2=0,a3=0;
        for (int64_t bi=0; bi*FP8_BLOCK<I; bi++) {
            int base=(int)(bi*FP8_BLOCK), blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
            float c0=0,c1=0,c2=0,c3=0;
            for (int i=base;i<base+blen;i++) {
                float xv=x[i];
                c0 += E4M3[w0[i]]*xv; c1 += E4M3[w1[i]]*xv;
                c2 += E4M3[w2[i]]*xv; c3 += E4M3[w3[i]]*xv;
            }
            a0 += (double)c0*s0[bi]; a1 += (double)c1*s1[bi];
            a2 += (double)c2*s2[bi]; a3 += (double)c3*s3[bi];
        }
        y[o]=(float)a0;
        if(o1!=o) y[o1]=(float)a1; if(o2!=o) y[o2]=(float)a2; if(o3!=o) y[o3]=(float)a3;
    }
}

#ifdef __ARM_NEON
/* ---- A: lane-per-row over the ROW-MAJOR layout (4 scalar byte loads per step) ---- */
static void neon_a(float *y, const float *x, const uint8_t *q8, const float *bscale,
                   int I, int O) {
    int64_t nb = nblk(I);
    for (int o = 0; o < O; o += 4) {
        int o1=o+1<O?o+1:o, o2=o+2<O?o+2:o, o3=o+3<O?o+3:o;
        const uint8_t *w0=q8+(int64_t)o*I, *w1=q8+(int64_t)o1*I;
        const uint8_t *w2=q8+(int64_t)o2*I, *w3=q8+(int64_t)o3*I;
        const float *s0=bscale+(o/FP8_BLOCK)*nb,  *s1=bscale+(o1/FP8_BLOCK)*nb;
        const float *s2=bscale+(o2/FP8_BLOCK)*nb, *s3=bscale+(o3/FP8_BLOCK)*nb;
        double a0=0,a1=0,a2=0,a3=0;
        for (int64_t bi=0; bi*FP8_BLOCK<I; bi++) {
            int base=(int)(bi*FP8_BLOCK), blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i=base;i<base+blen;i++) {
                float32x4_t w = { E4M3[w0[i]], E4M3[w1[i]], E4M3[w2[i]], E4M3[w3[i]] };
                /* NOT vfmaq: the reference rounds the product before adding. */
                acc = vaddq_f32(acc, vmulq_f32(w, vdupq_n_f32(x[i])));
            }
            a0+=(double)vgetq_lane_f32(acc,0)*s0[bi]; a1+=(double)vgetq_lane_f32(acc,1)*s1[bi];
            a2+=(double)vgetq_lane_f32(acc,2)*s2[bi]; a3+=(double)vgetq_lane_f32(acc,3)*s3[bi];
        }
        y[o]=(float)a0;
        if(o1!=o) y[o1]=(float)a1; if(o2!=o) y[o2]=(float)a2; if(o3!=o) y[o3]=(float)a3;
    }
}

/* rows4 pack: packed[(tile*I + i)*4 + lane] = q8[(tile*4+lane)*I + i] */
static void pack_rows4(uint8_t *p, const uint8_t *q8, int I, int O) {
    for (int o = 0; o < O; o++) {
        int t = o/4, l = o%4;
        for (int i = 0; i < I; i++) p[((size_t)t*I + i)*4 + l] = q8[(size_t)o*I + i];
    }
}

/* vectorised E4M3 -> f32 for 4 codes, no table. Mirrors e4m3_decode_ref exactly:
 *   e!=0 : (-1)^s * 2^(e-7) * (1 + m/8)      e==0 : (-1)^s * 2^-9 * m
 * 0x7F/0xFF (e==15,m==7) must be NaN. */
static inline float32x4_t decode4(uint32x4_t v) {
    uint32x4_t s = vshrq_n_u32(v, 7);
    uint32x4_t e = vandq_u32(vshrq_n_u32(v, 3), vdupq_n_u32(15));
    uint32x4_t m = vandq_u32(v, vdupq_n_u32(7));
    uint32x4_t enz = vcgtq_u32(e, vdupq_n_u32(0));
    /* normal: exponent field (e-7)+127 = e+120, mantissa m<<20 */
    uint32x4_t nrm = vorrq_u32(vshlq_n_u32(vaddq_u32(e, vdupq_n_u32(120)), 23),
                               vshlq_n_u32(m, 20));
    /* subnormal: m * 2^-9 computed exactly */
    float32x4_t sub = vmulq_n_f32(vcvtq_f32_u32(m), 1.0f/512.0f);
    float32x4_t val = vbslq_f32(enz, vreinterpretq_f32_u32(nrm), sub);
    /* NaN where e==15 && m==7 */
    uint32x4_t isnan = vandq_u32(vceqq_u32(e, vdupq_n_u32(15)), vceqq_u32(m, vdupq_n_u32(7)));
    val = vbslq_f32(isnan, vdupq_n_f32(NAN), val);
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(val), vshlq_n_u32(s, 31)));
}

/* ---- B: lane-per-row over ROWS4-PACKED weights + arithmetic decode ---- */
static void neon_b(float *y, const float *x, const uint8_t *packed, const float *bscale,
                   int I, int O) {
    int64_t nb = nblk(I);
    for (int o = 0; o < O; o += 4) {
        int o1=o+1<O?o+1:o, o2=o+2<O?o+2:o, o3=o+3<O?o+3:o;
        const uint8_t *w = packed + (size_t)(o/4)*I*4;
        const float *s0=bscale+(o/FP8_BLOCK)*nb,  *s1=bscale+(o1/FP8_BLOCK)*nb;
        const float *s2=bscale+(o2/FP8_BLOCK)*nb, *s3=bscale+(o3/FP8_BLOCK)*nb;
        double a0=0,a1=0,a2=0,a3=0;
        for (int64_t bi=0; bi*FP8_BLOCK<I; bi++) {
            int base=(int)(bi*FP8_BLOCK), blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i=base;i<base+blen;i++) {
                uint32_t four; memcpy(&four, w + (size_t)i*4, 4);          /* one 4-byte load */
                uint32x4_t codes = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(four))));
                acc = vaddq_f32(acc, vmulq_f32(decode4(codes), vdupq_n_f32(x[i])));
            }
            a0+=(double)vgetq_lane_f32(acc,0)*s0[bi]; a1+=(double)vgetq_lane_f32(acc,1)*s1[bi];
            a2+=(double)vgetq_lane_f32(acc,2)*s2[bi]; a3+=(double)vgetq_lane_f32(acc,3)*s3[bi];
        }
        y[o]=(float)a0;
        if(o1!=o) y[o1]=(float)a1; if(o2!=o) y[o2]=(float)a2; if(o3!=o) y[o3]=(float)a3;
    }
}
#endif /* __ARM_NEON */

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec/1e9; }
static int mism(const float*a,const float*b,int n){int c=0;for(int i=0;i<n;i++){
    uint32_t u,v;memcpy(&u,&a[i],4);memcpy(&v,&b[i],4);if(u!=v)c++;}return c;}

static void shape(const char *name, int I, int O, int iters) {
    int64_t nb = nblk(I); size_t sr = (O + FP8_BLOCK - 1)/FP8_BLOCK;
    uint8_t *q8 = malloc((size_t)O*I), *pk = malloc((size_t)((O+3)/4*4)*I);
    float *bs = malloc(sr*nb*sizeof(float)), *x = malloc((size_t)I*sizeof(float));
    float *yr = malloc((size_t)O*4), *ya = malloc((size_t)O*4), *yb = malloc((size_t)O*4);
    srandom(11);
    for (size_t i=0;i<(size_t)O*I;i++) q8[i]=(uint8_t)(random()&0xFF);
    for (size_t i=0;i<sr*nb;i++) bs[i]=0.5f+(float)(random()%1000)/1000.0f;
    for (int i=0;i<I;i++) x[i]=((float)(random()%2001)-1000)/337.0f;

    double t=now_s(); for(int k=0;k<iters;k++) ref_matvec(yr,x,q8,bs,I,O);
    double tr=(now_s()-t)/iters;
    printf("  %-26s I=%-5d O=%-5d   ref %8.3f ms\n", name, I, O, tr*1e3);
#ifdef __ARM_NEON
    memset(pk,0,(size_t)((O+3)/4*4)*I); pack_rows4(pk,q8,I,O);
    t=now_s(); for(int k=0;k<iters;k++) neon_a(ya,x,q8,bs,I,O); double ta=(now_s()-t)/iters;
    t=now_s(); for(int k=0;k<iters;k++) neon_b(yb,x,pk,bs,I,O); double tb=(now_s()-t)/iters;
    printf("      A row-major lane-per-row  %8.3f ms  %5.2fx   mismatches=%d %s\n",
           ta*1e3, tr/ta, mism(yr,ya,O), mism(yr,ya,O)?"*** NOT BIT-EXACT ***":"BIT-EXACT");
    printf("      B rows4-packed + decode4  %8.3f ms  %5.2fx   mismatches=%d %s\n",
           tb*1e3, tr/tb, mism(yr,yb,O), mism(yr,yb,O)?"*** NOT BIT-EXACT ***":"BIT-EXACT");
#else
    printf("      (no __ARM_NEON in this build)\n");
#endif
    printf("\n");
    free(q8);free(pk);free(bs);free(x);free(yr);free(ya);free(yb);
}

int main(void) {
    build_lut();
    printf("\n  fp8_neon_probe -- can matmul_fp8 get a bit-exact arm64 kernel?\n");
    printf("  single-threaded on purpose: this measures the KERNEL, not the OpenMP scaling\n\n");
    /* Real V4 attention projection shapes. o_rank/group_width vary by config; these bracket it. */
    shape("wo_b   (out proj B)",   4096, 4096, 20);
    shape("wo_a   (out proj A)",   2048, 1024, 40);
    shape("wq_b   (query B)",      1536, 4096, 30);
    shape("wkv    (kv proj)",      4096,  576, 40);
    return 0;
}
