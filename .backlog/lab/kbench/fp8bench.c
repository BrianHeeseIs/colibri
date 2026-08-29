/* Self-contained microbench for the fp8 E4M3 matrix-VECTOR kernel that dominates decode.
 *
 * WHY STANDALONE: iterating inside the engine costs a ~35-50 s model load per experiment. This
 * reproduces matmul_fp8()'s exact semantics (c/quant.h:502) on synthetic data so kernel variants
 * can be compared in milliseconds, then the winner is ported back and validated end-to-end.
 *
 * FIDELITY (must match c/quant.h:502 exactly or the numbers are meaningless):
 *   - o += 4 blocking, four independent accumulators
 *   - per-128-element block: float accumulator, then  a += (double)acc * scale
 *   - scale index: bscale[(o/128)*nblkI + bi]
 *   - weights row-major: q8[o*I + i], one byte per element
 * Variant outputs are checked bit-for-bit against the scalar reference.
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
#ifdef _OPENMP
#include <omp.h>
#endif

#define FP8_BLOCK 128
static inline int64_t fp8_nblk(int n){ return ((int64_t)n + FP8_BLOCK - 1)/FP8_BLOCK; }
static float E4M3_LUT[256];

/* OCP e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa. 0x7F/0xFF are NaN. exp==0 => subnormal. */
static float e4m3_decode_ref(uint8_t b){
    int s = (b>>7)&1, e = (b>>3)&0xF, m = b&0x7;
    if(e==0xF && m==0x7) return NAN;
    float v = (e==0) ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m/8.0f, e-7);
    return s ? -v : v;
}
static void lut_init(void){ for(int i=0;i<256;i++) E4M3_LUT[i]=e4m3_decode_ref((uint8_t)i); }
static inline float e4m3_decode(uint8_t b){ return E4M3_LUT[b]; }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* ---- V0: exact transcription of matmul_fp8() with S=1 ---- */
static void v0_scalar(float *y,const float *x,const uint8_t *q8,const float *bscale,int I,int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o+=4){
        int o1=o+1<O?o+1:o, o2=o+2<O?o+2:o, o3=o+3<O?o+3:o;
        const uint8_t *w0=q8+(int64_t)o*I,*w1=q8+(int64_t)o1*I,*w2=q8+(int64_t)o2*I,*w3=q8+(int64_t)o3*I;
        const float *s0=bscale+(o/FP8_BLOCK)*nblkI,*s1=bscale+(o1/FP8_BLOCK)*nblkI;
        const float *s2=bscale+(o2/FP8_BLOCK)*nblkI,*s3=bscale+(o3/FP8_BLOCK)*nblkI;
        double a0=0,a1=0,a2=0,a3=0;
        for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
            int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
            float c0=0,c1=0,c2=0,c3=0;
            for(int i=base;i<base+blen;i++){
                float xv=x[i];
                c0+=e4m3_decode(w0[i])*xv; c1+=e4m3_decode(w1[i])*xv;
                c2+=e4m3_decode(w2[i])*xv; c3+=e4m3_decode(w3[i])*xv;
            }
            a0+=(double)c0*s0[bi]; a1+=(double)c1*s1[bi];
            a2+=(double)c2*s2[bi]; a3+=(double)c3*s3[bi];
        }
        y[o]=(float)a0; if(o1!=o)y[o1]=(float)a1; if(o2!=o)y[o2]=(float)a2; if(o3!=o)y[o3]=(float)a3;
    }
}


/* ---- V1: ROW-INTERLEAVED layout, still scalar LUT decode ----
 * Isolates the LAYOUT effect alone, which is the hypothesis from E124: the scalar kernel reads 4
 * output rows from 4 DIFFERENT cache lines per iteration. Repacked as [o/R][i][r], the R rows for
 * column i are adjacent, so one 128-byte line now serves 128/R consecutive columns instead of one
 * column of one row. Accumulation ORDER per output row is unchanged, so results stay bit-exact. */
#define R1 4
static void repack_r4(uint8_t *dst,const uint8_t *q8,int I,int O){
    for(int o=0;o<O;o+=R1)
        for(int i=0;i<I;i++)
            for(int r=0;r<R1;r++){
                int oo=o+r<O?o+r:o;
                dst[((size_t)(o/R1)*I+i)*R1+r]=q8[(size_t)oo*I+i];
            }
}
static void v1_interleaved(float *y,const float *x,const uint8_t *packed,const float *bscale,int I,int O){
    int64_t nblkI=fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o+=R1){
        int o1=o+1<O?o+1:o,o2=o+2<O?o+2:o,o3=o+3<O?o+3:o;
        const uint8_t *w=packed+(size_t)(o/R1)*I*R1;
        const float *s0=bscale+(o/FP8_BLOCK)*nblkI,*s1=bscale+(o1/FP8_BLOCK)*nblkI;
        const float *s2=bscale+(o2/FP8_BLOCK)*nblkI,*s3=bscale+(o3/FP8_BLOCK)*nblkI;
        double a0=0,a1=0,a2=0,a3=0;
        for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
            int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK; if(base+blen>I)blen=I-base;
            float c0=0,c1=0,c2=0,c3=0;
            for(int i=base;i<base+blen;i++){
                const uint8_t *p=w+(size_t)i*R1; float xv=x[i];
                c0+=e4m3_decode(p[0])*xv; c1+=e4m3_decode(p[1])*xv;
                c2+=e4m3_decode(p[2])*xv; c3+=e4m3_decode(p[3])*xv;
            }
            a0+=(double)c0*s0[bi]; a1+=(double)c1*s1[bi];
            a2+=(double)c2*s2[bi]; a3+=(double)c3*s3[bi];
        }
        y[o]=(float)a0; if(o1!=o)y[o1]=(float)a1; if(o2!=o)y[o2]=(float)a2; if(o3!=o)y[o3]=(float)a3;
    }
}


/* ---- V2: 16-row interleave + BRANCH-FREE NEON e4m3 decode (no table) ----
 * V1 showed the layout alone is worth only ~6%, so the cost is the DEPENDENT LUT LOADS: NEON has no
 * gather, so a table decode cannot be vectorised at all. Decoding arithmetically removes the memory
 * dependency entirely and lets 16 weights become 16 floats in pure ALU.
 *
 * OCP e4m3fn: s.eeee.mmm, exponent bias 7, exp==0 subnormal, 0x7F/0xFF NaN.
 *   normal    : bits = sign | ((e + 120) << 23) | (m << 20)        [120 = 127 - 7]
 *   subnormal : value = m * 2^-9, signed
 *   NaN       : (b & 0x7F) == 0x7F  -> quiet NaN, matching quant.h's documented propagate policy
 * All 256 e4m3 values are exactly representable in fp32, so this is EXACT, not approximate.
 *
 * R=16 makes one column's 16 weights exactly one vld1q_u8. Rows o..o+15 never cross a 128-row scale
 * boundary (128 % 16 == 0), so one scale per block still applies.
 * mul+add, never vfma: the scalar reference rounds twice, so fusing would change results. */
#define R2 16
static void repack_r16(uint8_t *dst,const uint8_t *q8,int I,int O){
    int nb=(O+R2-1)/R2;
    for(int b=0;b<nb;b++)
        for(int i=0;i<I;i++)
            for(int r=0;r<R2;r++){
                int oo=b*R2+r; if(oo>=O) oo=O-1;
                dst[((size_t)b*I+i)*R2+r]=q8[(size_t)oo*I+i];
            }
}
#ifdef __ARM_NEON
/* 16 e4m3 bytes -> 4 x float32x4, exact. */
static inline void e4m3_decode16(uint8x16_t b, float32x4_t out[4]){
    const uint16x8_t lo16=vmovl_u8(vget_low_u8(b)), hi16=vmovl_u8(vget_high_u8(b));
    uint32x4_t u[4];
    u[0]=vmovl_u16(vget_low_u16(lo16));  u[1]=vmovl_u16(vget_high_u16(lo16));
    u[2]=vmovl_u16(vget_low_u16(hi16));  u[3]=vmovl_u16(vget_high_u16(hi16));
    const uint32x4_t m7=vdupq_n_u32(0x7), mF=vdupq_n_u32(0xF), m80=vdupq_n_u32(0x80),
                     m7F=vdupq_n_u32(0x7F);
    const float32x4_t inv512=vdupq_n_f32(1.0f/512.0f);
    const uint32x4_t nanbits=vdupq_n_u32(0x7FC00000);
    for(int k=0;k<4;k++){
        uint32x4_t v=u[k];
        uint32x4_t sign=vshlq_n_u32(vandq_u32(v,m80),24);
        uint32x4_t e=vandq_u32(vshrq_n_u32(v,3),mF);
        uint32x4_t m=vandq_u32(v,m7);
        uint32x4_t norm=vorrq_u32(sign,
                        vorrq_u32(vshlq_n_u32(vaddq_u32(e,vdupq_n_u32(120)),23),
                                  vshlq_n_u32(m,20)));
        /* subnormal: (+-) m/512 */
        float32x4_t sub=vmulq_f32(vcvtq_f32_u32(m),inv512);
        sub=vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(sub),sign));
        uint32x4_t is_sub=vceqq_u32(e,vdupq_n_u32(0));
        uint32x4_t is_nan=vceqq_u32(vandq_u32(v,m7F),m7F);
        uint32x4_t r=vbslq_u32(is_sub,vreinterpretq_u32_f32(sub),norm);
        r=vbslq_u32(is_nan,nanbits,r);
        out[k]=vreinterpretq_f32_u32(r);
    }
}
static void v2_neon(float *y,const float *x,const uint8_t *packed,const float *bscale,int I,int O){
    int64_t nblkI=fp8_nblk(I); int nb=(O+R2-1)/R2;
    #pragma omp parallel for schedule(static)
    for(int blk=0;blk<nb;blk++){
        int o=blk*R2;
        const uint8_t *w=packed+(size_t)blk*I*R2;
        const float *sc=bscale+(o/FP8_BLOCK)*nblkI;
        double a[R2]; for(int r=0;r<R2;r++) a[r]=0.0;
        for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
            int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK; if(base+blen>I)blen=I-base;
            float32x4_t c0=vdupq_n_f32(0),c1=vdupq_n_f32(0),c2=vdupq_n_f32(0),c3=vdupq_n_f32(0);
            for(int i=base;i<base+blen;i++){
                uint8x16_t wb=vld1q_u8(w+(size_t)i*R2);
                float32x4_t d[4]; e4m3_decode16(wb,d);
                const float xv=x[i];
                c0=vaddq_f32(c0,vmulq_n_f32(d[0],xv));
                c1=vaddq_f32(c1,vmulq_n_f32(d[1],xv));
                c2=vaddq_f32(c2,vmulq_n_f32(d[2],xv));
                c3=vaddq_f32(c3,vmulq_n_f32(d[3],xv));
            }
            const float sv=sc[bi];
            float tmp[R2];
            vst1q_f32(tmp+0,c0); vst1q_f32(tmp+4,c1); vst1q_f32(tmp+8,c2); vst1q_f32(tmp+12,c3);
            for(int r=0;r<R2;r++) a[r]+=(double)tmp[r]*sv;
        }
        for(int r=0;r<R2;r++){ int oo=o+r; if(oo<O) y[oo]=(float)a[r]; }
    }
}
#endif


#ifdef __ARM_NEON
/* ---- V3: 16-row interleave + f16-REINTERPRET decode ----
 * V2's arithmetic decode was ~60 ALU ops per 16 weights and LOST to the table (M3 has high L1 load
 * throughput and the LUT is 1 KB, so table lookup is cheap). The cheap decode is a reinterpret:
 *
 *   e4m3 (bias 7, 3-bit mantissa) -> f16 (bias 15, 10-bit mantissa) by placing e in f16's exponent
 *   field and m in the top of its mantissa:   h = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
 *   gives  h_value = e4m3_value * 2^(7-15) = e4m3_value / 256.
 *
 * EXACT for every non-NaN e4m3 code, including subnormals: an e4m3 subnormal m*2^-9 lands on the
 * f16 subnormal m*2^-17, and f16's smallest subnormal is 2^-24. The 256 is a power of two, so
 * folding it into the block scale (scale * 256) is exact -- no extra per-element multiply.
 * e4m3 NaN (0x7F/0xFF) would map to a finite f16, so it gets an explicit compare+select. */
#define R3 16
static inline void e4m3_decode16_f16(uint8x16_t b, float32x4_t out[4]){
    const uint16x8_t lo=vmovl_u8(vget_low_u8(b)), hi=vmovl_u8(vget_high_u8(b));
    const uint16x8_t m80=vdupq_n_u16(0x80), m7F=vdupq_n_u16(0x7F), qnan=vdupq_n_u16(0x7E00);
    uint16x8_t h0=vorrq_u16(vshlq_n_u16(vandq_u16(lo,m80),8),vshlq_n_u16(vandq_u16(lo,m7F),7));
    uint16x8_t h1=vorrq_u16(vshlq_n_u16(vandq_u16(hi,m80),8),vshlq_n_u16(vandq_u16(hi,m7F),7));
    h0=vbslq_u16(vceqq_u16(vandq_u16(lo,m7F),m7F),qnan,h0);
    h1=vbslq_u16(vceqq_u16(vandq_u16(hi,m7F),m7F),qnan,h1);
    const float16x8_t f0=vreinterpretq_f16_u16(h0), f1=vreinterpretq_f16_u16(h1);
    out[0]=vcvt_f32_f16(vget_low_f16(f0));  out[1]=vcvt_f32_f16(vget_high_f16(f0));
    out[2]=vcvt_f32_f16(vget_low_f16(f1));  out[3]=vcvt_f32_f16(vget_high_f16(f1));
}
static void v3_neon_f16(float *y,const float *x,const uint8_t *packed,const float *bscale,int I,int O){
    int64_t nblkI=fp8_nblk(I); int nb=(O+R3-1)/R3;
    #pragma omp parallel for schedule(static)
    for(int blk=0;blk<nb;blk++){
        int o=blk*R3;
        const uint8_t *w=packed+(size_t)blk*I*R3;
        const float *sc=bscale+(o/FP8_BLOCK)*nblkI;
        double a[R3]; for(int r=0;r<R3;r++) a[r]=0.0;
        for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
            int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK; if(base+blen>I)blen=I-base;
            float32x4_t c0=vdupq_n_f32(0),c1=vdupq_n_f32(0),c2=vdupq_n_f32(0),c3=vdupq_n_f32(0);
            for(int i=base;i<base+blen;i++){
                uint8x16_t wb=vld1q_u8(w+(size_t)i*R3);
                float32x4_t d[4]; e4m3_decode16_f16(wb,d);
                const float xv=x[i];
                c0=vfmaq_n_f32(c0,d[0],xv); c1=vfmaq_n_f32(c1,d[1],xv);
                c2=vfmaq_n_f32(c2,d[2],xv); c3=vfmaq_n_f32(c3,d[3],xv);
            }
            const float sv=sc[bi]*256.0f;      /* fold the reinterpret's 2^8; exact */
            float tmp[R3];
            vst1q_f32(tmp+0,c0); vst1q_f32(tmp+4,c1); vst1q_f32(tmp+8,c2); vst1q_f32(tmp+12,c3);
            for(int r=0;r<R3;r++) a[r]+=(double)tmp[r]*sv;
        }
        for(int r=0;r<R3;r++){ int oo=o+r; if(oo<O) y[oo]=(float)a[r]; }
    }
}
#endif


#ifdef __ARM_NEON
/* ---- V4: V3 + 2-column unroll ----
 * V3 reached 63% of achievable bandwidth. The remaining gap is latency, not throughput: each column
 * is load -> decode -> fma, a dependent chain into the SAME four accumulators. Processing two
 * columns per iteration into two accumulator SETS doubles the independent work in flight; the two
 * sets are summed at the block boundary. Per-lane accumulation order changes (even columns then odd),
 * so this is NOT bit-exact -- it is reported honestly and gated on taskcheck if adopted. */
static void v4_neon_u2(float *y,const float *x,const uint8_t *packed,const float *bscale,int I,int O){
    int64_t nblkI=fp8_nblk(I); int nb=(O+R3-1)/R3;
    #pragma omp parallel for schedule(static)
    for(int blk=0;blk<nb;blk++){
        int o=blk*R3;
        const uint8_t *w=packed+(size_t)blk*I*R3;
        const float *sc=bscale+(o/FP8_BLOCK)*nblkI;
        double a[R3]; for(int r=0;r<R3;r++) a[r]=0.0;
        for(int64_t bi=0;bi*FP8_BLOCK<I;bi++){
            int base=(int)(bi*FP8_BLOCK),blen=FP8_BLOCK; if(base+blen>I)blen=I-base;
            float32x4_t c0=vdupq_n_f32(0),c1=vdupq_n_f32(0),c2=vdupq_n_f32(0),c3=vdupq_n_f32(0);
            float32x4_t e0=vdupq_n_f32(0),e1=vdupq_n_f32(0),e2=vdupq_n_f32(0),e3=vdupq_n_f32(0);
            int i=base, end=base+blen;
            for(; i+1<end; i+=2){
                uint8x16_t wa=vld1q_u8(w+(size_t)i*R3), wb2=vld1q_u8(w+(size_t)(i+1)*R3);
                float32x4_t da[4],db[4];
                e4m3_decode16_f16(wa,da); e4m3_decode16_f16(wb2,db);
                const float xa=x[i], xb=x[i+1];
                c0=vfmaq_n_f32(c0,da[0],xa); e0=vfmaq_n_f32(e0,db[0],xb);
                c1=vfmaq_n_f32(c1,da[1],xa); e1=vfmaq_n_f32(e1,db[1],xb);
                c2=vfmaq_n_f32(c2,da[2],xa); e2=vfmaq_n_f32(e2,db[2],xb);
                c3=vfmaq_n_f32(c3,da[3],xa); e3=vfmaq_n_f32(e3,db[3],xb);
            }
            for(; i<end; i++){
                uint8x16_t wa=vld1q_u8(w+(size_t)i*R3); float32x4_t da[4];
                e4m3_decode16_f16(wa,da); const float xa=x[i];
                c0=vfmaq_n_f32(c0,da[0],xa); c1=vfmaq_n_f32(c1,da[1],xa);
                c2=vfmaq_n_f32(c2,da[2],xa); c3=vfmaq_n_f32(c3,da[3],xa);
            }
            c0=vaddq_f32(c0,e0); c1=vaddq_f32(c1,e1); c2=vaddq_f32(c2,e2); c3=vaddq_f32(c3,e3);
            const float sv=sc[bi]*256.0f;
            float tmp[R3];
            vst1q_f32(tmp+0,c0); vst1q_f32(tmp+4,c1); vst1q_f32(tmp+8,c2); vst1q_f32(tmp+12,c3);
            for(int r=0;r<R3;r++) a[r]+=(double)tmp[r]*sv;
        }
        for(int r=0;r<R3;r++){ int oo=o+r; if(oo<O) y[oo]=(float)a[r]; }
    }
}
#endif

typedef void (*kern)(float*,const float*,const uint8_t*,const float*,int,int);
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
static double bench(kern f,float*y,const float*x,const uint8_t*q,const float*s,int I,int O,int reps){
    double *v=malloc((size_t)reps*sizeof(double));
    f(y,x,q,s,I,O);                                   /* warm */
    for(int r=0;r<reps;r++){ double t0=now(); f(y,x,q,s,I,O); v[r]=now()-t0; }
    qsort(v,reps,sizeof(double),dcmp);
    double med=v[reps/2]; free(v); return med;         /* median: robust to scheduler noise */
}
int main(int argc,char**argv){
    int I = argc>1?atoi(argv[1]):7168;     /* input dim  (columns) */
    int O = argc>2?atoi(argv[2]):7168;     /* output dim (rows)    */
    int reps = argc>3?atoi(argv[3]):20;
    if(I%128){fprintf(stderr,"I must be a multiple of 128\n");return 2;}
    lut_init();
    size_t wn=(size_t)I*O; int64_t nblkI=fp8_nblk(I); size_t sn=(size_t)((O+127)/128)*nblkI;
    uint8_t *q=aligned_alloc(128,wn); float *sc=aligned_alloc(64,sn*sizeof(float));
    float *x=aligned_alloc(64,(size_t)I*sizeof(float)), *y0=aligned_alloc(64,(size_t)O*sizeof(float));
    if(!q||!sc||!x||!y0){fprintf(stderr,"alloc failed\n");return 1;}
    srand(12345);
    for(size_t i=0;i<wn;i++){ uint8_t b; do{ b=(uint8_t)(rand()&0xFF); }while((b&0x7F)==0x7F); q[i]=b; }
    for(size_t i=0;i<sn;i++) sc[i]=0.01f+0.001f*(float)(i%17);
    for(int i=0;i<I;i++) x[i]=((float)rand()/RAND_MAX)-0.5f;

    double bytes=(double)wn;               /* weights dominate; one pass per call */
    printf("shape I=%d O=%d  weights=%.1f MB  reps=%d  threads=%d\n",
           I,O,bytes/1e6,reps,
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
           );
    double t=bench(v0_scalar,y0,x,q,sc,I,O,reps);
    printf("V0 scalar (matmul_fp8)   %8.3f ms   %7.2f GB/s   1.00x\n",t*1e3,bytes/t/1e9);

    uint8_t *pk=aligned_alloc(128,wn); float *y1=aligned_alloc(64,(size_t)O*sizeof(float));
    if(!pk||!y1){fprintf(stderr,"alloc failed\n");return 1;}
    repack_r4(pk,q,I,O);
    double t1=bench(v1_interleaved,y1,x,pk,sc,I,O,reps);
    int bad=0; for(int i=0;i<O;i++) if(memcmp(&y0[i],&y1[i],sizeof(float))) bad++;
    printf("V1 interleaved R=4 LUT   %8.3f ms   %7.2f GB/s   %.2fx   bitexact=%s\n",
           t1*1e3,bytes/t1/1e9,t/t1,bad?"NO":"yes");
    if(bad) printf("   MISMATCH in %d of %d outputs\n",bad,O);
    free(pk);free(y1);
#ifdef __ARM_NEON
    {
        int nb=(O+R2-1)/R2; size_t pn=(size_t)nb*I*R2;
        uint8_t *pk2=aligned_alloc(128,((pn+127)/128)*128);
        float *y2=aligned_alloc(64,(size_t)O*sizeof(float));
        if(!pk2||!y2){fprintf(stderr,"alloc failed\n");return 1;}
        repack_r16(pk2,q,I,O);
        double t2=bench(v2_neon,y2,x,pk2,sc,I,O,reps);
        int bad2=0,ulp1=0;
        for(int i=0;i<O;i++) if(memcmp(&y0[i],&y2[i],sizeof(float))){
            bad2++; float d=fabsf(y0[i]-y2[i]); if(d<=1e-6f*fabsf(y0[i])) ulp1++;
        }
        printf("V2 interleave16 + NEON   %8.3f ms   %7.2f GB/s   %.2fx   bitexact=%s\n",
               t2*1e3,bytes/t2/1e9,t/t2,bad2?"NO":"yes");
        if(bad2) printf("   %d of %d differ (%d within 1e-6 rel)\n",bad2,O,ulp1);
        free(pk2);free(y2);
        uint8_t *pk3=aligned_alloc(128,((pn+127)/128)*128);
        float *y3=aligned_alloc(64,(size_t)O*sizeof(float));
        if(!pk3||!y3){fprintf(stderr,"alloc failed\n");return 1;}
        repack_r16(pk3,q,I,O);
        double t3=bench(v3_neon_f16,y3,x,pk3,sc,I,O,reps);
        int bad3=0; double worst=0;
        for(int i=0;i<O;i++) if(memcmp(&y0[i],&y3[i],sizeof(float))){
            bad3++; double rel=fabs((double)y0[i]-y3[i])/(fabs((double)y0[i])+1e-30);
            if(rel>worst)worst=rel;
        }
        printf("V3 interleave16 + f16cvt %8.3f ms   %7.2f GB/s   %.2fx   bitexact=%s\n",
               t3*1e3,bytes/t3/1e9,t/t3,bad3?"NO":"yes");
        if(bad3) printf("   %d of %d differ, worst rel %.3g\n",bad3,O,worst);
        free(pk3);free(y3);
        uint8_t *pk4=aligned_alloc(128,((pn+127)/128)*128);
        float *y4=aligned_alloc(64,(size_t)O*sizeof(float));
        if(!pk4||!y4){fprintf(stderr,"alloc failed\n");return 1;}
        repack_r16(pk4,q,I,O);
        double t4=bench(v4_neon_u2,y4,x,pk4,sc,I,O,reps);
        int bad4=0; double worst4=0;
        for(int i=0;i<O;i++) if(memcmp(&y0[i],&y4[i],sizeof(float))){
            bad4++; double rel=fabs((double)y0[i]-y4[i])/(fabs((double)y0[i])+1e-30);
            if(rel>worst4)worst4=rel;
        }
        printf("V4 V3 + 2-col unroll    %8.3f ms   %7.2f GB/s   %.2fx   bitexact=%s\n",
               t4*1e3,bytes/t4/1e9,t/t4,bad4?"NO":"yes");
        if(bad4) printf("   %d of %d differ, worst rel %.3g\n",bad4,O,worst4);
        free(pk4);free(y4);
    }
#endif
    free(q);free(sc);free(x);free(y0); return 0;
}
