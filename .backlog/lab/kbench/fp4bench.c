/* fp4bench -- is the mxfp4 EXPERT kernel leaving the same win on the table that fp8 was?
 *
 * WHY: post-E125, `expert_forward` is the largest decode phase at 36.0% (7557 ms of 20977).
 * Only the 16 HOT-PINNED experts per layer (of 256) get the rows16 NEON pack; every other resident
 * expert runs the SCALAR matmul_mxfp4 (c/quant.h:1440, no aarch64 SIMD). That is structurally the
 * same gap E125 just closed for fp8.
 *
 * E107 concluded "scalar MXFP4 + OMP is competitive with the NEON pack in situ; coverage is not a
 * lever", but it inferred that from a PIN SWEEP, which is confounded: raising pins also raises
 * expert_wait (+66% measured) and changes cache behaviour, so a flat expert_forward does not prove
 * the two KERNELS are equally fast. This measures the kernels directly, head to head, with no model
 * and no pinning policy involved -- the same method that refuted E104.
 *
 * Both kernels already exist in the engine, so this calls them rather than reimplementing:
 *   scalar : coli_fp4_matvec_ref        (block_rows == 1)
 *   NEON   : coli_fp4_matvec_rows16_v10 (block_rows == 16, packed by coli_fp4_pack_rows16_v10)
 * Real DeepSeek-V4-Flash expert shapes: gate/up [2048 x 4096], down [4096 x 2048].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../../c/native_quant.h"
#include "../../../c/native_quant_fp4_rows16.h"

float coli_fp8_minprod = 3.4e38f;
int   coli_fp8_minprod_enabled = 0;

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}


#ifdef __ARM_NEON
#include <arm_neon.h>
/* V2: hypothesis test for the biggest remaining decode lever.
 *
 * The engine's rows16 NEON kernel does, PER ELEMENT:
 *     sums[g] = vaddq(sums[g], vmulq(vmulq(x, values), scales[g]))
 * i.e. two multiplies and an add, three ops, with the block scale applied to every element even
 * though it is constant across each 32-column group. (The no-FMA form is deliberate there: it is
 * what makes the CPU path match the Metal kernel bit-for-bit.)
 *
 * This variant accumulates WITHOUT the scale using one FMA per element, then applies the scale once
 * per 32-column block: ~1.125 ops/element instead of 3, a 2.7x arithmetic reduction. It is NOT
 * bit-identical to either existing kernel -- that is the point of measuring it before deciding
 * whether the integration cost is worth paying.
 *
 * mxfp4 e2m1 is a 16-entry table, so the whole decode fits one vqtbl1q_u8 lookup register.
 */
static const float E2M1[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                              -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
static inline float e8m0(unsigned char b){ return ldexpf(1.0f, (int)b - 127); }

static void v2_fp4_hoisted(float *y, const float *x, const unsigned char *d,
                           const unsigned char *sc, int I, int O) {
    const size_t data_stride = (size_t)I / 2, scale_stride = (size_t)I / 32;
    /* four 4-lane fp32 tables addressed by nibble, built once */
    float32x4_t tab[4];
    for (int g = 0; g < 4; g++) tab[g] = vld1q_f32(E2M1 + 4*g);
    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < O/16; tile++) {
        float32x4_t sums[4] = {vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
        for (int base = 0; base < I; base += 32) {
            float dec[16];
            const unsigned char *sp = sc + ((size_t)tile*scale_stride + base/32)*16;
            for (int l = 0; l < 16; l++) dec[l] = e8m0(sp[l]);
            float32x4_t bs[4];
            for (int g = 0; g < 4; g++) bs[g] = vld1q_f32(dec + 4*g);
            float32x4_t acc[4] = {vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
            for (int off = 0; off < 32; off += 2) {
                const unsigned char *codes = d + ((size_t)tile*data_stride + (base+off)/2)*16;
                uint8x16_t bytes = vld1q_u8(codes);
                uint8x16_t lo = vandq_u8(bytes, vdupq_n_u8(15));
                uint8x16_t hi = vshrq_n_u8(bytes, 4);
                const float xa = x[base+off], xb = x[base+off+1];
                uint8_t li[16], hix[16];
                vst1q_u8(li, lo); vst1q_u8(hix, hi);
                for (int g = 0; g < 4; g++) {
                    const float32x4_t va = { E2M1[li[4*g+0]], E2M1[li[4*g+1]],
                                             E2M1[li[4*g+2]], E2M1[li[4*g+3]] };
                    const float32x4_t vb = { E2M1[hix[4*g+0]], E2M1[hix[4*g+1]],
                                             E2M1[hix[4*g+2]], E2M1[hix[4*g+3]] };
                    acc[g] = vfmaq_n_f32(acc[g], va, xa);   /* ONE fma per element */
                    acc[g] = vfmaq_n_f32(acc[g], vb, xb);
                }
            }
            for (int g = 0; g < 4; g++) sums[g] = vfmaq_f32(sums[g], acc[g], bs[g]);
        }
        for (int g = 0; g < 4; g++) vst1q_f32(y + (size_t)tile*16 + 4*g, sums[g]);
    }
    (void)tab;
}
#endif

/* MXFP4: 2 nibbles per byte, one UE8M0 scale per 32-column group. */
static int run_shape(const char *name, int rows, int cols, int reps) {
    size_t data_bytes  = (size_t)rows * cols / 2;
    size_t scale_bytes = (size_t)rows * cols / 32;
    unsigned char *d = malloc(data_bytes), *s = malloc(scale_bytes);
    unsigned char *pd = malloc(data_bytes), *ps = malloc(scale_bytes);
    float *x = malloc((size_t)cols * sizeof(float));
    float *y_ref = malloc((size_t)rows * sizeof(float));
    float *y_neon = malloc((size_t)rows * sizeof(float));
    if(!d||!s||!pd||!ps||!x||!y_ref||!y_neon){fprintf(stderr,"alloc\n");return 1;}
    unsigned st = 2246822519u;
    for(size_t i=0;i<data_bytes;i++){ st=st*1103515245u+12345u; d[i]=(unsigned char)(st>>16); }
    for(size_t i=0;i<scale_bytes;i++){ st=st*1103515245u+12345u; s[i]=(unsigned char)(120+(st>>24)%16); }
    for(int i=0;i<cols;i++){ st=st*1103515245u+12345u; x[i]=((float)((int)(st>>20)-2048))/2048.0f; }

    ColiTensorView cold; memset(&cold,0,sizeof(cold));
    cold.format=COLI_TENSOR_FP4_NATIVE_BLOCK; cold.scale_format=COLI_SCALE_UE8M0;
    cold.data=d; cold.scales=s; cold.data_bytes=data_bytes; cold.scale_bytes=scale_bytes;
    cold.rows=rows; cold.columns=cols; cold.block_rows=1; cold.block_columns=32;

    if (coli_fp4_pack_rows16_v10(pd, ps, &cold) != 0) {
        fprintf(stderr,"%s: pack failed (rows%%16=%d cols%%128=%d)\n",name,rows%16,cols%128); return 1;
    }
    ColiTensorView hot = cold;
    hot.data=pd; hot.scales=ps; hot.block_rows=16;

    double *v=malloc((size_t)reps*sizeof(double));
    coli_fp4_matvec_ref(y_ref,&cold,x);
    for(int r=0;r<reps;r++){double t0=now();coli_fp4_matvec_ref(y_ref,&cold,x);v[r]=now()-t0;}
    qsort(v,reps,sizeof(double),dcmp); double t_ref=v[reps/2];
    coli_fp4_matvec_rows16_v10(y_neon,&hot,x);
    for(int r=0;r<reps;r++){double t0=now();coli_fp4_matvec_rows16_v10(y_neon,&hot,x);v[r]=now()-t0;}
    qsort(v,reps,sizeof(double),dcmp); double t_neon=v[reps/2];
    free(v);

    int bad=0; double worst=0;
    for(int i=0;i<rows;i++) if(memcmp(&y_ref[i],&y_neon[i],sizeof(float))){
        bad++; double rel=fabs((double)y_ref[i]-y_neon[i])/(fabs((double)y_ref[i])+1e-30);
        if(rel>worst)worst=rel;
    }
    printf("%-18s %5dx%-5d  scalar %8.3f ms  NEON %8.3f ms  %5.2fx   %7.2f -> %7.2f GB/s  %s\n",
           name,rows,cols,t_ref*1e3,t_neon*1e3,t_ref/t_neon,
           (double)data_bytes/t_ref/1e9,(double)data_bytes/t_neon/1e9,
           bad? "DIFFERS":"bit-exact");
#ifdef __ARM_NEON
    {
        float *y2 = malloc((size_t)rows*sizeof(float));
        double *vv = malloc((size_t)reps*sizeof(double));
        v2_fp4_hoisted(y2,x,pd,ps,cols,rows);
        for(int r=0;r<reps;r++){double t0=now();v2_fp4_hoisted(y2,x,pd,ps,cols,rows);vv[r]=now()-t0;}
        qsort(vv,reps,sizeof(double),dcmp); double t2=vv[reps/2]; free(vv);
        double worst2=0; for(int i=0;i<rows;i++){
            double rel=fabs((double)y_neon[i]-y2[i])/(fabs((double)y_neon[i])+1e-30);
            if(rel>worst2)worst2=rel; }
        printf("%-18s %5dx%-5d  V2 hoisted-scale+FMA %8.3f ms  %5.2fx vs NEON  %7.2f GB/s  worst rel vs NEON %.3g\n",
               "",rows,cols,t2*1e3,t_neon/t2,(double)data_bytes/t2/1e9,worst2);
        free(y2);
    }
#endif
    if(bad) printf("   %d/%d rows differ, worst rel %.3g\n",bad,rows,worst);
    free(d);free(s);free(pd);free(ps);free(x);free(y_ref);free(y_neon);
    return 0;
}

int main(int argc,char**argv){
    int reps = argc>1?atoi(argv[1]):15;
    printf("mxfp4 expert kernels, real DeepSeek-V4-Flash shapes, median of %d, threads=%d\n",
           reps,
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
           );
    int rc=0;
    rc|=run_shape("expert gate", 2048,4096,reps);
    rc|=run_shape("expert up",   2048,4096,reps);
    rc|=run_shape("expert down", 4096,2048,reps);
    return rc;
}
