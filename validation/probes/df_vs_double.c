/* Decisive probe for E65: can double-float (2 floats) reproduce the CPU's `double` block
 * accumulation in matmul_fp8 (quant.h:505-521) EXACTLY, after the final cast to float?
 * Metal has no fp64, so this decides whether a bit-exact Metal attention kernel is possible. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

typedef struct { float hi, lo; } df;

/* Dekker TwoSum / TwoProduct using FMA (exact on ARM) */
static inline df two_sum(float a, float b){
    float s=a+b, bb=s-a, err=(a-(s-bb))+(b-bb);
    df r={s,err}; return r;
}
static inline df two_prod(float a, float b){
    float p=a*b, e=fmaf(a,b,-p);
    df r={p,e}; return r;
}
static inline df df_add_prod(df acc, float a, float b){
    df p = two_prod(a,b);
    df s = two_sum(acc.hi, p.hi);
    float lo = s.lo + (acc.lo + p.lo);
    df t = two_sum(s.hi, lo);
    return t;
}

static uint32_t bits(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

int main(int argc, char**argv){
    int NB   = argc>1? atoi(argv[1]) : 32;    /* blocks: I/128, 32 for I=4096 */
    long TRIALS = argc>2? atol(argv[2]) : 200000;
    unsigned seed = argc>3? (unsigned)atoi(argv[3]) : 12345;
    int mode = argc>4? atoi(argv[4]) : 0;
    srandom(seed);
    long diff=0, ulp1=0, worse=0;
    for(long t=0;t<TRIALS;t++){
        double a_dbl = 0.0;
        df     a_df  = {0.0f,0.0f};
        for(int bi=0; bi<NB; ++bi){
            /* acc: a float block sum with realistic magnitude; scl: an fp8 block scale */
            float acc, scl;
            switch (mode) {
            case 0: /* benign */
                acc = (float)((random()/(double)RAND_MAX)*2.0-1.0) * (float)(1<<(random()%8));
                scl = (float)((random()/(double)RAND_MAX)) * 0.01f + 1e-5f;
                break;
            case 1: /* WIDE dynamic range: exponents spanning ~1e-20..1e20 */
                acc = ldexpf((float)((random()/(double)RAND_MAX)*2.0-1.0), (int)(random()%120)-60);
                scl = ldexpf((float)((random()/(double)RAND_MAX)),        (int)(random()%120)-60);
                break;
            case 2: /* NEAR-CANCELLATION: alternating huge opposite terms */
                acc = (bi & 1) ? -1.0e7f : 1.0e7f;
                acc += (float)((random()/(double)RAND_MAX)-0.5)*1e-3f;
                scl = 1.0f + (float)((random()/(double)RAND_MAX)-0.5)*1e-6f;
                break;
            default: /* DENORMAL / tiny */
                acc = ldexpf((float)((random()/(double)RAND_MAX)*2.0-1.0), -(int)(random()%20)-120);
                scl = ldexpf((float)((random()/(double)RAND_MAX)),         -(int)(random()%20)-120);
                break;
            }
            a_dbl += (double)acc * (double)scl;
            a_df   = df_add_prod(a_df, acc, scl);
        }
        float r_dbl = (float)a_dbl;
        float r_df  = a_df.hi + a_df.lo;
        if (bits(r_dbl)!=bits(r_df)) {
            diff++;
            int64_t d = (int64_t)bits(r_dbl) - (int64_t)bits(r_df);
            if (d==1||d==-1) ulp1++; else worse++;
        }
    }
    static const char *mn[4]={"benign","wide-range","near-cancel","denormal"};
    printf("  mode=%-12s blocks=%d trials=%ld seed=%u\n", mn[mode&3], NB, TRIALS, seed);
    printf("  differing = %ld  (%.6f%%)   of which 1ULP=%ld  >1ULP=%ld\n",
           diff, 100.0*diff/TRIALS, ulp1, worse);
    printf("  VERDICT: %s\n", diff==0 ? "double-float is EXACT here -> bit-exact Metal kernel POSSIBLE"
                                      : "NOT exact -> bit-exact Metal fp8 kernel IMPOSSIBLE this way");
    return diff==0?0:1;
}
