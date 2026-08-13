/* M15 microbench: isolate the B=1 MXFP4 expert matvec that M11 measured at 1.06 ms/call
 * (28.2% of decode). Compares the shipping scalar path against candidate optimisations.
 * Bit-exactness is the gate: any variant that changes a single bit is a FAILURE, not a
 * tradeoff - the two-level accumulation order (serial within a 32-col group, then a += ga*sc
 * across groups) must be preserved per output row.
 *
 * build: cc -O3 -mcpu=native -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *           -L/opt/homebrew/opt/libomp/lib -lomp -o tests/bench_mxfp4 tests/bench_mxfp4.c -lm
 * run:   ./tests/bench_mxfp4 [iters]      (real dims: 4096->2048 gate/up, 2048->4096 down)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

static const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){ union{uint32_t u;float f;}b; b.u=(uint32_t)s<<23; return b.f; }

/* ---- BASELINE: verbatim shipping scalar path (c/quant.h:1401-1412) ---- */
static void mm_baseline(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                        int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb; const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                for(int i=base;i<base+glen;i+=2){
                    uint8_t byte=w[i>>1];
                    ga+=xs[i]*mx4_lut[byte&0xF];
                    if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4];
                }
                a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- V1: hoist the nibble branch out of the inner loop (whole 32-col groups have no tail) ----
 * Order-identical: same adds, same sequence; only the per-iteration bounds test is removed. */
static void mm_v1_notail(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                         int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb; const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                const uint8_t *wg=w+(base>>1);
                if(glen==32){                      /* fast path: 16 whole bytes */
                    for(int k=0;k<16;k++){
                        uint8_t byte=wg[k];
                        ga+=xs[base+2*k]  *mx4_lut[byte&0xF];
                        ga+=xs[base+2*k+1]*mx4_lut[byte>>4];
                    }
                }else{
                    for(int i=base;i<base+glen;i+=2){
                        uint8_t byte=w[i>>1];
                        ga+=xs[i]*mx4_lut[byte&0xF];
                        if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4];
                    }
                }
                a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- V2: two output rows per pass — walk the activation ONCE for 2 rows ----
 * Rows are independent, so per-row accumulation order is untouched. Targets the
 * activation-reread traffic that dominates a memory-bound matvec. */
static void mm_v2_2row(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                       int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o+=2){
        int o2 = (o+1<O) ? o+1 : o;
        const uint8_t *w0=q4+(int64_t)o *rb, *s0=e8s+(int64_t)o *ng;
        const uint8_t *w1=q4+(int64_t)o2*rb, *s1=e8s+(int64_t)o2*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a0=0, a1=0;
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float c0=mx4_scale(s0[g]), c1=mx4_scale(s1[g]), g0=0, g1=0;
                for(int i=base;i<base+glen;i+=2){
                    float xa=xs[i]; uint8_t b0=w0[i>>1], b1=w1[i>>1];
                    g0+=xa*mx4_lut[b0&0xF];  g1+=xa*mx4_lut[b1&0xF];
                    if(i+1<base+glen){ float xb=xs[i+1];
                        g0+=xb*mx4_lut[b0>>4]; g1+=xb*mx4_lut[b1>>4]; }
                }
                a0+=g0*c0; a1+=g1*c1;
            }
            y[(int64_t)s*O+o]=a0; if(o2!=o) y[(int64_t)s*O+o2]=a1;
        }
    }
}

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t);
                           return t.tv_sec + t.tv_nsec/1e9; }

typedef void (*mmfn)(float*,const float*,const uint8_t*,const uint8_t*,int,int,int);

static int run_case(const char*name,int S,int I,int O,int iters){
    int rb=(I+1)/2, ng=(I+31)/32;
    float *x=malloc((size_t)S*I*4);
    uint8_t *q4=malloc((size_t)O*rb), *e8=malloc((size_t)O*ng);
    float *yref=malloc((size_t)S*O*4), *y=malloc((size_t)S*O*4);
    srandom(4242);
    for(int i=0;i<S*I;i++) x[i]=((float)(random()%2001)-1000)/337.0f;
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(random()&0xFF);
    for(size_t i=0;i<(size_t)O*ng;i++) e8[i]=(uint8_t)(110+(random()%30));
    mm_baseline(yref,x,q4,e8,S,I,O);

    struct { const char*n; mmfn f; } V[] = {
        {"baseline", mm_baseline}, {"v1_notail", mm_v1_notail}, {"v2_2row", mm_v2_2row} };
    printf("  %s  S=%d I=%d O=%d\n", name, S, I, O);
    double base_mean=0;
    for(int v=0; v<3; v++){
        double sum=0,sum2=0,best=1e9;
        for(int it=0; it<iters; it++){
            double t0=now_s(); V[v].f(y,x,q4,e8,S,I,O); double d=now_s()-t0;
            if(it){ sum+=d; sum2+=d*d; }
            if(d<best) best=d;
        }
        int n=iters-1; double m=sum/n, sd=sqrt(fmax(0,sum2/n-m*m));
        int bad=0; for(int i=0;i<S*O;i++){ uint32_t a,b;
            memcpy(&a,&yref[i],4); memcpy(&b,&y[i],4); if(a!=b) bad++; }
        if(!v) base_mean=m;
        printf("    %-10s %7.3f ms +-%.3f  best %7.3f  %s%s\n", V[v].n, m*1e3, sd*1e3, best*1e3,
               bad? "BIT-DIFF " : "bit-exact",
               v? (bad? " <-- REJECT" : (m<base_mean? " <-- faster" : "")) : "");
        if(v && bad){ free(x);free(q4);free(e8);free(yref);free(y); return 1; }
    }
    free(x);free(q4);free(e8);free(yref);free(y);
    return 0;
}

int main(int argc,char**argv){
    int iters = argc>1? atoi(argv[1]) : 20;
    printf("M15 mxfp4 microbench, iters=%d (first dropped as warmup)\n", iters);
    int bad = 0;
    bad |= run_case("gate/up decode", 1, 4096, 2048, iters);
    bad |= run_case("down decode",    1, 2048, 4096, iters);
    bad |= run_case("gate/up S=8",    8, 4096, 2048, iters);
    printf("%s\n", bad? "FAIL: a variant broke bit-exactness" : "all variants bit-exact");
    return bad;
}
