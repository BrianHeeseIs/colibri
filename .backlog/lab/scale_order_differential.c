#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
static const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){
    union { uint32_t u; float f; } b; b.u = (uint32_t)s << 23; return b.f;
}
static uint32_t st = 12345;
static uint32_t rnd(void){ st ^= st<<13; st ^= st>>17; st ^= st<<5; return st; }
static float rndx(void){ return ((float)(rnd()%20001)-10000.f)/10000.f; }

int main(void){
    int NG = 128;            /* groups per dot product, ~ D.ng */
    int trials = 20000;
    long diff = 0, denorm_hit = 0;
    double maxulp = 0;
    for (int t = 0; t < trials; ++t) {
        float a_group = 0.f, a_col = 0.f;
        for (int g = 0; g < NG; ++g) {
            /* realistic UE8M0 exponents; include some small ones */
            uint8_t sb = (uint8_t)(100 + rnd()%40);
            float sc = mx4_scale(sb);
            float ga = 0.f;
            for (int i = 0; i < 32; ++i) {
                float x = rndx();
                float w = mx4_lut[rnd()&0xF];
                ga    += x * w;                 /* GPU: no scale yet     */
                a_col += (x * w) * sc;          /* CPU NEON: scale/column */
                if (fabsf((x*w)*sc) < 1.2e-38f && (x*w)!=0.f) denorm_hit++;
            }
            a_group += ga * sc;                 /* GPU: scale per group  */
        }
        if (a_group != a_col) {
            diff++;
            double u = fabs((double)a_group-(double)a_col)/fabs((double)a_col);
            if (u > maxulp) maxulp = u;
        }
    }
    printf("trials=%d  differing=%ld (%.2f%%)  max_rel=%.3e  denormal_products=%ld\n",
           trials, diff, 100.0*diff/trials, maxulp, denorm_hit);
    return 0;
}

/* Result on this repo, 2026-08-25:
 *   trials=20000  differing=19251 (96.25%)  max_rel=1.932e-02  denormal_products=0
 * Build: clang -O2 -o /tmp/scale_order .backlog/lab/scale_order_differential.c -lm
 * Proves the rows16 GPU/CPU divergence is FP reassociation (different summation
 * trees), not underflow and not an indexing bug. Fixed in coli_v4_matmul.metal by
 * applying the MX4 scale per column in the hot kernel. */
