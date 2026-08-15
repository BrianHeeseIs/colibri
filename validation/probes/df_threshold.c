/* Where exactly does double-float stop reproducing double? Sweep the exponent range. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
typedef struct { float hi, lo; } df;
static inline df two_sum(float a,float b){float s=a+b,bb=s-a,e=(a-(s-bb))+(b-bb);df r={s,e};return r;}
static inline df two_prod(float a,float b){float p=a*b,e=fmaf(a,b,-p);df r={p,e};return r;}
static inline df dfadd(df acc,float a,float b){
    df p=two_prod(a,b); df s=two_sum(acc.hi,p.hi);
    float lo=s.lo+(acc.lo+p.lo); return two_sum(s.hi,lo);
}
static uint32_t B(float f){uint32_t u;memcpy(&u,&f,4);return u;}
int main(void){
    srandom(4242);
    printf("  %8s %14s %12s %10s\n","exp_lo","magnitude","differing","pct");
    for(int e=-150; e<=0; e+=10){
        long diff=0, T=100000;
        for(long t=0;t<T;t++){
            double ad=0; df af={0,0};
            for(int bi=0;bi<32;bi++){
                float acc=ldexpf((float)((random()/(double)RAND_MAX)*2.0-1.0), e+(int)(random()%10));
                float scl=ldexpf((float)((random()/(double)RAND_MAX)),         e+(int)(random()%10));
                ad += (double)acc*(double)scl;
                af  = dfadd(af,acc,scl);
            }
            if(B((float)ad)!=B(af.hi+af.lo)) diff++;
        }
        printf("  %8d %14.3e %12ld %9.4f%%\n", e, ldexp(1.0,e), diff, 100.0*diff/T);
    }
    printf("\n  float min NORMAL = %.6e (2^-126);  float min denormal = %.6e\n", FLT_MIN, 1.4e-45);
    return 0;
}
