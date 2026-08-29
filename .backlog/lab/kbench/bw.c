/* Physical ceiling probe for the fp8 GEMV question: what read bandwidth can this host actually
 * sustain? A 1-byte-per-weight matvec must stream the whole weight matrix once per token, so this
 * number is the denominator for "are we bandwidth-bound or latency-bound". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
    size_t mb = argc>1?(size_t)atoi(argv[1]):2048;      /* buffer size MB */
    int reps  = argc>2?atoi(argv[2]):5;
    size_t n = mb*1024*1024;
    uint8_t *b = aligned_alloc(64,n);
    if(!b){fprintf(stderr,"alloc failed\n");return 1;}
    memset(b,1,n);                                        /* fault it in */
    /* Byte-sum read, the access pattern an fp8 matvec performs over its weights. */
    double best=1e18;
    for(int r=0;r<reps;r++){
        double t0=now(); uint64_t s=0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for(size_t i=0;i<n;i+=64) s+=b[i];               /* one touch per cache line */
        double dt=now()-t0; if(dt<best)best=dt;
        if(s==0)fprintf(stderr,"x");
    }
    printf("linescan  %6.2f GB/s  (%zu MB, best of %d)\n",(double)n/best/1e9,mb,reps);
    best=1e18;
    for(int r=0;r<reps;r++){
        double t0=now(); uint64_t s=0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for(size_t i=0;i<n;i++) s+=b[i];                 /* every byte, like a real matvec */
        double dt=now()-t0; if(dt<best)best=dt;
        if(s==0)fprintf(stderr,"x");
    }
    printf("bytesum   %6.2f GB/s  (%zu MB, best of %d)\n",(double)n/best/1e9,mb,reps);
#ifdef _OPENMP
    printf("omp_max_threads=%d\n",omp_get_max_threads());
#endif
    free(b); return 0;
}
