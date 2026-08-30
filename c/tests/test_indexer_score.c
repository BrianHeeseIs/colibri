/* test_indexer_score -- parallelising the indexer's candidate scoring must be BITWISE identical,
 * and identical across thread counts. Each candidate writes one score and its inner dot products
 * are untouched, so nothing is reordered -- this test is what proves that rather than assuming it.
 * Real shapes: index_n_heads=64, index_head_dim=128. memcmp on float bits, never an epsilon. */
#include "../indexer_score.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADS 64
#define DIM   128

static int run(const char *name, int count, int adversarial) {
    float *q = malloc((size_t)HEADS*DIM*sizeof(float));
    float *kv = malloc((size_t)count*DIM*sizeof(float));
    float *hw = malloc((size_t)HEADS*sizeof(float));
    float *s0 = malloc((size_t)count*sizeof(float));
    float *s1 = malloc((size_t)count*sizeof(float));
    if(!q||!kv||!hw||!s0||!s1) return 2;
    unsigned s = 7654321u;
    #define NX ((s=s*1103515245u+12345u), ((float)((int)(s>>18)-8192))/8192.0f)
    for (size_t i=0;i<(size_t)HEADS*DIM;i++) q[i]=NX;
    for (size_t i=0;i<(size_t)count*DIM;i++) kv[i]=NX;
    for (int i=0;i<HEADS;i++) hw[i]=NX;
    if (adversarial) {
        /* cancellation + signed zero: the shapes most sensitive to any regrouping, plus values
         * that drive the fmaxf relu to exactly zero from both sides */
        for (size_t i=0;i<(size_t)HEADS*DIM;i++) q[i] = (i%4==0)?1e20f:(i%4==2?-1e20f:1.0f);
        for (size_t i=0;i<(size_t)count*DIM;i++) kv[i] = (i%3==0)?0.0f:(i%5==0?-0.0f:1.0f);
        for (int i=0;i<HEADS;i++) hw[i] = (i%2)? -1.0f : 1e-30f;
    }
    coli_v4_indexer_scores(s0,q,kv,hw,count,HEADS,DIM,0);
    coli_v4_indexer_scores(s1,q,kv,hw,count,HEADS,DIM,1);
    int bad=0;
    if (memcmp(s0,s1,(size_t)count*sizeof(float))!=0) {
        int first=-1,n=0;
        for(int i=0;i<count;i++) if(memcmp(&s0[i],&s1[i],sizeof(float))){if(first<0)first=i;n++;}
        fprintf(stderr,"%s: NOT bit-exact: %d/%d differ, first %d serial=%.9g parallel=%.9g\n",
                name,n,count,first,(double)s0[first],(double)s1[first]);
        bad=1;
    }
    if(!bad) printf("  ok %-24s count=%d\n",name,count);
    free(q);free(kv);free(hw);free(s0);free(s1);
    return bad;
}

int main(void){
    int rc=0;
    rc |= run("random count=512", 512, 0);
    rc |= run("random count=47",   47, 0);   /* not a multiple of any thread count */
    rc |= run("adversarial",      512, 1);
    rc |= run("count=1",            1, 0);
    if(rc){fprintf(stderr,"FAIL test_indexer_score\n");return 1;}
    printf("PASS test_indexer_score\n");return 0;
}
