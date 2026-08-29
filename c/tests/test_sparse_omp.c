/* test_sparse_omp -- parallelising the sparse-attention head loop must be BITWISE identical.
 *
 * coli_v4_sparse_attention_ref loops 64 independent heads serially; each head writes only its own
 * slice of `output`, so parallelising over `head` cannot reorder any summation. Two things make it
 * non-trivial and both are covered here: the `scores` scratch was SHARED across heads (it must
 * become per-iteration), and the original loop can `return -1` from inside, which is illegal in an
 * OpenMP region and must become a flag.
 *
 * Real shapes: heads=64, head_dimension=512, index_topk=512. Comparison is memcmp on float bits.
 * The error path (an index >= kv_count) is exercised explicitly, because converting a `return` into
 * a flag is exactly the kind of change that silently stops reporting failures.
 */
/* quant.h's matmul_fp8 references these; the engine TU defines them, a standalone test must. */
float coli_fp8_minprod = 3.4e38f;
int   coli_fp8_minprod_enabled = 0;

#include "../deepseek_v4_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void coli_v4_sparse_omp_set(int on);
extern unsigned long long coli_v4_sparse_omp_heads(void);
extern void coli_v4_sparse_omp_reset(void);

#define HEADS 64
#define HDIM  512
#define TOPK  512
#define KVN   1024

static int run(const char *name, int corrupt_index) {
    float *q = malloc((size_t)HEADS*HDIM*sizeof(float));
    float *kv = malloc((size_t)KVN*HDIM*sizeof(float));
    float *sinks = malloc((size_t)HEADS*sizeof(float));
    int *idx = malloc((size_t)TOPK*sizeof(int));
    float *o_off = malloc((size_t)HEADS*HDIM*sizeof(float));
    float *o_on  = malloc((size_t)HEADS*HDIM*sizeof(float));
    if(!q||!kv||!sinks||!idx||!o_off||!o_on) return 2;
    unsigned s = 5150u;
    #define NX ((s=s*1103515245u+12345u), ((float)((int)(s>>18)-8192))/8192.0f)
    for (size_t i=0;i<(size_t)HEADS*HDIM;i++) q[i]=NX;
    for (size_t i=0;i<(size_t)KVN*HDIM;i++) kv[i]=NX;
    for (int i=0;i<HEADS;i++) sinks[i]=NX;
    for (int i=0;i<TOPK;i++) idx[i] = (i%7==0) ? -1 : (int)((s=s*1103515245u+12345u)>>20)%KVN;
    if (corrupt_index) idx[TOPK/2] = KVN + 5;      /* must make BOTH arms return -1 */

    coli_v4_sparse_omp_set(0); coli_v4_sparse_omp_reset();
    int r0 = coli_v4_sparse_attention_ref(o_off,q,kv,sinks,idx,HEADS,HDIM,KVN,TOPK,0.08838f);
    unsigned long long h_off = coli_v4_sparse_omp_heads();
    coli_v4_sparse_omp_set(1); coli_v4_sparse_omp_reset();
    int r1 = coli_v4_sparse_attention_ref(o_on ,q,kv,sinks,idx,HEADS,HDIM,KVN,TOPK,0.08838f);
    unsigned long long h_on = coli_v4_sparse_omp_heads();

    int bad = 0;
    if (r0 != r1) { fprintf(stderr,"%s: rc mismatch %d vs %d\n",name,r0,r1); bad++; }
    if (corrupt_index) {
        if (r0 != -1 || r1 != -1) { fprintf(stderr,"%s: expected -1 from both, got %d/%d\n",name,r0,r1); bad++; }
        if (!bad) printf("  ok %-26s both reject (rc=-1)\n",name);
    } else {
        if (h_off != 0) { fprintf(stderr,"%s: counter %llu with flag off\n",name,h_off); bad++; }
        if (h_on == 0)  { fprintf(stderr,"%s: parallel path did not run\n",name); bad++; }
        if (memcmp(o_off,o_on,(size_t)HEADS*HDIM*sizeof(float))!=0) {
            int first=-1,n=0;
            for (size_t i=0;i<(size_t)HEADS*HDIM;i++)
                if (memcmp(&o_off[i],&o_on[i],sizeof(float))) { if(first<0)first=(int)i; n++; }
            fprintf(stderr,"%s: NOT bit-exact: %d differ, first %d off=%.9g on=%.9g\n",
                    name,n,first,(double)o_off[first],(double)o_on[first]); bad++;
        }
        if (!bad) printf("  ok %-26s heads_on=%llu\n",name,h_on);
    }
    free(q);free(kv);free(sinks);free(idx);free(o_off);free(o_on);
    return bad?1:0;
}

int main(void){
    int rc = 0;
    rc |= run("bit-exactness", 0);
    rc |= run("error path preserved", 1);
    if (rc) { fprintf(stderr,"FAIL test_sparse_omp\n"); return 1; }
    printf("PASS test_sparse_omp\n"); return 0;
}
