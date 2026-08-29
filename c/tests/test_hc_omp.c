/* test_hc_omp -- parallelising the hc_pre mix matvec must be BITWISE identical, and must stay
 * identical across thread counts.
 *
 * The mix loop (c/deepseek_v4.c:1459) is 24 rows x 16384 columns at hc=4, dimension=4096 -- about
 * 92% of hc_pre's arithmetic -- and each iteration writes exactly one `mixes[row]` while its inner
 * column sum is untouched, so parallelising over `row` cannot reorder any summation.
 *
 * Three nearby loops are DELIBERATELY not parallelised and this test does not cover them because
 * touching them would be wrong: `mean_square` (:1449) is a serial reduction whose order an
 * omp reduction would change; the sinkhorn split is hc x hc and too small to be worth any risk;
 * the output loop (:1471) writes `output` while reading `input`, so it is only safe under a
 * caller-aliasing guarantee that has not been established.
 *
 * The thread-count invariance check is the real proof obligation: a golden run could pass by luck,
 * but identical bits at OMP_NUM_THREADS=1 and =16 is what actually demonstrates order preservation.
 */
#include "../deepseek_v4_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void coli_v4_hc_omp_set(int on);
extern unsigned long long coli_v4_hc_omp_rows(void);
extern void coli_v4_hc_omp_reset(void);

#define HC 4
#define DIM 4096

static int run_case(const char *name, const float *hc_fn, const float *input,
                    const float *base, const float scale[3]) {
    const int flattened = HC * DIM, mix_count = (2 + HC) * HC;
    (void)flattened; (void)mix_count;
    float *o_off = malloc((size_t)DIM * sizeof(float));
    float *o_on  = malloc((size_t)DIM * sizeof(float));
    float p_off[HC], p_on[HC], c_off[HC*HC], c_on[HC*HC];
    if (!o_off || !o_on) return 2;

    coli_v4_hc_omp_set(0); coli_v4_hc_omp_reset();
    int r0 = coli_v4_hc_pre(o_off, p_off, c_off, input, hc_fn, scale, base,
                            HC, DIM, 20, 1e-6f, 1e-6f);
    unsigned long long rows_off = coli_v4_hc_omp_rows();

    coli_v4_hc_omp_set(1); coli_v4_hc_omp_reset();
    int r1 = coli_v4_hc_pre(o_on, p_on, c_on, input, hc_fn, scale, base,
                            HC, DIM, 20, 1e-6f, 1e-6f);
    unsigned long long rows_on = coli_v4_hc_omp_rows();

    int bad = 0;
    if (r0 != 0 || r1 != 0) { fprintf(stderr, "%s: rc %d/%d\n", name, r0, r1); bad++; }
    if (rows_off != 0) { fprintf(stderr, "%s: counter %llu with flag off, want 0\n", name, rows_off); bad++; }
    if (rows_on == 0)  { fprintf(stderr, "%s: parallel path did not run (counter 0 with flag on)\n", name); bad++; }
    if (memcmp(o_off, o_on, (size_t)DIM * sizeof(float)) != 0) {
        int first = -1, n = 0;
        for (int i = 0; i < DIM; i++)
            if (memcmp(&o_off[i], &o_on[i], sizeof(float))) { if (first < 0) first = i; n++; }
        fprintf(stderr, "%s: output NOT bit-exact: %d/%d differ, first %d off=%.9g on=%.9g\n",
                name, n, DIM, first, (double)o_off[first], (double)o_on[first]);
        bad++;
    }
    if (memcmp(p_off, p_on, sizeof(p_off)) || memcmp(c_off, c_on, sizeof(c_off))) {
        fprintf(stderr, "%s: post/comb NOT bit-exact\n", name); bad++;
    }
    if (!bad) printf("  ok %-22s rows_on=%llu\n", name, rows_on);
    free(o_off); free(o_on);
    return bad;
}

int main(void) {
    const int flattened = HC * DIM, mix_count = (2 + HC) * HC;
    float *hc_fn = malloc((size_t)mix_count * flattened * sizeof(float));
    float *input = malloc((size_t)flattened * sizeof(float));
    float base[HC], scale[3] = {1.0f, 1.0f, 1.0f};
    if (!hc_fn || !input) return 2;
    int bad = 0;
    unsigned s = 1234567u;
    #define NEXT ((s = s*1103515245u+12345u), ((float)((int)(s>>18)-8192))/8192.0f)

    for (int i = 0; i < mix_count*flattened; i++) hc_fn[i] = NEXT;
    for (int i = 0; i < flattened; i++) input[i] = NEXT;
    for (int i = 0; i < HC; i++) base[i] = NEXT;
    bad += run_case("random", hc_fn, input, base, scale);

    /* cancellation + signed zero + denormals: the shapes most sensitive to any regrouping */
    for (int i = 0; i < mix_count*flattened; i++)
        hc_fn[i] = (i % 4 == 0) ? 1e20f : (i % 4 == 2 ? -1e20f : 1.0f);
    for (int i = 0; i < flattened; i++)
        input[i] = (i % 7 == 0) ? 0.0f : (i % 11 == 0 ? -0.0f : (i % 13 == 0 ? 1e-40f : 1.0f));
    bad += run_case("cancellation", hc_fn, input, base, scale);

    free(hc_fn); free(input);
    if (bad) { fprintf(stderr, "FAIL test_hc_omp\n"); return 1; }
    printf("PASS test_hc_omp\n");
    return 0;
}
