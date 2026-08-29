/* test_head_ilp -- the 4-row LM head kernel must be BITWISE identical to the 1-row reference.
 *
 * Compared with memcmp on the float object representation, never an epsilon: an epsilon would
 * silently accept a reordered summation, which is the exact defect this test exists to catch.
 * The head is the argmax layer, so a one-ulp logit change can flip a near-tie into a different
 * token -- that is why nothing less than bitwise equality is acceptable here.
 */
#include "../head_ilp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t bf16_of(float f) { uint32_t b; memcpy(&b,&f,4); return (uint16_t)(b >> 16); }

static int check(const char *name, const uint16_t *w, const float *h, int d, int rows) {
    int bad = 0;
    for (int r0 = 0; r0 + 4 <= rows; r0 += 4) {
        float ref[4], got[4];
        for (int k = 0; k < 4; k++) ref[k] = coli_v4_head_dot1(w + (size_t)(r0+k)*d, h, d);
        coli_v4_head_dot4(w + (size_t)(r0+0)*d, w + (size_t)(r0+1)*d,
                          w + (size_t)(r0+2)*d, w + (size_t)(r0+3)*d, h, d, got);
        for (int k = 0; k < 4; k++)
            if (memcmp(&ref[k], &got[k], sizeof(float)) != 0) {
                if (!bad) fprintf(stderr, "%s: row %d ref=%.9g got=%.9g\n",
                                  name, r0+k, (double)ref[k], (double)got[k]);
                bad++;
            }
    }
    if (!bad) printf("  ok %-28s d=%-5d rows=%d\n", name, d, rows);
    return bad;
}

int main(void) {
    const int d = 4096, rows = 8;
    uint16_t *w = malloc((size_t)rows * d * sizeof(*w));
    float *h = malloc((size_t)d * sizeof(*h));
    if (!w || !h) return 2;
    int bad = 0;

    /* 1. random-ish realistic values */
    unsigned s = 22695477u;
    /* Build from finite floats: a raw random u16 is often an inf/NaN bf16 encoding, which makes
     * every dot product NaN and hides real arithmetic differences behind NaN propagation. */
    for (int i = 0; i < rows*d; i++) {
        s = s*1103515245u+12345u;
        w[i] = bf16_of(((float)((int)(s>>18)-8192))/4096.0f);
    }
    for (int i = 0; i < d; i++) { s = s*1103515245u+12345u; h[i] = ((float)((int)(s>>20)-2048))/2048.0f; }
    bad += check("random", w, h, d, rows);

    /* 2. adversarial bf16 encodings: +0 -0 +inf -inf NaN denormal max-normal */
    static const uint16_t special[] = {0x0000,0x8000,0x7F80,0xFF80,0x7FC0,0x0001,0x7F7F,0x3F80};
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < d; i++)
            w[(size_t)r*d+i] = special[(i + r) % (int)(sizeof(special)/sizeof(special[0]))];
    for (int i = 0; i < d; i++) h[i] = (i % 3 == 0) ? 0.0f : ((i % 5 == 0) ? -0.0f : 1.5f);
    bad += check("adversarial codes", w, h, d, rows);

    /* 3. REORDERING CANARY. Alternating +1e20 / -1e20 with 1.0 between: the sequential sum
     *    cancels the huge terms pairwise and leaves the small ones, whereas ANY regrouping
     *    (e.g. 4 lanes striding one row, or summing partials out of order) loses the small
     *    terms to absorption and yields a different bit pattern. This case is GREEN only for a
     *    genuinely order-preserving implementation. */
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < d; i++) {
            float v = (i % 4 == 0) ? 1e20f : (i % 4 == 2 ? -1e20f : 1.0f);
            w[(size_t)r*d+i] = bf16_of(v);
        }
    for (int i = 0; i < d; i++) h[i] = 1.0f;
    bad += check("reordering canary", w, h, d, rows);

    free(w); free(h);
    if (bad) { fprintf(stderr, "FAIL test_head_ilp (%d mismatching rows)\n", bad); return 1; }
    printf("PASS test_head_ilp\n");
    return 0;
}
