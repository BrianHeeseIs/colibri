/* test_fp8_rows16 -- the aarch64 NEON rows16 path in coli_fp8_matvec_ref must be BIT-EXACT.
 *
 * WHY THIS TEST EXISTS AND WHAT IT GUARDS
 * `coli_fp8_matvec_ref` is 32.4% of decode (E123). Its only vector path is #ifdef __AVX2__, so ARM
 * runs the scalar `matmul_fp8`. The new path decodes 16 weights per step by reinterpreting e4m3 as
 * f16 -- and that reinterpret maps the e4m3 NaN codes 0x7F/0xFF onto a FINITE f16 unless an explicit
 * select rescues them. A uniformly random weight fill essentially never produces 0x7F, so the NaN
 * bug would ship silently. This test therefore plants 0x7F/0xFF, subnormals and signed zero
 * deliberately, and compares with memcmp -- NO epsilon. An epsilon would accept exactly the
 * order-breaking variants that were already measured and rejected (E124, and V2/V4 in the
 * microbench), which is the whole point of the exercise.
 *
 * Build: make -C c tests/test_fp8_rows16
 */
/* quant.h's matmul_fp8 references these (E66d min-product tracker); the engine TU defines them,
 * a standalone test must -- same as validation/probes/attn_fp8_exact.m:19-21. */
float coli_fp8_minprod = 3.4e38f;
int   coli_fp8_minprod_enabled = 0;

#include "../native_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Engagement counter: 0 when the path never ran. Required by the project's E101 rule -- a passing
 * comparison proves nothing about whether the code under test actually executed. */
extern unsigned long long coli_v4_fp8_rows16_tiles(void);
extern void coli_v4_fp8_rows16_reset(void);
extern void coli_v4_fp8_rows16_set(int on);

#define FP8_BLK 128

static int build_case(const char *name, int I, int O, int plant_specials) {
    size_t wn = (size_t)I * O;
    size_t sn = (size_t)((O + 127) / 128) * ((I + FP8_BLK - 1) / FP8_BLK);
    uint8_t *w = malloc(wn);
    float *sc = malloc(sn * sizeof(float));
    float *x = malloc((size_t)I * sizeof(float));
    float *y_ref = malloc((size_t)O * sizeof(float));
    float *y_neon = malloc((size_t)O * sizeof(float));
    if (!w || !sc || !x || !y_ref || !y_neon) { fprintf(stderr, "alloc\n"); return 1; }

    unsigned s = 987654321u;
    for (size_t i = 0; i < wn; i++) { s = s * 1103515245u + 12345u; w[i] = (uint8_t)(s >> 16); }
    if (plant_specials) {
        /* The cases the reinterpret decode gets wrong without help, placed where they cannot be
         * skipped: first tile, first columns, and again deep inside a later block. */
        const uint8_t specials[] = { 0x7F, 0xFF, 0x00, 0x80, 0x01, 0x81, 0x08, 0x88 };
        for (int r = 0; r < (O < 16 ? O : 16); r++)
            for (unsigned k = 0; k < sizeof(specials); k++)
                w[(size_t)r * I + k] = specials[k];
        if (I > FP8_BLK + 8 && O > 0)
            for (unsigned k = 0; k < sizeof(specials); k++)
                w[(size_t)0 * I + FP8_BLK + k] = specials[k];
    }
    for (size_t i = 0; i < sn; i++) sc[i] = 0.00390625f * (float)(1 + (i % 5));
    for (int i = 0; i < I; i++) { s = s * 1103515245u + 12345u; x[i] = (float)((int)(s >> 20) - 2048) / 2048.0f; }

    ColiTensorView v;
    memset(&v, 0, sizeof(v));
    v.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    v.scale_format = COLI_SCALE_F32;
    v.data = w; v.scales = (const uint8_t *)sc;
    v.rows = O; v.columns = I;
    v.block_rows = 128; v.block_columns = 128;
    v.data_bytes = wn; v.scale_bytes = sn * sizeof(float);

    /* Arm 1: reference. Flag OFF. */
    coli_v4_fp8_rows16_set(0);
    coli_v4_fp8_rows16_reset();
    if (coli_fp8_matvec_ref(y_ref, &v, x) != 0) { fprintf(stderr, "%s: ref failed\n", name); return 1; }
    unsigned long long tiles_off = coli_v4_fp8_rows16_tiles();

    /* Arm 2: NEON rows16. Flag ON. */
    coli_v4_fp8_rows16_set(1);
    coli_v4_fp8_rows16_reset();
    if (coli_fp8_matvec_ref(y_neon, &v, x) != 0) { fprintf(stderr, "%s: neon failed\n", name); return 1; }
    unsigned long long tiles_on = coli_v4_fp8_rows16_tiles();

    int rc = 0;
    if (tiles_off != 0) {
        fprintf(stderr, "%s: counter must be 0 with flag off, got %llu\n", name, tiles_off); rc = 1;
    }
#if defined(__aarch64__)
    /* Fewer than 16 rows means no whole tile exists, so declining to scalar is CORRECT, not a
     * failure. Only demand engagement where a tile is actually available. */
    if (O >= 16 && tiles_on == 0) {
        fprintf(stderr, "%s: NEON path did not execute (counter 0 with flag on)\n", name); rc = 1;
    }
    if (O < 16 && tiles_on != 0) {
        fprintf(stderr, "%s: expected scalar fallback below one tile, got tiles=%llu\n",
                name, tiles_on); rc = 1;
    }
    if (O >= 16 && tiles_on != (unsigned long long)(O / 16)) {
        fprintf(stderr, "%s: tile count %llu != expected %d\n", name, tiles_on, O / 16); rc = 1;
    }
#endif
    if (memcmp(y_ref, y_neon, (size_t)O * sizeof(float)) != 0) {
        int bad = 0, first = -1;
        for (int i = 0; i < O; i++)
            if (memcmp(&y_ref[i], &y_neon[i], sizeof(float))) { if (first < 0) first = i; bad++; }
        fprintf(stderr, "%s: NOT BIT-EXACT: %d/%d rows differ, first at %d ref=%.9g neon=%.9g\n",
                name, bad, O, first, (double)y_ref[first], (double)y_neon[first]);
        rc = 1;
    }
    if (!rc) printf("  ok %-22s I=%-5d O=%-6d tiles=%llu\n", name, I, O, tiles_on);
    free(w); free(sc); free(x); free(y_ref); free(y_neon);
    return rc;
}

int main(void) {
    int rc = 0;
    /* the four real DeepSeek-V4-Flash attention shapes */
    rc |= build_case("wq_b",        1024, 32768, 1);
    rc |= build_case("wkv",         4096,   512, 1);
    rc |= build_case("wo_a group",  4096,  1024, 1);
    rc |= build_case("wo_b",        8192,  4096, 1);
    /* tail: O not a multiple of 16 must still be exact (scalar remainder) */
    rc |= build_case("tail O=1000", 1024,  1000, 1);
    rc |= build_case("tail O=17",   256,     17, 1);
    /* degenerate widths */
    rc |= build_case("O=1",         256,      1, 1);
    if (rc) { fprintf(stderr, "FAIL test_fp8_rows16\n"); return 1; }
    printf("PASS test_fp8_rows16\n");
    return 0;
}
