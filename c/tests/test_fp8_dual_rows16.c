/* test_fp8_dual_rows16 -- the aarch64 rows16 arm of coli_fp8_dual_matvec_ref must be BITWISE
 * identical to the scalar matmul_fp8_dual path.
 *
 * This is the gate/up pair of the shared expert. The single-matvec rows16 kernel (E125) already
 * covers its down projection, which is why shared_expert fell 1619.8 -> 1372.6 ms; this closes the
 * remaining two thirds.
 *
 * The e4m3 NaN codes 0x7F/0xFF are planted deliberately: the reinterpret-as-f16 decode maps them to
 * a FINITE half and only an explicit select rescues them, and a random fill essentially never
 * produces them. Comparison is memcmp on the float bits -- no epsilon, because an epsilon would
 * accept a reordered summation, which is exactly the defect being guarded against.
 */
#include "../native_quant.h"
#include "../native_quant_dual.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float coli_fp8_minprod = 3.4e38f;
int   coli_fp8_minprod_enabled = 0;

extern void coli_v4_fp8_dual_rows16_set(int on);
extern unsigned long long coli_v4_fp8_dual_rows16_tiles(void);
extern void coli_v4_fp8_dual_rows16_reset(void);

static int run(const char *name, int rows, int cols) {
    size_t wn = (size_t)rows * cols;
    size_t sn = (size_t)((rows + 127) / 128) * (cols / 128);
    uint8_t *wa = malloc(wn), *wb = malloc(wn);
    float *sa = malloc(sn * sizeof(float)), *sb = malloc(sn * sizeof(float));
    float *x = malloc((size_t)cols * sizeof(float));
    float *ra = malloc((size_t)rows * sizeof(float)), *rb = malloc((size_t)rows * sizeof(float));
    float *na = malloc((size_t)rows * sizeof(float)), *nb = malloc((size_t)rows * sizeof(float));
    if (!wa||!wb||!sa||!sb||!x||!ra||!rb||!na||!nb) return 2;

    unsigned s = 99887766u;
    for (size_t i = 0; i < wn; i++) { s=s*1103515245u+12345u; wa[i]=(uint8_t)(s>>16); }
    for (size_t i = 0; i < wn; i++) { s=s*1103515245u+12345u; wb[i]=(uint8_t)(s>>16); }
    /* plant the codes the reinterpret decode gets wrong without an explicit rescue */
    static const uint8_t specials[] = {0x7F,0xFF,0x00,0x80,0x01,0x81,0x08,0x88};
    for (int r = 0; r < (rows < 16 ? rows : 16); r++)
        for (unsigned k = 0; k < sizeof(specials); k++) {
            wa[(size_t)r*cols+k] = specials[k];
            wb[(size_t)r*cols+k] = specials[(k+3) % sizeof(specials)];
        }
    /* block scales spanning several binades so the folded 256.0f is exercised */
    for (size_t i = 0; i < sn; i++) { sa[i]=ldexpf(1.0f,-8+(int)(i%7)); sb[i]=ldexpf(1.0f,-5+(int)(i%5)); }
    for (int i = 0; i < cols; i++) { s=s*1103515245u+12345u; x[i]=((float)((int)(s>>20)-2048))/2048.0f; }

    ColiTensorView A, B; memset(&A,0,sizeof(A));
    A.format=COLI_TENSOR_FP8_E4M3_BLOCK; A.scale_format=COLI_SCALE_F32;
    A.rows=rows; A.columns=cols; A.block_rows=128; A.block_columns=128;
    A.data_bytes=wn; A.scale_bytes=sn*sizeof(float);
    B=A; A.data=wa; A.scales=(const uint8_t*)sa; B.data=wb; B.scales=(const uint8_t*)sb;

    coli_v4_fp8_dual_rows16_set(0); coli_v4_fp8_dual_rows16_reset();
    if (coli_fp8_dual_matvec_ref(ra, rb, &A, &B, x) != 0) { fprintf(stderr,"%s: ref failed\n",name); return 1; }
    unsigned long long off = coli_v4_fp8_dual_rows16_tiles();

    coli_v4_fp8_dual_rows16_set(1); coli_v4_fp8_dual_rows16_reset();
    if (coli_fp8_dual_matvec_ref(na, nb, &A, &B, x) != 0) { fprintf(stderr,"%s: neon failed\n",name); return 1; }
    unsigned long long on = coli_v4_fp8_dual_rows16_tiles();

    int bad = 0;
    if (off != 0) { fprintf(stderr,"%s: counter %llu with flag off, want 0\n",name,off); bad++; }
    if (on == 0)  { fprintf(stderr,"%s: rows16 arm did not run (counter 0 with flag on)\n",name); bad++; }
    for (int i = 0; i < rows; i++) {
        if (memcmp(&ra[i],&na[i],sizeof(float))) {
            if (!bad) fprintf(stderr,"%s: A NOT bit-exact at %d ref=%.9g got=%.9g\n",name,i,(double)ra[i],(double)na[i]);
            bad++; break;
        }
    }
    for (int i = 0; i < rows; i++) {
        if (memcmp(&rb[i],&nb[i],sizeof(float))) {
            if (!bad) fprintf(stderr,"%s: B NOT bit-exact at %d ref=%.9g got=%.9g\n",name,i,(double)rb[i],(double)nb[i]);
            bad++; break;
        }
    }
    if (!bad) printf("  ok %-22s %5dx%-5d tiles=%llu\n",name,rows,cols,on);
    free(wa);free(wb);free(sa);free(sb);free(x);free(ra);free(rb);free(na);free(nb);
    return bad ? 1 : 0;
}

int main(void) {
    int rc = 0;
    rc |= run("shared w1/w3", 2048, 4096);   /* real shared-expert gate/up shape */
    rc |= run("tail rows=2040", 2040, 512);  /* rows % 16 != 0 -> scalar remainder must still match */
    rc |= run("small rows=16", 16, 256);
    if (rc) { fprintf(stderr,"FAIL test_fp8_dual_rows16\n"); return 1; }
    printf("PASS test_fp8_dual_rows16\n");
    return 0;
}
