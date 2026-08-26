// probe_simd_parity -- correctness gate for coli_v4_matmul_mxfp4_simd.
//
// WHY THIS EXISTS: validation/metal/bench_matmul.m dispatches every variant except `ordered`
// with out=NULL (:99 for simd), and computes its mismatch count only against the `ordered`
// result (:95). So the simd kernel has been TIMED many times and never once CHECKED. A kernel
// returning garbage benchmarks exactly as fast as a correct one. This probe closes that.
//
// simd replaces the serial 32-column accumulation with simd_sum(), a tree reduction. It is
// therefore NOT expected to be bit-exact against the scalar reference; it IS expected to be
// numerically correct. Bit mismatches are reported for information; the PASS/FAIL gate is the
// relative error.
//
// Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation \
//          -o validation/metal/probe_simd_parity validation/metal/probe_simd_parity.m
// Run from the REPO ROOT (it reads c/metal/coli_v4_matmul.metal).
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

static const float mx4_lut[16] = { 0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f };
static inline float mx4_scale(uint8_t s) { union { uint32_t u; float f; } b; b.u = (uint32_t)s << 23; return b.f; }

// Scalar reference: byte-for-byte the loop nest of coli_v4_matmul_mxfp4_ordered
// (c/metal/coli_v4_matmul.metal:25), which is itself the c/quant.h:1401-1412 scalar path.
static void cpu_mm(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I, int O) {
    int rb = (I + 1) / 2, ng = (I + 31) / 32;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb, *scl = e8s + (int64_t)o * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I; float a = 0;
            for (int g = 0; g < ng; g++) {
                int base = g * 32, gl = 32; if (base + gl > I) gl = I - base;
                float sc = mx4_scale(scl[g]), ga = 0;
                for (int i = base; i < base + gl; i += 2) {
                    uint8_t by = w[i >> 1];
                    ga += xs[i] * mx4_lut[by & 0xF];
                    if (i + 1 < base + gl) ga += xs[i + 1] * mx4_lut[by >> 4];
                }
                a += ga * sc;
            }
            y[(int64_t)s * O + o] = a;
        }
    }
}

typedef struct { int S, I, O, rb, ng; } Dims;
static id<MTLDevice> DEV; static id<MTLCommandQueue> QUE; static id<MTLLibrary> LIB;

static int gpu_run(const char *entry, int S, int I, int O,
                   const uint8_t *q4, const uint8_t *e8, const float *x,
                   size_t wb, size_t sb, float *out) {
    int isSimd = (strstr(entry, "simd") != NULL);
    NSError *e = nil;
    id<MTLFunction> fn = [LIB newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if (!fn) { printf("    !! no such kernel: %s\n", entry); return 0; }
    id<MTLComputePipelineState> ps = [DEV newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) { printf("    !! pipeline fail: %s\n", entry); return 0; }
    int rb = (I + 1) / 2, ng = (I + 31) / 32; Dims dm = { S, I, O, rb, ng };
    id<MTLBuffer> bx = [DEV newBufferWithBytes:x length:(size_t)S * I * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bq = [DEV newBufferWithBytes:q4 length:wb options:MTLResourceStorageModeShared];
    id<MTLBuffer> be = [DEV newBufferWithBytes:e8 length:sb options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [DEV newBufferWithLength:(size_t)S * O * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bd = [DEV newBufferWithBytes:&dm length:sizeof(dm) options:0];
    id<MTLCommandBuffer> cb = [QUE commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:ps];
    [en setBuffer:bx offset:0 atIndex:0]; [en setBuffer:bq offset:0 atIndex:1];
    [en setBuffer:be offset:0 atIndex:2]; [en setBuffer:by offset:0 atIndex:3];
    [en setBuffer:bd offset:0 atIndex:4];
    NSUInteger tg = ps.maxTotalThreadsPerThreadgroup > 256 ? 256 : ps.maxTotalThreadsPerThreadgroup;
    if (isSimd) [en dispatchThreads:MTLSizeMake(32, S * O, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    else        [en dispatchThreads:MTLSizeMake(S * O, 1, 1)  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(out, by.contents, (size_t)S * O * 4);
    return 1;
}

static int fails = 0;

static void check_seed(const char *name, int S, int I, int O, unsigned seed, int spread) {
    int rb = (I + 1) / 2, ng = (I + 31) / 32, n = S * O;
    float *x = malloc((size_t)S * I * 4);
    uint8_t *q4 = malloc((size_t)O * rb), *e8 = malloc((size_t)O * ng);
    srandom(seed);
    for (int i = 0; i < S * I; i++) x[i] = ((float)(random() % 2001) - 1000) / 337.0f;
    for (size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)(random() & 0xFF);
    /* `spread` is the width of the UE8M0 exponent band. The original probe used 30, which puts
     * per-group scales across 2^-17..2^12 within a single row -- far wider than real MXFP4
     * weights and deliberately adversarial for the outer accumulation. spread=4 approximates a
     * realistic band. It changes the ERROR of the non-exact `simd`; it must not change
     * simd_exact, which is bit-exact for any input. */
    for (size_t i = 0; i < (size_t)O * ng; i++) e8[i] = (uint8_t)(127 - spread / 2 + (random() % (spread ? spread : 1)));
    float *yc = malloc((size_t)n * 4), *yo = malloc((size_t)n * 4);
    float *ys = malloc((size_t)n * 4), *ye = malloc((size_t)n * 4);
    cpu_mm(yc, x, q4, e8, S, I, O);
    int ok_o = gpu_run("coli_v4_matmul_mxfp4_ordered",    S, I, O, q4, e8, x, (size_t)O * rb, (size_t)O * ng, yo);
    int ok_s = gpu_run("coli_v4_matmul_mxfp4_simd",       S, I, O, q4, e8, x, (size_t)O * rb, (size_t)O * ng, ys);
    int ok_e = gpu_run("coli_v4_matmul_mxfp4_simd_exact", S, I, O, q4, e8, x, (size_t)O * rb, (size_t)O * ng, ye);

    int bit_o = 0, bit_s = 0, bit_e = 0, nan_s = 0, nan_e = 0;
    double maxabs = 0, maxrel = 0, maxabs_e = 0, maxrel_e = 0;
    for (int i = 0; i < n; i++) {
        unsigned a, b, c, d2; memcpy(&a, &yc[i], 4); memcpy(&b, &yo[i], 4);
        memcpy(&c, &ys[i], 4); memcpy(&d2, &ye[i], 4);
        if (a != b) bit_o++;
        if (a != c) bit_s++;
        if (a != d2) bit_e++;
        if (isnan(ys[i]) || isinf(ys[i])) nan_s++;
        if (isnan(ye[i]) || isinf(ye[i])) nan_e++;
        double ref = fabs((double)yc[i]);
        double d = fabs((double)ys[i] - (double)yc[i]);
        if (d > maxabs) maxabs = d;
        if (ref > 1e-4) { double r = d / ref; if (r > maxrel) maxrel = r; }
        double de = fabs((double)ye[i] - (double)yc[i]);
        if (de > maxabs_e) maxabs_e = de;
        if (ref > 1e-4) { double r = de / ref; if (r > maxrel_e) maxrel_e = r; }
    }
    const double TOL = 1e-5;
    int pass_s = ok_s && nan_s == 0 && maxrel < TOL;
    int pass_e = ok_e && nan_e == 0 && bit_e == 0;          // exact variant: BIT-exactness is the gate
    int pass = ok_o && bit_o == 0 && pass_e;                // simd_exact failing is a hard fail
    if (!pass) fails++;
    printf("  %-30s S=%d I=%d O=%d  seed=%u scale_spread=%d\n", name, S, I, O, seed, spread);
    printf("      ordered    vs cpu : %d bit-mismatches   (control, MUST be 0)\n", bit_o);
    printf("      simd       vs cpu : %d/%d bit-mism  nan=%d  max_abs=%.3e max_rel=%.3e  [%s tol %.0e]\n",
           bit_s, n, nan_s, maxabs, maxrel, pass_s ? "ok" : "OVER", TOL);
    printf("      simd_exact vs cpu : %d/%d bit-mism  nan=%d  max_abs=%.3e max_rel=%.3e  [%s]\n",
           bit_e, n, nan_e, maxabs_e, maxrel_e, pass_e ? "BIT-EXACT" : "*** NOT BIT-EXACT ***");
    printf("      => %s\n\n", pass ? "PASS" : "*** FAIL ***");
    free(x); free(q4); free(e8); free(yc); free(yo); free(ys); free(ye);
}

int main(void) { @autoreleasepool {
    FILE *f = fopen("c/metal/coli_v4_matmul.metal", "rb");
    if (!f) { printf("run me from the repo root (need c/metal/coli_v4_matmul.metal)\n"); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = malloc(n + 1); size_t rd = fread(src, 1, n, f); src[rd] = 0; fclose(f);
    DEV = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    MTLCompileOptions *o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe;   // fast-math MUST stay off
    LIB = [DEV newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if (!LIB) { printf("compile fail: %s\n", [[e localizedDescription] UTF8String]); return 3; }
    QUE = [DEV newCommandQueue];
    printf("\n  probe_simd_parity -- is coli_v4_matmul_mxfp4_simd CORRECT?\n");
    printf("  threadExecutionWidth assumption: one 32-thread threadgroup == one simdgroup\n\n");

    // S1: production shapes, adversarial 30-exponent scale band (the original probe's data)
    check_seed("gate|up S=1 (production)",  1, 4096, 2048, 7, 30);
    check_seed("down    S=1 (production)",  1, 2048, 4096, 7, 30);
    check_seed("gate|up S=8 (prefill-ish)", 8, 4096, 2048, 7, 30);
    // S1b: realistic narrow scale band -- isolates how much of `simd`'s error is the data
    check_seed("gate|up S=1 narrow-scale",  1, 4096, 2048, 7, 4);
    check_seed("down    S=1 narrow-scale",  1, 2048, 4096, 7, 4);
    // S1c: seed variation -- guards the bit-exactness claim against one lucky dataset
    check_seed("gate|up S=1 seed=101",      1, 4096, 2048, 101, 30);
    check_seed("gate|up S=1 seed=99991",    1, 4096, 2048, 99991, 12);
    check_seed("down    S=1 seed=424242",   1, 2048, 4096, 424242, 20);
    // S2: ragged tails -- I not a multiple of 32, O not a multiple of 16, odd I (odd-nibble tail)
    check_seed("ragged  I=100 O=37",        1,  100,   37, 7, 30);
    check_seed("ragged  I=33  O=17 (odd)",  3,   33,   17, 7, 30);
    check_seed("ragged  I=31  O=1  (sub)",  1,   31,    1, 7, 30);
    check_seed("ragged  I=65  O=33 seed=5", 2,   65,    33, 5, 16);

    printf("  %s  (%d failing shape%s)\n\n", fails ? "*** OVERALL FAIL ***" : "OVERALL PASS",
           fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
} }
