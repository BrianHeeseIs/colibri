/* DECISIVE PROBE: does FMA contraction break CPU/GPU bit-exactness?
 *
 * The plan agent flagged this as the #1 risk, and it is a sharper objection than the
 * reduction-order argument that the original DEFER rested on.
 *
 * My earlier reduction probe used `acc += x[i]` -- a bare add, which cannot contract.
 * The REAL inner loop of matmul_mxfp4 (quant.h:1405-1409) is:
 *       ga += xs[i] * mx4_lut[nibble];
 * a multiply-then-add. Both clang (CPU, -mcpu=native) and the Metal compiler are free to
 * contract that into a single fused multiply-add, which rounds ONCE instead of TWICE.
 * If the two sides make different choices, the "exact" kernel is not bit-identical no
 * matter how faithfully the summation order is reproduced.
 *
 * This probe measures four things:
 *   1. CPU with contraction ON  (default -ffp-contract=fast under -O2/-mcpu)
 *   2. CPU with contraction OFF (#pragma clang fp contract(off))
 *   3. GPU default             (Metal defaults to fast-math: contraction + reassoc ON)
 *   4. GPU with fast-math OFF  (MTLCompileOptions.fastMathEnabled = NO)
 * and reports which combinations agree BIT-EXACTLY.
 *
 * Inputs are chosen so that fma(a,b,c) != a*b+c in the last bit -- otherwise the probe
 * would produce a falsely reassuring "everything matches".
 *
 * Tiny: one dispatch of 1 thread. The user is asleep; this costs microseconds.
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
 *     validation/metal/probe_fma_contraction.m -o validation/metal/probe_fma_contraction
 */
#import <Metal/Metal.h>
#import <stdio.h>
#import <math.h>

#define N 8

/* Values engineered so the product needs rounding before the add. */
static void make_inputs(float *x, float *w){
    /* 1 + 2^-23 style perturbations: the product's low bits fall off the end of the
     * accumulator, which is exactly where FMA and mul-then-add diverge. */
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f + (float)(i + 1) * 0x1p-23f;
        w[i] = 1.0f + (float)(i + 1) * 0x1p-23f;
    }
}

/* CPU accumulation with contraction explicitly ALLOWED. */
__attribute__((noinline))
static float cpu_contract_on(const float *x, const float *w){
#pragma clang fp contract(fast)
    float a = 0.0f;
    for (int i = 0; i < N; i++) a += x[i] * w[i];
    return a;
}

/* CPU accumulation with contraction explicitly FORBIDDEN. */
__attribute__((noinline))
static float cpu_contract_off(const float *x, const float *w){
#pragma clang fp contract(off)
    float a = 0.0f;
    for (int i = 0; i < N; i++) a += x[i] * w[i];
    return a;
}

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void mac_chain(device const float *x [[buffer(0)]],\n"
"                      device const float *w [[buffer(1)]],\n"
"                      device float *out     [[buffer(2)]],\n"
"                      uint tid [[thread_position_in_grid]]){\n"
"    if (tid != 0) return;\n"
"    float a = 0.0f;\n"
"    for (int i = 0; i < 8; ++i) a += x[i] * w[i];\n"
"    out[0] = a;\n"
"}\n";

static float gpu_run(id<MTLDevice> d, BOOL fastMath, const float *x, const float *w, BOOL *ok){
    NSError *err = nil;
    MTLCompileOptions *opt = [MTLCompileOptions new];
    /* fastMathEnabled controls contraction + reassociation. Deprecated in newer SDKs in
     * favour of mathMode, so set whichever this SDK exposes. */
#if defined(__MAC_26_0)
    opt.mathMode = fastMath ? MTLMathModeFast : MTLMathModeSafe;
#else
    opt.fastMathEnabled = fastMath;
#endif
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                         options:opt error:&err];
    if(!lib){ printf("  compile failed (%s): %s\n", fastMath?"fast":"safe",
                     [[err localizedDescription] UTF8String]); *ok=NO; return 0; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mac_chain"];
    id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&err];
    if(!ps){ *ok=NO; return 0; }
    id<MTLBuffer> bx=[d newBufferWithBytes:x length:N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw=[d newBufferWithBytes:w length:N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q=[d newCommandQueue];
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bo offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    *ok = (cb.error == nil);
    return ((const float*)bo.contents)[0];
}

static void show(const char *label, float v){
    unsigned u; memcpy(&u, &v, 4);
    printf("  %-28s %.9e  0x%08x\n", label, v, u);
}

int main(void){
  @autoreleasepool {
    float x[N], w[N];
    make_inputs(x, w);

    /* Prove the inputs actually discriminate: fma() vs mul-then-add on one term. */
    float m  = x[3] * w[3];
    float f  = fmaf(x[3], w[3], 1.0f);
    float ma = m + 1.0f;
    printf("discriminating inputs: fma=%.9e (0x%08x)  mul+add=%.9e (0x%08x)  differ=%s\n",
           f, *(unsigned*)&f, ma, *(unsigned*)&ma, (memcmp(&f,&ma,4)!=0) ? "YES" : "NO");

    float c_on  = cpu_contract_on(x, w);
    float c_off = cpu_contract_off(x, w);
    show("cpu contract(fast)", c_on);
    show("cpu contract(off)",  c_off);
    printf("  cpu on == cpu off : %s\n", memcmp(&c_on,&c_off,4)==0 ? "YES (compiler did not contract)" : "NO (contraction is REAL here)");

    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    BOOL ok1=YES, ok2=YES;
    float g_fast = gpu_run(d, YES, x, w, &ok1);
    float g_safe = gpu_run(d, NO,  x, w, &ok2);
    if(ok1) show("gpu fastMath ON",  g_fast);
    if(ok2) show("gpu fastMath OFF", g_safe);

    printf("\nMATCH MATRIX (bit-exact?)\n");
    printf("  gpu_safe == cpu_off  : %s   <-- the pairing the EXACT kernel needs\n",
           (ok2 && memcmp(&g_safe,&c_off,4)==0) ? "YES" : "NO");
    printf("  gpu_safe == cpu_on   : %s\n", (ok2 && memcmp(&g_safe,&c_on,4)==0)  ? "YES" : "NO");
    printf("  gpu_fast == cpu_on   : %s\n", (ok1 && memcmp(&g_fast,&c_on,4)==0)  ? "YES" : "NO");
    printf("  gpu_fast == cpu_off  : %s\n", (ok1 && memcmp(&g_fast,&c_off,4)==0) ? "YES" : "NO");
    return 0;
  }
}
