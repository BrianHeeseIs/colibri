// probe_rows16_parity -- WHICH reference does each hot (rows16) Metal kernel actually match?
//
// Motivation: the single-expert Metal entry hard-rejects block_rows != 1, so every PINNED hot
// expert (rows16) runs on the CPU. Measured on a 24-token decode run: 2218 of 4902 expert calls,
// 45%, refused there. Project memory says widening that acceptance "BREAKS GOLDEN" but records
// no mechanism. This probe finds the mechanism.
//
// There are TWO different CPU references in this engine and they are NOT the same function:
//
//   COLD  (c/quant.h scalar matmul_mxfp4, mirrored by coli_v4_matmul_mxfp4_ordered):
//         per 32-column group:  ga += x*w      <- CONTRACTED to one fma by the compiler
//         then                  a  += ga*sc     <- scale applied ONCE per group
//
//   ROWS16 (neon_rows16_accumulate in c/deepseek_v4.c, the live path for pinned hot experts):
//         sums = vaddq_f32(sums, vmulq_f32(vmulq_f32(x, values), scales));
//         i.e.  sum = sum + ((x*w)*scale)      <- scale applied PER COLUMN, and these are three
//         SEPARATELY ROUNDED NEON instructions: mul, mul, add. There is NO fma.
//
// coli_v4_matmul_mxfp4_ordered_hot_xcache writes `a += (x*w)*sc`, which a clang-based compiler
// is free to contract into fma(x*w, sc, a) -- fusing away the second rounding that NEON performs.
// If that is happening, the hot GPU kernel is not bit-exact to the CPU path it claims to mirror.
//
// Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation \
//          -o validation/metal/probe_rows16_parity validation/metal/probe_rows16_parity.m
// Run from the REPO ROOT.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

static const float mx4_lut[16] = { 0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f };
static inline float mx4_scale(uint8_t s){ union{uint32_t u;float f;}b; b.u=(uint32_t)s<<23; return b.f; }

// COLD reference. Contraction LEFT ON deliberately: the cold GPU kernels match this form.
static void cpu_cold(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                     int I, int O) {
    int rb=(I+1)/2, ng=(I+31)/32;
    for (int o=0;o<O;o++){ const uint8_t*w=q4+(size_t)o*rb,*scl=e8s+(size_t)o*ng; float a=0;
        for(int g=0;g<ng;g++){ int base=g*32,gl=32; if(base+gl>I)gl=I-base;
            float sc=mx4_scale(scl[g]), ga=0;
            for(int i=base;i<base+gl;i+=2){ uint8_t by=w[i>>1];
                ga += x[i]*mx4_lut[by&0xF];
                if(i+1<base+gl) ga += x[i+1]*mx4_lut[by>>4]; }
            a += ga*sc; }
        y[o]=a; } }

// ROWS16 reference, replicating neon_rows16_accumulate EXACTLY.
// contract(off) is load-bearing: NEON does vmul, vmul, vadd as three rounded instructions, so
// the reference must not fuse them either.
#pragma clang fp contract(off)
static void cpu_rows16(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                       int I, int O) {
    int rb=(I+1)/2, ng=(I+31)/32;
    for (int o=0;o<O;o++){ const uint8_t*w=q4+(size_t)o*rb,*scl=e8s+(size_t)o*ng; float sum=0;
        for(int g=0;g<ng;g++){ int base=g*32,gl=32; if(base+gl>I)gl=I-base;
            float sc=mx4_scale(scl[g]);
            for(int i=base;i<base+gl;i++){ uint8_t by=w[i>>1];
                float wv = (i&1) ? mx4_lut[by>>4] : mx4_lut[by&0xF];
                sum = sum + ((x[i]*wv)*sc); } }      /* three roundings, per column */
        y[o]=sum; } }
#pragma clang fp contract(on)

// rows16 pack: packed[(tile*stride + col)*16 + lane] = data[row*stride + col]
static void pack16(uint8_t *p, const uint8_t *d, int rows, int stride) {
    for (int r=0;r<rows;r++){ int t=r/16,l=r%16;
        for(int c=0;c<stride;c++) p[(size_t)(t*stride+c)*16+l]=d[(size_t)r*stride+c]; } }

typedef struct { int S,I,O,rb,ng; } Dims;
static id<MTLDevice> DEV; static id<MTLCommandQueue> QUE; static id<MTLLibrary> LIB;

static int gpu(const char *entry, int I, int O, const uint8_t *q4, const uint8_t *e8,
               const float *x, size_t wb, size_t sb, float *out) {
    NSError *e=nil;
    id<MTLFunction> fn=[LIB newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if(!fn){ printf("    !! missing kernel %s\n", entry); return 0; }
    id<MTLComputePipelineState> ps=[DEV newComputePipelineStateWithFunction:fn error:&e];
    if(!ps){ printf("    !! pipeline fail %s\n", entry); return 0; }
    int rb=(I+1)/2,ng=(I+31)/32; Dims dm={1,I,O,rb,ng};
    id<MTLBuffer> bx=[DEV newBufferWithBytes:x length:(size_t)I*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bq=[DEV newBufferWithBytes:q4 length:wb options:MTLResourceStorageModeShared];
    id<MTLBuffer> be=[DEV newBufferWithBytes:e8 length:sb options:MTLResourceStorageModeShared];
    id<MTLBuffer> by=[DEV newBufferWithLength:(size_t)O*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bd=[DEV newBufferWithBytes:&dm length:sizeof(dm) options:0];
    id<MTLCommandBuffer> cb=[QUE commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];
    [en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bq offset:0 atIndex:1];
    [en setBuffer:be offset:0 atIndex:2];[en setBuffer:by offset:0 atIndex:3];
    [en setBytes:&dm length:sizeof(dm) atIndex:4];
    int isX=(strstr(entry,"xcache")!=NULL), isS=(strstr(entry,"simd")!=NULL);
    if(isS)      [en dispatchThreads:MTLSizeMake(32,O,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    else if(isX) [en dispatchThreads:MTLSizeMake(O,1,1)  threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    else         [en dispatchThreads:MTLSizeMake(O,1,1)  threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(out, by.contents, (size_t)O*4);
    return 1; }

static int mism(const float*a,const float*b,int n){int m=0;for(int i=0;i<n;i++){
    unsigned u,v;memcpy(&u,&a[i],4);memcpy(&v,&b[i],4);if(u!=v)m++;}return m;}

static void shape(const char *name, int I, int O, int scale_lo, int scale_span) {
    int rb=(I+1)/2, ng=(I+31)/32, padO=((O+15)/16)*16;
    float *x=malloc((size_t)I*4);
    uint8_t *q4=malloc((size_t)O*rb), *e8=malloc((size_t)O*ng);
    srandom(7);
    for(int i=0;i<I;i++) x[i]=((float)(random()%2001)-1000)/337.0f;
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(random()&0xFF);
    /* scale_lo controls the UE8M0 exponent band. Near 127 the products are normal floats.
     * Low codes make (x*w)*sc DENORMAL: 2^(code-127), so code 20 is ~2^-107 and the per-column
     * product underflows. That matters because Metal GPUs flush denormals to zero while the
     * NEON path does not, and the rows16 form multiplies by the scale on EVERY column whereas
     * the cold form multiplies once per 32-column group. */
    for(size_t i=0;i<(size_t)O*ng;i++) e8[i]=(uint8_t)(scale_lo+(random()%(scale_span?scale_span:1)));
    uint8_t *q4h=calloc((size_t)padO*rb,1), *e8h=calloc((size_t)padO*ng,1);
    pack16(q4h,q4,O,rb); pack16(e8h,e8,O,ng);

    float *ref_cold=malloc((size_t)O*4), *ref_r16=malloc((size_t)O*4);
    float *g_hot=malloc((size_t)O*4), *g_hotx=malloc((size_t)O*4), *g_hotxn=malloc((size_t)O*4);
    cpu_cold(ref_cold,x,q4,e8,I,O);
    cpu_rows16(ref_r16,x,q4,e8,I,O);

    gpu("coli_v4_matmul_mxfp4_ordered_hot",           I,O,q4h,e8h,x,(size_t)padO*rb,(size_t)padO*ng,g_hot);
    gpu("coli_v4_matmul_mxfp4_ordered_hot_xcache",    I,O,q4h,e8h,x,(size_t)padO*rb,(size_t)padO*ng,g_hotx);
    int have_nc = gpu("coli_v4_matmul_mxfp4_hot_xcache_nofma",
                                                      I,O,q4h,e8h,x,(size_t)padO*rb,(size_t)padO*ng,g_hotxn);

    printf("  %s  I=%d O=%d  scale_band=[%d,%d)\n", name, I, O, scale_lo, scale_lo+scale_span);
    printf("    cold ref vs rows16 ref : %d/%d differ  (the two CPU references are NOT the same function)\n",
           mism(ref_cold,ref_r16,O), O);
    printf("    ordered_hot        vs cold=%-5d  vs rows16=%-5d\n", mism(g_hot,ref_cold,O),  mism(g_hot,ref_r16,O));
    printf("    ordered_hot_xcache vs cold=%-5d  vs rows16=%-5d   <-- the kernel the seam uses for rows16\n",
           mism(g_hotx,ref_cold,O), mism(g_hotx,ref_r16,O));
    if (have_nc)
        printf("    hot_xcache_nofma   vs cold=%-5d  vs rows16=%-5d   <-- contraction suppressed\n",
               mism(g_hotxn,ref_cold,O), mism(g_hotxn,ref_r16,O));
    else
        printf("    hot_xcache_nofma   (kernel not present yet)\n");
    printf("\n");
    free(x);free(q4);free(e8);free(q4h);free(e8h);
    free(ref_cold);free(ref_r16);free(g_hot);free(g_hotx);free(g_hotxn); }

int main(void){ @autoreleasepool {
    FILE *f=fopen("c/metal/coli_v4_matmul.metal","rb");
    if(!f){ printf("run from the repo root\n"); return 2; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *src=malloc(n+1); size_t rd=fread(src,1,n,f); src[rd]=0; fclose(f);
    DEV=MTLCreateSystemDefaultDevice(); NSError *e=nil;
    MTLCompileOptions *o=[MTLCompileOptions new]; o.mathMode=MTLMathModeSafe;
    LIB=[DEV newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!LIB){ printf("compile fail: %s\n",[[e localizedDescription]UTF8String]); return 3; }
    QUE=[DEV newCommandQueue];
    printf("\n  probe_rows16_parity -- which CPU reference does each hot kernel match?\n\n");
    shape("gate|up NORMAL scales ", 4096, 2048, 120, 15);
    shape("down    NORMAL scales ", 2048, 4096, 120, 15);
    /* denormal-producing bands: 2^(code-127) small enough that (x*w)*sc underflows fp32 */
    shape("gate|up TINY   scales ", 4096, 2048,  10, 15);
    shape("gate|up WIDE   scales ", 4096, 2048,  10, 120);
    return 0; } }
