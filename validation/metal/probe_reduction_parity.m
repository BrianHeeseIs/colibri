/* Blocker B2, the open half: is a GPU reduction REPRODUCIBLE, and can it match a CPU
 * sequential accumulation?
 *
 * The DEFER verdict claimed bit-exactness "may be structurally impossible" because the CPU
 * rows16 path accumulates sequentially per row while the Metal moe_gemv tree-reduces across
 * 32 lanes via simd_sum. Floating-point addition is not associative, so a tree and a chain
 * genuinely CAN differ.
 *
 * But that argument conflates two very different claims:
 *   (1) GPU result != CPU result           -- plausible, and measurable here
 *   (2) GPU result is not REPRODUCIBLE     -- a much stronger claim, and testable
 *
 * (2) is what actually matters for a trustworthy engine. A deterministic GPU result that
 * differs from CPU by a bounded amount is a legitimate engineering position; a
 * nondeterministic one is not. This probe measures BOTH, plus a third case that the DEFER
 * did not consider: a GPU kernel that deliberately reproduces the CPU's sequential order.
 *
 * Tiny by design (one buffer, a few dispatches) -- the user is asleep, fans must stay quiet.
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
 *     validation/metal/probe_reduction_parity.m -o validation/metal/probe_reduction_parity
 */
#import <Metal/Metal.h>
#import <stdio.h>
#import <math.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"// (A) tree reduction across a simdgroup -- what moe_gemv does today\n"
"kernel void reduce_simd(device const float *x [[buffer(0)]],\n"
"                        device float *out     [[buffer(1)]],\n"
"                        constant uint &n      [[buffer(2)]],\n"
"                        uint tid [[thread_position_in_threadgroup]],\n"
"                        uint tpg [[threads_per_threadgroup]]){\n"
"    float acc = 0.0f;\n"
"    for(uint i = tid; i < n; i += tpg) acc += x[i];\n"
"    acc = simd_sum(acc);\n"
"    if(tid == 0) out[0] = acc;\n"
"}\n"
"// (B) single-thread sequential accumulation -- mirrors the CPU chain exactly\n"
"kernel void reduce_serial(device const float *x [[buffer(0)]],\n"
"                          device float *out     [[buffer(1)]],\n"
"                          constant uint &n      [[buffer(2)]],\n"
"                          uint tid [[thread_position_in_threadgroup]]){\n"
"    if(tid != 0) return;\n"
"    float acc = 0.0f;\n"
"    for(uint i = 0; i < n; ++i) acc += x[i];\n"
"    out[0] = acc;\n"
"}\n";

static id<MTLComputePipelineState> mkps(id<MTLDevice> d, id<MTLLibrary> lib, NSString *name){
    NSError *e=nil;
    id<MTLFunction> f=[lib newFunctionWithName:name];
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&e];
    if(!p) printf("pipeline %s FAILED: %s\n",[name UTF8String],[[e localizedDescription] UTF8String]);
    return p;
}

static float run(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                 id<MTLBuffer> bx, id<MTLBuffer> bo, uint n, uint tpg){
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:bx offset:0 atIndex:0];
    [e setBuffer:bo offset:0 atIndex:1];
    [e setBytes:&n length:sizeof(n) atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return ((const float*)bo.contents)[0];
}

int main(void){
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    NSError *err=nil;
    id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                       options:[MTLCompileOptions new] error:&err];
    if(!lib){ printf("COMPILE FAILED: %s\n",[[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> psTree = mkps(d,lib,@"reduce_simd");
    id<MTLComputePipelineState> psSer  = mkps(d,lib,@"reduce_serial");
    if(!psTree||!psSer) return 1;
    id<MTLCommandQueue> q=[d newCommandQueue];

    /* Values chosen so summation order MATTERS: mixing large and tiny magnitudes is where
     * float non-associativity actually bites. A benign vector would hide the effect. */
    enum { N = 4096 };
    static float x[N];
    for(uint i=0;i<N;i++){
        float base = (i % 7 == 0) ? 1.0e6f : 1.0e-4f;
        x[i] = base * (((i*2654435761u) >> 16 & 1023) + 1) / 512.0f;
        if(i & 1) x[i] = -x[i];
    }
    id<MTLBuffer> bx=[d newBufferWithBytes:x length:sizeof(x) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[d newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    /* CPU reference: plain sequential chain, the order rows16 uses. */
    float cpu=0.0f; for(uint i=0;i<N;i++) cpu+=x[i];

    printf("N=%u  (mixed 1e6 / 1e-4 magnitudes so order matters)\n", (unsigned)N);
    printf("cpu sequential      : %.9e  (0x%08x)\n", cpu, *(unsigned*)&cpu);

    /* (2) REPRODUCIBILITY: same kernel, same input, many runs. */
    float first = run(d,q,psTree,bx,bo,N,32);
    int stable = 1;
    for(int k=0;k<64;k++){
        float v = run(d,q,psTree,bx,bo,N,32);
        if(memcmp(&v,&first,sizeof(float))!=0){ stable=0; break; }
    }
    printf("gpu simd_sum tree   : %.9e  (0x%08x)\n", first, *(unsigned*)&first);
    printf("  reproducible over 64 runs: %s\n", stable ? "YES (bit-identical)" : "NO");

    /* (3) Can a GPU kernel deliberately reproduce the CPU chain? */
    float ser = run(d,q,psSer,bx,bo,N,32);
    printf("gpu serial chain    : %.9e  (0x%08x)\n", ser, *(unsigned*)&ser);
    printf("  bit-exact vs CPU  : %s\n",
           memcmp(&ser,&cpu,sizeof(float))==0 ? "YES" : "NO");

    /* (1) How far apart are tree and chain, in ulps of the CPU answer? */
    double rel = (cpu!=0.0f) ? fabs((double)first-(double)cpu)/fabs((double)cpu) : 0.0;
    printf("tree vs cpu rel err : %.3e\n", rel);

    printf("\nVERDICT INPUTS:\n");
    printf("  reproducible tree     = %s\n", stable?"yes":"no");
    printf("  serial kernel == cpu  = %s\n", memcmp(&ser,&cpu,sizeof(float))==0?"yes":"no");
    return 0;
  }
}
