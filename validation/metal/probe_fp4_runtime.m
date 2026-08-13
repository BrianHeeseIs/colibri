/* Capability probe: does RUNTIME MSL compilation work on this host, and does a GPU
 * FP4(E2M1) + UE8M0-scale decode match the CPU reference BIT-EXACTLY?
 *
 * Why this exists: the offline Metal toolchain is NOT installed on this machine
 * (`xcrun metal` -> "missing Metal Toolchain"). newLibraryWithSource: compiles MSL at
 * runtime through Metal.framework, which IS present. llama.cpp ships shader source and
 * compiles at runtime for the same portability reason. This probe proves the whole
 * pipeline -- compile, pipeline-state, buffer, dispatch, readback -- before any real
 * kernel is written.
 *
 * The decode itself is the single most important primitive for DeepSeek-V4: routed
 * experts are native FP4 with UE8M0 (power-of-two, 8-bit exponent) block scales. If the
 * GPU cannot reproduce this exactly, nothing downstream can be bit-exact either.
 *
 * Deliberately tiny (8 elements, one dispatch): the user is asleep and the fans must not
 * spin. This costs microseconds of GPU time.
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
 *         validation/metal/probe_fp4_runtime.m -o validation/metal/probe_fp4_runtime
 */
#import <Metal/Metal.h>
#import <stdio.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"inline float fp4_e2m1(uint n){\n"
"    const float mag[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};\n"
"    float v = mag[n & 7u];\n"
"    return (n & 8u) ? -v : v;\n"
"}\n"
"kernel void probe_fp4(device const uchar *packed [[buffer(0)]],\n"
"                      device const uchar *e8     [[buffer(1)]],\n"
"                      device float       *out    [[buffer(2)]],\n"
"                      uint gid [[thread_position_in_grid]]){\n"
"    uchar b   = packed[gid >> 1];\n"
"    uint  nib = (gid & 1u) ? (b >> 4) : (b & 0x0Fu);\n"
"    float s   = ldexp(1.0f, int(e8[gid >> 5]) - 127);\n"
"    out[gid]  = fp4_e2m1(nib) * s;\n"
"}\n";

/* CPU reference: must mirror the MSL above exactly. */
static float cpu_fp4_e2m1(unsigned n){
    static const float mag[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    float v = mag[n & 7u];
    return (n & 8u) ? -v : v;
}

int main(void){
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if(!d){ printf("NO METAL DEVICE\n"); return 1; }
        printf("device: %s  unified=%s\n", [[d name] UTF8String],
               [d hasUnifiedMemory] ? "yes" : "no");

        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                             options:opt error:&err];
        if(!lib){ printf("COMPILE FAILED: %s\n",[[err localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"probe_fp4"];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&err];
        if(!ps){ printf("PIPELINE FAILED: %s\n",[[err localizedDescription] UTF8String]); return 1; }
        printf("runtime MSL compile: OK\n");
        printf("  maxTotalThreadsPerThreadgroup=%lu threadExecutionWidth=%lu\n",
               (unsigned long)ps.maxTotalThreadsPerThreadgroup,
               (unsigned long)ps.threadExecutionWidth);

        enum { N = 8 };
        /* nibbles 0..7 packed low-then-high per byte */
        unsigned char packed[4] = {0x10,0x32,0x54,0x76};
        unsigned char e8[1]     = {127};            /* 2^(127-127) = 1.0 */

        id<MTLBuffer> bp = [d newBufferWithBytes:packed length:sizeof(packed)
                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> be = [d newBufferWithBytes:e8 length:sizeof(e8)
                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [d newBufferWithLength:N*sizeof(float)
                                          options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> q = [d newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:bp offset:0 atIndex:0];
        [enc setBuffer:be offset:0 atIndex:1];
        [enc setBuffer:bo offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(N,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if(cb.error){ printf("DISPATCH FAILED: %s\n",[[cb.error localizedDescription] UTF8String]); return 1; }

        const float *got = (const float *)bo.contents;
        int ok = 1;
        printf("  gpu: ");
        for(unsigned i=0;i<N;i++) printf("%.1f ", got[i]);
        printf("\n  cpu: ");
        for(unsigned i=0;i<N;i++){
            unsigned nib = (i & 1u) ? (packed[i>>1] >> 4) : (packed[i>>1] & 0x0Fu);
            float want = cpu_fp4_e2m1(nib);            /* scale 2^0 = 1 */
            printf("%.1f ", want);
            if(got[i] != want) ok = 0;
        }
        printf("\n  BIT-EXACT vs CPU reference: %s\n", ok ? "YES" : "NO");
        return ok ? 0 : 1;
    }
}
