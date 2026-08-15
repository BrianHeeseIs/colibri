/* What is the FIXED cost of one Metal command-buffer round trip on this host?
 * Decides whether batching 8 wo_a dispatches into 1 is worth a restructure (A5). */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
static const char*SRC=
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void nop(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]){ if(i==0) o[0]=1.0f; }\n";
int main(void){ @autoreleasepool {
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[d newCommandQueue];
  NSError*e=nil;
  MTLCompileOptions *opt=[MTLCompileOptions new]; opt.fastMathEnabled=NO;
  id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:opt error:&e];
  id<MTLComputePipelineState> pso=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nop"] error:&e];
  id<MTLBuffer> b=[d newBufferWithLength:64 options:MTLResourceStorageModeShared];
  for(int warm=0;warm<50;warm++){
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:pso];[en setBuffer:b offset:0 atIndex:0];
    [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
  }
  int N=2000; double t0=now_s();
  for(int i=0;i<N;i++){
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:pso];[en setBuffer:b offset:0 atIndex:0];
    [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
  }
  double per=(now_s()-t0)/N;
  printf("  empty dispatch + blocking wait: %.3f ms per round trip (n=%d)\n", per*1e3, N);
  printf("\n  === what A5 (batching wo_a 8 -> 1) would save on p064 ===\n");
  int saved = 7 * 43 * 2;                 /* 7 fewer round trips per layer-chunk */
  printf("  round trips eliminated: 7 x 43 layers x 2 chunks = %d\n", saved);
  printf("  fixed-overhead saving : %.0f ms\n", saved*per*1e3);
  printf("  measured dispatch_wait: 3586 ms  ->  saving is %.1f%% of it\n", saved*per*1e3/3586.0*100);
  return 0; } }
