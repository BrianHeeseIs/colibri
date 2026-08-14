#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static double now_s(void){
  static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9;
}

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void bwread(device const float4* src [[buffer(0)]],\n"
"                   device float* out [[buffer(1)]],\n"
"                   constant uint& n4 [[buffer(2)]],\n"
"                   uint gid [[thread_position_in_grid]],\n"
"                   uint gsz [[threads_per_grid]]) {\n"
"  float4 a = float4(0.0f);\n"
"  for (uint i = gid; i < n4; i += gsz) a += src[i];\n"
"  float s = a.x+a.y+a.z+a.w;\n"
"  if (s == 12345.678f) out[0] = s;\n"   // never true; defeats DCE without a write race
"}\n";

int main(int argc,char**argv){ @autoreleasepool {
  size_t MB = (argc>1)? atoll(argv[1]) : 4096;
  size_t bytes = MB*1024ull*1024ull;
  size_t n4 = bytes/16;
  int iters = 3;

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  // shared buffer: SAME physical memory for CPU and GPU (unified) - this is the fair test
  id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
  float *p = (float*)buf.contents;
  for (size_t i=0;i<bytes/4;i+=1024) p[i]=1.0f;   // touch pages
  memset(p, 1, bytes);

  printf("  buffer %zu MB (shared / unified)\n\n", MB);

  // ---------- CPU: multi-threaded streaming read via GCD ----------
  size_t nth = 12;                      // P-cores
  double best_cpu = 0;
  for (int it=0; it<iters; ++it){
    __block double dummy=0;
    double t0=now_s();
    dispatch_apply(nth, DISPATCH_APPLY_AUTO, ^(size_t t){
      const float *q = (const float*)p;
      size_t N = bytes/4, chunk = N/nth, s = t*chunk, e = (t==nth-1)? N : s+chunk;
      float acc0=0,acc1=0,acc2=0,acc3=0;
      for (size_t i=s;i+3<e;i+=4){ acc0+=q[i]; acc1+=q[i+1]; acc2+=q[i+2]; acc3+=q[i+3]; }
      if (acc0+acc1+acc2+acc3 == 12345.678f) dummy += 1;
    });
    double dt=now_s()-t0; double gbs=bytes/dt/1e9;
    if (gbs>best_cpu) best_cpu=gbs;
    printf("  CPU  read  iter%d  %6.1f GB/s  (%.3f s)%s\n", it, gbs, dt, dummy>0?" ":"");
  }

  // ---------- GPU: streaming read of the SAME buffer ----------
  NSError *err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
  if(!lib){ printf("  shader compile failed: %s\n",[[err localizedDescription]UTF8String]); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bwread"] error:&err];
  id<MTLCommandQueue> q=[dev newCommandQueue];
  uint32_t n4u=(uint32_t)n4;
  double best_gpu=0;
  for(int it=0; it<iters; ++it){
    double t0=now_s();
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:buf offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&n4u length:4 atIndex:2];
    NSUInteger tg=pso.maxTotalThreadsPerThreadgroup;      // saturate
    NSUInteger grid=tg*1024;
    [e dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    double dt=now_s()-t0; double gbs=bytes/dt/1e9;
    if(gbs>best_gpu) best_gpu=gbs;
    printf("  GPU  read  iter%d  %6.1f GB/s  (%.3f s)\n", it, gbs, dt);
  }

  printf("\n  BEST  CPU %.1f GB/s   GPU %.1f GB/s   ratio GPU/CPU = %.2fx\n",
         best_cpu, best_gpu, best_gpu/best_cpu);
  printf("  => premise 'unified memory gives the GPU no bandwidth advantage' is %s\n",
         (best_gpu > best_cpu*1.15) ? "**FALSE** on this host" : "supported");
  return 0; }}
