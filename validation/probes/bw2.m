#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <string.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
// GPU kernel now writes a REAL per-thread result -> loads cannot be dead-code eliminated.
static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void bwread(device const float4* src [[buffer(0)]],\n"
"                   device float* out [[buffer(1)]],\n"
"                   constant uint& n4 [[buffer(2)]],\n"
"                   uint gid [[thread_position_in_grid]],\n"
"                   uint gsz [[threads_per_grid]]) {\n"
"  float4 a = float4(0.0f);\n"
"  for (uint i = gid; i < n4; i += gsz) a += src[i];\n"
"  out[gid] = a.x+a.y+a.z+a.w;\n"   // UNCONDITIONAL write - result is observable
"}\n";
int main(int argc,char**argv){ @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError*err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bwread"] error:&err];
  printf("  %8s %12s %12s %10s   %s\n","sizeMB","CPU GB/s","GPU GB/s","ratio","checksum match");
  size_t sizes[]={512,1024,2048,4096,8192};
  for(int si=0; si<5; ++si){
    size_t MB=sizes[si], bytes=MB*1024ull*1024ull, n4=bytes/16;
    id<MTLBuffer> buf=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    float *p=(float*)buf.contents;
    for(size_t i=0;i<bytes/4;i++) p[i]=1.0f;                    // real data, sum is known
    NSUInteger tg=pso.maxTotalThreadsPerThreadgroup, grid=tg*1024;
    id<MTLBuffer> out=[dev newBufferWithLength:grid*sizeof(float) options:MTLResourceStorageModeShared];
    // CPU: accumulate into a real array, written out -> not eliminable
    size_t nth=12; double best_cpu=0; static double cpu_sums[12];
    for(int it=0; it<4; ++it){
      double t0=now_s();
      dispatch_apply(nth, DISPATCH_APPLY_AUTO, ^(size_t t){
        const float *x=(const float*)p; size_t N=bytes/4, c=N/nth, s=t*c, e=(t==nth-1)?N:s+c;
        float a0=0,a1=0,a2=0,a3=0;
        for(size_t i=s;i+3<e;i+=4){ a0+=x[i]; a1+=x[i+1]; a2+=x[i+2]; a3+=x[i+3]; }
        cpu_sums[t]=(double)a0+a1+a2+a3;
      });
      double dt=now_s()-t0, g=bytes/dt/1e9; if(g>best_cpu) best_cpu=g;
    }
    double cpu_total=0; for(size_t t=0;t<nth;t++) cpu_total+=cpu_sums[t];
    uint32_t n4u=(uint32_t)n4; double best_gpu=0;
    for(int it=0; it<4; ++it){
      double t0=now_s();
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      [e setBuffer:buf offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
      [e setBytes:&n4u length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double dt=now_s()-t0, g=bytes/dt/1e9; if(g>best_gpu) best_gpu=g;
    }
    double gpu_total=0; float*op=(float*)out.contents; for(NSUInteger i=0;i<grid;i++) gpu_total+=op[i];
    double expect=(double)(bytes/4);
    int ok = (fabs(cpu_total-expect)/expect < 1e-3) && (fabs(gpu_total-expect)/expect < 1e-3);
    printf("  %8zu %12.1f %12.1f %9.2fx   %s (cpu=%.0f gpu=%.0f expect=%.0f)\n",
           MB,best_cpu,best_gpu,best_gpu/best_cpu, ok?"YES":"NO", cpu_total,gpu_total,expect);
  }
  return 0; }}
