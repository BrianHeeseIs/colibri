#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
#define I_DIM 4096
#define O_DIM 2048
#define BLK   32
#define TILE_I 256          // I-chunk staged in threadgroup memory
#define TG     256          // threads per threadgroup

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"constant float E2M1[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};\n"
"kernel void mxfp4_tiled(device const uchar4* W  [[buffer(0)]],\n"
"                        device const uchar*  SC [[buffer(1)]],\n"
"                        device const float*  X  [[buffer(2)]],\n"
"                        device float*        Y  [[buffer(3)]],\n"
"                        constant uint&       S  [[buffer(4)]],\n"
"                        threadgroup float*   xs [[threadgroup(0)]],\n"
"                        uint tid  [[thread_index_in_threadgroup]],\n"
"                        uint tgid [[threadgroup_position_in_grid]],\n"
"                        uint tgsz [[threads_per_threadgroup]]) {\n"
"  const uint I=%d, NB=I/%d, TI=%d;\n"
"  uint row = tgid*tgsz + tid;\n"
"  bool live = (row < %du);\n"
"  device const uchar4 *w4 = W + (ulong)row*(I/8);\n"   // 8 nibbles per uchar4
"  device const uchar  *sc = SC + (ulong)row*NB;\n"
"  float acc[%d];\n"
"  for (uint s=0;s<S;++s) acc[s]=0.0f;\n"
"  for (uint t0=0; t0<I; t0+=TI) {\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s=0;s<S;++s)\n"
"      for (uint i=tid; i<TI; i+=tgsz) xs[s*TI+i] = X[(ulong)s*I + t0 + i];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (live) {\n"
"      for (uint b=0; b<TI/%d; ++b) {\n"
"        uint blk = (t0/%d)+b;\n"
"        float scale = exp2((float)sc[blk]-127.0f);\n"
"        uint wbase = (t0 + b*%d)/8;\n"
"        float g[%d];\n"
"        for (uint s=0;s<S;++s) g[s]=0.0f;\n"
"        for (uint j=0;j<%d/8;++j) {\n"
"          uchar4 pk = w4[wbase+j];\n"
"          float wv[8];\n"
"          wv[0]=E2M1[pk.x&0xF]; wv[1]=E2M1[(pk.x>>4)&0xF];\n"
"          wv[2]=E2M1[pk.y&0xF]; wv[3]=E2M1[(pk.y>>4)&0xF];\n"
"          wv[4]=E2M1[pk.z&0xF]; wv[5]=E2M1[(pk.z>>4)&0xF];\n"
"          wv[6]=E2M1[pk.w&0xF]; wv[7]=E2M1[(pk.w>>4)&0xF];\n"
"          uint xo = b*%d + j*8;\n"
"          for (uint s=0;s<S;++s) {\n"
"            threadgroup const float *xp = xs + s*TI + xo;\n"
"            float p=0.0f;\n"
"            for (uint k=0;k<8;++k) p += wv[k]*xp[k];\n"
"            g[s]+=p;\n"
"          }\n"
"        }\n"
"        for (uint s=0;s<S;++s) acc[s] += g[s]*scale;\n"
"      }\n"
"    }\n"
"  }\n"
"  if (live) for (uint s=0;s<S;++s) Y[(ulong)s*%du + row] = acc[s];\n"
"}\n";

int main(void){ @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  int MAXS=16;
  char src[8192]; snprintf(src,sizeof src,SRC,I_DIM,BLK,TILE_I,O_DIM,MAXS,BLK,BLK,BLK,MAXS,BLK,BLK,O_DIM);
  NSError*err=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
  if(!lib){ printf("compile fail: %s\n",[[err localizedDescription]UTF8String]); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mxfp4_tiled"] error:&err];
  if(!pso){ printf("pso fail: %s\n",[[err localizedDescription]UTF8String]); return 1; }
  size_t wb=(size_t)O_DIM*(I_DIM/2), sb=(size_t)O_DIM*(I_DIM/BLK);
  id<MTLBuffer> W=[dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
  id<MTLBuffer> SC=[dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
  uint8_t*wp=W.contents,*sp=SC.contents;
  for(size_t i=0;i<wb;i++) wp[i]=(uint8_t)(i*31u);
  for(size_t i=0;i<sb;i++) sp[i]=127;
  printf("  tiled+uchar4 kernel, TILE_I=%d TG=%d\n\n",TILE_I,TG);
  printf("  %4s %10s %10s %12s %12s %9s\n","S","time_ms","GB/s","GFLOP/s","us/token","speedup");
  double base=0;
  int Ss[]={1,2,4,8,16};
  for(int t=0;t<5;t++){
    uint32_t S=Ss[t];
    id<MTLBuffer> X=[dev newBufferWithLength:(size_t)S*I_DIM*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y=[dev newBufferWithLength:(size_t)S*O_DIM*4 options:MTLResourceStorageModeShared];
    float*xp=X.contents; for(size_t i=0;i<(size_t)S*I_DIM;i++) xp[i]=(float)((i%17)-8)*0.125f;
    double best=1e9;
    for(int it=0;it<5;it++){
      double t0=now_s();
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:SC offset:0 atIndex:1];
      [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
      [e setBytes:&S length:4 atIndex:4];
      [e setThreadgroupMemoryLength:S*TILE_I*sizeof(float) atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(O_DIM/TG,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double dt=now_s()-t0; if(dt<best) best=dt;
    }
    double flops=2.0*(double)S*I_DIM*O_DIM, bytes=(double)(wb+sb), us=best*1e6/S;
    if(t==0) base=us;
    printf("  %4u %10.3f %10.1f %12.1f %12.1f %8.2fx\n",S,best*1e3,bytes/best/1e9,flops/best/1e9,us,base/us);
  }
  return 0; }}
