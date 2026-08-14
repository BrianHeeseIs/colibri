#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }

// Real V4 gate/up shape: I=4096 reduction, O=2048 outputs, MXFP4 blocks of 32 (ng=I/32=128)
#define I_DIM 4096
#define O_DIM 2048
#define BLK   32

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"constant float E2M1[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};\n"
// one threadgroup = one output row block; each thread owns one output row; loops over S vectors.
// Weights for a row are read ONCE and reused across all S -> arithmetic intensity scales with S.
"kernel void mxfp4_sweep(device const uchar*  W   [[buffer(0)]],\n"   // [O][I/2] packed nibbles
"                        device const uchar*  SC  [[buffer(1)]],\n"   // [O][I/BLK] ue8m0 exponents
"                        device const float*  X   [[buffer(2)]],\n"   // [S][I]
"                        device float*        Y   [[buffer(3)]],\n"   // [S][O]
"                        constant uint&       S   [[buffer(4)]],\n"
"                        uint row [[thread_position_in_grid]]) {\n"
"  if (row >= %d) return;\n"
"  const uint I = %d, NB = I/%d, rb = I/2;\n"
"  device const uchar *w  = W  + (ulong)row*rb;\n"
"  device const uchar *sc = SC + (ulong)row*NB;\n"
"  for (uint s = 0; s < S; ++s) {\n"
"    device const float *x = X + (ulong)s*I;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < NB; ++b) {\n"
"      float g = 0.0f;\n"                                     // two-level: serial inside block
"      uint base = b*%d;\n"
"      for (uint k = 0; k < %d; k += 2) {\n"
"        uchar byte = w[(base+k)>>1];\n"
"        g += E2M1[byte & 0xF]      * x[base+k];\n"
"        g += E2M1[(byte >> 4)&0xF] * x[base+k+1];\n"
"      }\n"
"      acc += g * exp2((float)sc[b] - 127.0f);\n"             // then across blocks
"    }\n"
"    Y[(ulong)s*%d + row] = acc;\n"
"  }\n"
"}\n";

int main(void){ @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  char src[4096]; snprintf(src,sizeof src, SRC, O_DIM, I_DIM, BLK, BLK, BLK, O_DIM);
  NSError *err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
  if(!lib){ printf("compile fail: %s\n",[[err localizedDescription]UTF8String]); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mxfp4_sweep"] error:&err];

  size_t wbytes=(size_t)O_DIM*(I_DIM/2), sbytes=(size_t)O_DIM*(I_DIM/BLK);
  id<MTLBuffer> W =[dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> SC=[dev newBufferWithLength:sbytes options:MTLResourceStorageModeShared];
  uint8_t *wp=W.contents,*sp=SC.contents;
  for(size_t i=0;i<wbytes;i++) wp[i]=(uint8_t)(i*31u);
  for(size_t i=0;i<sbytes;i++) sp[i]=127;

  printf("  shape I=%d O=%d blk=%d   weights=%.2f MB + scales=%.2f MB = %.2f MB/expert-matrix\n\n",
         I_DIM,O_DIM,BLK, wbytes/1e6, sbytes/1e6, (wbytes+sbytes)/1e6);
  printf("  %4s %10s %10s %12s %12s %10s\n","S","time_ms","GB/s","GFLOP/s","us/token","speedup");
  double base=0;
  int Ss[]={1,2,4,8,16,32,64,128};
  for(int t=0;t<8;t++){
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
      [e dispatchThreads:MTLSizeMake(O_DIM,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double dt=now_s()-t0; if(dt<best) best=dt;
    }
    double flops=2.0*(double)S*I_DIM*O_DIM;
    double bytes=(double)(wbytes+sbytes);          // weights read once per dispatch
    double us_tok=best*1e6/S;
    if(t==0) base=us_tok;
    printf("  %4u %10.3f %10.1f %12.1f %12.1f %9.2fx\n",
           S,best*1e3,bytes/best/1e9,flops/best/1e9,us_tok, base/us_tok);
  }
  printf("\n  (bytes = weight matrix read ONCE per dispatch; us/token = cost amortised per token)\n");
  return 0; }}
