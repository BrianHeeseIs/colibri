/* Isolate: does the INNER float loop already diverge between C and Metal?
 * Single 128-element block => exactly ONE outer accumulation, so the outer accumulator
 * cannot be the explanation. If this differs, the cause is the inner loop (FMA contraction). */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float coli_fp8_minprod=3.4e38f; int coli_fp8_minprod_enabled=0;
#include "../../c/quant.h"
static uint32_t B32(float f){uint32_t u;memcpy(&u,&f,4);return u;}

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void inner(device const uchar* W [[buffer(0)]],\n"
"                  device const float* X [[buffer(1)]],\n"
"                  device float* Y       [[buffer(2)]],\n"
"                  device const float* LUT[[buffer(3)]],\n"
"                  constant uint& N      [[buffer(4)]],\n"
"                  constant uint& mode   [[buffer(5)]],\n"
"                  uint gid [[thread_position_in_grid]]) {\n"
"  if (gid!=0) return;\n"
"  float acc = 0.0f;\n"
"  if (mode==0u) { for(uint i=0;i<N;++i) acc += LUT[W[i]] * X[i]; }            // may contract to FMA\n"
"  else          { for(uint i=0;i<N;++i) acc = fma(LUT[W[i]], X[i], acc); }    // explicit FMA\n"
"  Y[0]=acc;\n"
"}\n";

int main(int argc,char**argv){ @autoreleasepool {
  int N=128;
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError*e=nil;
  MTLCompileOptions *opt=[MTLCompileOptions new];
  opt.fastMathEnabled = NO;   /* CRITICAL: fast math destroys Dekker two_sum/two_prod */
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:opt error:&e];
  if(!lib){printf("compile: %s\n",[[e localizedDescription]UTF8String]);return 2;}
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"inner"] error:&e];
  id<MTLBuffer> W=[dev newBufferWithLength:N options:MTLResourceStorageModeShared];
  id<MTLBuffer> X=[dev newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Y=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> L=[dev newBufferWithLength:256*4 options:MTLResourceStorageModeShared];
  { float*l=L.contents; for(int c=0;c<256;c++) l[c]=e4m3_decode((uint8_t)c); }

  printf("  %6s %14s %14s %10s   %s\n","trial","C (float)","Metal","ulp","mode");
  for(int mode=0;mode<2;mode++){
    long bad=0;
    for(int t=0;t<2000;t++){
      srandom(7000+t);
      uint8_t*wp=W.contents; float*xp=X.contents;
      for(int i=0;i<N;i++){ uint8_t b=(uint8_t)(random()&0xFF); if(b==0x7F||b==0xFF)b=0x3C; wp[i]=b; }
      for(int i=0;i<N;i++) xp[i]=(float)((random()/(double)RAND_MAX)*2.0-1.0);
      /* C reference: exactly matmul_fp8's inner statement */
      float acc=0.0f;
      for(int i=0;i<N;i++){ float xv=xp[i]; acc += e4m3_decode(wp[i])*xv; }
      unsigned n=N,m=mode;
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
      [en setComputePipelineState:pso];
      [en setBuffer:W offset:0 atIndex:0];[en setBuffer:X offset:0 atIndex:1];
      [en setBuffer:Y offset:0 atIndex:2];[en setBuffer:L offset:0 atIndex:3];
      [en setBytes:&n length:4 atIndex:4];[en setBytes:&m length:4 atIndex:5];
      [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
      [en endEncoding];[cb commit];[cb waitUntilCompleted];
      float got=*(float*)Y.contents;
      if(B32(acc)!=B32(got)){ if(bad<2) printf("  %6d %14.9g %14.9g %10ld   %s\n",t,acc,got,
          (long)B32(acc)-(long)B32(got), mode?"explicit fma":"a+=b*c"); bad++; }
    }
    printf("  mode=%-14s differing = %ld / 2000\n\n", mode?"explicit fma":"a+=b*c", bad);
  }
  return 0; } }
