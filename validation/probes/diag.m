#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float coli_fp8_minprod=3.4e38f; int coli_fp8_minprod_enabled=0;
#include "../../c/quant.h"
static uint32_t B32(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static const char*SRC=
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void k(device const uchar* W[[buffer(0)]], device const float* SC[[buffer(1)]],\n"
"              device const float* X[[buffer(2)]], device float* Y[[buffer(3)]],\n"
"              device const float* LUT[[buffer(4)]], constant uint4& d[[buffer(5)]],\n"
"              uint2 gid[[thread_position_in_grid]]){\n"
"  uint O=d.x,I=d.y,S=d.z,NB=d.w; uint o=gid.x,s=gid.y; if(o>=O||s>=S) return;\n"
"  device const uchar* w=W+(ulong)o*I; device const float* x=X+(ulong)s*I;\n"
"  device const float* scl=SC+(ulong)(o/128u)*NB;\n"
"  float a=0.0f;\n"
"  for(uint bi=0;bi<NB;++bi){ uint base=bi*128u; uint blen=(base+128u>I)?(I-base):128u;\n"
"    float acc=0.0f; for(uint i=0;i<blen;++i) acc += LUT[w[base+i]]*x[base+i];\n"
"    a += acc*scl[bi]; }\n"
"  Y[(ulong)s*O+o]=a;\n}\n";
int main(void){ @autoreleasepool {
  int O=128,I=128,S=1; int64_t NB=fp8_nblk(I);
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError*e=nil;
  MTLCompileOptions *opt=[MTLCompileOptions new];
  opt.fastMathEnabled = NO;   /* CRITICAL: fast math destroys Dekker two_sum/two_prod */ id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:opt error:&e];
  if(!lib){printf("%s\n",[[e localizedDescription]UTF8String]);return 2;}
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&e];
  size_t wb=(size_t)O*I, sb=(size_t)((O+127)/128)*NB*4;
  id<MTLBuffer> W=[dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
  id<MTLBuffer> SC=[dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
  id<MTLBuffer> X=[dev newBufferWithLength:(size_t)S*I*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Y=[dev newBufferWithLength:(size_t)S*O*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> L=[dev newBufferWithLength:1024 options:MTLResourceStorageModeShared];
  {float*l=L.contents; for(int c=0;c<256;c++) l[c]=e4m3_decode((uint8_t)c);}
  uint8_t*wp=W.contents; float*sp=SC.contents,*xp=X.contents;
  srandom(99);
  for(size_t i=0;i<wb;i++){uint8_t b=(uint8_t)(random()&0xFF); if(b==0x7F||b==0xFF)b=0x3C; wp[i]=b;}
  for(size_t i=0;i<sb/4;i++) sp[i]=(float)((random()/(double)RAND_MAX)*0.02+0.001);
  for(size_t i=0;i<(size_t)S*I;i++) xp[i]=(float)((random()/(double)RAND_MAX)*2.0-1.0);
  printf("  scale buffer floats = %zu  (expect ceil(O/128)*nblkI = %lld)\n", sb/4, (long long)(((O+127)/128)*NB));
  float*ref=malloc((size_t)S*O*4);
  matmul_fp8(ref,xp,wp,sp,S,I,O);
  unsigned d[4]={(unsigned)O,(unsigned)I,(unsigned)S,(unsigned)NB};
  id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:pso];
  [en setBuffer:W offset:0 atIndex:0];[en setBuffer:SC offset:0 atIndex:1];[en setBuffer:X offset:0 atIndex:2];
  [en setBuffer:Y offset:0 atIndex:3];[en setBuffer:L offset:0 atIndex:4];[en setBytes:d length:16 atIndex:5];
  [en dispatchThreads:MTLSizeMake(O,S,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  float*got=Y.contents; long bad=0;
  printf("  %4s %16s %16s %8s\n","o","ref","metal","ulp");
  for(int o=0;o<O;o++){ if(B32(ref[o])!=B32(got[o])){ if(bad<6) printf("  %4d %16.9g %16.9g %8ld\n",o,ref[o],got[o],(long)B32(ref[o])-(long)B32(got[o])); bad++; } }
  printf("  differing = %ld / %d   (single block, so outer accumulator is irrelevant)\n",bad,O);
  return 0; } }
