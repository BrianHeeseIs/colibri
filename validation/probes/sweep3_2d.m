// MXFP4 sweep v3: 2D dispatch (O, S) like the fp8 attention kernel.
// v1/v2 dispatched only O=2048 threads on a 40-core GPU -> massive under-occupancy.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
#define I_DIM 4096
#define O_DIM 2048
#define BLK   32
static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"constant float E2M1[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};\n"
"kernel void mxfp4_2d(device const uchar* W [[buffer(0)]],\n"
"                     device const uchar* SC[[buffer(1)]],\n"
"                     device const float* X [[buffer(2)]],\n"
"                     device float* Y       [[buffer(3)]],\n"
"                     constant uint4& d     [[buffer(4)]],\n"   // O, I, S, NB
"                     uint2 gid [[thread_position_in_grid]]) {\n"
"  uint O=d.x, I=d.y, S=d.z, NB=d.w;\n"
"  uint row=gid.x, s=gid.y;\n"
"  if(row>=O || s>=S) return;\n"
"  device const uchar *w = W + (ulong)row*(I/2);\n"
"  device const uchar *sc= SC+ (ulong)row*NB;\n"
"  device const float *x = X + (ulong)s*I;\n"
"  float acc=0.0f;\n"
"  for(uint b=0;b<NB;++b){\n"
"    float g=0.0f; uint base=b*32u;\n"
"    for(uint k=0;k<32u;k+=2){\n"
"      uchar byte=w[(base+k)>>1];\n"
"      g += E2M1[byte & 0xF]      * x[base+k];\n"
"      g += E2M1[(byte>>4)&0xF]   * x[base+k+1];\n"
"    }\n"
"    acc += g*exp2((float)sc[b]-127.0f);\n"
"  }\n"
"  Y[(ulong)s*O+row]=acc;\n"
"}\n";
int main(void){ @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError*e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
  if(!lib){printf("compile: %s\n",[[e localizedDescription]UTF8String]);return 1;}
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mxfp4_2d"] error:&e];
  size_t wb=(size_t)O_DIM*(I_DIM/2), sb=(size_t)O_DIM*(I_DIM/BLK);
  id<MTLBuffer> W=[dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
  id<MTLBuffer> SC=[dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
  unsigned char*wp=W.contents,*sp=SC.contents;
  for(size_t i=0;i<wb;i++) wp[i]=(unsigned char)(i*31u);
  for(size_t i=0;i<sb;i++) sp[i]=127;
  printf("  2D dispatch (O,S). v1/v2 used 1D (O only) = %d threads on 40 GPU cores.\n\n", O_DIM);
  printf("  %4s %11s %12s %12s %10s\n","S","time_ms","GFLOP/s","us/token","vs v2 best");
  double v2[]={968.5,304.8,210.6,163.5,128.9};
  int Ss[]={1,2,4,8,16,32,64}; 
  for(int t=0;t<7;t++){
    uint32_t S=Ss[t];
    id<MTLBuffer> X=[dev newBufferWithLength:(size_t)S*I_DIM*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y=[dev newBufferWithLength:(size_t)S*O_DIM*4 options:MTLResourceStorageModeShared];
    float*xp=X.contents; for(size_t i=0;i<(size_t)S*I_DIM;i++) xp[i]=(float)((i%17)-8)*0.125f;
    unsigned int d[4]={O_DIM,I_DIM,S,I_DIM/BLK};
    double best=1e9;
    for(int it=0;it<5;it++){
      double t0=now_s();
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
      [en setComputePipelineState:pso];
      [en setBuffer:W offset:0 atIndex:0];[en setBuffer:SC offset:0 atIndex:1];
      [en setBuffer:X offset:0 atIndex:2];[en setBuffer:Y offset:0 atIndex:3];
      [en setBytes:d length:sizeof(d) atIndex:4];
      [en dispatchThreads:MTLSizeMake(O_DIM,S,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [en endEncoding];[cb commit];[cb waitUntilCompleted];
      double dt=now_s()-t0; if(dt<best)best=dt;
    }
    double fl=2.0*S*I_DIM*O_DIM, us=best*1e6/S;
    printf("  %4u %11.3f %12.1f %12.1f %9s\n",S,best*1e3,fl/best/1e9,us,
           t<5? ({static char b[32]; snprintf(b,32,"%.2fx",v2[t]/us); b;}) : "-");
  }
  return 0; }}
