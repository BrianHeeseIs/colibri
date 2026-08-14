// Feasibility probe: dense fp8 attention matmul, CPU vs GPU, at REAL DeepSeek-V4 shapes.
// Attention is 33.18% of prefill (E54), already batches to S=64, and its weights are RESIDENT
// (dense set in RAM) - so unlike MoE it has no SSD bound, no lease capacity, no eviction risk.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }

// e4m3fn decode (matches the engine's table-driven decode closely enough for a throughput probe)
static float e4m3_decode(unsigned char v){
  int s=(v>>7)&1, e=(v>>3)&0xF, m=v&7;
  float mag;
  if(e==0) mag = ldexpf((float)m, -9);            // subnormal: m * 2^-9
  else     mag = ldexpf(1.0f + (float)m/8.0f, e-7);
  return s? -mag : mag;
}

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"inline float e4m3(uchar v){ int s=(v>>7)&1,e=(v>>3)&0xF,m=v&7; float mag;\n"
"  if(e==0) mag = ldexp((float)m,-9); else mag = ldexp(1.0f+(float)m/8.0f, e-7);\n"
"  return s? -mag: mag; }\n"
// one thread per (output row, token); weights read once per row and reused across the S tokens
"kernel void fp8mm(device const uchar* W [[buffer(0)]],\n"
"                  device const float* SC[[buffer(1)]],\n"
"                  device const float* X [[buffer(2)]],\n"
"                  device float* Y       [[buffer(3)]],\n"
"                  constant uint4& dims  [[buffer(4)]],\n"   // rows, cols, S, scale_cols
"                  uint2 gid [[thread_position_in_grid]]) {\n"
"  uint rows=dims.x, cols=dims.y, S=dims.z, scols=dims.w;\n"
"  uint r = gid.x, s = gid.y;\n"
"  if (r>=rows || s>=S) return;\n"
"  device const uchar *w = W + (ulong)r*cols;\n"
"  device const float *x = X + (ulong)s*cols;\n"
"  uint srow = r/128;\n"
"  float acc = 0.0f;\n"
"  for (uint b=0; b<scols; ++b){\n"
"    float sc = SC[(ulong)srow*scols + b];\n"
"    float g = 0.0f; uint base=b*128;\n"
"    for (uint k=0;k<128;++k) g += e4m3(w[base+k]) * x[base+k];\n"
"    acc += g*sc;\n"
"  }\n"
"  Y[(ulong)s*rows + r] = acc;\n"
"}\n";

typedef struct { const char*name; int rows, cols; } Shape;

static float LUT[256];
int main(void){ @autoreleasepool {
  for(int c=0;c<256;c++) LUT[c]=e4m3_decode((unsigned char)c);   /* engine does this at :12350 */
  // real V4 attention shapes (rows = output dim, cols = input dim)
  Shape shapes[] = {
    {"wq_a  4096->1024",   1024, 4096},
    {"wq_b  1024->32768", 32768, 1024},
    {"wkv   4096->512",     512, 4096},
    {"wo_b  1024->4096",   4096, 1024},
  };
  int Ss[] = {1, 16, 64};
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError*err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
  if(!lib){ printf("compile fail: %s\n",[[err localizedDescription]UTF8String]); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fp8mm"] error:&err];

  printf("  %-20s %4s %11s %11s %10s %11s %11s\n","shape","S","CPU ms","GPU ms","GPU/CPU","CPU GF/s","GPU GF/s");
  for(int si=0; si<4; ++si){
    int rows=shapes[si].rows, cols=shapes[si].cols;
    size_t scols=cols/128, srows=(rows+127)/128;
    id<MTLBuffer> W=[dev newBufferWithLength:(size_t)rows*cols options:MTLResourceStorageModeShared];
    id<MTLBuffer> SC=[dev newBufferWithLength:srows*scols*sizeof(float) options:MTLResourceStorageModeShared];
    unsigned char*wp=W.contents; float*sp=SC.contents;
    for(size_t i=0;i<(size_t)rows*cols;i++) wp[i]=(unsigned char)((i*37)&0xFF);
    for(size_t i=0;i<srows*scols;i++) sp[i]=0.0035f;
    for(int ti=0; ti<3; ++ti){
      int S=Ss[ti];
      id<MTLBuffer> X=[dev newBufferWithLength:(size_t)S*cols*4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> Y=[dev newBufferWithLength:(size_t)S*rows*4 options:MTLResourceStorageModeShared];
      float*xp=X.contents; for(size_t i=0;i<(size_t)S*cols;i++) xp[i]=(float)((i%13)-6)*0.1f;
      double flops=2.0*S*rows*cols;
      // ---- CPU: 12 threads over output rows ----
      double bc=1e9;
      for(int it=0; it<3; ++it){
        double t0=now_s();
        dispatch_apply(12, DISPATCH_APPLY_AUTO, ^(size_t th){
          size_t chunk=(rows+11)/12, r0=th*chunk, r1=r0+chunk; if(r1>(size_t)rows) r1=rows;
          float *Yp=(float*)Y.contents; const float*Xp=(const float*)X.contents;
          for(size_t r=r0;r<r1;r++){
            const unsigned char *w=wp+r*cols; size_t srow=r/128;
            for(int s=0;s<S;s++){
              const float*x=Xp+(size_t)s*cols; float acc=0.f;
              for(size_t b=0;b<scols;b++){
                float sc=sp[srow*scols+b]; size_t base=b*128;
                float g0=0,g1=0,g2=0,g3=0;          /* 4-way ILP, LUT decode */
                for(int k=0;k<128;k+=4){
                  g0+=LUT[w[base+k  ]]*x[base+k  ];
                  g1+=LUT[w[base+k+1]]*x[base+k+1];
                  g2+=LUT[w[base+k+2]]*x[base+k+2];
                  g3+=LUT[w[base+k+3]]*x[base+k+3];
                }
                acc+=(g0+g1+g2+g3)*sc;
              }
              Yp[(size_t)s*rows+r]=acc;
            }
          }
        });
        double dt=now_s()-t0; if(dt<bc) bc=dt;
      }
      // ---- GPU ----
      unsigned int dims[4]={ (unsigned)rows,(unsigned)cols,(unsigned)S,(unsigned)scols };
      double bg=1e9;
      for(int it=0; it<3; ++it){
        double t0=now_s();
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:SC offset:0 atIndex:1];
        [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
        [e setBytes:dims length:sizeof(dims) atIndex:4];
        [e dispatchThreads:MTLSizeMake(rows,S,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double dt=now_s()-t0; if(dt<bg) bg=dt;
      }
      printf("  %-20s %4d %11.2f %11.2f %9.2fx %11.1f %11.1f\n",
             si? "":shapes[si].name, S, bc*1e3, bg*1e3, bc/bg, flops/bc/1e9, flops/bg/1e9);
      if(!si && ti==0) printf("  %-20s\n", shapes[si].name);
    }
    printf("  %-20s %4s %11s %11s %10s %11s %11s\n","","","","","","","");
  }
  return 0; }}
