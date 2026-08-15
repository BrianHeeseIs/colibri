/* A6 feasibility: is ONE fused dispatch over the 8 wo_a groups faster than 8 separate dispatches?
 * Identical total arithmetic; the only differences are occupancy and dispatch count.
 * Real shape: o_rank=1024 rows, group_width=4096 cols, o_groups=8. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
#define O_RANK 1024
#define GW     4096
#define GROUPS 8

static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"struct df{float hi;float lo;};\n"
"inline df ts(float a,float b){float s=a+b,bb=s-a,e=(a-(s-bb))+(b-bb);return df{s,e};}\n"
"inline df tp(float a,float b){float p=a*b,e=fma(a,b,-p);return df{p,e};}\n"
"inline df dap(df ac,float a,float b){df p=tp(ac.hi,0.0f);p=tp(a,b);df s=ts(ac.hi,p.hi);"
"float lo=s.lo+(ac.lo+p.lo);return ts(s.hi,lo);}\n"
"kernel void one(device const uchar*W[[buffer(0)]],device const float*SC[[buffer(1)]],\n"
"                device const float*X[[buffer(2)]],device float*Y[[buffer(3)]],\n"
"                device const float*L[[buffer(4)]],constant uint4&d[[buffer(5)]],\n"
"                uint2 g[[thread_position_in_grid]]){\n"
"  uint O=d.x,I=d.y,S=d.z,NB=d.w; uint o=g.x,s=g.y; if(o>=O||s>=S)return;\n"
"  device const uchar*w=W+(ulong)o*I; device const float*x=X+(ulong)s*I;\n"
"  device const float*sc=SC+(ulong)(o/128u)*NB; df a=df{0.0f,0.0f};\n"
"  for(uint b=0;b<NB;++b){uint base=b*128u;float acc=0.0f;\n"
"    for(uint i=0;i<128u;++i) acc+=L[w[base+i]]*x[base+i];\n"
"    a=dap(a,acc,sc[b]);}\n"
"  Y[(ulong)s*O+o]=a.hi+a.lo;}\n"
"kernel void fused(device const uchar*W[[buffer(0)]],device const float*SC[[buffer(1)]],\n"
"                  device const float*X[[buffer(2)]],device float*Y[[buffer(3)]],\n"
"                  device const float*L[[buffer(4)]],constant uint4&d[[buffer(5)]],\n"
"                  constant uint&NG[[buffer(6)]],uint3 g[[thread_position_in_grid]]){\n"
"  uint O=d.x,I=d.y,S=d.z,NB=d.w; uint o=g.x,s=g.y,gr=g.z;\n"
"  if(o>=O||s>=S||gr>=NG)return;\n"
"  device const uchar*w=W+(ulong)gr*O*I+(ulong)o*I;\n"
"  device const float*x=X+(ulong)gr*S*I+(ulong)s*I;\n"
"  device const float*sc=SC+(ulong)gr*((O+127u)/128u)*NB+(ulong)(o/128u)*NB;\n"
"  df a=df{0.0f,0.0f};\n"
"  for(uint b=0;b<NB;++b){uint base=b*128u;float acc=0.0f;\n"
"    for(uint i=0;i<128u;++i) acc+=L[w[base+i]]*x[base+i];\n"
"    a=dap(a,acc,sc[b]);}\n"
"  Y[((ulong)gr*S+s)*O+o]=a.hi+a.lo;}\n";

int main(void){ @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  NSError *e = nil;
  MTLCompileOptions *opt = [MTLCompileOptions new];
  opt.fastMathEnabled = NO;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                         options:opt error:&e];
  if (!lib) { printf("  compile: %s\n", [[e localizedDescription] UTF8String]); return 2; }
  id<MTLComputePipelineState> p1 = [dev newComputePipelineStateWithFunction:
                                    [lib newFunctionWithName:@"one"] error:&e];
  id<MTLComputePipelineState> pf = [dev newComputePipelineStateWithFunction:
                                    [lib newFunctionWithName:@"fused"] error:&e];
  if (!p1 || !pf) { printf("  pipeline failed\n"); return 2; }

  unsigned NB = GW / 128;
  size_t wb = (size_t)GROUPS * O_RANK * GW;
  size_t sb = (size_t)GROUPS * ((O_RANK + 127) / 128) * NB * sizeof(float);
  id<MTLBuffer> W  = [dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
  id<MTLBuffer> SC = [dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
  id<MTLBuffer> L  = [dev newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModeShared];
  unsigned char *wp = (unsigned char*)W.contents;
  float *sp = (float*)SC.contents, *lp = (float*)L.contents;
  for (size_t i = 0; i < wb; i++) wp[i] = (unsigned char)(i * 37u);
  for (size_t i = 0; i < sb/sizeof(float); i++) sp[i] = 0.003f;
  for (int c = 0; c < 256; c++) lp[c] = (float)((c % 16) - 8) * 0.25f;

  printf("  %5s %14s %14s %10s %14s\n","S","8 separate","1 fused","speedup","threads/disp");
  int Ss[] = {64, 32, 16, 6, 1};
  for (int t = 0; t < 5; t++) {
    unsigned S = (unsigned)Ss[t];
    id<MTLBuffer> X = [dev newBufferWithLength:(size_t)GROUPS*S*GW*sizeof(float)
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y = [dev newBufferWithLength:(size_t)GROUPS*S*O_RANK*sizeof(float)
                                       options:MTLResourceStorageModeShared];
    float *xp = (float*)X.contents;
    for (size_t i = 0; i < (size_t)GROUPS*S*GW; i++) xp[i] = (float)((i % 9) - 4) * 0.1f;
    unsigned d[4] = { O_RANK, GW, S, NB };
    unsigned ng = GROUPS;
    double b1 = 1e9, bf = 1e9;
    for (int it = 0; it < 5; it++) {
      double t0 = now_s();
      for (unsigned g = 0; g < GROUPS; g++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:p1];
        [en setBuffer:W  offset:(NSUInteger)g*O_RANK*GW atIndex:0];
        [en setBuffer:SC offset:(NSUInteger)g*((O_RANK+127)/128)*NB*sizeof(float) atIndex:1];
        [en setBuffer:X  offset:(NSUInteger)g*S*GW*sizeof(float) atIndex:2];
        [en setBuffer:Y  offset:(NSUInteger)g*S*O_RANK*sizeof(float) atIndex:3];
        [en setBuffer:L  offset:0 atIndex:4];
        [en setBytes:d length:sizeof(d) atIndex:5];
        [en dispatchThreads:MTLSizeMake(O_RANK,S,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      double dt = now_s() - t0; if (dt < b1) b1 = dt;
    }
    for (int it = 0; it < 5; it++) {
      double t0 = now_s();
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:pf];
      [en setBuffer:W offset:0 atIndex:0]; [en setBuffer:SC offset:0 atIndex:1];
      [en setBuffer:X offset:0 atIndex:2]; [en setBuffer:Y offset:0 atIndex:3];
      [en setBuffer:L offset:0 atIndex:4];
      [en setBytes:d length:sizeof(d) atIndex:5];
      [en setBytes:&ng length:sizeof(ng) atIndex:6];
      [en dispatchThreads:MTLSizeMake(O_RANK,S,GROUPS) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double dt = now_s() - t0; if (dt < bf) bf = dt;
    }
    printf("  %5u %11.3f ms %11.3f ms %9.2fx %14u\n", S, b1*1e3, bf*1e3, b1/bf, O_RANK*S);
  }
  return 0; } }
