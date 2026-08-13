#import <Metal/Metal.h>
#import <stdio.h>
#import <math.h>
#import <string.h>
#define N 8192
static const char*SRC=
"#include <metal_stdlib>\n using namespace metal;\n"
"inline float sg(float v){ if(v>=0.0f){float d=precise::exp(-v);return 1.0f/(1.0f+d);}\n"
" float g=precise::exp(v); return g/(1.0f+g);}\n"
"kernel void k(device const float*x[[buffer(0)]],device float*o[[buffer(1)]],\n"
"              uint t[[thread_position_in_grid]]){ o[t]=sg(x[t]); }\n";
static float cpu_sg(float v){ if(v>=0.0f){float d=expf(-v);return 1.0f/(1.0f+d);} float g=expf(v); return g/(1.0f+g);}
int main(void){@autoreleasepool{
  float*x=malloc(N*4);
  for(int i=0;i<N;i++) x[i]=-40.0f+80.0f*((float)i/(float)(N-1));  // [-40,40]
  id<MTLDevice>d=MTLCreateSystemDefaultDevice(); NSError*e=nil;
  MTLCompileOptions*o=[MTLCompileOptions new]; o.mathMode=MTLMathModeSafe;
  id<MTLLibrary>l=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:o error:&e];
  id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e];
  id<MTLBuffer>bx=[d newBufferWithBytes:x length:N*4 options:0],bo=[d newBufferWithLength:N*4 options:0];
  id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];
  id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
  [en setComputePipelineState:p];[en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bo offset:0 atIndex:1];
  [en dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  float*g=(float*)bo.contents;
  int diff=0,maxulp=0; double maxrel=0; float worst=0;
  for(int i=0;i<N;i++){
    float c=cpu_sg(x[i]); unsigned hc,hg; memcpy(&hc,&c,4); memcpy(&hg,&g[i],4);
    if(hc!=hg){ diff++; int u=abs((int)hc-(int)hg); if(u>maxulp){maxulp=u;}
      double rel=c!=0?fabs((double)g[i]-(double)c)/fabs((double)c):0;
      if(rel>maxrel){maxrel=rel;worst=x[i];} }
  }
  printf("  sweep [-40,40] n=%d\n",N);
  printf("  differing        : %d (%.2f%%)\n",diff,100.0*diff/N);
  printf("  max ULP distance : %d\n",maxulp);
  printf("  max rel error    : %.6e  at x=%.6f\n",maxrel,worst);
  return 0;}}
