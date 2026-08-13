#import <Metal/Metal.h>
#import <stdio.h>
#import <math.h>
#define N 8
static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void k(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],\n"
"              device float*o[[buffer(2)]],uint t[[thread_position_in_grid]]){\n"
" if(t!=0)return;\n"
" float a=0.0f; for(int i=0;i<8;++i) a+=x[i]*w[i];      o[0]=a;   // may contract\n"
" float b=0.0f; for(int i=0;i<8;++i) b=fma(x[i],w[i],b); o[1]=b;   // explicit fma\n"
" float c=0.0f; for(int i=0;i<8;++i){float p=x[i]*w[i]; c=c+p;} o[2]=c; // forced split\n"
"}\n";
static float gpu[3];
static void run(id<MTLDevice>d,MTLMathMode m,const float*x,const float*w){
  NSError*e=nil; MTLCompileOptions*o=[MTLCompileOptions new]; o.mathMode=m;
  id<MTLLibrary>l=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:o error:&e];
  if(!l){printf("  compile fail: %s\n",[[e localizedDescription]UTF8String]);return;}
  id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e];
  id<MTLBuffer>bx=[d newBufferWithBytes:x length:N*4 options:0],bw=[d newBufferWithBytes:w length:N*4 options:0],
               bo=[d newBufferWithLength:3*4 options:0];
  id<MTLCommandQueue>q=[d newCommandQueue]; id<MTLCommandBuffer>cb=[q commandBuffer];
  id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
  [en setComputePipelineState:p];[en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bw offset:0 atIndex:1];
  [en setBuffer:bo offset:0 atIndex:2];
  [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  memcpy(gpu,bo.contents,12);
}
__attribute__((noinline)) static float cpu_split(const float*x,const float*w){
#pragma clang fp contract(off)
  float a=0; for(int i=0;i<N;i++){ float p=x[i]*w[i]; a=a+p; } return a; }
__attribute__((noinline)) static float cpu_fma(const float*x,const float*w){
  float a=0; for(int i=0;i<N;i++) a=fmaf(x[i],w[i],a); return a; }
#define HX(v) ({unsigned _u;memcpy(&_u,&(v),4);_u;})
int main(void){@autoreleasepool{
  float x[N],w[N];
  for(int i=0;i<N;i++){ x[i]=1.0f+(i+1)*0x1p-12f; w[i]=1.0f+(i+1)*0x1p-12f; }
  float cs=cpu_split(x,w), cf=cpu_fma(x,w);
  printf("  cpu split(mul,add) 0x%08x  %.9e\n",HX(cs),cs);
  printf("  cpu explicit fma   0x%08x  %.9e\n",HX(cf),cf);
  printf("  cpu split==cpu fma : %s  <- if NO, contraction is observable here\n", cs==cf?"YES":"NO");
  id<MTLDevice>d=MTLCreateSystemDefaultDevice();
  const char*nm[3]={"Safe","Relaxed","Fast"};
  for(int m=0;m<3;m++){
    run(d,(MTLMathMode)m,x,w);
    printf("  gpu %-7s: a+=x*w 0x%08x | fma 0x%08x | split 0x%08x\n",nm[m],HX(gpu[0]),HX(gpu[1]),HX(gpu[2]));
    printf("            match cpu_split=%s cpu_fma=%s\n",
      gpu[0]==cs?"YES":"no", gpu[0]==cf?"YES":"no");
  }
  return 0;}}
