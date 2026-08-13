// T5b: GPU port of coli_v4_swiglu -- measures divergence where it ACTUALLY lands.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#define SRCPATH "c/metal/coli_v4_swiglu.metal"
#define N 4096
static float sigmoidf_stable(float v){ if(v>=0.0f){float d=expf(-v);return 1.0f/(1.0f+d);}
    float g=expf(v); return g/(1.0f+g); }
// VERBATIM from c/deepseek_v4.c:1383-1396
static void cpu_swiglu(float*out,const float*gate,const float*up,int dim,float limit){
    for(int i=0;i<dim;i++){
        float gv=gate[i], uv=up[i];
        if(limit>0.0f){ gv=fminf(gv,limit); uv=fmaxf(-limit,fminf(uv,limit)); }
        out[i]=gv*sigmoidf_stable(gv)*uv;
    }
}
static int run_case(id<MTLDevice>d,id<MTLLibrary>lib,float limit,const char*label){
    float*gate=malloc(N*4),*up=malloc(N*4),*cpu=malloc(N*4);
    srandom(777);
    for(int i=0;i<N;i++){
        gate[i]=-100.0f+200.0f*((float)i/(float)(N-1));           // sweep incl. deep negatives
        up[i]=((float)(random()%2001)-1000.f)/97.0f;
    }
    cpu_swiglu(cpu,gate,up,N,limit);
    NSError*e=nil;
    id<MTLFunction>fn=[lib newFunctionWithName:@"coli_v4_swiglu"];
    if(!fn){printf("    entry missing\n");return 1;}
    id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];
    struct{int dim;float limit;}P={N,limit};
    id<MTLBuffer>bg=[d newBufferWithBytes:gate length:N*4 options:0];
    id<MTLBuffer>bu=[d newBufferWithBytes:up length:N*4 options:0];
    id<MTLBuffer>bo=[d newBufferWithLength:N*4 options:0];
    id<MTLBuffer>bp=[d newBufferWithBytes:&P length:sizeof(P) options:0];
    id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];
    id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];[en setBuffer:bo offset:0 atIndex:0];[en setBuffer:bg offset:0 atIndex:1];
    [en setBuffer:bu offset:0 atIndex:2];[en setBuffer:bp offset:0 atIndex:3];
    [en dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    float*g=(float*)bo.contents;
    int bad=0; double maxrel=0; float worst=0; int flush=0;
    for(int i=0;i<N;i++){
        unsigned hc,hg; memcpy(&hc,&cpu[i],4); memcpy(&hg,&g[i],4);
        if(hc!=hg){ bad++;
            if(g[i]==0.0f && cpu[i]!=0.0f){ flush++; continue; }   // denormal flush class
            double r=cpu[i]!=0?fabs((double)g[i]-(double)cpu[i])/fabs((double)cpu[i]):0;
            if(r>maxrel){maxrel=r;worst=gate[i];}
        }
    }
    printf("    limit=%-5.1f %-9s diff=%4d/%d (%.2f%%)  denormal-flush=%d  maxRel(non-flush)=%.4e at gate=%.3f\n",
           limit,label,bad,N,100.0*bad/N,flush,maxrel,worst);
    free(gate);free(up);free(cpu);
    return 0;
}
int main(void){@autoreleasepool{
    FILE*f=fopen(SRCPATH,"rb");
    if(!f){printf("RED: %s absent.\n",SRCPATH);return 2;}
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;
    MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;
    id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){printf("RED: compile failed:\n%s\n",[[e localizedDescription]UTF8String]);return 3;}
    printf("  swiglu divergence at the level that actually matters:\n");
    run_case(d,lib,0.0f,"no-clamp");
    run_case(d,lib,7.0f,"clamped");
    return 0;
}}
