// T5a: GPU port of matmul_mxfp4 (scalar/arm64 reference path) -- bit-exactness probe.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#define SRCPATH "c/metal/coli_v4_matmul.metal"

static const float mx4_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){union{uint32_t u;float f;}b;b.u=(uint32_t)s<<23;return b.f;}

// VERBATIM scalar path from c/quant.h:1401-1412 (the branch arm64 actually takes).
static void cpu_matmul_mxfp4(float*y,const float*x,const uint8_t*q4,const uint8_t*e8s,int S,int I,int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    for(int o=0;o<O;o++){
        const uint8_t*w=q4+(int64_t)o*rb; const uint8_t*scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float*xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                for(int i=base;i<base+glen;i+=2){
                    uint8_t byte=w[i>>1];
                    ga+=xs[i]*mx4_lut[byte&0xF];
                    if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4];
                }
                a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}
typedef struct { int S,I,O,rb,ng; } Dims;
static int run_case(id<MTLDevice>d,id<MTLLibrary>lib,const char*entry,int S,int I,int O,const char*label){
    int rb=(I+1)/2, ng=(I+31)/32;
    srandom(12345);
    float*x=malloc((size_t)S*I*4); uint8_t*q4=malloc((size_t)O*rb); uint8_t*e8=malloc((size_t)O*ng);
    for(int i=0;i<S*I;i++) x[i]=((float)(random()%2001)-1000.f)/337.0f;
    for(int i=0;i<O*rb;i++) q4[i]=(uint8_t)(random()&0xFF);
    for(int i=0;i<O*ng;i++) e8[i]=(uint8_t)(110+(random()%30));   // realistic exponents
    float*cpu=calloc((size_t)S*O,4);
    cpu_matmul_mxfp4(cpu,x,q4,e8,S,I,O);

    NSError*e=nil;
    id<MTLFunction>fn=[lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if(!fn){printf("    %-8s entry %s MISSING\n",label,entry);return 1;}
    id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];
    if(!ps){printf("    %-8s pipeline FAIL %s\n",label,[[e localizedDescription]UTF8String]);return 1;}
    Dims dm={S,I,O,rb,ng};
    id<MTLBuffer>bx=[d newBufferWithBytes:x length:(size_t)S*I*4 options:0];
    id<MTLBuffer>bq=[d newBufferWithBytes:q4 length:(size_t)O*rb options:0];
    id<MTLBuffer>be=[d newBufferWithBytes:e8 length:(size_t)O*ng options:0];
    id<MTLBuffer>by=[d newBufferWithLength:(size_t)S*O*4 options:0];
    id<MTLBuffer>bd=[d newBufferWithBytes:&dm length:sizeof(dm) options:0];
    id<MTLCommandQueue>q=[d newCommandQueue]; id<MTLCommandBuffer>cb=[q commandBuffer];
    id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];
    [en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bq offset:0 atIndex:1];
    [en setBuffer:be offset:0 atIndex:2];[en setBuffer:by offset:0 atIndex:3];
    [en setBuffer:bd offset:0 atIndex:4];
    int isSimd = (strstr(entry,"simd")!=NULL);
    int isX    = (strstr(entry,"xcache")!=NULL);
    if(isX)         [en dispatchThreads:MTLSizeMake(O,S,1)    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    else if(isSimd) [en dispatchThreads:MTLSizeMake(32,S*O,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    else            [en dispatchThreads:MTLSizeMake(S*O,1,1)  threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    float*g=(float*)by.contents;
    int bad=0,maxulp=0; double maxrel=0;
    for(int i=0;i<S*O;i++){
        unsigned hc,hg; memcpy(&hc,&cpu[i],4); memcpy(&hg,&g[i],4);
        if(hc!=hg){ bad++; int u=abs((int)hc-(int)hg); if(u>maxulp)maxulp=u;
            double r=cpu[i]!=0?fabs((double)g[i]-(double)cpu[i])/fabs((double)cpu[i]):0;
            if(r>maxrel)maxrel=r;
            if(bad<=3) printf("      [%d] cpu=0x%08x(%+.9e) gpu=0x%08x(%+.9e)\n",i,hc,cpu[i],hg,g[i]);
        }
    }
    printf("    %-8s S=%d I=%d O=%d ng=%d : %s  diff=%d/%d maxULP=%d maxRel=%.3e\n",
           label,S,I,O,ng,bad?"MISMATCH":"BIT-EXACT",bad,S*O,maxulp,maxrel);
    free(x);free(q4);free(e8);free(cpu);
    return bad?1:0;
}
static void pack16(uint8_t*p,const uint8_t*d,int rows,int stride){
    for(int r=0;r<rows;r++){int t=r/16,l=r%16;for(int c=0;c<stride;c++)p[(t*stride+c)*16+l]=d[r*stride+c];}}
static int run_case_hot(id<MTLDevice>d,id<MTLLibrary>lib,const char*entry,int S,int I,int O,const char*label){
    int rb=(I+1)/2,ng=(I+31)/32,padO=((O+15)/16)*16;
    srandom(12345);
    float*x=malloc((size_t)S*I*4); uint8_t*q4=malloc((size_t)O*rb),*e8=malloc((size_t)O*ng);
    for(int i=0;i<S*I;i++) x[i]=((float)(random()%2001)-1000.f)/337.0f;
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(random()&0xFF);
    for(size_t i=0;i<(size_t)O*ng;i++) e8[i]=(uint8_t)(110+(random()%30));
    uint8_t*q4h=calloc((size_t)padO*rb,1),*e8h=calloc((size_t)padO*ng,1);
    pack16(q4h,q4,O,rb); pack16(e8h,e8,O,ng);
    float*cpu=calloc((size_t)S*O,4); cpu_matmul_mxfp4(cpu,x,q4,e8,S,I,O);
    NSError*e=nil; id<MTLFunction>fn=[lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if(!fn){printf("    %-16s entry MISSING\n",label);return 1;}
    id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];
    Dims dm={S,I,O,rb,ng};
    id<MTLBuffer>bx=[d newBufferWithBytes:x length:(size_t)S*I*4 options:0],
      bq=[d newBufferWithBytes:q4h length:(size_t)padO*rb options:0],
      be=[d newBufferWithBytes:e8h length:(size_t)padO*ng options:0],
      by=[d newBufferWithLength:(size_t)S*O*4 options:0],
      bd=[d newBufferWithBytes:&dm length:sizeof(dm) options:0];
    id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];
    id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];[en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bq offset:0 atIndex:1];
    [en setBuffer:be offset:0 atIndex:2];[en setBuffer:by offset:0 atIndex:3];[en setBuffer:bd offset:0 atIndex:4];
    [en dispatchThreads:MTLSizeMake(O,S,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    float*g=(float*)by.contents; int bad=0;
    for(int i=0;i<S*O;i++){unsigned a,b;memcpy(&a,&cpu[i],4);memcpy(&b,&g[i],4);if(a!=b)bad++;}
    printf("    %-16s S=%d I=%d O=%d : %s  diff=%d/%d\n",label,S,I,O,bad?"MISMATCH":"BIT-EXACT",bad,S*O);
    free(x);free(q4);free(e8);free(q4h);free(e8h);free(cpu);
    return bad?1:0;
}
int main(void){@autoreleasepool{
    FILE*f=fopen(SRCPATH,"rb");
    if(!f){printf("RED: %s absent -- kernel not written.\n",SRCPATH);return 2;}
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    id<MTLDevice>d=MTLCreateSystemDefaultDevice(); NSError*e=nil;
    MTLCompileOptions*o=[MTLCompileOptions new]; o.mathMode=MTLMathModeSafe;
    id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){printf("RED: compile failed:\n%s\n",[[e localizedDescription]UTF8String]);return 3;}
    int bad=0;
    printf("  ordered variant (must be BIT-EXACT):\n");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered",2,64,8,"aligned");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered",2,48,8,"tail16");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered",1,32,4,"single");
    printf("  ordered_xcache (MUST be BIT-EXACT -- this is the ship candidate):\n");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered_xcache",2,64,8,"aligned");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered_xcache",2,48,8,"tail16");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered_xcache",1,4096,64,"REALDIM I=4096");
    printf("  ordered_hot_xcache (MUST be BIT-EXACT):\n");
    bad|=run_case_hot(d,lib,"coli_v4_matmul_mxfp4_ordered_hot_xcache",2,64,32,"aligned");
    bad|=run_case_hot(d,lib,"coli_v4_matmul_mxfp4_ordered_hot_xcache",1,4096,64,"REALDIM I=4096");
    printf("  ordered (cold, real dim regression):\n");
    bad|=run_case(d,lib,"coli_v4_matmul_mxfp4_ordered",1,4096,64,"REALDIM I=4096");
    printf("  simd variant (tolerance expected, reported not asserted):\n");
    run_case(d,lib,"coli_v4_matmul_mxfp4_simd",2,64,8,"aligned");
    run_case(d,lib,"coli_v4_matmul_mxfp4_simd",2,48,8,"tail16");
    return bad;
}}
