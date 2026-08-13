#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
static const float mx4_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){union{uint32_t u;float f;}b;b.u=(uint32_t)s<<23;return b.f;}
static void cpu_mm(float*y,const float*x,const uint8_t*q4,const uint8_t*e8s,int S,int I,int O){
    int rb=(I+1)/2,ng=(I+31)/32;
    for(int o=0;o<O;o++){const uint8_t*w=q4+(int64_t)o*rb;const uint8_t*scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){const float*xs=x+(int64_t)s*I;float a=0;
            for(int g=0;g<ng;g++){int base=g*32,gl=32;if(base+gl>I)gl=I-base;float sc=mx4_scale(scl[g]),ga=0;
                for(int i=base;i<base+gl;i+=2){uint8_t by=w[i>>1];ga+=xs[i]*mx4_lut[by&0xF];if(i+1<base+gl)ga+=xs[i+1]*mx4_lut[by>>4];}
                a+=ga*sc;}y[(int64_t)s*O+o]=a;}}}
// rows16 pack (matches synth_v4_rows16_pack)
static void pack16(uint8_t*packed,const uint8_t*data,int rows,int stride){
    for(int row=0;row<rows;row++){int tile=row/16,lane=row%16;
        for(int c=0;c<stride;c++) packed[(tile*stride+c)*16+lane]=data[row*stride+c];}}
typedef struct{int S,I,O,rb,ng;}Dims;
static int run(id<MTLDevice>d,id<MTLLibrary>lib,const char*entry,int S,int I,int O,
               const uint8_t*q4,const uint8_t*e8,const float*x,float*out,
               size_t wbytes,size_t sbytes){
    NSError*e=nil; id<MTLFunction>fn=[lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if(!fn){printf("    %s MISSING\n",entry);return 1;}
    id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];
    int rb=(I+1)/2,ng=(I+31)/32; Dims dm={S,I,O,rb,ng};
    id<MTLBuffer>bx=[d newBufferWithBytes:x length:(size_t)S*I*4 options:0];
    id<MTLBuffer>bq=[d newBufferWithBytes:q4 length:wbytes options:0];
    id<MTLBuffer>be=[d newBufferWithBytes:e8 length:sbytes options:0];
    id<MTLBuffer>by=[d newBufferWithLength:(size_t)S*O*4 options:0];
    id<MTLBuffer>bd=[d newBufferWithBytes:&dm length:sizeof(dm) options:0];
    id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];
    id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];[en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bq offset:0 atIndex:1];
    [en setBuffer:be offset:0 atIndex:2];[en setBuffer:by offset:0 atIndex:3];[en setBuffer:bd offset:0 atIndex:4];
    [en dispatchThreads:MTLSizeMake(S*O,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(out,by.contents,(size_t)S*O*4); return 0;}
int main(void){@autoreleasepool{
    FILE*f=fopen("c/metal/coli_v4_matmul.metal","rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;
    id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){printf("compile fail:\n%s\n",[[e localizedDescription]UTF8String]);return 3;}
    int bad=0;
    struct{int S,I,O;const char*n;}C[]={{2,64,32,"tile-aligned O=32"},{2,64,48,"O=48 (3 tiles)"},{2,48,16,"tail16 O=16"},{1,128,17,"O=17 ragged tile"}};
    for(int c=0;c<4;c++){int S=C[c].S,I=C[c].I,O=C[c].O,rb=(I+1)/2,ng=(I+31)/32;
        srandom(4242+c);
        float*x=malloc((size_t)S*I*4);uint8_t*q4=malloc((size_t)O*rb),*e8=malloc((size_t)O*ng);
        for(int i=0;i<S*I;i++)x[i]=((float)(random()%2001)-1000)/337.0f;
        for(int i=0;i<O*rb;i++)q4[i]=(uint8_t)(random()&0xFF);
        for(int i=0;i<O*ng;i++)e8[i]=(uint8_t)(110+(random()%30));
        int padO=((O+15)/16)*16;
        uint8_t*q4h=calloc((size_t)padO*rb,1),*e8h=calloc((size_t)padO*ng,1);
        pack16(q4h,q4,O,rb); pack16(e8h,e8,O,ng);
        float*cpu=malloc((size_t)S*O*4),*cold=malloc((size_t)S*O*4),*hot=malloc((size_t)S*O*4);
        cpu_mm(cpu,x,q4,e8,S,I,O);
        run(d,lib,"coli_v4_matmul_mxfp4_ordered",S,I,O,q4,e8,x,cold,(size_t)O*rb,(size_t)O*ng);
        run(d,lib,"coli_v4_matmul_mxfp4_ordered_hot",S,I,O,q4h,e8h,x,hot,(size_t)padO*rb,(size_t)padO*ng);
        int hc=0,hh=0,ch=0;
        for(int i=0;i<S*O;i++){unsigned a,b,g;memcpy(&a,&cpu[i],4);memcpy(&b,&cold[i],4);memcpy(&g,&hot[i],4);
            if(a!=b)hc++; if(a!=g)hh++; if(b!=g)ch++;}
        printf("    %-18s S=%d I=%d O=%d : cold==cpu %s | hot==cpu %s | hot==cold %s\n",
            C[c].n,S,I,O, hc?"NO":"yes", hh?"NO":"yes", ch?"NO":"yes");
        if(hh||ch)bad++;
        free(x);free(q4);free(e8);free(q4h);free(e8h);free(cpu);free(cold);free(hot);}
    printf("  => hot layout bit-identical to cold+cpu on all shapes: %s\n",bad?"NO":"YES");
    return bad;}}
