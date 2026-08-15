// T5c: GPU port of coli_fp8_activation_qdq_ref (steps 1 & 5 of the MoE chain) -- bit-exact probe.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <float.h>
#define SRCPATH "c/metal/coli_v4_fp8qdq.metal"

// ---- CPU originals, verbatim ----
static int ceil_log2_positive(float value){int e;float fr=frexpf(value,&e);return fr==0.5f?e-1:e;}
static float coli_e8m0_decode(uint8_t v){ if(v==0xff)return NAN; return ldexpf(1.0f,(int)v-127); }
static float coli_e4m3fn_decode(uint8_t value){
    int sign=value>>7,exponent=(value>>3)&15,mantissa=value&7;
    if(exponent==15&&mantissa==7)return NAN;
    float number; if(!exponent)number=ldexpf((float)mantissa,-9);
    else number=ldexpf(1.0f+(float)mantissa/8.0f,exponent-7); return sign?-number:number;
}
static uint8_t coli_e4m3fn_encode(float value){
    if(isnan(value))return 0x7f;
    int negative=signbit(value)!=0; float magnitude=fabsf(value);
    if(!magnitude)return negative?0x80:0;
    if(magnitude>=448.0f)return (uint8_t)((negative?0x80:0)|0x7e);
    uint8_t best=0;
    if(magnitude<0.015625f){
        float scaled=magnitude*512.0f; uint8_t rounded=(uint8_t)scaled; float fraction=scaled-rounded;
        if(fraction>0.5f||(fraction==0.5f&&(rounded&1)))rounded++; best=rounded;
    }else{
        uint32_t bits; memcpy(&bits,&magnitude,4);
        int exponent=(int)((bits>>23)&0xff)-127;
        uint32_t significand=0x800000u|(bits&0x7fffffu);
        uint32_t rounded=significand>>20; uint32_t remainder=significand&0xfffffu;
        if(remainder>0x80000u||(remainder==0x80000u&&(rounded&1u)))rounded++;
        if(rounded==16u){rounded=8u;exponent++;}
        best=(uint8_t)((exponent+7)*8+(int)rounded-8);
    }
    return (uint8_t)(best|(negative?0x80:0));
}
static int cpu_fp8_qdq(float*output,uint8_t*scales,const float*input,int length,int block_size){
    for(int base=0;base<length;base+=block_size){
        int count=length-base<block_size?length-base:block_size;
        float maximum=0.0f;
        for(int i=0;i<count;i++)maximum=fmaxf(maximum,fabsf(input[base+i]));
        maximum=fmaxf(maximum,1e-4f);
        int se=ceil_log2_positive(maximum/448.0f);
        if(se<-127)se=-127; if(se>127)se=127;
        uint8_t enc=(uint8_t)(se+127); float scale=coli_e8m0_decode(enc);
        scales[base/block_size]=enc;
        for(int i=0;i<count;i++){
            float nrm=fmaxf(-448.0f,fminf(448.0f,input[base+i]/scale));
            output[base+i]=coli_e4m3fn_decode(coli_e4m3fn_encode(nrm))*scale;
        }
    }
    return 0;
}
// A7 feasibility: is a GPU fp8 QDQ actually faster than the CPU one, at realistic size?
// CPU and GPU are timed IN THE SAME PROCESS ON THE SAME BUFFER -- no cross-run subtraction.
#include <QuartzCore/QuartzCore.h>
int main(void){@autoreleasepool{
    FILE*f=fopen(SRCPATH,"rb");
    if(!f){printf("RED: %s absent.\n",SRCPATH);return 2;}
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;
    MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;   // match production -fno-fast-math
    id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){printf("RED: compile failed:\n%s\n",[[e localizedDescription]UTF8String]);return 3;}
    id<MTLFunction>fn=[lib newFunctionWithName:@"coli_v4_fp8_qdq"];
    id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];
    id<MTLCommandQueue>q=[d newCommandQueue];

    const int BLK=128, NB=131072, LEN=BLK*NB;          // 16.8 Melems ~ 67 MB in / 67 MB out
    float*in=malloc((size_t)LEN*4); srandom(9001);
    for(int i=0;i<LEN;i++){ int cls=i%5;
        double u=(double)(random()%2000001-1000000)/1000000.0;
        in[i]= cls==0? (float)(u*1e-5) : cls==1? (float)(u*3.0) : cls==2? (float)(u*120.0)
             : cls==3? (float)(u*0.02) : (float)(u*800.0); }
    float*cpu_o=malloc((size_t)LEN*4); uint8_t*cpu_s=malloc(NB);

    const int IT=5;
    cpu_fp8_qdq(cpu_o,cpu_s,in,LEN,BLK);                                   // warm
    double t0=CACurrentMediaTime();
    for(int it=0;it<IT;it++) cpu_fp8_qdq(cpu_o,cpu_s,in,LEN,BLK);
    double cpu_s_each=(CACurrentMediaTime()-t0)/IT;

    id<MTLBuffer>bi=[d newBufferWithBytes:in length:(size_t)LEN*4 options:0];
    id<MTLBuffer>bo=[d newBufferWithLength:(size_t)LEN*4 options:0];
    id<MTLBuffer>bs=[d newBufferWithLength:NB options:0];
    struct{int length,block_size;}P={LEN,BLK};
    void(^run)(void)=^{
        id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
        [en setComputePipelineState:ps];[en setBuffer:bo offset:0 atIndex:0];[en setBuffer:bs offset:0 atIndex:1];
        [en setBuffer:bi offset:0 atIndex:2];[en setBytes:&P length:sizeof(P) atIndex:3];
        [en dispatchThreadgroups:MTLSizeMake(NB,1,1) threadsPerThreadgroup:MTLSizeMake(BLK,1,1)];
        [en endEncoding];[cb commit];[cb waitUntilCompleted];};
    run();                                                                  // warm
    t0=CACurrentMediaTime(); for(int it=0;it<IT;it++) run();
    double gpu_s_each=(CACurrentMediaTime()-t0)/IT;

    const float*go=(const float*)bo.contents; const uint8_t*gs=(const uint8_t*)bs.contents;
    long sbad=0,obad=0; for(int i=0;i<NB;i++) if(gs[i]!=cpu_s[i]) sbad++;
    for(int i=0;i<LEN;i++){unsigned hc,hg;memcpy(&hc,&cpu_o[i],4);memcpy(&hg,&go[i],4);if(hc!=hg)obad++;}

    double cm=LEN/1e6/cpu_s_each, gm=LEN/1e6/gpu_s_each;
    printf("  exactness at scale : scales %s (%ld bad), outputs %s (%ld/%d bad)\n",
        sbad?"MISMATCH":"BIT-EXACT",sbad, obad?"MISMATCH":"BIT-EXACT",obad,LEN);
    printf("  CPU qdq  : %8.2f ms  -> %7.0f Melem/s\n", cpu_s_each*1e3, cm);
    printf("  GPU qdq  : %8.2f ms  -> %7.0f Melem/s\n", gpu_s_each*1e3, gm);
    printf("  speedup  : %.2fx  (kernel-level, before in-situ dilution)\n", cpu_s_each/gpu_s_each);
    double eng_cpu_ms=1273.5, saved=eng_cpu_ms*(1.0-gpu_s_each/cpu_s_each);
    printf("  engine   : CPU qdq costs %.1f ms -> nominal saving %.0f ms; /1.75 in-situ -> ~%.0f ms\n",
        eng_cpu_ms, saved, saved/1.75);
    printf("  wall     : p064 on-lane 35.782 s -> ~%.4fx incremental\n", 35.782/(35.782-saved/1.75/1000.0));
    return (sbad||obad)?1:0;
}}
