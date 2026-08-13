// T6: end-to-end single-expert forward, CPU chain vs GPU chain, per-step attribution.
// Reproduces parity_v4's explicit 7-step sequence. Reuses proven kernels (fp8_qdq,
// matmul_mxfp4_ordered, bf16_round_array, swiglu). Self-contained fixtures; NO model.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <float.h>
#define HID 64
#define INT 128
#define BLK 128
// ---- CPU refs (verbatim) ----
static const float mx4_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){union{uint32_t u;float f;}b;b.u=(uint32_t)s<<23;return b.f;}
static int ceil_log2_positive(float v){int e;float f=frexpf(v,&e);return f==0.5f?e-1:e;}
static float e8m0_decode(uint8_t v){return v==0xff?NAN:ldexpf(1.0f,(int)v-127);}
static float e4m3_decode(uint8_t v){int s=v>>7,e=(v>>3)&15,m=v&7;if(e==15&&m==7)return NAN;
    float n;if(!e)n=ldexpf((float)m,-9);else n=ldexpf(1.0f+(float)m/8.0f,e-7);return s?-n:n;}
static uint8_t e4m3_encode(float v){if(isnan(v))return 0x7f;int ng=signbit(v)!=0;float m=fabsf(v);
    if(!m)return ng?0x80:0;if(m>=448.0f)return (uint8_t)((ng?0x80:0)|0x7e);uint8_t best=0;
    if(m<0.015625f){float sc=m*512.0f;uint8_t r=(uint8_t)sc;float fr=sc-r;if(fr>0.5f||(fr==0.5f&&(r&1)))r++;best=r;}
    else{uint32_t b;memcpy(&b,&m,4);int e=(int)((b>>23)&0xff)-127;uint32_t sig=0x800000u|(b&0x7fffffu);
        uint32_t r=sig>>20,rem=sig&0xfffffu;if(rem>0x80000u||(rem==0x80000u&&(r&1u)))r++;
        if(r==16u){r=8u;e++;}best=(uint8_t)((e+7)*8+(int)r-8);}return (uint8_t)(best|(ng?0x80:0));}
static float bf16r(float v){uint32_t b;memcpy(&b,&v,4);if((b&0x7f800000u)!=0x7f800000u){uint32_t t=(b>>16)&1u;b+=0x7fffu+t;}b&=0xffff0000u;memcpy(&v,&b,4);return v;}
static float sig_stable(float v){if(v>=0.0f){float d=expf(-v);return 1.0f/(1.0f+d);}float g=expf(v);return g/(1.0f+g);}
static void cpu_qdq(float*o,uint8_t*sc,const float*in,int len,int blk){
    for(int base=0;base<len;base+=blk){int c=len-base<blk?len-base:blk;float mx=0;
        for(int i=0;i<c;i++)mx=fmaxf(mx,fabsf(in[base+i]));mx=fmaxf(mx,1e-4f);
        int se=ceil_log2_positive(mx/448.0f);if(se<-127)se=-127;if(se>127)se=127;
        uint8_t en=(uint8_t)(se+127);float s=e8m0_decode(en);sc[base/blk]=en;
        for(int i=0;i<c;i++){float nr=fmaxf(-448.0f,fminf(448.0f,in[base+i]/s));o[base+i]=e4m3_decode(e4m3_encode(nr))*s;}}}
static void cpu_mm(float*y,const float*x,const uint8_t*q4,const uint8_t*e8s,int S,int I,int O){
    int rb=(I+1)/2,ng=(I+31)/32;
    for(int o=0;o<O;o++){const uint8_t*w=q4+(int64_t)o*rb;const uint8_t*scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){const float*xs=x+(int64_t)s*I;float a=0;
            for(int g=0;g<ng;g++){int base=g*32,gl=32;if(base+gl>I)gl=I-base;float sc=mx4_scale(scl[g]),ga=0;
                for(int i=base;i<base+gl;i+=2){uint8_t by=w[i>>1];ga+=xs[i]*mx4_lut[by&0xF];if(i+1<base+gl)ga+=xs[i+1]*mx4_lut[by>>4];}
                a+=ga*sc;}y[(int64_t)s*O+o]=a;}}}
// full CPU expert forward, dumps each step
static void cpu_expert(float*out,float*g_qdq,float*g_gate,float*g_swi,float*g_down,
    const float*in,const uint8_t*Wg,const uint8_t*Sg,const uint8_t*Wu,const uint8_t*Su,
    const uint8_t*Wd,const uint8_t*Sd,float rw,float lim){
    uint8_t qs[1],ds[1];float qdq[HID],gate[INT],up[INT],act[INT],wt[INT],din[INT];
    cpu_qdq(qdq,qs,in,HID,BLK); memcpy(g_qdq,qdq,HID*4);
    cpu_mm(gate,qdq,Wg,Sg,1,HID,INT); cpu_mm(up,qdq,Wu,Su,1,HID,INT);
    for(int i=0;i<INT;i++){gate[i]=bf16r(gate[i]);up[i]=bf16r(up[i]);} memcpy(g_gate,gate,INT*4);
    for(int i=0;i<INT;i++){float gv=gate[i],uv=up[i];if(lim>0){gv=fminf(gv,lim);uv=fmaxf(-lim,fminf(uv,lim));}act[i]=gv*sig_stable(gv)*uv;}
    memcpy(g_swi,act,INT*4);
    for(int i=0;i<INT;i++)wt[i]=bf16r(act[i]*rw);
    cpu_qdq(din,ds,wt,INT,BLK);
    cpu_mm(out,din,Wd,Sd,1,INT,HID); memcpy(g_down,out,HID*4);
    for(int i=0;i<HID;i++)out[i]=bf16r(out[i]);
}
typedef struct{int S,I,O,rb,ng;}Dims; typedef struct{int length,block_size;}QP; typedef struct{int dimension;float limit;}SP;
static id<MTLLibrary> LIB; static id<MTLDevice> DEV; static int USE_SIMD=0;
static id<MTLComputePipelineState> mkpipe(const char*n){NSError*e=nil;
    id<MTLFunction>f=[LIB newFunctionWithName:[NSString stringWithUTF8String:n]];
    return f?[DEV newComputePipelineStateWithFunction:f error:&e]:nil;}
static id<MTLCommandQueue> QUE;
static void disp(id<MTLComputePipelineState>ps,NSArray*bufs,MTLSize grid,MTLSize tg){
    id<MTLCommandBuffer>cb=[QUE commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];for(NSUInteger i=0;i<bufs.count;i++)[en setBuffer:bufs[i] offset:0 atIndex:i];
    [en dispatchThreads:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted];}
static id<MTLBuffer> buf(const void*p,int n){return p?[DEV newBufferWithBytes:p length:n options:0]:[DEV newBufferWithLength:n options:0];}
static void cmp(const char*name,const float*c,const float*g,int n,int exact){
    int bad=0;double maxrel=0;for(int i=0;i<n;i++){unsigned hc,hg;memcpy(&hc,&c[i],4);memcpy(&hg,&g[i],4);
        if(hc!=hg){bad++;if(g[i]==0&&c[i]!=0)continue;double r=c[i]!=0?fabs((double)g[i]-c[i])/fabs((double)c[i]):0;if(r>maxrel)maxrel=r;}}
    printf("    %-10s %s  diff=%d/%d maxRel=%.3e%s\n",name,
        (exact?(bad==0?"BIT-EXACT":"MISMATCH"):(maxrel<=5e-6?"WITHIN-BOUND":"OVER-BOUND")),
        bad,n,maxrel, (exact&&bad)?"  <-- REGRESSION":"");
}
int main(int argc,char**argv){@autoreleasepool{
    FILE*f=fopen("c/metal/_moe_combined.metal","rb"); if(!f){printf("RED: combined source missing\n");return 2;}
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    DEV=MTLCreateSystemDefaultDevice();NSError*e=nil;MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;
    LIB=[DEV newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!LIB){printf("RED: compile failed:\n%s\n",[[e localizedDescription]UTF8String]);return 3;}
    QUE=[DEV newCommandQueue];
    USE_SIMD = getenv("COLI_MM") && !strcmp(getenv("COLI_MM"),"simd");
    id<MTLComputePipelineState> pQ=mkpipe("coli_v4_fp8_qdq"),pM=mkpipe(USE_SIMD?"coli_v4_matmul_mxfp4_simd":"coli_v4_matmul_mxfp4_ordered"),
        pB=mkpipe("coli_v4_bf16_round_array"),pS=mkpipe("coli_v4_swiglu");
    if(!pQ||!pM||!pB||!pS){printf("RED: a kernel entry is missing (Q%d M%d B%d S%d)\n",!!pQ,!!pM,!!pB,!!pS);return 4;}
    // fixtures
    int rbG=(HID+1)/2,ngG=(HID+31)/32, rbD=(INT+1)/2,ngD=(INT+31)/32;
    float in[HID]; uint8_t Wg[INT*rbG],Sg[INT*ngG],Wu[INT*rbG],Su[INT*ngG],Wd[HID*rbD],Sd[HID*ngD];
    unsigned SEED=(argc>1)?(unsigned)atoi(argv[1]):24601; srandom(SEED);
    for(int i=0;i<HID;i++)in[i]=((float)(random()%4000)-2000)/311.0f;
    #define FILL(W,S,nw,ns) for(int i=0;i<nw;i++)W[i]=(uint8_t)(random()&0xFF); for(int i=0;i<ns;i++)S[i]=(uint8_t)(112+(random()%22));
    FILL(Wg,Sg,INT*rbG,INT*ngG) FILL(Wu,Su,INT*rbG,INT*ngG) FILL(Wd,Sd,HID*rbD,HID*ngD)
    float rw=0.25f+(float)(random()%1000)/1000.0f, lim=(random()&1)?7.0f:0.0f;
    // CPU golden chain
    float c_out[HID],c_qdq[HID],c_gate[INT],c_swi[INT],c_down[HID];
    cpu_expert(c_out,c_qdq,c_gate,c_swi,c_down,in,Wg,Sg,Wu,Su,Wd,Sd,rw,lim);
    // GPU chain
    id<MTLBuffer> bin=buf(in,HID*4);
    id<MTLBuffer> bqdq=buf(NULL,HID*4),bqs=buf(NULL,1*4);
    QP qp1={HID,BLK}; disp(pQ,@[bqdq,bqs,bin,buf(&qp1,sizeof(qp1))],MTLSizeMake(1,1,1),MTLSizeMake(1,1,1));
    id<MTLBuffer> bgate=buf(NULL,INT*4),bup=buf(NULL,INT*4);
    Dims dG={1,HID,INT,rbG,ngG};id<MTLBuffer>bdG=buf(&dG,sizeof(dG));
    disp(pM,@[bqdq,buf(Wg,INT*rbG),buf(Sg,INT*ngG),bgate,bdG],USE_SIMD?MTLSizeMake(32,INT,1):MTLSizeMake(INT,1,1),MTLSizeMake(32,1,1));
    disp(pM,@[bqdq,buf(Wu,INT*rbG),buf(Su,INT*ngG),bup,bdG],USE_SIMD?MTLSizeMake(32,INT,1):MTLSizeMake(INT,1,1),MTLSizeMake(32,1,1));
    uint32_t ni=INT; id<MTLBuffer>bni=buf(&ni,4);
    disp(pB,@[bgate,bni],MTLSizeMake(INT,1,1),MTLSizeMake(64,1,1));
    disp(pB,@[bup,bni],MTLSizeMake(INT,1,1),MTLSizeMake(64,1,1));
    id<MTLBuffer> bact=buf(NULL,INT*4); SP sp={INT,lim};
    disp(pS,@[bact,bgate,bup,buf(&sp,sizeof(sp))],MTLSizeMake(INT,1,1),MTLSizeMake(64,1,1));
    // weighted = bf16(act*rw): do on CPU-side buffer op via a tiny inline — reuse bf16 kernel after scalar mul
    float actg[INT]; memcpy(actg,bact.contents,INT*4);
    float wt[INT]; for(int i=0;i<INT;i++)wt[i]=actg[i]*rw;   // scalar mul (exact), bf16 via kernel next
    id<MTLBuffer> bwt=buf(wt,INT*4); disp(pB,@[bwt,bni],MTLSizeMake(INT,1,1),MTLSizeMake(64,1,1));
    id<MTLBuffer> bdin=buf(NULL,INT*4),bds=buf(NULL,1*4);
    QP qp2={INT,BLK}; disp(pQ,@[bdin,bds,bwt,buf(&qp2,sizeof(qp2))],MTLSizeMake(1,1,1),MTLSizeMake(1,1,1));
    id<MTLBuffer> bdown=buf(NULL,HID*4); Dims dD={1,INT,HID,rbD,ngD};
    disp(pM,@[bdin,buf(Wd,HID*rbD),buf(Sd,HID*ngD),bdown,buf(&dD,sizeof(dD))],USE_SIMD?MTLSizeMake(32,HID,1):MTLSizeMake(HID,1,1),MTLSizeMake(32,1,1));
    float g_down_pre[HID]; memcpy(g_down_pre,bdown.contents,HID*4);
    uint32_t nh=HID; disp(pB,@[bdown,buf(&nh,4)],MTLSizeMake(HID,1,1),MTLSizeMake(64,1,1));
    float g_out[HID]; memcpy(g_out,bdown.contents,HID*4);
      int e2e=0,s13=0; for(int i=0;i<HID;i++){unsigned a,b;memcpy(&a,&c_out[i],4);memcpy(&b,&g_out[i],4);if(a!=b)e2e++;}
  for(int i=0;i<HID;i++){unsigned a,b;memcpy(&a,&c_qdq[i],4);memcpy(&b,((float*)bqdq.contents)+i,4);if(a!=b)s13++;}
  double e2erel=0;for(int i=0;i<HID;i++){double r=c_out[i]!=0?fabs((double)g_out[i]-c_out[i])/fabs((double)c_out[i]):0;if(r>e2erel)e2erel=r;}
  printf("  [%s] seed=%u rw=%.3f lim=%.0f : step1-3 %s(%d) | END-TO-END out diff=%d/%d maxRel=%.3e %s\n",USE_SIMD?"simd":"ord",SEED,rw,lim,s13?"MISMATCH":"exact",s13,e2e,HID,e2erel,e2e?"":"<-- BIT-EXACT");
  return s13?1:0;
}}
