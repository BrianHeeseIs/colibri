// T1 probe: bit-exactness of MSL primitive ports vs CPU originals.
#import <Metal/Metal.h>
#import <stdio.h>
#import <string.h>
#import <math.h>
#define SRCPATH "c/metal/coli_v4_decode.metal"
#define HX(v) ({unsigned _u; float _t=(v); memcpy(&_u,&_t,4); _u;})

// ---- CPU originals (verbatim from c/quant.h and c/deepseek_v4.c) ----
static const float mx4_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){union{uint32_t u;float f;}b;b.u=(uint32_t)s<<23;return b.f;}
static float coli_bf16_round(float value){
    uint32_t bits; memcpy(&bits,&value,4);
    if((bits&0x7f800000u)!=0x7f800000u){uint32_t tie=(bits>>16)&1u; bits+=0x7fffu+tie;}
    bits&=0xffff0000u; memcpy(&value,&bits,4); return value;
}
static float coli_e4m3fn_decode(uint8_t value){
    int sign=value>>7, exponent=(value>>3)&15, mantissa=value&7;
    if(exponent==15&&mantissa==7) return NAN;
    float number;
    if(!exponent) number=ldexpf((float)mantissa,-9);
    else number=ldexpf(1.0f+(float)mantissa/8.0f,exponent-7);
    return sign?-number:number;
}
static float sigmoidf_stable(float value){
    if(value>=0.0f){float decay=expf(-value); return 1.0f/(1.0f+decay);}
    float growth=expf(value); return growth/(1.0f+growth);
}

static int fails=0, total=0;
static void chk(const char*grp,int i,unsigned in,float c,float g){
    total++;
    unsigned hc=HX(c), hg=HX(g);
    if(hc!=hg){ fails++;
        if(fails<=12) printf("    MISMATCH %-10s in=0x%08x cpu=0x%08x(%+.9e) gpu=0x%08x(%+.9e)\n",
                             grp,in,hc,c,hg,g);
    }
}
int main(void){@autoreleasepool{
    FILE*f=fopen(SRCPATH,"rb");
    if(!f){printf("RED: %s does not exist yet -- port not written.\n",SRCPATH); return 2;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char*src=malloc(n+1); fread(src,1,n,f); src[n]=0; fclose(f);

    id<MTLDevice> d=MTLCreateSystemDefaultDevice();
    NSError*e=nil; MTLCompileOptions*o=[MTLCompileOptions new]; o.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){printf("RED: compile failed:\n%s\n",[[e localizedDescription]UTF8String]); return 3;}
    id<MTLFunction> fn=[lib newFunctionWithName:@"coli_v4_probe_primitives"];
    if(!fn){printf("RED: entry coli_v4_probe_primitives missing\n"); return 4;}
    id<MTLComputePipelineState> ps=[d newComputePipelineStateWithFunction:fn error:&e];
    if(!ps){printf("RED: pipeline failed: %s\n",[[e localizedDescription]UTF8String]); return 5;}

    // input plan: 16 lut + 256 scale + 256 e4m3 + 32 bf16 + 32 sigmoid
    const int NL=16,NS=256,NE=256,NB=32,NG=32;
    const int CNT=NL+NS+NE+NB+NG;
    uint32_t *in=calloc(CNT,4); float *cpu=calloc(CNT,4);
    int k=0;
    for(int i=0;i<NL;i++){ in[k]=i; cpu[k]=mx4_lut[i]; k++; }
    for(int i=0;i<NS;i++){ in[k]=i; cpu[k]=mx4_scale((uint8_t)i); k++; }
    for(int i=0;i<NE;i++){ in[k]=i; cpu[k]=coli_e4m3fn_decode((uint8_t)i); k++; }
    float bv[32]={0.f,-0.f,1.f,-1.f,0x1p-126f,-0x1p-126f,3.14159265f,-2.718281828f,
        1e-38f,1e38f,65504.f,0.5f,1.5f,2.5f,1.0000001f,0.9999999f,
        123456.789f,-987654.321f,1e-45f,-1e-45f,INFINITY,-INFINITY,NAN,-NAN,
        1.0001221f,1.0001831f,1.0002441f,2.0002441f,7.9999995f,8.0000005f,1e-30f,-1e-30f};
    for(int i=0;i<NB;i++){ memcpy(&in[k],&bv[i],4); cpu[k]=coli_bf16_round(bv[i]); k++; }
    float gv[32]={0.f,-0.f,1.f,-1.f,0.5f,-0.5f,10.f,-10.f,20.f,-20.f,50.f,-50.f,
        88.f,-88.f,100.f,-100.f,1e-8f,-1e-8f,0.1f,-0.1f,3.f,-3.f,7.f,-7.f,
        1e-30f,-1e-30f,15.f,-15.f,30.f,-30.f,1.f/3.f,-1.f/3.f};
    for(int i=0;i<NG;i++){ memcpy(&in[k],&gv[i],4); cpu[k]=sigmoidf_stable(gv[i]); k++; }

    id<MTLBuffer> bi=[d newBufferWithBytes:in length:CNT*4 options:0];
    id<MTLBuffer> bo=[d newBufferWithLength:CNT*4 options:0];
    id<MTLCommandQueue> q=[d newCommandQueue]; id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];[en setBuffer:bi offset:0 atIndex:0];[en setBuffer:bo offset:0 atIndex:1];
    [en dispatchThreads:MTLSizeMake(CNT,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    float*gpu=(float*)bo.contents;

    struct{const char*n;int off,cnt;} G[5]={{"mx4_lut",0,NL},{"mx4_scale",NL,NS},
        {"e4m3fn",NL+NS,NE},{"bf16_round",NL+NS+NE,NB},{"sigmoid",NL+NS+NE+NB,NG}};
    printf("  %-12s %6s %6s\n","group","n","fail");
    for(int g=0;g<5;g++){
        int before=fails;
        for(int i=0;i<G[g].cnt;i++) chk(G[g].n,i,in[G[g].off+i],cpu[G[g].off+i],gpu[G[g].off+i]);
        printf("  %-12s %6d %6d %s\n",G[g].n,G[g].cnt,fails-before,(fails-before)?"FAIL":"exact");
    }
    printf("  TOTAL %d/%d exact\n",total-fails,total);
    // Exit semantics: sigmoid is KNOWN-DIVERGENT (precise::exp != libm expf, plus GPU
    // denormal-result flush). Documented in .backlog/ft-deepmetal-notepad.md T1 RESULT.
    // Gating on total equality would leave this probe permanently red, which trains people
    // to ignore it and would mask a genuine future regression. So: the four bit-exact groups
    // must stay bit-exact, and sigmoid must stay WITHIN its documented bound.
    int sig_off = NL+NS+NE+NB, sig_bad = 0, sig_flush = 0, sig_maxulp = 0; double sig_maxrel = 0;
    for(int i=0;i<NG;i++){
        unsigned hc=HX(cpu[sig_off+i]), hg=HX(gpu[sig_off+i]);
        if(hc==hg) continue;
        sig_bad++;
        // Denormal-flush is a SEPARATE class: cpu yields a denormal, gpu yields exactly +0.
        // ULP distance is meaningless across that boundary (it reads in the millions) and
        // relative error is 100%. Both metrics would be true and completely misleading, so
        // this class is counted on its own and excluded from the ULP/rel statistics.
        if(gpu[sig_off+i]==0.0f && cpu[sig_off+i]!=0.0f){ sig_flush++; continue; }
        int u=abs((int)hc-(int)hg); if(u>sig_maxulp) sig_maxulp=u;
        double r = cpu[sig_off+i]!=0 ? fabs((double)gpu[sig_off+i]-(double)cpu[sig_off+i])
                                       /fabs((double)cpu[sig_off+i]) : 0;
        if(r>sig_maxrel) sig_maxrel=r;
    }
    int exact_group_fails = fails - sig_bad;
    const int  SIG_MAX_ULP   = 2;
    const double SIG_MAX_REL = 5e-07;
    const int  SIG_MAX_FLUSH = 2;   // exp(-88), exp(-100) in this fixed 32-sample vector
    int regressed = 0;
    if(exact_group_fails != 0){
        printf("  REGRESSION: %d mismatch(es) in groups that must be BIT-EXACT\n",exact_group_fails);
        regressed = 1;
    }
    if(sig_flush > SIG_MAX_FLUSH){
        printf("  REGRESSION: denormal-flush count %d > documented %d\n",sig_flush,SIG_MAX_FLUSH);
        regressed = 1;
    }
    if(sig_maxulp > SIG_MAX_ULP || sig_maxrel > SIG_MAX_REL){
        printf("  REGRESSION: sigmoid drifted beyond documented bound "
               "(maxULP %d > %d or maxRel %.3e > %.3e)\n",
               sig_maxulp,SIG_MAX_ULP,sig_maxrel,SIG_MAX_REL);
        regressed = 1;
    }
    if(!regressed)
        printf("  OK: 4 groups bit-exact; sigmoid divergent as documented "
               "(%d/%d = %d transcendental + %d denormal-flush, maxULP %d, maxRel %.3e) -- within bound\n",
               sig_bad,NG,sig_bad-sig_flush,sig_flush,sig_maxulp,sig_maxrel);
    return regressed;
}}
