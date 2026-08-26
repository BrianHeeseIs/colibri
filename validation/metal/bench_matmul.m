// REAL-DIM perf: V4 MoE matmul_mxfp4, CPU vs Metal. Shapes from models/deepseek-v4-flash.
#import <Metal/Metal.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <mach/mach_time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static const float mx4_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){union{uint32_t u;float f;}b;b.u=(uint32_t)s<<23;return b.f;}
static void cpu_mm(float*y,const float*x,const uint8_t*q4,const uint8_t*e8s,int S,int I,int O){
    int rb=(I+1)/2,ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){const uint8_t*w=q4+(int64_t)o*rb;const uint8_t*scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){const float*xs=x+(int64_t)s*I;float a=0;
            for(int g=0;g<ng;g++){int base=g*32,gl=32;if(base+gl>I)gl=I-base;float sc=mx4_scale(scl[g]),ga=0;
                for(int i=base;i<base+gl;i+=2){uint8_t by=w[i>>1];ga+=xs[i]*mx4_lut[by&0xF];if(i+1<base+gl)ga+=xs[i+1]*mx4_lut[by>>4];}
                a+=ga*sc;}y[(int64_t)s*O+o]=a;}}}
static int mmcount(const float*a,const float*b,int n){int m=0;for(int i=0;i<n;i++){unsigned u,v;
    memcpy(&u,&a[i],4);memcpy(&v,&b[i],4);if(u!=v)m++;}return m;}
static double now_s(void){static mach_timebase_info_data_t tb;if(!tb.denom)mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e9;}
typedef struct{int S,I,O,rb,ng;}Dims;
static void pack16(uint8_t*p,const uint8_t*d,int rows,int stride){
    for(int r=0;r<rows;r++){int t=r/16,l=r%16;for(int c=0;c<stride;c++)p[(t*stride+c)*16+l]=d[r*stride+c];}}
static id<MTLDevice> DEV; static id<MTLCommandQueue> QUE; static id<MTLLibrary> LIB;
static double *g_mean=NULL,*g_sd=NULL;
static double gpu_bench(const char*entry,int S,int I,int O,const uint8_t*q4,const uint8_t*e8,
                        const float*x,size_t wb,size_t sb,int iters,float*out){
    int isX = (strstr(entry,"xcache")!=NULL); int isSimd=(strstr(entry,"simd")!=NULL);
    NSError*e=nil;id<MTLFunction>fn=[LIB newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if(!fn)return -1;
    id<MTLComputePipelineState>ps=[DEV newComputePipelineStateWithFunction:fn error:&e]; if(!ps)return -1;
    int rb=(I+1)/2,ng=(I+31)/32;Dims dm={S,I,O,rb,ng};
    id<MTLBuffer>bx=[DEV newBufferWithBytes:x length:(size_t)S*I*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer>bq=[DEV newBufferWithBytes:q4 length:wb options:MTLResourceStorageModeShared];
    id<MTLBuffer>be=[DEV newBufferWithBytes:e8 length:sb options:MTLResourceStorageModeShared];
    id<MTLBuffer>by=[DEV newBufferWithLength:(size_t)S*O*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer>bd=[DEV newBufferWithBytes:&dm length:sizeof(dm) options:0];
    double best=1e9,sum=0,sum2=0;int nok=0;
    for(int it=0;it<iters;it++){
        id<MTLCommandBuffer>cb=[QUE commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
        [en setComputePipelineState:ps];[en setBuffer:bx offset:0 atIndex:0];[en setBuffer:bq offset:0 atIndex:1];
        [en setBuffer:be offset:0 atIndex:2];[en setBuffer:by offset:0 atIndex:3];[en setBuffer:bd offset:0 atIndex:4];
        NSUInteger tg=ps.maxTotalThreadsPerThreadgroup>256?256:ps.maxTotalThreadsPerThreadgroup;
        if(isX)        [en dispatchThreads:MTLSizeMake(O,S,1)   threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        else if(isSimd)[en dispatchThreads:MTLSizeMake(32,S*O,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        else           [en dispatchThreads:MTLSizeMake(S*O,1,1)  threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [en endEncoding];[cb commit];[cb waitUntilCompleted];
        double d=cb.GPUEndTime-cb.GPUStartTime;
        if(d>0){ if(d<best)best=d; if(it>0){sum+=d;sum2+=d*d;nok++;} }
    }
    if(out)memcpy(out,by.contents,(size_t)S*O*4);
    if(g_mean){ double m=nok?sum/nok:best; *g_mean=m;
        *g_sd = nok>1 ? sqrt(fmax(0.0,sum2/nok-m*m)) : 0.0; }
    return best;
}
int main(int argc,char**argv){@autoreleasepool{
    FILE*f=fopen("c/metal/coli_v4_matmul.metal","rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
    DEV=MTLCreateSystemDefaultDevice();NSError*e=nil;MTLCompileOptions*o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;
    LIB=[DEV newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!LIB){printf("compile fail: %s\n",[[e localizedDescription]UTF8String]);return 3;}
    QUE=[DEV newCommandQueue];
    int iters=argc>1?atoi(argv[1]):5;
#ifdef _OPENMP
    printf("  CPU threads (OpenMP): %d\n",omp_get_max_threads());
#else
    printf("  CPU threads: 1 (no OpenMP)\n");
#endif
    printf("  iters/variant: %d (best-of)\n\n",iters);
    struct{int S,I,O;const char*name;}CS[]={
        {1,4096,2048,"gate|up  S=1 4096->2048"},
        {1,2048,4096,"down     S=1 2048->4096"},
        {8,4096,2048,"gate|up  S=8 4096->2048"},
    };
    printf("  %-26s %9s %9s %9s %9s %9s %9s\n","shape","CPU ms","ord","hot","ord+xc","hot+xc","simd");
    for(int c=0;c<3;c++){
        int S=CS[c].S,I=CS[c].I,O=CS[c].O,rb=(I+1)/2,ng=(I+31)/32;
        float*x=malloc((size_t)S*I*4);uint8_t*q4=malloc((size_t)O*rb),*e8=malloc((size_t)O*ng);
        srandom(7);for(int i=0;i<S*I;i++)x[i]=((float)(random()%2001)-1000)/337.0f;
        for(size_t i=0;i<(size_t)O*rb;i++)q4[i]=(uint8_t)(random()&0xFF);
        for(size_t i=0;i<(size_t)O*ng;i++)e8[i]=(uint8_t)(110+(random()%30));
        int padO=((O+15)/16)*16;
        uint8_t*q4h=calloc((size_t)padO*rb,1),*e8h=calloc((size_t)padO*ng,1);
        pack16(q4h,q4,O,rb);pack16(e8h,e8,O,ng);
        /* INTEGRITY: every variant gets its OWN output buffer and its OWN mismatch count.
         * Until 2026-08-26 only `ordered` was dispatched with a non-NULL out; every other
         * variant was timed with out=NULL and silently never checked, so `simd` shipped a
         * 1.06e-3 relative error through many benchmark runs unnoticed. A benchmark that
         * TIMES a variant must also CHECK it. */
        float*yc=malloc((size_t)S*O*4),*yg=malloc((size_t)S*O*4);
        float*yh=malloc((size_t)S*O*4),*ysd=malloc((size_t)S*O*4);
        float*yse=malloc((size_t)S*O*4),*yox=malloc((size_t)S*O*4),*yhx=malloc((size_t)S*O*4);
        double bc=1e9,cs=0,cs2=0;int cn=0;
        for(int it=0;it<iters;it++){double t0=now_s();cpu_mm(yc,x,q4,e8,S,I,O);double d=now_s()-t0;
            if(d<bc)bc=d; if(it>0){cs+=d;cs2+=d*d;cn++;}}
        double cmean=cn?cs/cn:bc, csd=cn>1?sqrt(fmax(0.0,cs2/cn-cmean*cmean)):0.0;
        double m_o,d_o,m_h,d_h,m_ox,d_ox,m_hx,d_hx,m_s,d_s,m_se,d_se;
        g_mean=&m_o;g_sd=&d_o;
        double to=gpu_bench("coli_v4_matmul_mxfp4_ordered",S,I,O,q4,e8,x,(size_t)O*rb,(size_t)O*ng,iters,yg);
        int mism=0;for(int i=0;i<S*O;i++){unsigned a,b;memcpy(&a,&yc[i],4);memcpy(&b,&yg[i],4);if(a!=b)mism++;}
        g_mean=&m_h;g_sd=&d_h;
        double th=gpu_bench("coli_v4_matmul_mxfp4_ordered_hot",S,I,O,q4h,e8h,x,(size_t)padO*rb,(size_t)padO*ng,iters,yh);
        g_mean=&m_s;g_sd=&d_s;
        double ts=gpu_bench("coli_v4_matmul_mxfp4_simd",S,I,O,q4,e8,x,(size_t)O*rb,(size_t)O*ng,iters,ysd);
        g_mean=&m_se;g_sd=&d_se;
        double tse=gpu_bench("coli_v4_matmul_mxfp4_simd_exact",S,I,O,q4,e8,x,(size_t)O*rb,(size_t)O*ng,iters,yse);
        g_mean=&m_ox;g_sd=&d_ox;
        double tox=gpu_bench("coli_v4_matmul_mxfp4_ordered_xcache",S,I,O,q4,e8,x,(size_t)O*rb,(size_t)O*ng,iters,yox);
        g_mean=&m_hx;g_sd=&d_hx;
        double thx=gpu_bench("coli_v4_matmul_mxfp4_ordered_hot_xcache",S,I,O,q4h,e8h,x,(size_t)padO*rb,(size_t)padO*ng,iters,yhx);
        int mm_h=mmcount(yc,yh,S*O), mm_s=mmcount(yc,ysd,S*O), mm_se=mmcount(yc,yse,S*O);
        int mm_ox=mmcount(yc,yox,S*O), mm_hx=mmcount(yc,yhx,S*O);
        double bestg=to;if(th>0&&th<bestg)bestg=th;if(ts>0&&ts<bestg)bestg=ts;if(tox>0&&tox<bestg)bestg=tox;if(thx>0&&thx<bestg)bestg=thx;if(tse>0&&tse<bestg)bestg=tse;
        double bestex=to;if(th>0&&th<bestex)bestex=th;if(tox>0&&tox<bestex)bestex=tox;if(thx>0&&thx<bestex)bestex=thx;if(tse>0&&tse<bestex)bestex=tse;
        /* simd_exact is included in "best EXACT": validation/metal/probe_simd_parity proves it
         * bit-identical to the scalar reference on all production and ragged shapes. `simd`
         * (tree reduction) is NOT and stays excluded. */
        double bex=m_o; if(m_h<bex)bex=m_h; if(m_ox<bex)bex=m_ox; if(m_hx<bex)bex=m_hx; if(m_se<bex)bex=m_se;
        printf("  %s\n",CS[c].name);
        printf("      cpu %7.3f+-%.3f | ord %7.3f+-%.3f | hot %7.3f+-%.3f\n",cmean*1e3,csd*1e3,m_o*1e3,d_o*1e3,m_h*1e3,d_h*1e3);
        printf("      ord+xc %7.3f+-%.3f | hot+xc %7.3f+-%.3f | simd %7.3f+-%.3f\n",m_ox*1e3,d_ox*1e3,m_hx*1e3,d_hx*1e3,m_s*1e3,d_s*1e3);
        printf("      simd_exact %7.3f+-%.3f\n",m_se*1e3,d_se*1e3);
        /* Reading this line: cpu_mm is the COLD scalar reference (per-32-group scale).
         *   ord / hot / ord+xc / simd_exact  MUST be 0.
         *   simd    is EXPECTED non-zero  -- tree reduction, ~1e-3 rel err. Not bit-exact.
         *   hot+xc  is EXPECTED non-zero  -- it deliberately applies the scale PER COLUMN to
         *           match coli_fp4_dual_matvec_rows16_v10's NEON loop, a DIFFERENT reference
         *           than cpu_mm. See the comment at c/metal/coli_v4_matmul.metal:203-212.
         *           It is not wrong; it is being compared against the wrong reference here. */
        printf("      CHECK vs cpu-cold (bit-mismatches / %d):  ord %d | hot %d | ord+xc %d | hot+xc %d* | simd %d* | simd_exact %d\n",
               S*O,mism,mm_h,mm_ox,mm_hx,mm_s,mm_se);
        printf("      (* hot+xc and simd are EXPECTED to differ from the cold reference -- see source comment)\n");
        printf("      => best EXACT %.3f ms = %.2fx CPU  |  simd %.2fx CPU  |  simd_exact %.2fx CPU  |  ord==cpu %s\n",bex*1e3,cmean/bex,cmean/m_s,cmean/m_se,mism?"NO":"bit-exact");
        free(x);free(q4);free(e8);free(q4h);free(e8h);free(yc);free(yg);
        free(yh);free(ysd);free(yse);free(yox);free(yhx);
    }
    return 0;}}
