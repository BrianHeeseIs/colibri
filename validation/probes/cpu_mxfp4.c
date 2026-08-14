#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <math.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
#define I_DIM 4096
#define O_DIM 2048
#define BLK 32
static const float E2M1[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
int main(void){
  size_t wb=(size_t)O_DIM*(I_DIM/2), sb=(size_t)O_DIM*(I_DIM/BLK);
  unsigned char *W=malloc(wb), *SC=malloc(sb);
  for(size_t i=0;i<wb;i++) W[i]=(unsigned char)(i*31u);
  for(size_t i=0;i<sb;i++) SC[i]=127;
  printf("  CPU (12 threads via GCD), same shape I=%d O=%d blk=%d\n\n",I_DIM,O_DIM,BLK);
  printf("  %4s %10s %12s %12s %9s\n","S","time_ms","GFLOP/s","us/token","speedup");
  double base=0; int Ss[]={1,2,4,8,16};
  for(int t=0;t<5;t++){
    int S=Ss[t];
    float *X=malloc((size_t)S*I_DIM*4), *Y=malloc((size_t)S*O_DIM*4);
    for(size_t i=0;i<(size_t)S*I_DIM;i++) X[i]=(float)((i%17)-8)*0.125f;
    double best=1e9;
    for(int it=0;it<5;it++){
      double t0=now_s();
      dispatch_apply(12, DISPATCH_APPLY_AUTO, ^(size_t th){
        size_t rows=O_DIM/12, r0=th*rows, r1=(th==11)?O_DIM:r0+rows;
        for(size_t row=r0; row<r1; ++row){
          const unsigned char *w=W+row*(I_DIM/2); const unsigned char *sc=SC+row*(I_DIM/BLK);
          for(int s=0;s<S;++s){
            const float *x=X+(size_t)s*I_DIM; float acc=0.f;
            for(int b=0;b<I_DIM/BLK;++b){
              float g=0.f; int base2=b*BLK;
              for(int k=0;k<BLK;k+=2){
                unsigned char by=w[(base2+k)>>1];
                g+=E2M1[by&0xF]*x[base2+k];
                g+=E2M1[(by>>4)&0xF]*x[base2+k+1];
              }
              acc+=g*ldexpf(1.0f,(int)sc[b]-127);
            }
            Y[(size_t)s*O_DIM+row]=acc;
          }
        }
      });
      double dt=now_s()-t0; if(dt<best) best=dt;
    }
    double flops=2.0*S*I_DIM*O_DIM, us=best*1e6/S;
    if(t==0) base=us;
    printf("  %4d %10.3f %12.1f %12.1f %8.2fx\n",S,best*1e3,flops/best/1e9,us,base/us);
    free(X); free(Y);
  }
  return 0;
}
