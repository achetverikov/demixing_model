// Ordinary likelihood EM; same arithmetic as the validated persistent CUDA solver.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

constexpr float PI = 3.14159265358979323846f;
constexpr float OMEGA = PI / 180.f;

// Stable local subtraction, followed by exact half-open range correction.
// In particular, nextafterf(180,-inf)+180 can round to360 in float32.

__device__ float wrap(float x) {
  float r=x-360.f*floorf((x+180.f)/360.f);
  if(r < -180.f) r+=360.f;
  if(r >= 180.f) r-=360.f;
  return r;
}


__device__ float warp_sum(float x) {
  for (int shift=16; shift>0; shift/=2)
    x += __shfl_down_sync(0xffffffff, x, shift);
  return __shfl_sync(0xffffffff, x, 0);
}

struct Axis { float logp, first, second; };
__device__ __forceinline__ Axis wrapped_body(float x, float mean, float sd) {
  float d=wrap(x-mean), var=sd*sd;
  if (sd < 60.f) {
    float z0=d-360.f, z1=d, z2=d+360.f;
    float a0=-.5f*z0*z0/var, a1=-.5f*z1*z1/var, a2=-.5f*z2*z2/var;
    float mx=fmaxf(a0,fmaxf(a1,a2));
    float p0=expf(a0-mx), p1=expf(a1-mx), p2=expf(a2-mx);
    float sum=p0+p1+p2;
    return {mx+logf(sum)-logf(sd)-.5f*logf(2.f*PI),
      (p0*z0+p1*z1+p2*z2)/sum,
      (p0*z0*z0+p1*z1*z1+p2*z2*z2)/sum};
  }
  float c,s; sincosf(d*OMEGA,&s,&c);
  float cn=c,sn=s,denom=1.f,first=0.f,second=0.f;
  #pragma unroll
  for (int h=1; h<=6; ++h) {
    float a=expf(-.5f*var*OMEGA*OMEGA*h*h);
    denom += 2.f*a*cn;
    first += h*a*sn;
    second += h*h*a*cn;
    float next=cn*c-sn*s;
    sn=sn*c+cn*s; cn=next;
  }
  return {logf(denom)-logf(360.f),2.f*var*OMEGA*first/denom,
    var-2.f*var*var*OMEGA*OMEGA*second/denom};
}

// Outline the same density formula where smaller code beats call overhead.
// At N=500 the measured tradeoff favors keeping it inside the statistics loop.
__device__ __noinline__ Axis wrapped_outlined(float x,float mean,float sd) {
  return wrapped_body(x,mean,sd);
}

template<int N> __device__ Axis wrapped(float x,float mean,float sd) {
  if constexpr(N==500)return wrapped_body(x,mean,sd);
  else return wrapped_outlined(x,mean,sd);
}

template<int N> __device__ void statistics(const float *xf,const float *xi,
    const float *theta,float &ll,float *count,float *mom) {
  float sums[11]={};
  constexpr int ITEMS=(N+31)/32;
  int lane=threadIdx.x%32;
  // Keep the observation loop rolled to limit kernel code size at larger N.
  #pragma unroll 1
  for(int j=0;j<ITEMS;++j) {
    if(lane+j*32<N) {
      Axis f0=wrapped<N>(xf[j],theta[2],theta[6]);
      Axis f1=wrapped<N>(xf[j],theta[3],theta[7]);
      float di0=xi[j]-theta[4],di1=xi[j]-theta[5];
      float v0=theta[8]*theta[8],v1=theta[9]*theta[9];
      float l0=f0.logp-.5f*di0*di0/v0-logf(theta[8])-.5f*logf(2.f*PI)+logf(theta[0]);
      float l1=f1.logp-.5f*di1*di1/v1-logf(theta[9])-.5f*logf(2.f*PI)+logf(theta[1]);
      float mx=fmaxf(l0,l1),a0=expf(l0-mx),a1=expf(l1-mx);
      float norm=mx+logf(a0+a1),r0=expf(l0-norm),r1=expf(l1-norm);
      sums[0]+=norm; sums[1]+=r0; sums[2]+=r1;
      sums[3]+=r0*f0.first; sums[4]+=r1*f1.first;
      sums[5]+=r0*f0.second; sums[6]+=r1*f1.second;
      sums[7]+=r0*di0; sums[8]+=r1*di1;
      sums[9]+=r0*di0*di0; sums[10]+=r1*di1*di1;
    }
  }
  #pragma unroll
  for(int k=0;k<11;++k) sums[k]=warp_sum(sums[k]);
  ll=sums[0]; count[0]=sums[1]; count[1]=sums[2];
  #pragma unroll
  for(int k=0;k<8;++k) {
    float den=count[k%2]>0.f?count[k%2]:1.f;
    mom[k]=sums[k+3]/den;
  }
}

struct EMState {
  float theta[10],count[2],mom[8],ll,relative,change,worst;
  int iteration,decreases;
};

__device__ bool done(const EMState &s) {
  return isfinite(s.ll) && s.relative<=1.e-6f && s.change<=1.e-5f;
}

template<int N> __device__ void step(const float *xf,const float *xi,EMState &s) {
  float next[10];
  next[0]=fminf(.9f,fmaxf(.1f,s.count[0]/N)); next[1]=1.f-next[0];
  #pragma unroll
  for(int c=0;c<2;++c) {
    next[2+c]=s.count[c]>0.f?wrap(s.theta[2+c]+s.mom[c]):s.theta[2+c];
    next[4+c]=s.count[c]>0.f?s.theta[4+c]+s.mom[4+c]:s.theta[4+c];
    next[6+c]=s.count[c]>0.f?sqrtf(fmaxf(s.mom[2+c]-s.mom[c]*s.mom[c],1.f)):s.theta[6+c];
    next[8+c]=s.count[c]>0.f?sqrtf(fmaxf(s.mom[6+c]-s.mom[4+c]*s.mom[4+c],1.f)):s.theta[8+c];
  }
  s.change=0.f;
  #pragma unroll
  for(int c=0;c<2;++c) {
    s.change=fmaxf(s.change,fabsf(next[c]-s.theta[c]));
    float df=fmaxf(fabsf(wrap(next[2+c]-s.theta[2+c]))-4.f*FLT_EPSILON*360.f,0.f)/s.theta[6+c];
    float di=fmaxf(fabsf(next[4+c]-s.theta[4+c])-2.f*FLT_EPSILON*(fabsf(next[4+c])+fabsf(s.theta[4+c])),0.f)/s.theta[8+c];
    s.change=fmaxf(s.change,fmaxf(df,di));
    s.change=fmaxf(s.change,fabsf((next[6+c]-s.theta[6+c])/s.theta[6+c]));
    s.change=fmaxf(s.change,fabsf((next[8+c]-s.theta[8+c])/s.theta[8+c]));
  }
  #pragma unroll
  for(int k=0;k<10;++k) s.theta[k]=next[k];
  float old=s.ll;
  statistics<N>(xf,xi,s.theta,s.ll,s.count,s.mom);
  float delta=s.ll-old;
  s.relative=fabsf(delta)/(1.e-12f+fabsf(old));
  s.worst=fminf(s.worst,delta);
  s.decreases += delta < -8.f*FLT_EPSILON*(1.f+fabsf(old));
  ++s.iteration;
}


__device__ void store_state(const EMState&s,float*out) {
  #pragma unroll
  for(int k=0;k<10;++k)out[k]=s.theta[k];
  out[10]=s.ll;out[11]=s.count[0];out[12]=s.count[1];
  #pragma unroll
  for(int k=0;k<8;++k)out[13+k]=s.mom[k];
  out[21]=s.iteration;out[22]=s.relative;out[23]=s.change;
  out[24]=s.worst;out[25]=s.decreases;
}

__device__ EMState load_state(const float*in) {
  EMState s;
  #pragma unroll
  for(int k=0;k<10;++k)s.theta[k]=in[k];
  s.ll=in[10];s.count[0]=in[11];s.count[1]=in[12];
  #pragma unroll
  for(int k=0;k<8;++k)s.mom[k]=in[13+k];
  s.iteration=int(in[21]);s.relative=in[22];s.change=in[23];
  s.worst=in[24];s.decreases=int(in[25]);return s;
}
