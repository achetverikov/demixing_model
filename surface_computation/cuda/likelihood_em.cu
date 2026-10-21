// JAX FFI owns buffers and the CUDA stream; no global queue or raw-pointer borrowing.
#include "likelihood_em.cuh"
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;

__device__ bool finite_state(const EMState&s) {
  bool valid=isfinite(s.ll)&&isfinite(s.count[0])&&isfinite(s.count[1]);
  #pragma unroll
  for(int k=0;k<10;++k)valid=valid&&isfinite(s.theta[k]);
  #pragma unroll
  for(int k=0;k<8;++k)valid=valid&&isfinite(s.mom[k]);
  return valid;
}

template<int N> __device__ void squarem(const float*xf,const float*xi,EMState&s,int max_iter) {
  float stepmax=1.f;
  while(s.iteration<max_iter&&isfinite(s.ll)&&!done(s)) {
    if(s.iteration<10||max_iter-s.iteration<3){step<N>(xf,xi,s);continue;}
    float base[10],first[10];
    #pragma unroll
    for(int k=0;k<10;++k)base[k]=s.theta[k];
    step<N>(xf,xi,s);
    if(done(s)||!isfinite(s.ll))break;
    #pragma unroll
    for(int k=0;k<10;++k)first[k]=s.theta[k];
    step<N>(xf,xi,s);
    if(done(s)||!isfinite(s.ll))break;
    // Use one local chart per cycle: scaled means, wrapped angular differences,
    // log SDs, and the original coupled weights (including active bounds).
    float r[10],v[10],r2=0.f,v2=0.f;
    #pragma unroll
    for(int k=0;k<10;++k) {
      if(k<2){r[k]=first[k]-base[k];v[k]=s.theta[k]-first[k]-r[k];}
      else if(k<6) {
        float d1=first[k]-base[k],d2=s.theta[k]-first[k];
        if(k<4){d1=wrap(d1);d2=wrap(d2);}
        r[k]=d1/base[k+4];v[k]=d2/base[k+4]-r[k];
      }else{r[k]=logf(first[k]/base[k]);v[k]=logf(s.theta[k]/first[k])-r[k];}
    }
    #pragma unroll
    for(int k=0;k<10;++k){r2+=r[k]*r[k];v2+=v[k]*v[k];}
    float alpha=fminf(stepmax,fmaxf(1.f,sqrtf(r2/fmaxf(v2,1.e-30f))));
    // A near-unit step offers no useful extrapolation: retain the exact F^2 state.
    if(alpha<=1.01f){stepmax=fminf(256.f,stepmax*4.f);continue;}
    float delta[10];
    #pragma unroll
    for(int k=0;k<10;++k)delta[k]=2.f*alpha*r[k]+alpha*alpha*v[k];
    EMState trial=s;
    trial.theta[0]=fminf(.9f,fmaxf(.1f,base[0]+delta[0]));trial.theta[1]=1.f-trial.theta[0];
    #pragma unroll
    for(int k=2;k<6;++k){trial.theta[k]=base[k]+delta[k]*base[k+4];if(k<4)trial.theta[k]=wrap(trial.theta[k]);}
    #pragma unroll
    for(int k=6;k<10;++k)trial.theta[k]=fmaxf(1.f,base[k]*expf(delta[k]));
    statistics<N>(xf,xi,trial.theta,trial.ll,trial.count,trial.mom);
    bool valid=finite_state(trial);
    if(valid){step<N>(xf,xi,trial);valid=finite_state(trial);}
    // Stabilize before testing ascent against F^2; reject without losing its
    // cached moments. A rejected stabilization attempt still counts toward the cap.
    valid=valid&&trial.ll>=s.ll;
    if(valid)s=trial;else ++s.iteration;
    if(alpha==stepmax)stepmax=valid?fminf(256.f,stepmax*4.f):fmaxf(1.f,stepmax/4.f);
  }
}

__device__ int sum_i(int v) {
  for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);
  return v;
}
__device__ float min_f(float v) {
  for(int d=16;d;d/=2) {
    float b=__shfl_down_sync(0xffffffff,v,d);
    v=(isnan(v)||isnan(b))?NAN:fminf(v,b);
  }
  return v;
}
__device__ bool converged(const float*s) {
  bool ok=isfinite(s[10]) && s[22]<=1.e-6f && s[23]<=1.e-5f;
  #pragma unroll
  for(int k=0;k<10;++k)ok=ok && isfinite(s[k]);
  return ok;
}
__device__ void max_pair(float&ll,int&index) {
  for(int d=16;d;d/=2) {
    float other=__shfl_down_sync(0xffffffff,ll,d);
    int oi=__shfl_down_sync(0xffffffff,index,d);
    if(other>ll || (other==ll && oi<index)){ll=other;index=oi;}
  }
}

__global__ void finish_kernel(const float*raw,float*compact,int starts) {
  int dataset=blockIdx.x,t=threadIdx.x,lane=t%32,warp=t/32;
  __shared__ float sh_ll[8],sh_worst[8],winning_ll;
  __shared__ int sh_index[8],sh_cap[8],sh_bad[8],sh_decr[8],sh_tie[8],sh_ok_tie[8],winning_index;
  float best=-INFINITY,worst=0.f;int index=0x7fffffff,cap=0,bad=0,decr=0;
  for(int i=t;i<starts;i+=256) {
    const float*s=raw+(size_t(dataset)*starts+i)*26;
    bool finite=isfinite(s[10]);float ll=finite?s[10]:-INFINITY;
    if(ll>best || (ll==best && i<index)){best=ll;index=i;}
    bad+=!finite;cap+=finite && !converged(s);decr+=int(s[25]);
    worst=(isnan(worst)||isnan(s[24]))?NAN:fminf(worst,s[24]);
  }
  max_pair(best,index);cap=sum_i(cap);bad=sum_i(bad);decr=sum_i(decr);worst=min_f(worst);
  if(lane==0) {
    sh_ll[warp]=best;sh_index[warp]=index;sh_cap[warp]=cap;
    sh_bad[warp]=bad;sh_decr[warp]=decr;sh_worst[warp]=worst;
  }
  __syncthreads();
  if(warp==0) {
    best=lane<8?sh_ll[lane]:-INFINITY;index=lane<8?sh_index[lane]:0x7fffffff;
    cap=lane<8?sh_cap[lane]:0;bad=lane<8?sh_bad[lane]:0;decr=lane<8?sh_decr[lane]:0;
    worst=lane<8?sh_worst[lane]:0.f;
    max_pair(best,index);cap=sum_i(cap);bad=sum_i(bad);decr=sum_i(decr);worst=min_f(worst);
    if(lane==0) {
      winning_ll=best;winning_index=index;
      compact[dataset*33+26]=index;compact[dataset*33+28]=cap;
      compact[dataset*33+29]=bad;compact[dataset*33+30]=decr;compact[dataset*33+31]=worst;
    }
  }
  __syncthreads();
  if(t<26)compact[dataset*33+t]=raw[(size_t(dataset)*starts+winning_index)*26+t];
  int ties=0,ok_ties=0;
  for(int i=t;i<starts;i+=256) {
    const float*s=raw+(size_t(dataset)*starts+i)*26;
    bool tied=isfinite(s[10]) && s[10]==winning_ll;
    ties+=tied;ok_ties+=tied && converged(s);
  }
  ties=sum_i(ties);ok_ties=sum_i(ok_ties);
  if(lane==0){sh_tie[warp]=ties;sh_ok_tie[warp]=ok_ties;}
  __syncthreads();
  if(warp==0) {
    ties=sum_i(lane<8?sh_tie[lane]:0);ok_ties=sum_i(lane<8?sh_ok_tie[lane]:0);
    if(lane==0){compact[dataset*33+27]=ties;compact[dataset*33+32]=ok_ties;}
  }
}


template<int N,bool Accelerate> __global__ void queued_kernel(const float*x,const float*initial,
    float*out,unsigned*head,int datasets,int starts,int max_iter) {
  int lane=threadIdx.x%32;
  while(true) {
    unsigned start=0;
    if(lane==0)start=atomicAdd(head,1u);
    start=__shfl_sync(0xffffffff,start,0);
    if(start>=datasets*starts)return;
    int dataset=start/starts;
    constexpr int ITEMS=(N+31)/32;
    float xf[ITEMS],xi[ITEMS];
    #pragma unroll
    for(int j=0;j<ITEMS;++j)if(lane+j*32<N) {
      xf[j]=x[(size_t(dataset)*N+lane+j*32)*2];
      xi[j]=x[(size_t(dataset)*N+lane+j*32)*2+1];
    }
    EMState s;
    #pragma unroll
    for(int k=0;k<10;++k)s.theta[k]=initial[size_t(start)*10+k];
    statistics<N>(xf,xi,s.theta,s.ll,s.count,s.mom);
    s.iteration=0;s.decreases=0;s.relative=INFINITY;s.change=INFINITY;s.worst=0.f;
    if constexpr(Accelerate)squarem<N>(xf,xi,s,max_iter);
    else while(s.iteration<max_iter && isfinite(s.ll) && !done(s))step<N>(xf,xi,s);
    if(lane==0)store_state(s,out+size_t(start)*26);
  }
}

template<int N> __global__ void continue_kernel(const float*x,float*raw,
    int datasets,int starts,int max_iter) {
  int dataset=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
  if(dataset>=datasets)return;
  constexpr int ITEMS=(N+31)/32;
  float xf[ITEMS],xi[ITEMS];
  #pragma unroll
  for(int j=0;j<ITEMS;++j)if(lane+j*32<N) {
    xf[j]=x[(size_t(dataset)*N+lane+j*32)*2];
    xi[j]=x[(size_t(dataset)*N+lane+j*32)*2+1];
  }
  float*states=raw+size_t(dataset)*starts*26;
  while(true) {
    float ll=-INFINITY;int best=0x7fffffff;
    for(int i=lane;i<starts;i+=32) {
      float value=isfinite(states[i*26+10])?states[i*26+10]:-INFINITY;
      if(value>ll || (value==ll && i<best)){ll=value;best=i;}
    }
    max_pair(ll,best);best=__shfl_sync(0xffffffff,best,0);
    EMState s=load_state(states+best*26);
    if(!isfinite(s.ll) || done(s) || s.iteration>=max_iter)return;
    // Continue from cached moments, then reselect if rounding changes the winner.
    while(s.iteration<max_iter && isfinite(s.ll) && !done(s))step<N>(xf,xi,s);
    if(lane==0)store_state(s,states+best*26);
    __syncwarp();
  }
}

template<int N,bool Accelerate> cudaError_t launch(cudaStream_t stream,const float*x,
    const float*theta,float*raw,float*compact,unsigned*head,int datasets,
    int starts,int first_cap,int max_iter,bool occupancy) {
  int device,sm_count,active=4;
  cudaError_t status=cudaGetDevice(&device);
  if(status!=cudaSuccess)return status;
  status=cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device);
  if(status!=cudaSuccess)return status;
  if(occupancy) {
    status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,queued_kernel<N,Accelerate>,128,0);
    if(status!=cudaSuccess)return status;
  }
  int needed=(datasets*starts+3)/4,blocks=std::min(needed,sm_count*active);
  status=cudaMemsetAsync(head,0,sizeof(unsigned),stream);
  if(status!=cudaSuccess)return status;
  queued_kernel<N,Accelerate><<<blocks,128,0,stream>>>(x,theta,raw,head,datasets,starts,first_cap);
  if(max_iter>first_cap)
    continue_kernel<N><<<(datasets+3)/4,128,0,stream>>>(x,raw,datasets,starts,max_iter);
  finish_kernel<<<datasets,256,0,stream>>>(raw,compact,starts);
  return cudaGetLastError();
}

ffi::Error fit(cudaStream_t stream,ffi::BufferR3<ffi::F32>x,
    ffi::BufferR4<ffi::F32>theta,ffi::ResultBufferR3<ffi::F32>raw,
    ffi::ResultBufferR2<ffi::F32>compact,ffi::ResultBufferR1<ffi::S32>queue,
    int64_t first_cap,int64_t max_iter,int64_t occupancy,int64_t accelerate) {
  auto xd=x.dimensions(),td=theta.dimensions();
  int datasets=xd[0],n=xd[1],starts=td[1];
  if(datasets<1 || starts<1 || xd[2]!=2 || td[0]!=datasets || td[2]!=5 || td[3]!=2
      || first_cap<1 || max_iter<first_cap)
    return ffi::Error::InvalidArgument("Invalid CUDA EM shape or iteration limit");
  if(raw->dimensions()[0]!=datasets || raw->dimensions()[1]!=starts || raw->dimensions()[2]!=26
      || compact->dimensions()[0]!=datasets || compact->dimensions()[1]!=33 || queue->dimensions()[0]!=1)
    return ffi::Error::InvalidArgument("Invalid CUDA EM result shape");
  cudaError_t status;
  auto head=reinterpret_cast<unsigned*>(queue->typed_data());
  #define DM_LAUNCH_IMPL(N,A) launch<N,A>(stream,x.typed_data(),theta.typed_data(),raw->typed_data(), \
      compact->typed_data(),head,datasets,starts,first_cap,max_iter,occupancy)
  #define DM_LAUNCH(N) (accelerate?DM_LAUNCH_IMPL(N,true):DM_LAUNCH_IMPL(N,false))
  switch(n) {
    case 4:status=DM_LAUNCH(4);break;
    case 10:status=DM_LAUNCH(10);break;
    case 20:status=DM_LAUNCH(20);break;
    case 100:status=DM_LAUNCH(100);break;
    case 500:status=DM_LAUNCH(500);break;
    default:return ffi::Error::InvalidArgument("CUDA EM supports N=4,10,20,100,500");
  }
  #undef DM_LAUNCH
  #undef DM_LAUNCH_IMPL
  return status==cudaSuccess?ffi::Error::Success():ffi::Error::Internal(cudaGetErrorString(status));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(dm_likelihood_em,fit,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::BufferR3<ffi::F32>>().Arg<ffi::BufferR4<ffi::F32>>()
    .Ret<ffi::BufferR3<ffi::F32>>().Ret<ffi::BufferR2<ffi::F32>>().Ret<ffi::BufferR1<ffi::S32>>()
    .Attr<int64_t>("first_cap").Attr<int64_t>("max_iter").Attr<int64_t>("occupancy")
    .Attr<int64_t>("accelerate"));
