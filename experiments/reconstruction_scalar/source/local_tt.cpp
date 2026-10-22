#include <cstdint>
#include <algorithm>
extern "C" void local_tt(const int32_t *low,const uint64_t *frac,const uint64_t *cdf,const int32_t *sizes,const uint32_t *net,const uint32_t *shift,int32_t *counts,int rows,int L,int states,int pad){
 for(int row=0;row<rows;row++)for(int t=0;t<L;t++){
  int bond=0;
  for(int j=0;j<16;j++){
   int s=low[row*16+j]+(uint64_t(net[t*32+j]^shift[row*32+j])<frac[row*16+j]);
   uint64_t u=net[t*32+16+j]^shift[row*32+16+j];
   const uint64_t *c=cdf+((int64_t(j)*states+s)*pad+bond)*pad;
   int lo=0,hi=sizes[j];
   while(lo<hi){int mid=(lo+hi)/2;if(c[mid]<=u)lo=mid+1;else hi=mid;}
   bond=std::min(lo,sizes[j]-1);
  }
  counts[row*16+bond]++;
 }
}
extern "C" void bern_qmc(const uint64_t *threshold,const uint64_t *coef,const uint32_t *net,const uint32_t *shift,int32_t *counts,int size,int L,int degree){
 for(int s=0;s<size;s++){
  int c=0;
  for(int t=0;t<L;t++){
   int k=0;
   for(int j=0;j<degree;j++)k+=(uint64_t(net[t*(degree+1)+j]^shift[s*(degree+1)+j])<threshold[s]);
   c+=(uint64_t(net[t*(degree+1)+degree]^shift[s*(degree+1)+degree])<coef[k]);
  }
  counts[s]=c;
 }
}
#include <vector>
static inline uint32_t rngstep(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static inline int split_count(int m,uint64_t threshold,uint32_t &rng){
 if(m==0)return 0;uint64_t phase=rngstep(rng);int count=0;
 for(int t=0;t<m;t++){uint64_t u=((uint64_t(t)<<32)+phase)/m;count+=(u<threshold);}
 return count;
}
extern "C" void strat_tt(const int32_t *low,const uint64_t *frac,const uint64_t *cdf,const int32_t *sizes,const uint32_t *seeds,int32_t *counts,int rows,int L,int states,int pad){
 for(int row=0;row<rows;row++){
  uint32_t rng=seeds[row];std::vector<int> current(pad,0),next(pad,0);current[0]=L;
  for(int j=0;j<16;j++){
   std::fill(next.begin(),next.end(),0);int left=j? sizes[j-1]:1;
   for(int bond=0;bond<left;bond++){
    int m=current[bond];int high=split_count(m,frac[row*16+j],rng);
    for(int side=0;side<2;side++){
     int num=side?high:m-high;if(!num)continue;int state=low[row*16+j]+side;
     const uint64_t *c=cdf+((int64_t(j)*states+state)*pad+bond)*pad;uint64_t phase=rngstep(rng);
     for(int t=0;t<num;t++){
      uint64_t u=((uint64_t(t)<<32)+phase)/num;int lo=0,hi=sizes[j];
      while(lo<hi){int mid=(lo+hi)/2;if(c[mid]<=u)lo=mid+1;else hi=mid;}
      next[std::min(lo,sizes[j]-1)]++;
     }
    }
   }
   current.swap(next);
  }
  for(int k=0;k<16;k++)counts[row*16+k]=current[k];
 }
}
extern "C" void strat_bern(const uint64_t *threshold,const uint64_t *coef,const uint32_t *seeds,int32_t *counts,int size,int L,int degree){
 for(int s=0;s<size;s++){
  uint32_t rng=seeds[s];std::vector<int> current(degree+1,0),next(degree+1,0);current[0]=L;
  for(int j=0;j<degree;j++){
   std::fill(next.begin(),next.end(),0);
   for(int k=0;k<=j;k++){int m=current[k],hi=split_count(m,threshold[s],rng);next[k]+=m-hi;next[k+1]+=hi;}
   current.swap(next);
  }
  int total=0;for(int k=0;k<=degree;k++)total+=split_count(current[k],coef[k],rng);counts[s]=total;
 }
}
extern "C" void antithetic_tt(const int32_t *low,const uint64_t *frac,const uint64_t *cdf,const int32_t *sizes,const uint32_t *phases,int32_t *counts,int rows,int L,int states,int pad){
 for(int row=0;row<rows;row++)for(int replica=0;replica<2;replica++){
  std::vector<int> current(pad,0),next(pad,0);current[0]=L/2;
  for(int j=0;j<16;j++){
   std::fill(next.begin(),next.end(),0);int left=j?sizes[j-1]:1;
   for(int bond=0;bond<left;bond++){
    const uint32_t *pp=phases+((row*16+j)*pad+bond)*3;
    uint64_t phase=replica?(~pp[0]):pp[0];int m=current[bond],high=0;
    for(int t=0;t<m;t++)high+=((((uint64_t(t)<<32)+phase)/m)<frac[row*16+j]);
    for(int side=0;side<2;side++){
     int num=side?high:m-high;if(!num)continue;int state=low[row*16+j]+side;
     const uint64_t *c=cdf+((int64_t(j)*states+state)*pad+bond)*pad;
     phase=replica?(~pp[side+1]):pp[side+1];
     for(int t=0;t<num;t++){
      uint64_t u=((uint64_t(t)<<32)+phase)/num;int lo=0,hi=sizes[j];
      while(lo<hi){int mid=(lo+hi)/2;if(c[mid]<=u)lo=mid+1;else hi=mid;}
      next[std::min(lo,sizes[j]-1)]++;
     }
    }
   }
   current.swap(next);
  }
  for(int k=0;k<16;k++)counts[row*16+k]+=current[k];
 }
}
