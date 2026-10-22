#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
template<class T> std::vector<T> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);auto n=f.tellg();if(n%sizeof(T))throw std::runtime_error("length");std::vector<T>a(n/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(a.data()),n);return a;}
template<class T> void write(const std::string&p,const std::vector<T>&a){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));if(!f)throw std::runtime_error(p);}
uint32_t xs(uint32_t x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
int main(int argc,char**argv){
 if(argc!=12){std::cerr<<"scores values cdf seeds counts sums rows rank L burn seed_out\n";return 2;}
 auto score=read<int16_t>(argv[1]),value=read<int16_t>(argv[2]);auto cdf=read<uint16_t>(argv[3]);auto seed=read<uint32_t>(argv[4]);
 int rows=std::stoi(argv[7]),rank=std::stoi(argv[8]),L=std::stoi(argv[9]),burn=std::stoi(argv[10]);
 if(score.size()<size_t(rows*16)||value.size()<size_t(rows*256)||seed.size()!=32)throw std::runtime_error("input shapes");
 std::array<int,16>offset;int total=0;for(int j=0;j<16;++j){offset[j]=total;total+=16*(j?rank:1)*(j==15?16:rank);}if(cdf.size()!=size_t(total))throw std::runtime_error("cdf shape");
 std::vector<uint16_t> counts(rows*16);std::vector<int32_t>sums(rows*16);
 for(int row=0;row<rows;++row){
  std::array<int,16>z,order,threshold,state;int sum=0;
  for(int j=0;j<16;++j){z[j]=score[row*16+j];order[j]=j;state[j]=7;sum+=z[j];}
  int vv=sum+8,mean=vv>=0?vv/16:-((-vv+15)/16);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return z[a]>z[b];});
  for(int j=0;j<16;++j){int c=z[order[j]]-mean;threshold[j]=c<=-4096?0:c>=4096?2048:(c+4098)/4;}
  for(int t=0;t<burn+L;++t){
   for(int j=0;j<16;++j){bool up=(seed[j]>>21)<uint32_t(threshold[j]);seed[j]=xs(seed[j]);state[j]=std::min(15,std::max(0,state[j]+(up?1:-1)));}
   if(t<burn)continue;int bond=0;
   for(int j=0;j<16;++j){uint32_t u=seed[16+j]>>21;seed[16+j]=xs(seed[16+j]);int right=j==15?16:rank,left=j?rank:1;int base=offset[j]+(state[j]*left+bond)*right;int k=0;while(k<right&&u>=cdf[base+k])++k;if(k==right)throw std::runtime_error("cdf terminal");bond=k;}
   int ix=order[bond];counts[row*16+ix]++;
   for(int d=0;d<16;++d)sums[row*16+d]+=value[(row*16+ix)*16+d];
  }
 }
 write(argv[5],counts);write(argv[6],sums);write(argv[11],seed);
}
