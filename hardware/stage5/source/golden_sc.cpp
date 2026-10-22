// Independent functional model: no copied RTL state machine or divider.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
static std::vector<uint32_t> readhex(const std::string& path){
 std::ifstream f(path);if(!f)throw std::runtime_error(path);
 std::vector<uint32_t> a;std::string s;while(f>>s)a.push_back(std::stoul(s,nullptr,16));return a;
}
static uint32_t next(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
int main(int argc,char** argv){
 if(argc!=8){std::cerr<<"MODE BURN scores.hex values.hex seed.hex rows output.csv\n";return 2;}
 const int mode=std::stoi(argv[1]),burn=std::stoi(argv[2]),rows=std::stoi(argv[6]);
 const auto s=readhex(argv[3]),v=readhex(argv[4]);auto rng=readhex(argv[5]);
 if(s.size()<size_t(rows*16)||v.size()<size_t((rows+15)/16*256)||rng.size()!=32)return 3;
 const unsigned coeff[9]={65536,511,6938,1,5,692,1,15,22};
 const unsigned hc[5]={65536,32768,21846,16384,13108};
 std::ofstream out(argv[7]);out<<"row,cycles";
 for(int j=0;j<16;j++)out<<",word"<<j;
 for(int j=0;j<16;j++)out<<",count"<<j;
 out<<",denominator\n";
 for(int row=0;row<rows;row++){
  int m=-32768;for(int k=0;k<16;k++)m=std::max(m,int(int16_t(s[row*16+k])));
  std::array<int,16> c{};
  for(int k=0;k<16;k++){
   int gap=std::min(8192,m-int(int16_t(s[row*16+k]))),state=7;
   uint32_t &rx=rng[2*k],&rc=rng[2*k+1];
   if(mode==0){
    unsigned th=std::min(65536,32768+gap*8);
    for(int t=0;t<burn+64;t++){
     state=std::clamp(state+((next(rx)>>16)<th ? 1 : -1),0,15);
     if(t>=burn)c[k]+=gap==0 || state<14;
    }
   }else if(mode==1){
    for(int t=0;t<64;t++){
     int pop=0;for(int j=0;j<8;j++)pop+=(next(rx)>>16)<unsigned(gap*8);
     c[k]+=(next(rc)>>16)<coeff[pop];
    }
   }else{
    for(int t=0;t<64;t++){
     std::array<bool,8> h;h.fill(true);
     for(int d=4;d>=0;d--)for(int f=0;f<8;f++){
      bool x=(next(rx)>>16)<unsigned(gap*8),a=(next(rc)>>16)<hc[d];
      h[f]=!(x&&a&&h[f]);
     }
     c[k]+=std::all_of(h.begin(),h.end(),[](bool b){return b;});
    }
   }
  }
  int den=0;for(auto x:c)den+=x;
  if(den<64 || den>1024)throw std::runtime_error("denominator outside guaranteed range");
  const int64_t inv=(int64_t(1)<<30)/den;
  int run=mode==0?burn+64:mode==1?64:320;
  out<<row<<","<<(162+run+(row%16==0?256:0));
  for(int d=0;d<16;d++){
   int64_t sum=0;
   for(int k=0;k<16;k++)sum+=int64_t(c[k])*int16_t(v[(row/16)*256+k*16+d]);
   int64_t y=(sum*inv+(int64_t(1)<<29))>>30;
   out<<","<<uint32_t(y);
  }
  for(auto x:c)out<<","<<x;
  out<<","<<den<<"\n";
 }
 return 0;
}
