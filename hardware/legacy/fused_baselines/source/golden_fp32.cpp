#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
template<class T>std::vector<T> read(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto n=f.tellg();if(n<0)throw std::runtime_error(p);std::vector<T> v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);return v;}
int main(int argc,char**argv){if(argc!=5)return 2;auto s=read<int16_t>(argv[1]),v=read<int16_t>(argv[2]);auto lut=read<float>(argv[3]);std::ofstream f(argv[4],std::ios::binary);for(size_t r=0;r<s.size()/16;r++){int m=-32768;for(int k=0;k<16;k++)m=std::max(m,(int)s[r*16+k]);float e[16],total=0;for(int k=0;k<16;k++){int ix=std::min(1024,(m-s[r*16+k]+8)>>4);e[k]=lut[ix];total+=e[k];}float reciprocal=1.0f/total;for(int d=0;d<16;d++){float acc=0;for(int k=0;k<16;k++)acc=std::fma(e[k],(float)v[(r*16+k)*16+d],acc);int32_t out=(int32_t)std::nearbyint(acc*reciprocal);f.write((char*)&out,4);}}}
