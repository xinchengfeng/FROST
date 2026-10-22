// Test all quantitative outputs through the public RTL ports.
#include "Vdut.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static std::vector<uint32_t> read_hex(const std::string &path){
 std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open "+path);
 std::vector<uint32_t> v;std::string s;
 while(f>>s)v.push_back(static_cast<uint32_t>(std::stoul(s,nullptr,16)));
 return v;
}
int main(int argc,char **argv){
 if(argc<6){std::cerr<<"Usage: SIM scores.hex rng.hex output.csv L rows [trace.vcd]\n";return 2;}
 Verilated::commandArgs(argc,argv);
 auto scores=read_hex(argv[1]);auto seeds=read_hex(argv[2]);auto values=read_hex("mem/values.hex");
 int L=std::stoi(argv[4]),rows=std::stoi(argv[5]);
 if(scores.size()<static_cast<size_t>(16*rows)||seeds.size()!=32)return 3;
 Vdut top;
 VerilatedVcdC *trace=nullptr;
 if(argc>6){Verilated::traceEverOn(true);trace=new VerilatedVcdC;top.trace(trace,99);trace->set_time_unit("1ps");trace->set_time_resolution("1ps");trace->open(argv[6]);}
 uint64_t cycle=0;
 uint64_t clock_mhz=15;
 if(const char *clock_env=std::getenv("FROST_SIM_CLOCK_MHZ"))clock_mhz=std::stoul(clock_env);
 if(clock_mhz==0||clock_mhz>1000)throw std::runtime_error("Invalid simulation clock");
 auto tick=[&](){
   // Integer-MHz clock at 1-ps resolution; default preserves old 15-MHz runs.
   // Stage 5 uses 8 MHz, a common point meeting every mapped design's timing.
   top.clk=0;top.eval();if(trace)trace->dump(cycle*1000000/clock_mhz);
   top.clk=1;top.eval();if(trace)trace->dump((2*cycle+1)*500000/clock_mhz);++cycle;
 };
 top.rst=1;top.value_load=0;top.value_index=0;top.value_word=0;top.load=0;top.start=0;top.seed_write=0;top.stream_length=L;
 tick();tick();top.rst=0;tick();
 for(int i=0;i<32;++i){top.seed_write=1;top.seed_index=i;top.seed_word=seeds[i];tick();}
 top.seed_write=0;
 std::string temporary=std::string(argv[3])+".tmp";
 std::ofstream out(temporary);if(!out)return 4;
 out<<"row,cycles";for(int j=0;j<16;++j)out<<",word"<<j;out<<"\n";
 for(int row=0;row<rows;++row){
   uint64_t begin=cycle+1;
   if(row%16==0){for(int i=0;i<256;++i){top.value_load=1;top.value_index=i;top.value_word=values[(row/16)*256+i];tick();}top.value_load=0;}
   for(int i=0;i<16;++i){top.load=1;top.load_index=i;top.score_word=scores[16*row+i];tick();}
   top.load=0;top.start=1;tick();top.start=0;
   std::array<uint32_t,16> words{};std::array<bool,16> seen{};
   while(!top.done){
     tick();
     if(top.out_valid){
       if(seen[top.out_index])throw std::runtime_error("Duplicate output index");
       seen[top.out_index]=true;words[top.out_index]=top.out_word;
     }
     if(cycle-begin>100000)throw std::runtime_error("Row timeout");
   }
   for(bool s:seen)if(!s)throw std::runtime_error("Missing output component");
   out<<row<<","<<(cycle-begin+1);
   for(auto w:words)out<<","<<w;out<<"\n";
 }
 out.close();if(!out)throw std::runtime_error("Output stream failed");
 if(std::rename(temporary.c_str(),argv[3])!=0)throw std::runtime_error("Atomic output rename failed");
 top.final();if(trace){top.clk=0;top.eval();trace->dump(cycle*1000000/clock_mhz);trace->close();delete trace;}
 std::cerr<<"Verified port transport for "<<rows<<" rows; "<<cycle<<" simulation clocks\n";
}
