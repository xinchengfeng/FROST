module dut(input clk,rst,load,input [3:0] load_index,input signed [15:0] score_word,
 input value_load,input [7:0] value_index,input signed [15:0] value_word,
 input start,input [10:0] stream_length,input seed_write,input [4:0] seed_index,input [31:0] seed_word,
 output busy,out_valid,output [3:0] out_index,output [31:0] out_word,output done);
fx16_fused core(clk,rst,load,load_index,score_word,value_load,value_index,value_word,start,stream_length,seed_write,seed_index,seed_word,busy,out_valid,out_index,out_word,done);endmodule
