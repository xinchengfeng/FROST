// LUT exponential, one reciprocal per query, LANES reused binary32 FMAs.
// Values enter as signed Q10; weighted averaging is performed in integer-unit
// binary32 and rounded back to signed Q10 at the declared public boundary.
module fp32_fused #(parameter LANES=1)(
 input clk,rst,load,input [3:0] load_index,input signed [15:0] score_word,
 input value_load,input [7:0] value_index,input signed [15:0] value_word,
 input start,input [10:0] stream_length,input seed_write,input [4:0] seed_index,input [31:0] seed_word,
 output busy,output reg out_valid,output reg [3:0] out_index,output reg [31:0] out_word,output reg done);
 localparam GROUPS=16/LANES;
 (* ram_style="block" *) reg [31:0] exp_rom[0:1024];
 initial $readmemh("mem/exp_fp32.hex",exp_rom);
 reg [255:0] scores;reg signed [15:0] maximum;
 reg [31:0] exp_data;reg [32:0] exps[0:15],accum[0:15];
 reg [31:0] outputs[0:15];reg [32:0] total,reciprocal;
 reg [3:0] phase,pos,key,group_pos;
 wire signed [16:0] delta=$signed(maximum)-$signed(scores[255:240]);
 wire [16:0] idx_unclip=(delta+17'd8)>>4;
 wire [10:0] idx=idx_unclip>=1025 ? 1024 : idx_unclip;
 wire [32:0] exp_rec,sum_result,one_rec,div_result;
 wire div_ready,div_valid,sqrt_out;wire [4:0] sum_flags,div_flags;
 fNToRecFN #(8,24) cv_exp(exp_data,exp_rec);
 fNToRecFN #(8,24) cv_one(32'h3f800000,one_rec);
 addRecFN #(8,24) adder(.control(1'b0),.subOp(1'b0),.a(total),.b(exp_rec),.roundingMode(3'b000),.out(sum_result),.exceptionFlags(sum_flags));
 divSqrtRecFN_small #(8,24,0) divider(.nReset(!rst),.clock(clk),.control(1'b0),.inReady(div_ready),.inValid(phase==3),.sqrtOp(1'b0),.a(one_rec),.b(total),.roundingMode(3'b000),.outValid(div_valid),.sqrtOpOut(sqrt_out),.out(div_result),.exceptionFlags(div_flags));
 assign busy=phase!=0;
 integer i;
 always @(posedge clk)begin
  if(phase==1)exp_data<=exp_rom[idx];
  if(rst)begin phase<=0;pos<=0;key<=0;group_pos<=0;maximum<=0;scores<=0;total<=0;reciprocal<=0;
   out_valid<=0;out_index<=0;out_word<=0;done<=0;
   for(i=0;i<16;i=i+1)exps[i]<=0;
  end else begin
   out_valid<=0;done<=0;
   if(load && phase==0)begin scores<={scores[239:0],score_word};if(load_index==0 || score_word>maximum)maximum<=score_word;end
   case(phase)
    0:if(start)begin phase<=1;pos<=0;total<=0;end
    1:phase<=2;
    2:begin total<=sum_result;exps[pos]<=exp_rec;scores<={scores[239:0],scores[255:240]};
      if(pos==15)begin phase<=3;pos<=0;end else begin pos<=pos+1;phase<=1;end end
    3:if(div_ready)phase<=4;
    4:if(div_valid)begin reciprocal<=div_result;phase<=5;key<=0;group_pos<=0;end
    5:if(group_pos==GROUPS-1)begin group_pos<=0;
      if(key==15)phase<=6;else key<=key+1;end else group_pos<=group_pos+1;
    6:if(group_pos==GROUPS-1)begin phase<=7;pos<=0;end else group_pos<=group_pos+1;
    7:begin out_valid<=1;out_index<=pos;out_word<=outputs[pos];
      if(pos==15)begin done<=1;phase<=0;end else pos<=pos+1;end
   endcase
  end
 end
 genvar g;
 generate for(g=0;g<LANES;g=g+1)begin:lane
  (* ram_style="distributed" *)reg signed [15:0] values[0:256/LANES-1];
  wire [3:0] channel=group_pos*LANES+g;
  wire signed [15:0] value=values[key*GROUPS+group_pos];
  wire [32:0] vrec,result;
  wire [4:0] cvflags,fmaflags;wire [2:0] intflags;
  wire [31:0] rounded;
  iNToRecFN #(16,8,24) cv_v(.control(1'b0),.signedIn(1'b1),.in(value),.roundingMode(3'b000),.out(vrec),.exceptionFlags(cvflags));
  mulAddRecFN #(8,24) fma(.control(1'b0),.op(2'b00),
    .a(phase==6 ? accum[channel] : exps[key]),.b(phase==6 ? reciprocal : vrec),
    .c(phase==6 ? 33'b0 : accum[channel]),.roundingMode(3'b000),.out(result),.exceptionFlags(fmaflags));
  recFNToIN #(8,24,32) cv_out(.control(1'b0),.in(result),.roundingMode(3'b000),.signedOut(1'b1),.out(rounded),.intExceptionFlags(intflags));
  integer d;
  always @(posedge clk)begin
   if(value_load && value_index%LANES==g)values[value_index/LANES]<=value_word;
   for(d=g;d<16;d=d+LANES)begin
    if(rst || start)begin accum[d]<=0;outputs[d]<=0;end
    else if(channel==d)begin
     if(phase==5)accum[d]<=result;
     if(phase==6)outputs[d]<=rounded;
    end
   end
  end
 end endgenerate
endmodule
