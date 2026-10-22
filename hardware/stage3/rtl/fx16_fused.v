// Four reused integer multipliers; one restoring reciprocal per row.
module fx16_fused(
 input clk,rst,load,input [3:0] load_index,input signed [15:0] score_word,
 input value_load,input [7:0] value_index,input signed [15:0] value_word,
 input start,input [10:0] stream_length,input seed_write,input [4:0] seed_index,input [31:0] seed_word,
 output busy,output reg out_valid,output reg [3:0] out_index,output reg [31:0] out_word,output reg done);
 (* ram_style="block" *)reg [16:0] exp_rom[0:1024];
 initial $readmemh("mem/exp_fx16.hex",exp_rom);
 reg [255:0] scores;reg signed [15:0] maximum;reg [16:0] exp_data,exps[0:15];
 reg signed [37:0] accum[0:15];reg [20:0] total;reg [21:0] remainder;
 reg [31:0] numerator,quotient;reg [15:0] reciprocal;reg [5:0] bit_count;
 reg [3:0] phase,pos,key;reg [1:0] group_pos;
 wire signed [16:0] delta=$signed(maximum)-$signed(scores[255:240]);
 wire [16:0] idx_unclip=(delta+17'd8)>>4;
 wire [10:0] idx=idx_unclip>=1025 ? 1024 : idx_unclip;
 wire [22:0] trial={remainder,numerator[31]};wire take=trial>={2'b0,total};
 wire [31:0] next_quotient={quotient[30:0],take};
 wire signed [54:0] scaled=accum[pos]*$signed({1'b0,reciprocal});
 wire signed [54:0] rounded=(scaled+55'sd1073741824)>>>31;
 assign busy=phase!=0;integer i;
 always @(posedge clk)begin
  if(phase==1)exp_data<=exp_rom[idx];
  if(rst)begin phase<=0;pos<=0;key<=0;group_pos<=0;scores<=0;maximum<=0;total<=0;
   remainder<=0;numerator<=0;quotient<=0;reciprocal<=0;bit_count<=0;
   out_valid<=0;out_index<=0;out_word<=0;done<=0;for(i=0;i<16;i=i+1)exps[i]<=0;
  end else begin
   out_valid<=0;done<=0;
   if(load && phase==0)begin scores<={scores[239:0],score_word};if(load_index==0 || score_word>maximum)maximum<=score_word;end
   case(phase)
    0:if(start)begin phase<=1;pos<=0;total<=0;end
    1:phase<=2;
    2:begin total<=total+exp_data;exps[pos]<=exp_data;scores<={scores[239:0],scores[255:240]};
      if(pos==15)phase<=3;else begin pos<=pos+1;phase<=1;end end
    3:begin remainder<=0;quotient<=0;numerator<=32'h80000000;bit_count<=31;phase<=4;end
    4:begin remainder<=take ? trial-total : trial;quotient<=next_quotient;numerator<=numerator<<1;
      if(bit_count==0)begin reciprocal<=next_quotient[15:0];key<=0;group_pos<=0;phase<=5;end else bit_count<=bit_count-1;end
    5:if(group_pos==3)begin group_pos<=0;if(key==15)begin phase<=6;pos<=0;end else key<=key+1;end else group_pos<=group_pos+1;
    6:begin out_valid<=1;out_index<=pos;out_word<=rounded;
      if(pos==15)begin phase<=0;done<=1;end else pos<=pos+1;end
   endcase
  end
 end
 genvar g;
 generate for(g=0;g<4;g=g+1)begin:lane
  (* ram_style="distributed" *)reg signed [15:0] values[0:63];
  wire signed [15:0] value=values[{key,group_pos}];
  wire signed [33:0] product=$signed({1'b0,exps[key]})*value;
  integer d;
  always @(posedge clk)begin
   if(value_load && value_index[1:0]==g)values[value_index[7:2]]<=value_word;
   for(d=g;d<16;d=d+4)begin
    if(rst || start)accum[d]<=0;
    else if(phase==5 && group_pos==d/4)accum[d]<=accum[d]+product;
   end
  end
 end endgenerate
endmodule
