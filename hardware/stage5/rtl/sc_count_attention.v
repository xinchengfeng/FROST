// Literature-derived SC exponentiation with a shared integer count-weighted AV.
// MODE 0: 16-state SExp, gain 2; MODE 1: degree-8 Bernstein;
// MODE 2: eight fifth-order NAND-Horner factors, five clocks/output bit.
// L is fixed at 64, matching the optimized FROST kernel. Not a full accelerator.
module sc_count_attention #(parameter MODE=0, BURN=256)(
 input clk,rst,load,input [3:0] load_index,input signed [15:0] score_word,
 input value_load,input [7:0] value_index,input signed [15:0] value_word,
 input start,input [10:0] stream_length,input seed_write,input [4:0] seed_index,input [31:0] seed_word,
 output busy,output reg out_valid,output reg [3:0] out_index,output reg [31:0] out_word,output reg done);
 localparam RUN_CYCLES=MODE==0 ? BURN+64 : MODE==1 ? 64 : 320;
 reg [255:0] scores;
 reg signed [15:0] maximum;
 reg [13:0] gap[0:15];
 wire [6:0] counts[0:15];
 reg [3:0] phase,pos,key;
 reg [1:0] group_pos;
 reg [8:0] run_count;
 reg [2:0] horner_step;
 reg [10:0] total;
 reg [11:0] remainder;
 reg [31:0] numerator,quotient;
 reg [24:0] reciprocal;
 reg [4:0] bit_count;
 reg signed [25:0] accum[0:15];
 wire signed [16:0] delta=$signed(maximum)-$signed(scores[255:240]);
 wire [12:0] trial={remainder,numerator[31]};
 wire take=trial>={2'b0,total};
 wire [31:0] next_quotient={quotient[30:0],take};
 wire signed [51:0] scaled=accum[pos]*$signed({1'b0,reciprocal});
 wire signed [51:0] rounded=(scaled+52'sd536870912)>>>30;
 wire [16:0] horner_coefficient=horner_step==4 ? 17'd13108 :
        horner_step==3 ? 17'd16384 : horner_step==2 ? 17'd21846 :
        horner_step==1 ? 17'd32768 : 17'd65536;
 assign busy=phase!=0;
 integer i;
 always @(posedge clk)begin
  if(rst)begin
   phase<=0;pos<=0;key<=0;group_pos<=0;scores<=0;maximum<=0;
   run_count<=0;horner_step<=4;total<=0;remainder<=0;numerator<=0;
   quotient<=0;reciprocal<=0;bit_count<=0;
   out_valid<=0;out_index<=0;out_word<=0;done<=0;
   for(i=0;i<16;i=i+1)gap[i]<=0;
  end else begin
   out_valid<=0;done<=0;
   if(load && phase==0)begin
    scores<={scores[239:0],score_word};
    if(load_index==0 || score_word>maximum)maximum<=score_word;
   end
   case(phase)
    0:if(start)begin phase<=1;pos<=0;run_count<=0;horner_step<=4;total<=0;end
    1:begin
     gap[pos]<=delta>=8192 ? 14'd8192 : delta[13:0];
     scores<={scores[239:0],scores[255:240]};
     if(pos==15)begin phase<=2;pos<=0;end else pos<=pos+1;
    end
    2:begin
     run_count<=run_count+1;
     if(MODE==2)horner_step<=horner_step==0 ? 4 : horner_step-1;
     if(run_count==RUN_CYCLES-1)begin phase<=3;pos<=0;end
    end
    3:begin total<=total+counts[pos];if(pos==15)phase<=4;else pos<=pos+1;end
    4:begin remainder<=0;quotient<=0;numerator<=32'h40000000;bit_count<=31;phase<=5;end
    5:begin
     // Restoring division processes 32 bits of 2^30, including its leading zero.
     remainder<=take ? trial-total : trial;
     quotient<=next_quotient;numerator<=numerator<<1;
     if(bit_count==0)begin reciprocal<=next_quotient[24:0];key<=0;group_pos<=0;phase<=6;end
     else bit_count<=bit_count-1;
    end
    6:if(group_pos==3)begin
      group_pos<=0;if(key==15)begin phase<=7;pos<=0;end else key<=key+1;
     end else group_pos<=group_pos+1;
    7:begin
     out_valid<=1;out_index<=pos;out_word<=rounded;
     if(pos==15)begin phase<=0;done<=1;end else pos<=pos+1;
    end
   endcase
  end
 end

 function [31:0] xs;input [31:0] a;reg [31:0] b;
 begin b=a^(a<<13);b=b^(b>>17);xs=b^(b<<5);end endfunction
 function [16:0] bern_coefficient;input [3:0] k;
 begin
  case(k)
   `include "bernstein_coeff_case.vh"
   default:bern_coefficient=0;
  endcase
 end endfunction

 genvar g,h;
 generate for(g=0;g<16;g=g+1)begin:scalar
  reg [31:0] rx,rc;
  reg [6:0] count;
  reg [3:0] fsm;
  reg [7:0] horner;
  wire [16:0] x_threshold={gap[g],3'b000};
  wire [16:0] fsm_threshold=gap[g]>=4096 ? 17'd65536 : 17'd32768+{gap[g],3'b000};
  wire [31:0] nx[0:8] /* verilator split_var */;
  wire [31:0] nc[0:8] /* verilator split_var */;
  wire [7:0] xbits,cbits,hnext;
  assign nx[0]=rx;assign nc[0]=rc;
  for(h=0;h<8;h=h+1)begin:draw
   assign nx[h+1]=xs(nx[h]);assign nc[h+1]=xs(nc[h]);
   assign xbits[h]={1'b0,nx[h+1][31:16]}<x_threshold;
   assign cbits[h]={1'b0,nc[h+1][31:16]}<horner_coefficient;
   assign hnext[h]=~(xbits[h]&cbits[h]&horner[h]);
  end
  wire [3:0] pop={3'b0,xbits[0]}+{3'b0,xbits[1]}+{3'b0,xbits[2]}+{3'b0,xbits[3]}+
                 {3'b0,xbits[4]}+{3'b0,xbits[5]}+{3'b0,xbits[6]}+{3'b0,xbits[7]};
  wire up={1'b0,nx[1][31:16]}<fsm_threshold;
  wire [3:0] fnext=up ? (fsm==15 ? fsm : fsm+1'b1) : (fsm==0 ? fsm : fsm-1'b1);
  assign counts[g]=count;
  always @(posedge clk)begin
   if(rst)begin rx<=1;rc<=1;count<=0;fsm<=7;horner<=8'hff;end
   else begin
    if(seed_write && seed_index==2*g)rx<=seed_word;
    if(seed_write && seed_index==2*g+1)rc<=seed_word;
    if(start)begin count<=0;fsm<=7;horner<=8'hff;end
    else if(phase==2)begin
     if(MODE==0)begin
      rx<=nx[1];fsm<=fnext;
      if(run_count>=BURN && (gap[g]==0 || fnext<14))count<=count+1'b1;
     end else if(MODE==1)begin
      rx<=nx[8];rc<=nc[1];
      if({1'b0,nc[1][31:16]}<bern_coefficient(pop))count<=count+1'b1;
     end else begin
      rx<=nx[8];rc<=nc[8];
      horner<=horner_step==0 ? 8'hff : hnext;
      if(horner_step==0 && (&hnext))count<=count+1'b1;
     end
    end
   end
  end
 end endgenerate

 generate for(g=0;g<4;g=g+1)begin:lane
  (* ram_style="distributed" *)reg signed [15:0] values[0:63];
  wire signed [15:0] value=values[{key,group_pos}];
  wire signed [23:0] product=$signed({1'b0,counts[key]})*value;
  integer d;
  always @(posedge clk)begin
   if(value_load && value_index[1:0]==g)values[value_index[7:2]]<=value_word;
   for(d=g;d<16;d=d+4)begin
    if(rst || start)accum[d]<=0;
    else if(phase==6 && group_pos==d/4)accum[d]<=accum[d]+product;
   end
  end
 end endgenerate
endmodule
