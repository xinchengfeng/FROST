// Rank-2 categorical pipeline; terminal CDF entries (one) are implicit.
module tt_r2 #(parameter FIRST=0, LAST=0, STAGE=0, FILE="mem/tt00.hex")(
 input clk,rst,valid_in,input [3:0] state_in,input latent_in,
 input seed_write,input [31:0] seed_word,
 output reg valid_out,output reg [3:0] category);
 localparam W=LAST ? 120 : 8;
 reg [W-1:0] rom[0:(FIRST ? 16 : 32)-1];
 initial $readmemh(FILE,rom);
 wire [W-1:0] cdf=rom[FIRST ? {1'b0,state_in} : {state_in,latent_in}];
 reg [31:0] rng; reg [3:0] next_cat;integer i;
 function [31:0] xs;input [31:0] a;reg [31:0] b;
 begin b=a^(a<<13);b=b^(b>>17);xs=b^(b<<5);end endfunction
 generate if(LAST)begin:terminal_decode
  wire [14:0] ge;
  genvar t;
  for(t=0;t<15;t=t+1)begin:cmp
   assign ge[t]=({1'b0,rng[31:25]}>=cdf[t*8+:8]);
  end
  // The fixed, monotone CDF makes these disjoint intervals exhaustive.
  always @*begin
   next_cat[3]=ge[7];
   next_cat[2]=(ge[3]&!ge[7])|ge[11];
   next_cat[1]=(ge[1]&!ge[3])|(ge[5]&!ge[7])|(ge[9]&!ge[11])|ge[13];
   next_cat[0]=(ge[0]&!ge[1])|(ge[2]&!ge[3])|(ge[4]&!ge[5])|(ge[6]&!ge[7])|
               (ge[8]&!ge[9])|(ge[10]&!ge[11])|(ge[12]&!ge[13])|ge[14];
  end
 end else begin:latent_decode
  always @*next_cat={3'b0,({1'b0,rng[31:25]}>=cdf[7:0])};
 end endgenerate
 always @(posedge clk)begin
  if(rst)begin rng<=STAGE+17;valid_out<=0;category<=0;end
  else begin
   valid_out<=valid_in;
   if(seed_write)rng<=seed_word==0 ? 1 : seed_word;
   else if(valid_in)begin rng<=xs(rng);category<=next_cat;end
  end
 end
endmodule

module frost_perm_therm(
 input clk,rst,load,input [3:0] load_index,input signed [15:0] score_word,
 input value_load,input [7:0] value_index,input signed [15:0] value_word,
 input start,input [10:0] stream_length,input seed_write,input [4:0] seed_index,input [31:0] seed_word,
 output busy,output reg out_valid,output reg [3:0] out_index,output reg [31:0] out_word,output reg done);
 wire front_done;wire [191:0] threshold;wire [63:0] permutation;
 frost_frontend_perm front(clk,rst,load,load_index,score_word,start,front_done,threshold,permutation);
 reg [2:0] phase;reg [3:0] state[0:15],warm,output_pos;
 reg [31:0] drive_rng[0:15];reg [63:0] next_states;
 reg [5:0] launched;
 reg [14:0] update_delay;
 wire [15:0] update_en={update_delay,(phase==2 || phase==3)};
 always @(posedge clk)if(rst)update_delay<=0;else update_delay<={update_delay[13:0],(phase==2 || phase==3)};
 wire [15:0] pv;wire [63:0] pc;wire [3:0] pick=permutation[pc[63:60]*4+:4];
 reg signed [21:0] accum[0:15];
 integer j;
 function [31:0] xs;input [31:0] a;reg [31:0] b;
 begin b=a^(a<<13);b=b^(b>>17);xs=b^(b<<5);end endfunction
 always @*begin
  for(j=0;j<16;j=j+1)begin
   next_states[j*4+:4]=state[j];
   if({1'b0,drive_rng[j][31:21]}<threshold[j*12+:12])begin
    if(state[j]<15)next_states[j*4+:4]=state[j]+1;
   end else if(state[j]>0)next_states[j*4+:4]=state[j]-1;
  end
 end
 assign busy=phase!=0;
 wire signed [21:0] rounded=($signed(accum[output_pos])+22'sd32)>>>6;
 always @(posedge clk)begin
  if(rst)begin phase<=0;warm<=0;launched<=0;
   out_valid<=0;out_index<=0;out_word<=0;done<=0;output_pos<=0;
   for(j=0;j<16;j=j+1)begin state[j]<=7;drive_rng[j]<=j+1;end
  end else begin
   out_valid<=0;done<=0;
   if(seed_write && phase==0 && seed_index<16)drive_rng[seed_index]<=seed_word==0 ? 1 : seed_word;
   for(j=0;j<16;j=j+1)if(update_en[j])begin state[j]<=next_states[j*4+:4];drive_rng[j]<=xs(drive_rng[j]);end
   // Samples form one contiguous valid burst; its trailing edge marks completion.
   if(pv[15] && !pv[14])begin phase<=5;output_pos<=0;end
   case(phase)
    0:if(start)begin phase<=1;
      launched<=0;for(j=0;j<16;j=j+1)state[j]<=7;end
    1:if(front_done)begin warm<=0;phase<=2;end
    2:if(warm==15)phase<=3;else warm<=warm+1;
    3:begin launched<=launched+1;if(launched==6'd63)phase<=4;end
    5:begin out_valid<=1;out_index<=output_pos;out_word<=rounded;
      if(output_pos==15)begin phase<=0;done<=1;end else output_pos<=output_pos+1;end
   endcase
  end
 end
 genvar d;
 generate for(d=0;d<16;d=d+1)begin:values
  (* ram_style="distributed" *)reg signed [15:0] mem[0:15];
  always @(posedge clk)begin
   if(value_load && value_index[3:0]==d)mem[value_index[7:4]]<=value_word;
   if(rst || start)accum[d]<=0;
   else if(pv[15])accum[d]<=accum[d]+$signed(mem[pick]);
  end
 end endgenerate
 // Each stage consumes its local FSM state; only update enables are delayed.
 `include "pipeline_retime.vh"
endmodule
