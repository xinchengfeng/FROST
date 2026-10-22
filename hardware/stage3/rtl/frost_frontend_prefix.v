// Hardware score loading, integer centering, stable descending insertion sort,
// bounded 11-fraction-bit encoding and token permutation. No real arithmetic.
module frost_frontend_prefix(
 input clk, input rst, input load, input [3:0] load_index,
 input signed [15:0] score_word, input start,
 output reg done, output reg [191:0] thresholds, output reg [63:0] permutation);
 reg signed [15:0] scores[0:15];
 reg [3:0] order[0:15];
 reg signed [20:0] sum;
 reg signed [15:0] mean;
 reg [1:0] phase;
 reg [3:0] enc;
 wire [15:0] shift_prefix;
 genvar g;
 generate for(g=0;g<16;g=g+1)begin:prefix
  if(g==0)assign shift_prefix[g]=(g<load_index && score_word>scores[g]);
  else assign shift_prefix[g]=shift_prefix[g-1] | (g<load_index && score_word>scores[g]);
 end endgenerate
 wire signed [16:0] centred = $signed(scores[enc])-$signed(mean);
 wire [11:0] threshold = centred <= -17'sd4096 ? 12'd0 :
                        centred >= 17'sd4096 ? 12'd2048 :
                        ((centred+17'sd4098) >>> 2);
 integer i;
 always @(posedge clk) begin
   if(rst) begin phase<=0;done<=0;sum<=0;thresholds<=0;permutation<=0;
     enc<=0;mean<=0;
     for(i=0;i<16;i=i+1)begin scores[i]<=0;order[i]<=i;end
   end else begin
     done<=0;
     if(load && phase==0)begin
       for(i=0;i<16;i=i+1)begin
         if(i<=load_index)begin
           if(i==0)begin
             if(load_index==0 || shift_prefix[i])begin scores[i]<=score_word;order[i]<=load_index;end
           end else begin
             if(shift_prefix[i-1])begin scores[i]<=scores[i-1];order[i]<=order[i-1];end
             else if(i==load_index || shift_prefix[i])begin scores[i]<=score_word;order[i]<=load_index;end
           end
         end
       end
       if(load_index==0)sum<=score_word;else sum<=sum+score_word;
     end
     case(phase)
       0:if(start)begin mean<=(sum+21'sd8)>>>4;enc<=0;phase<=2;end
       2:begin
         // Decode the destination once. Static slices avoid synthesizing a
         // wide variable write-mask and a multiplier for the address 12*enc.
         for(i=0;i<16;i=i+1)if(enc==i)begin
           thresholds[i*12+:12]<=threshold;
           permutation[i*4+:4]<=order[i];
         end
         if(enc==15)begin phase<=0;done<=1;end else enc<=enc+1;
       end
     endcase
   end
 end
endmodule
