tt_r2 #(.FIRST(1),.LAST(0),.STAGE(0),.FILE("mem/tt00.hex")) s0(
.clk(clk),.rst(rst),.valid_in(phase==3),
.state_in(next_states[0+:4]),
.latent_in(1'b0),
.seed_write(seed_write && phase==0 && seed_index==16),.seed_word(seed_word),
.valid_out(pv[0]),.category(pc[0+:4]));
reg [3:0] delay1;
always @(posedge clk)if(rst)delay1<=0;else delay1<= {next_states[4+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(1),.FILE("mem/tt01.hex")) s1(
.clk(clk),.rst(rst),.valid_in(pv[0]),
.state_in(delay1[3-:4]),
.latent_in(pc[0]),
.seed_write(seed_write && phase==0 && seed_index==17),.seed_word(seed_word),
.valid_out(pv[1]),.category(pc[4+:4]));
reg [7:0] delay2;
always @(posedge clk)if(rst)delay2<=0;else delay2<= {delay2[3:0],next_states[8+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(2),.FILE("mem/tt02.hex")) s2(
.clk(clk),.rst(rst),.valid_in(pv[1]),
.state_in(delay2[7-:4]),
.latent_in(pc[4]),
.seed_write(seed_write && phase==0 && seed_index==18),.seed_word(seed_word),
.valid_out(pv[2]),.category(pc[8+:4]));
reg [11:0] delay3;
always @(posedge clk)if(rst)delay3<=0;else delay3<= {delay3[7:0],next_states[12+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(3),.FILE("mem/tt03.hex")) s3(
.clk(clk),.rst(rst),.valid_in(pv[2]),
.state_in(delay3[11-:4]),
.latent_in(pc[8]),
.seed_write(seed_write && phase==0 && seed_index==19),.seed_word(seed_word),
.valid_out(pv[3]),.category(pc[12+:4]));
reg [15:0] delay4;
always @(posedge clk)if(rst)delay4<=0;else delay4<= {delay4[11:0],next_states[16+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(4),.FILE("mem/tt04.hex")) s4(
.clk(clk),.rst(rst),.valid_in(pv[3]),
.state_in(delay4[15-:4]),
.latent_in(pc[12]),
.seed_write(seed_write && phase==0 && seed_index==20),.seed_word(seed_word),
.valid_out(pv[4]),.category(pc[16+:4]));
reg [19:0] delay5;
always @(posedge clk)if(rst)delay5<=0;else delay5<= {delay5[15:0],next_states[20+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(5),.FILE("mem/tt05.hex")) s5(
.clk(clk),.rst(rst),.valid_in(pv[4]),
.state_in(delay5[19-:4]),
.latent_in(pc[16]),
.seed_write(seed_write && phase==0 && seed_index==21),.seed_word(seed_word),
.valid_out(pv[5]),.category(pc[20+:4]));
reg [23:0] delay6;
always @(posedge clk)if(rst)delay6<=0;else delay6<= {delay6[19:0],next_states[24+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(6),.FILE("mem/tt06.hex")) s6(
.clk(clk),.rst(rst),.valid_in(pv[5]),
.state_in(delay6[23-:4]),
.latent_in(pc[20]),
.seed_write(seed_write && phase==0 && seed_index==22),.seed_word(seed_word),
.valid_out(pv[6]),.category(pc[24+:4]));
reg [27:0] delay7;
always @(posedge clk)if(rst)delay7<=0;else delay7<= {delay7[23:0],next_states[28+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(7),.FILE("mem/tt07.hex")) s7(
.clk(clk),.rst(rst),.valid_in(pv[6]),
.state_in(delay7[27-:4]),
.latent_in(pc[24]),
.seed_write(seed_write && phase==0 && seed_index==23),.seed_word(seed_word),
.valid_out(pv[7]),.category(pc[28+:4]));
reg [31:0] delay8;
always @(posedge clk)if(rst)delay8<=0;else delay8<= {delay8[27:0],next_states[32+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(8),.FILE("mem/tt08.hex")) s8(
.clk(clk),.rst(rst),.valid_in(pv[7]),
.state_in(delay8[31-:4]),
.latent_in(pc[28]),
.seed_write(seed_write && phase==0 && seed_index==24),.seed_word(seed_word),
.valid_out(pv[8]),.category(pc[32+:4]));
reg [35:0] delay9;
always @(posedge clk)if(rst)delay9<=0;else delay9<= {delay9[31:0],next_states[36+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(9),.FILE("mem/tt09.hex")) s9(
.clk(clk),.rst(rst),.valid_in(pv[8]),
.state_in(delay9[35-:4]),
.latent_in(pc[32]),
.seed_write(seed_write && phase==0 && seed_index==25),.seed_word(seed_word),
.valid_out(pv[9]),.category(pc[36+:4]));
reg [39:0] delay10;
always @(posedge clk)if(rst)delay10<=0;else delay10<= {delay10[35:0],next_states[40+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(10),.FILE("mem/tt10.hex")) s10(
.clk(clk),.rst(rst),.valid_in(pv[9]),
.state_in(delay10[39-:4]),
.latent_in(pc[36]),
.seed_write(seed_write && phase==0 && seed_index==26),.seed_word(seed_word),
.valid_out(pv[10]),.category(pc[40+:4]));
reg [43:0] delay11;
always @(posedge clk)if(rst)delay11<=0;else delay11<= {delay11[39:0],next_states[44+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(11),.FILE("mem/tt11.hex")) s11(
.clk(clk),.rst(rst),.valid_in(pv[10]),
.state_in(delay11[43-:4]),
.latent_in(pc[40]),
.seed_write(seed_write && phase==0 && seed_index==27),.seed_word(seed_word),
.valid_out(pv[11]),.category(pc[44+:4]));
reg [47:0] delay12;
always @(posedge clk)if(rst)delay12<=0;else delay12<= {delay12[43:0],next_states[48+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(12),.FILE("mem/tt12.hex")) s12(
.clk(clk),.rst(rst),.valid_in(pv[11]),
.state_in(delay12[47-:4]),
.latent_in(pc[44]),
.seed_write(seed_write && phase==0 && seed_index==28),.seed_word(seed_word),
.valid_out(pv[12]),.category(pc[48+:4]));
reg [51:0] delay13;
always @(posedge clk)if(rst)delay13<=0;else delay13<= {delay13[47:0],next_states[52+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(13),.FILE("mem/tt13.hex")) s13(
.clk(clk),.rst(rst),.valid_in(pv[12]),
.state_in(delay13[51-:4]),
.latent_in(pc[48]),
.seed_write(seed_write && phase==0 && seed_index==29),.seed_word(seed_word),
.valid_out(pv[13]),.category(pc[52+:4]));
reg [55:0] delay14;
always @(posedge clk)if(rst)delay14<=0;else delay14<= {delay14[51:0],next_states[56+:4]};
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(14),.FILE("mem/tt14.hex")) s14(
.clk(clk),.rst(rst),.valid_in(pv[13]),
.state_in(delay14[55-:4]),
.latent_in(pc[52]),
.seed_write(seed_write && phase==0 && seed_index==30),.seed_word(seed_word),
.valid_out(pv[14]),.category(pc[56+:4]));
reg [59:0] delay15;
always @(posedge clk)if(rst)delay15<=0;else delay15<= {delay15[55:0],next_states[60+:4]};
tt_r2 #(.FIRST(0),.LAST(1),.STAGE(15),.FILE("mem/tt15.hex")) s15(
.clk(clk),.rst(rst),.valid_in(pv[14]),
.state_in(delay15[59-:4]),
.latent_in(pc[56]),
.seed_write(seed_write && phase==0 && seed_index==31),.seed_word(seed_word),
.valid_out(pv[15]),.category(pc[60+:4]));
