tt_r2 #(.FIRST(1),.LAST(0),.STAGE(0),.FILE("mem/tt00.hex")) s0(
.clk(clk),.rst(rst),.valid_in(phase==3),
.state_in(next_states[0+:4]),
.latent_in(1'b0),
.seed_write(seed_write && phase==0 && seed_index==16),.seed_word(seed_word),
.valid_out(pv[0]),.category(pc[0+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(1),.FILE("mem/tt01.hex")) s1(
.clk(clk),.rst(rst),.valid_in(pv[0]),
.state_in(next_states[4+:4]),
.latent_in(pc[0]),
.seed_write(seed_write && phase==0 && seed_index==17),.seed_word(seed_word),
.valid_out(pv[1]),.category(pc[4+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(2),.FILE("mem/tt02.hex")) s2(
.clk(clk),.rst(rst),.valid_in(pv[1]),
.state_in(next_states[8+:4]),
.latent_in(pc[4]),
.seed_write(seed_write && phase==0 && seed_index==18),.seed_word(seed_word),
.valid_out(pv[2]),.category(pc[8+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(3),.FILE("mem/tt03.hex")) s3(
.clk(clk),.rst(rst),.valid_in(pv[2]),
.state_in(next_states[12+:4]),
.latent_in(pc[8]),
.seed_write(seed_write && phase==0 && seed_index==19),.seed_word(seed_word),
.valid_out(pv[3]),.category(pc[12+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(4),.FILE("mem/tt04.hex")) s4(
.clk(clk),.rst(rst),.valid_in(pv[3]),
.state_in(next_states[16+:4]),
.latent_in(pc[12]),
.seed_write(seed_write && phase==0 && seed_index==20),.seed_word(seed_word),
.valid_out(pv[4]),.category(pc[16+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(5),.FILE("mem/tt05.hex")) s5(
.clk(clk),.rst(rst),.valid_in(pv[4]),
.state_in(next_states[20+:4]),
.latent_in(pc[16]),
.seed_write(seed_write && phase==0 && seed_index==21),.seed_word(seed_word),
.valid_out(pv[5]),.category(pc[20+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(6),.FILE("mem/tt06.hex")) s6(
.clk(clk),.rst(rst),.valid_in(pv[5]),
.state_in(next_states[24+:4]),
.latent_in(pc[20]),
.seed_write(seed_write && phase==0 && seed_index==22),.seed_word(seed_word),
.valid_out(pv[6]),.category(pc[24+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(7),.FILE("mem/tt07.hex")) s7(
.clk(clk),.rst(rst),.valid_in(pv[6]),
.state_in(next_states[28+:4]),
.latent_in(pc[24]),
.seed_write(seed_write && phase==0 && seed_index==23),.seed_word(seed_word),
.valid_out(pv[7]),.category(pc[28+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(8),.FILE("mem/tt08.hex")) s8(
.clk(clk),.rst(rst),.valid_in(pv[7]),
.state_in(next_states[32+:4]),
.latent_in(pc[28]),
.seed_write(seed_write && phase==0 && seed_index==24),.seed_word(seed_word),
.valid_out(pv[8]),.category(pc[32+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(9),.FILE("mem/tt09.hex")) s9(
.clk(clk),.rst(rst),.valid_in(pv[8]),
.state_in(next_states[36+:4]),
.latent_in(pc[32]),
.seed_write(seed_write && phase==0 && seed_index==25),.seed_word(seed_word),
.valid_out(pv[9]),.category(pc[36+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(10),.FILE("mem/tt10.hex")) s10(
.clk(clk),.rst(rst),.valid_in(pv[9]),
.state_in(next_states[40+:4]),
.latent_in(pc[36]),
.seed_write(seed_write && phase==0 && seed_index==26),.seed_word(seed_word),
.valid_out(pv[10]),.category(pc[40+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(11),.FILE("mem/tt11.hex")) s11(
.clk(clk),.rst(rst),.valid_in(pv[10]),
.state_in(next_states[44+:4]),
.latent_in(pc[40]),
.seed_write(seed_write && phase==0 && seed_index==27),.seed_word(seed_word),
.valid_out(pv[11]),.category(pc[44+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(12),.FILE("mem/tt12.hex")) s12(
.clk(clk),.rst(rst),.valid_in(pv[11]),
.state_in(next_states[48+:4]),
.latent_in(pc[44]),
.seed_write(seed_write && phase==0 && seed_index==28),.seed_word(seed_word),
.valid_out(pv[12]),.category(pc[48+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(13),.FILE("mem/tt13.hex")) s13(
.clk(clk),.rst(rst),.valid_in(pv[12]),
.state_in(next_states[52+:4]),
.latent_in(pc[48]),
.seed_write(seed_write && phase==0 && seed_index==29),.seed_word(seed_word),
.valid_out(pv[13]),.category(pc[52+:4]));
tt_r2 #(.FIRST(0),.LAST(0),.STAGE(14),.FILE("mem/tt14.hex")) s14(
.clk(clk),.rst(rst),.valid_in(pv[13]),
.state_in(next_states[56+:4]),
.latent_in(pc[52]),
.seed_write(seed_write && phase==0 && seed_index==30),.seed_word(seed_word),
.valid_out(pv[14]),.category(pc[56+:4]));
tt_r2 #(.FIRST(0),.LAST(1),.STAGE(15),.FILE("mem/tt15.hex")) s15(
.clk(clk),.rst(rst),.valid_in(pv[14]),
.state_in(next_states[60+:4]),
.latent_in(pc[56]),
.seed_write(seed_write && phase==0 && seed_index==31),.seed_word(seed_word),
.valid_out(pv[15]),.category(pc[60+:4]));
