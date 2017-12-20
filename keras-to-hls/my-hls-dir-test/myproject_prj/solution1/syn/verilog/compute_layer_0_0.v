// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2016.4
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module compute_layer_0_0 (
        ap_clk,
        ap_rst,
        data_0_V_read,
        data_1_V_read,
        data_2_V_read,
        data_3_V_read,
        data_4_V_read,
        data_5_V_read,
        data_6_V_read,
        data_7_V_read,
        data_8_V_read,
        data_9_V_read,
        data_10_V_read,
        data_11_V_read,
        data_12_V_read,
        data_13_V_read,
        data_14_V_read,
        data_15_V_read,
        data_16_V_read,
        data_17_V_read,
        data_18_V_read,
        data_19_V_read,
        data_20_V_read,
        data_21_V_read,
        data_22_V_read,
        data_23_V_read,
        data_24_V_read,
        data_25_V_read,
        data_26_V_read,
        data_27_V_read,
        data_28_V_read,
        data_29_V_read,
        data_30_V_read,
        data_31_V_read,
        ap_return,
        ap_ce
);

parameter    ap_const_lv28_FFFFEB6 = 28'b1111111111111111111010110110;
parameter    ap_const_lv28_FFFFE73 = 28'b1111111111111111111001110011;
parameter    ap_const_lv28_FFFFED4 = 28'b1111111111111111111011010100;
parameter    ap_const_lv27_F2 = 27'b11110010;
parameter    ap_const_lv27_BF = 27'b10111111;
parameter    ap_const_lv27_7FFFF46 = 27'b111111111111111111101000110;
parameter    ap_const_lv26_5C = 26'b1011100;
parameter    ap_const_lv27_9F = 27'b10011111;
parameter    ap_const_lv26_3FFFF97 = 26'b11111111111111111110010111;
parameter    ap_const_lv26_3FFFF9F = 26'b11111111111111111110011111;
parameter    ap_const_lv25_32 = 25'b110010;
parameter    ap_const_lv26_4A = 26'b1001010;
parameter    ap_const_lv24_19 = 24'b11001;
parameter    ap_const_lv27_7FFFF4B = 27'b111111111111111111101001011;
parameter    ap_const_lv27_7FFFF52 = 27'b111111111111111111101010010;
parameter    ap_const_lv27_7FFFF0B = 27'b111111111111111111100001011;
parameter    ap_const_lv28_FFFFEB3 = 28'b1111111111111111111010110011;
parameter    ap_const_lv28_FFFFE92 = 28'b1111111111111111111010010010;
parameter    ap_const_lv27_7FFFF71 = 27'b111111111111111111101110001;
parameter    ap_const_lv28_FFFFE4A = 28'b1111111111111111111001001010;
parameter    ap_const_lv24_1D = 24'b11101;
parameter    ap_const_lv28_FFFFE87 = 28'b1111111111111111111010000111;
parameter    ap_const_lv27_7FFFF5C = 27'b111111111111111111101011100;
parameter    ap_const_lv6_0 = 6'b000000;
parameter    ap_const_lv2_0 = 2'b00;
parameter    ap_const_lv32_A = 32'b1010;
parameter    ap_const_lv32_18 = 32'b11000;
parameter    ap_const_lv7_0 = 7'b0000000;
parameter    ap_const_lv5_0 = 5'b00000;
parameter    ap_const_lv32_19 = 32'b11001;
parameter    ap_const_lv21_0 = 21'b000000000000000000000;
parameter    ap_const_lv32_14 = 32'b10100;
parameter    ap_const_lv32_3 = 32'b11;
parameter    ap_const_lv32_11 = 32'b10001;
parameter    ap_const_lv3_0 = 3'b000;
parameter    ap_const_lv22_0 = 22'b0000000000000000000000;
parameter    ap_const_lv32_15 = 32'b10101;
parameter    ap_const_lv32_1A = 32'b11010;
parameter    ap_const_lv9_0 = 9'b000000000;
parameter    ap_const_lv32_1B = 32'b11011;
parameter    ap_const_lv32_17 = 32'b10111;
parameter    ap_const_lv16_FF86 = 16'b1111111110000110;

input   ap_clk;
input   ap_rst;
input  [17:0] data_0_V_read;
input  [17:0] data_1_V_read;
input  [17:0] data_2_V_read;
input  [17:0] data_3_V_read;
input  [17:0] data_4_V_read;
input  [17:0] data_5_V_read;
input  [17:0] data_6_V_read;
input  [17:0] data_7_V_read;
input  [17:0] data_8_V_read;
input  [17:0] data_9_V_read;
input  [17:0] data_10_V_read;
input  [17:0] data_11_V_read;
input  [17:0] data_12_V_read;
input  [17:0] data_13_V_read;
input  [17:0] data_14_V_read;
input  [17:0] data_15_V_read;
input  [17:0] data_16_V_read;
input  [17:0] data_17_V_read;
input  [17:0] data_18_V_read;
input  [17:0] data_19_V_read;
input  [17:0] data_20_V_read;
input  [17:0] data_21_V_read;
input  [17:0] data_22_V_read;
input  [17:0] data_23_V_read;
input  [17:0] data_24_V_read;
input  [17:0] data_25_V_read;
input  [17:0] data_26_V_read;
input  [17:0] data_27_V_read;
input  [17:0] data_28_V_read;
input  [17:0] data_29_V_read;
input  [17:0] data_30_V_read;
input  [17:0] data_31_V_read;
output  [17:0] ap_return;
input   ap_ce;

reg   [17:0] data_2_V_read_3_reg_4450;
reg   [17:0] ap_pipeline_reg_pp0_iter1_data_2_V_read_3_reg_4450;
reg   [14:0] tmp_5_reg_4456;
reg   [14:0] ap_pipeline_reg_pp0_iter1_tmp_5_reg_4456;
reg   [14:0] ap_pipeline_reg_pp0_iter2_tmp_5_reg_4456;
reg   [14:0] tmp_14_reg_4471;
reg   [14:0] ap_pipeline_reg_pp0_iter1_tmp_14_reg_4471;
reg   [14:0] ap_pipeline_reg_pp0_iter2_tmp_14_reg_4471;
reg   [15:0] tmp_20_reg_4481;
reg   [15:0] ap_pipeline_reg_pp0_iter1_tmp_20_reg_4481;
reg   [15:0] ap_pipeline_reg_pp0_iter2_tmp_20_reg_4481;
reg   [14:0] tmp_29_reg_4506;
reg   [14:0] ap_pipeline_reg_pp0_iter1_tmp_29_reg_4506;
reg   [14:0] ap_pipeline_reg_pp0_iter2_tmp_29_reg_4506;
reg   [10:0] tmp_32_reg_4516;
reg   [10:0] ap_pipeline_reg_pp0_iter1_tmp_32_reg_4516;
reg   [10:0] ap_pipeline_reg_pp0_iter2_tmp_32_reg_4516;
reg   [14:0] tmp_41_reg_4531;
reg   [14:0] ap_pipeline_reg_pp0_iter1_tmp_41_reg_4531;
reg   [11:0] tmp_143_reg_4546;
reg   [11:0] ap_pipeline_reg_pp0_iter1_tmp_143_reg_4546;
reg   [15:0] tmp_55_reg_4591;
reg   [15:0] ap_pipeline_reg_pp0_iter1_tmp_55_reg_4591;
reg   [15:0] ap_pipeline_reg_pp0_iter2_tmp_55_reg_4591;
reg   [16:0] tmp_8_reg_4611;
reg   [17:0] tmp_10_2_reg_4616;
reg   [16:0] tmp_11_reg_4621;
reg   [16:0] tmp_17_reg_4626;
reg   [17:0] tmp_10_7_reg_4631;
reg   [16:0] tmp_23_reg_4636;
reg   [17:0] tmp_10_9_reg_4641;
reg   [16:0] tmp_26_reg_4646;
reg   [17:0] tmp_10_s_reg_4651;
reg   [16:0] tmp_35_reg_4656;
reg   [15:0] tmp_38_reg_4661;
reg   [13:0] tmp_43_reg_4666;
reg   [17:0] tmp_10_1_reg_4671;
reg   [15:0] tmp_48_reg_4676;
reg   [16:0] tmp_50_reg_4681;
reg   [17:0] tmp_10_3_reg_4686;
reg   [17:0] tmp_10_4_reg_4691;
reg   [17:0] tmp_10_5_reg_4696;
reg   [15:0] tmp_s_reg_4701;
reg   [16:0] tmp_53_reg_4706;
reg   [15:0] tmp_57_reg_4711;
reg   [16:0] tmp_59_reg_4716;
wire   [15:0] tmp34_fu_4205_p2;
reg   [15:0] tmp34_reg_4721;
wire   [15:0] tmp44_fu_4217_p2;
reg   [15:0] tmp44_reg_4726;
wire   [17:0] tmp16_fu_4317_p2;
reg   [17:0] tmp16_reg_4731;
wire   [17:0] tmp23_fu_4357_p2;
reg   [17:0] tmp23_reg_4736;
wire   [17:0] tmp32_fu_4371_p2;
reg   [17:0] tmp32_reg_4741;
wire   [17:0] tmp35_fu_4387_p2;
reg   [17:0] tmp35_reg_4746;
wire   [17:0] tmp38_fu_4425_p2;
reg   [17:0] tmp38_reg_4751;
wire  signed [9:0] grp_fu_402_p1;
wire  signed [9:0] grp_fu_403_p1;
wire  signed [9:0] grp_fu_404_p1;
wire   [8:0] grp_fu_405_p1;
wire   [8:0] grp_fu_408_p1;
wire  signed [8:0] grp_fu_409_p1;
wire   [7:0] grp_fu_410_p1;
wire   [8:0] grp_fu_411_p1;
wire  signed [7:0] grp_fu_412_p1;
wire  signed [7:0] grp_fu_417_p1;
wire   [6:0] grp_fu_418_p1;
wire   [7:0] grp_fu_419_p1;
wire   [5:0] grp_fu_420_p1;
wire  signed [8:0] grp_fu_422_p1;
wire  signed [8:0] grp_fu_423_p1;
wire  signed [8:0] grp_fu_424_p1;
wire  signed [9:0] grp_fu_425_p1;
wire  signed [9:0] grp_fu_426_p1;
wire  signed [8:0] grp_fu_428_p1;
wire  signed [9:0] grp_fu_429_p1;
wire   [5:0] grp_fu_430_p1;
wire  signed [9:0] grp_fu_432_p1;
wire  signed [8:0] grp_fu_433_p1;
wire   [23:0] p_shl10_fu_3540_p3;
wire   [19:0] p_shl12_fu_3552_p3;
wire  signed [24:0] p_shl14_cast_fu_3560_p1;
wire  signed [24:0] p_shl12_cast_fu_3548_p1;
wire   [24:0] p_Val2_1_fu_3564_p2;
wire   [23:0] p_shl7_fu_3590_p3;
wire   [18:0] p_shl8_fu_3602_p3;
wire  signed [24:0] p_shl7_cast_fu_3598_p1;
wire  signed [24:0] p_shl8_cast_fu_3610_p1;
wire   [24:0] p_Val2_1_4_fu_3614_p2;
wire   [24:0] p_shl5_fu_3635_p3;
wire   [22:0] p_shl6_fu_3647_p3;
wire  signed [25:0] p_shl5_cast_fu_3643_p1;
wire  signed [25:0] p_shl6_cast_fu_3655_p1;
wire   [25:0] p_Val2_1_6_fu_3659_p2;
wire   [23:0] p_shl4_fu_3699_p3;
wire  signed [24:0] p_shl4_cast_fu_3707_p1;
wire  signed [24:0] OP1_V_10_cast_fu_3695_p1;
wire   [24:0] p_Val2_1_10_fu_3711_p2;
wire   [19:0] p_shl3_fu_3736_p3;
wire  signed [20:0] p_shl3_cast_fu_3744_p1;
wire   [20:0] p_neg_fu_3748_p2;
wire  signed [20:0] OP1_V_12_cast_fu_3732_p1;
wire   [20:0] p_Val2_1_12_fu_3754_p2;
wire   [20:0] p_shl2_fu_3800_p3;
wire  signed [21:0] p_shl2_cast_fu_3808_p1;
wire   [21:0] p_Val2_1_17_fu_3812_p2;
wire   [24:0] p_shl_fu_3868_p3;
wire   [20:0] p_shl1_fu_3880_p3;
wire  signed [25:0] p_shl_cast_fu_3876_p1;
wire  signed [25:0] p_shl1_cast_fu_3888_p1;
wire   [25:0] p_Val2_1_26_fu_3892_p2;
wire   [26:0] grp_fu_428_p2;
wire   [26:0] p_shl9_fu_3933_p3;
wire   [23:0] p_shl11_fu_3944_p3;
wire  signed [27:0] p_shl11_cast_fu_3951_p1;
wire  signed [27:0] p_shl9_cast_fu_3940_p1;
wire   [27:0] p_Val2_1_2_fu_3955_p2;
wire   [26:0] grp_fu_422_p2;
wire   [26:0] grp_fu_408_p2;
wire   [27:0] grp_fu_426_p2;
wire   [26:0] grp_fu_423_p2;
wire   [27:0] grp_fu_425_p2;
wire   [26:0] grp_fu_405_p2;
wire   [27:0] grp_fu_403_p2;
wire   [26:0] grp_fu_433_p2;
wire   [25:0] grp_fu_410_p2;
wire   [23:0] grp_fu_420_p2;
wire   [27:0] grp_fu_429_p2;
wire   [24:0] grp_fu_418_p2;
wire   [14:0] tmp_144_fu_4087_p4;
wire   [25:0] grp_fu_417_p2;
wire   [26:0] grp_fu_424_p2;
wire   [27:0] grp_fu_404_p2;
wire   [27:0] grp_fu_402_p2;
wire   [27:0] grp_fu_432_p2;
wire   [25:0] grp_fu_419_p2;
wire   [26:0] grp_fu_411_p2;
wire   [25:0] grp_fu_412_p2;
wire   [26:0] grp_fu_409_p2;
wire   [23:0] grp_fu_430_p2;
wire   [13:0] tmp_145_fu_4191_p4;
wire  signed [15:0] tmp_47_cast_fu_4084_p1;
wire  signed [15:0] tmp_50_cast_fu_4097_p1;
wire  signed [15:0] tmp_10_15_cast_fu_4061_p1;
wire   [15:0] tmp45_fu_4211_p2;
wire  signed [15:0] tmp_68_cast_fu_4201_p1;
wire  signed [17:0] tmp_6_fu_4223_p1;
wire  signed [17:0] tmp_9_fu_4226_p1;
wire  signed [17:0] tmp_12_fu_4229_p1;
wire   [17:0] tmp19_fu_4289_p2;
wire   [17:0] tmp18_fu_4283_p2;
wire  signed [17:0] tmp_15_fu_4232_p1;
wire  signed [17:0] tmp_18_fu_4235_p1;
wire  signed [17:0] tmp_21_fu_4238_p1;
wire   [17:0] tmp22_fu_4306_p2;
wire   [17:0] tmp21_fu_4300_p2;
wire   [17:0] tmp20_fu_4311_p2;
wire   [17:0] tmp17_fu_4294_p2;
wire  signed [17:0] tmp_24_fu_4241_p1;
wire  signed [17:0] tmp_27_fu_4244_p1;
wire  signed [17:0] tmp_30_fu_4247_p1;
wire   [17:0] tmp26_fu_4328_p2;
wire   [17:0] tmp25_fu_4323_p2;
wire  signed [17:0] tmp_33_fu_4250_p1;
wire  signed [17:0] tmp_36_fu_4253_p1;
wire  signed [17:0] tmp_39_fu_4256_p1;
wire   [17:0] tmp29_fu_4345_p2;
wire   [17:0] tmp28_fu_4340_p2;
wire   [17:0] tmp27_fu_4351_p2;
wire   [17:0] tmp24_fu_4334_p2;
wire  signed [17:0] tmp_44_fu_4259_p1;
wire  signed [17:0] tmp34_cast_fu_4368_p1;
wire   [17:0] tmp33_fu_4363_p2;
wire  signed [17:0] tmp_49_fu_4262_p1;
wire  signed [17:0] tmp_51_fu_4265_p1;
wire   [17:0] tmp37_fu_4383_p2;
wire   [17:0] tmp36_fu_4377_p2;
wire  signed [17:0] tmp_52_fu_4268_p1;
wire  signed [17:0] tmp_54_fu_4271_p1;
wire  signed [17:0] tmp_56_fu_4274_p1;
wire   [17:0] tmp41_fu_4398_p2;
wire   [17:0] tmp40_fu_4393_p2;
wire  signed [17:0] tmp_58_fu_4277_p1;
wire  signed [17:0] tmp_60_fu_4280_p1;
wire  signed [17:0] tmp44_cast_fu_4416_p1;
wire   [17:0] tmp43_fu_4410_p2;
wire   [17:0] tmp42_fu_4419_p2;
wire   [17:0] tmp39_fu_4404_p2;
wire   [17:0] tmp31_fu_4435_p2;
wire   [17:0] tmp30_fu_4439_p2;
wire   [17:0] tmp15_fu_4431_p2;
reg    grp_fu_402_ce;
reg    grp_fu_403_ce;
reg    grp_fu_404_ce;
reg    grp_fu_405_ce;
reg    grp_fu_408_ce;
reg    grp_fu_409_ce;
reg    grp_fu_410_ce;
reg    grp_fu_411_ce;
reg    grp_fu_412_ce;
reg    grp_fu_417_ce;
reg    grp_fu_418_ce;
reg    grp_fu_419_ce;
reg    grp_fu_420_ce;
reg    grp_fu_422_ce;
reg    grp_fu_423_ce;
reg    grp_fu_424_ce;
reg    grp_fu_425_ce;
reg    grp_fu_426_ce;
reg    grp_fu_428_ce;
reg    grp_fu_429_ce;
reg    grp_fu_430_ce;
reg    grp_fu_432_ce;
reg    grp_fu_433_ce;

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U175(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_24_V_read),
    .din1(grp_fu_402_p1),
    .ce(grp_fu_402_ce),
    .dout(grp_fu_402_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U176(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_12_V_read),
    .din1(grp_fu_403_p1),
    .ce(grp_fu_403_ce),
    .dout(grp_fu_403_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U177(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_23_V_read),
    .din1(grp_fu_404_p1),
    .ce(grp_fu_404_ce),
    .dout(grp_fu_404_p2)
);

myproject_mul_18sfYi #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18sfYi_x_U178(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_10_V_read),
    .din1(grp_fu_405_p1),
    .ce(grp_fu_405_ce),
    .dout(grp_fu_405_p2)
);

myproject_mul_18sfYi #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18sfYi_x_U179(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_5_V_read),
    .din1(grp_fu_408_p1),
    .ce(grp_fu_408_ce),
    .dout(grp_fu_408_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U180(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_30_V_read),
    .din1(grp_fu_409_p1),
    .ce(grp_fu_409_ce),
    .dout(grp_fu_409_p2)
);

myproject_mul_18shbi #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 26 ))
myproject_mul_18shbi_x_U181(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_15_V_read),
    .din1(grp_fu_410_p1),
    .ce(grp_fu_410_ce),
    .dout(grp_fu_410_p2)
);

myproject_mul_18sfYi #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18sfYi_x_U182(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_27_V_read),
    .din1(grp_fu_411_p1),
    .ce(grp_fu_411_ce),
    .dout(grp_fu_411_p2)
);

myproject_mul_18sjbC #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 26 ))
myproject_mul_18sjbC_x_U183(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_29_V_read),
    .din1(grp_fu_412_p1),
    .ce(grp_fu_412_ce),
    .dout(grp_fu_412_p2)
);

myproject_mul_18sjbC #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 26 ))
myproject_mul_18sjbC_x_U184(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_21_V_read),
    .din1(grp_fu_417_p1),
    .ce(grp_fu_417_ce),
    .dout(grp_fu_417_p2)
);

myproject_mul_18slbW #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 7 ),
    .dout_WIDTH( 25 ))
myproject_mul_18slbW_x_U185(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_20_V_read),
    .din1(grp_fu_418_p1),
    .ce(grp_fu_418_ce),
    .dout(grp_fu_418_p2)
);

myproject_mul_18shbi #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 26 ))
myproject_mul_18shbi_x_U186(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_26_V_read),
    .din1(grp_fu_419_p1),
    .ce(grp_fu_419_ce),
    .dout(grp_fu_419_p2)
);

myproject_mul_18smb6 #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 6 ),
    .dout_WIDTH( 24 ))
myproject_mul_18smb6_U187(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_17_V_read),
    .din1(grp_fu_420_p1),
    .ce(grp_fu_420_ce),
    .dout(grp_fu_420_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U188(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_3_V_read),
    .din1(grp_fu_422_p1),
    .ce(grp_fu_422_ce),
    .dout(grp_fu_422_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U189(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_8_V_read),
    .din1(grp_fu_423_p1),
    .ce(grp_fu_423_ce),
    .dout(grp_fu_423_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U190(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_22_V_read),
    .din1(grp_fu_424_p1),
    .ce(grp_fu_424_ce),
    .dout(grp_fu_424_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U191(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_9_V_read),
    .din1(grp_fu_425_p1),
    .ce(grp_fu_425_ce),
    .dout(grp_fu_425_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U192(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_7_V_read),
    .din1(grp_fu_426_p1),
    .ce(grp_fu_426_ce),
    .dout(grp_fu_426_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U193(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_1_V_read),
    .din1(grp_fu_428_p1),
    .ce(grp_fu_428_ce),
    .dout(grp_fu_428_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U194(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_18_V_read),
    .din1(grp_fu_429_p1),
    .ce(grp_fu_429_ce),
    .dout(grp_fu_429_p2)
);

myproject_mul_18smb6 #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 6 ),
    .dout_WIDTH( 24 ))
myproject_mul_18smb6_U195(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_31_V_read),
    .din1(grp_fu_430_p1),
    .ce(grp_fu_430_ce),
    .dout(grp_fu_430_p2)
);

myproject_mul_18sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 28 ))
myproject_mul_18sbkb_x_U196(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_25_V_read),
    .din1(grp_fu_432_p1),
    .ce(grp_fu_432_ce),
    .dout(grp_fu_432_p2)
);

myproject_mul_18scud #(
    .ID( 1 ),
    .NUM_STAGE( 3 ),
    .din0_WIDTH( 18 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 27 ))
myproject_mul_18scud_x_U197(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(data_14_V_read),
    .din1(grp_fu_433_p1),
    .ce(grp_fu_433_ce),
    .dout(grp_fu_433_p2)
);

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_ce)) begin
        ap_pipeline_reg_pp0_iter1_data_2_V_read_3_reg_4450 <= data_2_V_read_3_reg_4450;
        ap_pipeline_reg_pp0_iter1_tmp_143_reg_4546 <= tmp_143_reg_4546;
        ap_pipeline_reg_pp0_iter1_tmp_14_reg_4471 <= tmp_14_reg_4471;
        ap_pipeline_reg_pp0_iter1_tmp_20_reg_4481 <= tmp_20_reg_4481;
        ap_pipeline_reg_pp0_iter1_tmp_29_reg_4506 <= tmp_29_reg_4506;
        ap_pipeline_reg_pp0_iter1_tmp_32_reg_4516 <= tmp_32_reg_4516;
        ap_pipeline_reg_pp0_iter1_tmp_41_reg_4531 <= tmp_41_reg_4531;
        ap_pipeline_reg_pp0_iter1_tmp_55_reg_4591 <= tmp_55_reg_4591;
        ap_pipeline_reg_pp0_iter1_tmp_5_reg_4456 <= tmp_5_reg_4456;
        ap_pipeline_reg_pp0_iter2_tmp_14_reg_4471 <= ap_pipeline_reg_pp0_iter1_tmp_14_reg_4471;
        ap_pipeline_reg_pp0_iter2_tmp_20_reg_4481 <= ap_pipeline_reg_pp0_iter1_tmp_20_reg_4481;
        ap_pipeline_reg_pp0_iter2_tmp_29_reg_4506 <= ap_pipeline_reg_pp0_iter1_tmp_29_reg_4506;
        ap_pipeline_reg_pp0_iter2_tmp_32_reg_4516 <= ap_pipeline_reg_pp0_iter1_tmp_32_reg_4516;
        ap_pipeline_reg_pp0_iter2_tmp_55_reg_4591 <= ap_pipeline_reg_pp0_iter1_tmp_55_reg_4591;
        ap_pipeline_reg_pp0_iter2_tmp_5_reg_4456 <= ap_pipeline_reg_pp0_iter1_tmp_5_reg_4456;
        data_2_V_read_3_reg_4450 <= data_2_V_read;
        tmp16_reg_4731 <= tmp16_fu_4317_p2;
        tmp23_reg_4736 <= tmp23_fu_4357_p2;
        tmp32_reg_4741 <= tmp32_fu_4371_p2;
        tmp34_reg_4721 <= tmp34_fu_4205_p2;
        tmp35_reg_4746 <= tmp35_fu_4387_p2;
        tmp38_reg_4751 <= tmp38_fu_4425_p2;
        tmp44_reg_4726 <= tmp44_fu_4217_p2;
        tmp_10_1_reg_4671 <= {{grp_fu_429_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_2_reg_4616 <= {{p_Val2_1_2_fu_3955_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_3_reg_4686 <= {{grp_fu_404_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_4_reg_4691 <= {{grp_fu_402_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_5_reg_4696 <= {{grp_fu_432_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_7_reg_4631 <= {{grp_fu_426_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_9_reg_4641 <= {{grp_fu_425_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_10_s_reg_4651 <= {{grp_fu_403_p2[ap_const_lv32_1B : ap_const_lv32_A]}};
        tmp_11_reg_4621 <= {{grp_fu_422_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_143_reg_4546 <= {{p_Val2_1_17_fu_3812_p2[ap_const_lv32_15 : ap_const_lv32_A]}};
        tmp_14_reg_4471 <= {{p_Val2_1_4_fu_3614_p2[ap_const_lv32_18 : ap_const_lv32_A]}};
        tmp_17_reg_4626 <= {{grp_fu_408_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_20_reg_4481 <= {{p_Val2_1_6_fu_3659_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
        tmp_23_reg_4636 <= {{grp_fu_423_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_26_reg_4646 <= {{grp_fu_405_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_29_reg_4506 <= {{p_Val2_1_10_fu_3711_p2[ap_const_lv32_18 : ap_const_lv32_A]}};
        tmp_32_reg_4516 <= {{p_Val2_1_12_fu_3754_p2[ap_const_lv32_14 : ap_const_lv32_A]}};
        tmp_35_reg_4656 <= {{grp_fu_433_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_38_reg_4661 <= {{grp_fu_410_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
        tmp_41_reg_4531 <= {{data_16_V_read[ap_const_lv32_11 : ap_const_lv32_3]}};
        tmp_43_reg_4666 <= {{grp_fu_420_p2[ap_const_lv32_17 : ap_const_lv32_A]}};
        tmp_48_reg_4676 <= {{grp_fu_417_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
        tmp_50_reg_4681 <= {{grp_fu_424_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_53_reg_4706 <= {{grp_fu_411_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_55_reg_4591 <= {{p_Val2_1_26_fu_3892_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
        tmp_57_reg_4711 <= {{grp_fu_412_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
        tmp_59_reg_4716 <= {{grp_fu_409_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_5_reg_4456 <= {{p_Val2_1_fu_3564_p2[ap_const_lv32_18 : ap_const_lv32_A]}};
        tmp_8_reg_4611 <= {{grp_fu_428_p2[ap_const_lv32_1A : ap_const_lv32_A]}};
        tmp_s_reg_4701 <= {{grp_fu_419_p2[ap_const_lv32_19 : ap_const_lv32_A]}};
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_402_ce = 1'b0;
    end else begin
        grp_fu_402_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_403_ce = 1'b0;
    end else begin
        grp_fu_403_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_404_ce = 1'b0;
    end else begin
        grp_fu_404_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_405_ce = 1'b0;
    end else begin
        grp_fu_405_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_408_ce = 1'b0;
    end else begin
        grp_fu_408_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_409_ce = 1'b0;
    end else begin
        grp_fu_409_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_410_ce = 1'b0;
    end else begin
        grp_fu_410_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_411_ce = 1'b0;
    end else begin
        grp_fu_411_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_412_ce = 1'b0;
    end else begin
        grp_fu_412_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_417_ce = 1'b0;
    end else begin
        grp_fu_417_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_418_ce = 1'b0;
    end else begin
        grp_fu_418_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_419_ce = 1'b0;
    end else begin
        grp_fu_419_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_420_ce = 1'b0;
    end else begin
        grp_fu_420_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_422_ce = 1'b0;
    end else begin
        grp_fu_422_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_423_ce = 1'b0;
    end else begin
        grp_fu_423_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_424_ce = 1'b0;
    end else begin
        grp_fu_424_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_425_ce = 1'b0;
    end else begin
        grp_fu_425_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_426_ce = 1'b0;
    end else begin
        grp_fu_426_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_428_ce = 1'b0;
    end else begin
        grp_fu_428_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_429_ce = 1'b0;
    end else begin
        grp_fu_429_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_430_ce = 1'b0;
    end else begin
        grp_fu_430_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_432_ce = 1'b0;
    end else begin
        grp_fu_432_ce = 1'b1;
    end
end

always @ (*) begin
    if (~(1'b1 == ap_ce)) begin
        grp_fu_433_ce = 1'b0;
    end else begin
        grp_fu_433_ce = 1'b1;
    end
end

assign OP1_V_10_cast_fu_3695_p1 = $signed(data_11_V_read);

assign OP1_V_12_cast_fu_3732_p1 = $signed(data_13_V_read);

assign ap_return = (tmp30_fu_4439_p2 + tmp15_fu_4431_p2);

assign grp_fu_402_p1 = ap_const_lv28_FFFFEB6;

assign grp_fu_403_p1 = ap_const_lv28_FFFFE73;

assign grp_fu_404_p1 = ap_const_lv28_FFFFED4;

assign grp_fu_405_p1 = ap_const_lv27_F2;

assign grp_fu_408_p1 = ap_const_lv27_BF;

assign grp_fu_409_p1 = ap_const_lv27_7FFFF46;

assign grp_fu_410_p1 = ap_const_lv26_5C;

assign grp_fu_411_p1 = ap_const_lv27_9F;

assign grp_fu_412_p1 = ap_const_lv26_3FFFF97;

assign grp_fu_417_p1 = ap_const_lv26_3FFFF9F;

assign grp_fu_418_p1 = ap_const_lv25_32;

assign grp_fu_419_p1 = ap_const_lv26_4A;

assign grp_fu_420_p1 = ap_const_lv24_19;

assign grp_fu_422_p1 = ap_const_lv27_7FFFF4B;

assign grp_fu_423_p1 = ap_const_lv27_7FFFF52;

assign grp_fu_424_p1 = ap_const_lv27_7FFFF0B;

assign grp_fu_425_p1 = ap_const_lv28_FFFFEB3;

assign grp_fu_426_p1 = ap_const_lv28_FFFFE92;

assign grp_fu_428_p1 = ap_const_lv27_7FFFF71;

assign grp_fu_429_p1 = ap_const_lv28_FFFFE4A;

assign grp_fu_430_p1 = ap_const_lv24_1D;

assign grp_fu_432_p1 = ap_const_lv28_FFFFE87;

assign grp_fu_433_p1 = ap_const_lv27_7FFFF5C;

assign p_Val2_1_10_fu_3711_p2 = ($signed(p_shl4_cast_fu_3707_p1) + $signed(OP1_V_10_cast_fu_3695_p1));

assign p_Val2_1_12_fu_3754_p2 = ($signed(p_neg_fu_3748_p2) - $signed(OP1_V_12_cast_fu_3732_p1));

assign p_Val2_1_17_fu_3812_p2 = ($signed(ap_const_lv22_0) - $signed(p_shl2_cast_fu_3808_p1));

assign p_Val2_1_26_fu_3892_p2 = ($signed(p_shl_cast_fu_3876_p1) - $signed(p_shl1_cast_fu_3888_p1));

assign p_Val2_1_2_fu_3955_p2 = ($signed(p_shl11_cast_fu_3951_p1) - $signed(p_shl9_cast_fu_3940_p1));

assign p_Val2_1_4_fu_3614_p2 = ($signed(p_shl7_cast_fu_3598_p1) + $signed(p_shl8_cast_fu_3610_p1));

assign p_Val2_1_6_fu_3659_p2 = ($signed(p_shl5_cast_fu_3643_p1) + $signed(p_shl6_cast_fu_3655_p1));

assign p_Val2_1_fu_3564_p2 = ($signed(p_shl14_cast_fu_3560_p1) - $signed(p_shl12_cast_fu_3548_p1));

assign p_neg_fu_3748_p2 = ($signed(ap_const_lv21_0) - $signed(p_shl3_cast_fu_3744_p1));

assign p_shl10_fu_3540_p3 = {{data_0_V_read}, {ap_const_lv6_0}};

assign p_shl11_cast_fu_3951_p1 = $signed(p_shl11_fu_3944_p3);

assign p_shl11_fu_3944_p3 = {{ap_pipeline_reg_pp0_iter1_data_2_V_read_3_reg_4450}, {ap_const_lv6_0}};

assign p_shl12_cast_fu_3548_p1 = $signed(p_shl10_fu_3540_p3);

assign p_shl12_fu_3552_p3 = {{data_0_V_read}, {ap_const_lv2_0}};

assign p_shl14_cast_fu_3560_p1 = $signed(p_shl12_fu_3552_p3);

assign p_shl1_cast_fu_3888_p1 = $signed(p_shl1_fu_3880_p3);

assign p_shl1_fu_3880_p3 = {{data_28_V_read}, {ap_const_lv3_0}};

assign p_shl2_cast_fu_3808_p1 = $signed(p_shl2_fu_3800_p3);

assign p_shl2_fu_3800_p3 = {{data_19_V_read}, {ap_const_lv3_0}};

assign p_shl3_cast_fu_3744_p1 = $signed(p_shl3_fu_3736_p3);

assign p_shl3_fu_3736_p3 = {{data_13_V_read}, {ap_const_lv2_0}};

assign p_shl4_cast_fu_3707_p1 = $signed(p_shl4_fu_3699_p3);

assign p_shl4_fu_3699_p3 = {{data_11_V_read}, {ap_const_lv6_0}};

assign p_shl5_cast_fu_3643_p1 = $signed(p_shl5_fu_3635_p3);

assign p_shl5_fu_3635_p3 = {{data_6_V_read}, {ap_const_lv7_0}};

assign p_shl6_cast_fu_3655_p1 = $signed(p_shl6_fu_3647_p3);

assign p_shl6_fu_3647_p3 = {{data_6_V_read}, {ap_const_lv5_0}};

assign p_shl7_cast_fu_3598_p1 = $signed(p_shl7_fu_3590_p3);

assign p_shl7_fu_3590_p3 = {{data_4_V_read}, {ap_const_lv6_0}};

assign p_shl8_cast_fu_3610_p1 = $signed(p_shl8_fu_3602_p3);

assign p_shl8_fu_3602_p3 = {{data_4_V_read}, {1'b0}};

assign p_shl9_cast_fu_3940_p1 = $signed(p_shl9_fu_3933_p3);

assign p_shl9_fu_3933_p3 = {{ap_pipeline_reg_pp0_iter1_data_2_V_read_3_reg_4450}, {ap_const_lv9_0}};

assign p_shl_cast_fu_3876_p1 = $signed(p_shl_fu_3868_p3);

assign p_shl_fu_3868_p3 = {{data_28_V_read}, {ap_const_lv7_0}};

assign tmp15_fu_4431_p2 = (tmp23_reg_4736 + tmp16_reg_4731);

assign tmp16_fu_4317_p2 = (tmp20_fu_4311_p2 + tmp17_fu_4294_p2);

assign tmp17_fu_4294_p2 = (tmp19_fu_4289_p2 + tmp18_fu_4283_p2);

assign tmp18_fu_4283_p2 = ($signed(tmp_6_fu_4223_p1) + $signed(tmp_9_fu_4226_p1));

assign tmp19_fu_4289_p2 = ($signed(tmp_10_2_reg_4616) + $signed(tmp_12_fu_4229_p1));

assign tmp20_fu_4311_p2 = (tmp22_fu_4306_p2 + tmp21_fu_4300_p2);

assign tmp21_fu_4300_p2 = ($signed(tmp_15_fu_4232_p1) + $signed(tmp_18_fu_4235_p1));

assign tmp22_fu_4306_p2 = ($signed(tmp_21_fu_4238_p1) + $signed(tmp_10_7_reg_4631));

assign tmp23_fu_4357_p2 = (tmp27_fu_4351_p2 + tmp24_fu_4334_p2);

assign tmp24_fu_4334_p2 = (tmp26_fu_4328_p2 + tmp25_fu_4323_p2);

assign tmp25_fu_4323_p2 = ($signed(tmp_24_fu_4241_p1) + $signed(tmp_10_9_reg_4641));

assign tmp26_fu_4328_p2 = ($signed(tmp_27_fu_4244_p1) + $signed(tmp_30_fu_4247_p1));

assign tmp27_fu_4351_p2 = (tmp29_fu_4345_p2 + tmp28_fu_4340_p2);

assign tmp28_fu_4340_p2 = ($signed(tmp_10_s_reg_4651) + $signed(tmp_33_fu_4250_p1));

assign tmp29_fu_4345_p2 = ($signed(tmp_36_fu_4253_p1) + $signed(tmp_39_fu_4256_p1));

assign tmp30_fu_4439_p2 = (tmp38_reg_4751 + tmp31_fu_4435_p2);

assign tmp31_fu_4435_p2 = (tmp35_reg_4746 + tmp32_reg_4741);

assign tmp32_fu_4371_p2 = ($signed(tmp34_cast_fu_4368_p1) + $signed(tmp33_fu_4363_p2));

assign tmp33_fu_4363_p2 = ($signed(tmp_44_fu_4259_p1) + $signed(tmp_10_1_reg_4671));

assign tmp34_cast_fu_4368_p1 = $signed(tmp34_reg_4721);

assign tmp34_fu_4205_p2 = ($signed(tmp_47_cast_fu_4084_p1) + $signed(tmp_50_cast_fu_4097_p1));

assign tmp35_fu_4387_p2 = (tmp37_fu_4383_p2 + tmp36_fu_4377_p2);

assign tmp36_fu_4377_p2 = ($signed(tmp_49_fu_4262_p1) + $signed(tmp_51_fu_4265_p1));

assign tmp37_fu_4383_p2 = (tmp_10_3_reg_4686 + tmp_10_4_reg_4691);

assign tmp38_fu_4425_p2 = (tmp42_fu_4419_p2 + tmp39_fu_4404_p2);

assign tmp39_fu_4404_p2 = (tmp41_fu_4398_p2 + tmp40_fu_4393_p2);

assign tmp40_fu_4393_p2 = ($signed(tmp_10_5_reg_4696) + $signed(tmp_52_fu_4268_p1));

assign tmp41_fu_4398_p2 = ($signed(tmp_54_fu_4271_p1) + $signed(tmp_56_fu_4274_p1));

assign tmp42_fu_4419_p2 = ($signed(tmp44_cast_fu_4416_p1) + $signed(tmp43_fu_4410_p2));

assign tmp43_fu_4410_p2 = ($signed(tmp_58_fu_4277_p1) + $signed(tmp_60_fu_4280_p1));

assign tmp44_cast_fu_4416_p1 = $signed(tmp44_reg_4726);

assign tmp44_fu_4217_p2 = ($signed(tmp45_fu_4211_p2) + $signed(tmp_68_cast_fu_4201_p1));

assign tmp45_fu_4211_p2 = ($signed(tmp_10_15_cast_fu_4061_p1) + $signed(ap_const_lv16_FF86));

assign tmp_10_15_cast_fu_4061_p1 = $signed(ap_pipeline_reg_pp0_iter1_tmp_41_reg_4531);

assign tmp_12_fu_4229_p1 = $signed(tmp_11_reg_4621);

assign tmp_144_fu_4087_p4 = {{grp_fu_418_p2[ap_const_lv32_18 : ap_const_lv32_A]}};

assign tmp_145_fu_4191_p4 = {{grp_fu_430_p2[ap_const_lv32_17 : ap_const_lv32_A]}};

assign tmp_15_fu_4232_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_14_reg_4471);

assign tmp_18_fu_4235_p1 = $signed(tmp_17_reg_4626);

assign tmp_21_fu_4238_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_20_reg_4481);

assign tmp_24_fu_4241_p1 = $signed(tmp_23_reg_4636);

assign tmp_27_fu_4244_p1 = $signed(tmp_26_reg_4646);

assign tmp_30_fu_4247_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_29_reg_4506);

assign tmp_33_fu_4250_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_32_reg_4516);

assign tmp_36_fu_4253_p1 = $signed(tmp_35_reg_4656);

assign tmp_39_fu_4256_p1 = $signed(tmp_38_reg_4661);

assign tmp_44_fu_4259_p1 = $signed(tmp_43_reg_4666);

assign tmp_47_cast_fu_4084_p1 = $signed(ap_pipeline_reg_pp0_iter1_tmp_143_reg_4546);

assign tmp_49_fu_4262_p1 = $signed(tmp_48_reg_4676);

assign tmp_50_cast_fu_4097_p1 = $signed(tmp_144_fu_4087_p4);

assign tmp_51_fu_4265_p1 = $signed(tmp_50_reg_4681);

assign tmp_52_fu_4268_p1 = $signed(tmp_s_reg_4701);

assign tmp_54_fu_4271_p1 = $signed(tmp_53_reg_4706);

assign tmp_56_fu_4274_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_55_reg_4591);

assign tmp_58_fu_4277_p1 = $signed(tmp_57_reg_4711);

assign tmp_60_fu_4280_p1 = $signed(tmp_59_reg_4716);

assign tmp_68_cast_fu_4201_p1 = $signed(tmp_145_fu_4191_p4);

assign tmp_6_fu_4223_p1 = $signed(ap_pipeline_reg_pp0_iter2_tmp_5_reg_4456);

assign tmp_9_fu_4226_p1 = $signed(tmp_8_reg_4611);

endmodule //compute_layer_0_0
