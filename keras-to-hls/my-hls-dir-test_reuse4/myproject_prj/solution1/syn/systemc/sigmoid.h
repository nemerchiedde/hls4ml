// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2016.4
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

#ifndef _sigmoid_HH_
#define _sigmoid_HH_

#include "systemc.h"
#include "AESL_pkg.h"

#include "sigmoid_sigmoid_tlbW.h"

namespace ap_rtl {

struct sigmoid : public sc_module {
    // Port declarations 8
    sc_in_clk ap_clk;
    sc_in< sc_logic > ap_rst;
    sc_in< sc_logic > ap_start;
    sc_out< sc_logic > ap_done;
    sc_out< sc_logic > ap_idle;
    sc_out< sc_logic > ap_ready;
    sc_in< sc_lv<18> > data_V_read;
    sc_out< sc_lv<10> > ap_return;


    // Module declarations
    sigmoid(sc_module_name name);
    SC_HAS_PROCESS(sigmoid);

    ~sigmoid();

    sc_trace_file* mVcdFile;

    sigmoid_sigmoid_tlbW* sigmoid_table3_U;
    sc_signal< sc_lv<1> > ap_CS_fsm;
    sc_signal< sc_lv<1> > ap_CS_fsm_pp0_stage0;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter0;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter1;
    sc_signal< sc_logic > ap_enable_reg_pp0_iter2;
    sc_signal< sc_lv<10> > sigmoid_table3_address0;
    sc_signal< sc_logic > sigmoid_table3_ce0;
    sc_signal< sc_lv<10> > sigmoid_table3_q0;
    sc_signal< sc_lv<10> > tmp_7_fu_172_p1;
    sc_signal< sc_lv<10> > tmp_7_reg_203;
    sc_signal< sc_lv<4> > tmp_8_reg_208;
    sc_signal< sc_lv<64> > tmp_s_fu_198_p1;
    sc_signal< sc_lv<14> > tmp_1_fu_78_p4;
    sc_signal< sc_lv<4> > tmp_3_fu_100_p1;
    sc_signal< sc_lv<10> > p_Result_2_fu_104_p3;
    sc_signal< sc_lv<15> > ret_V_cast_fu_88_p1;
    sc_signal< sc_lv<1> > tmp_5_fu_112_p2;
    sc_signal< sc_lv<15> > ret_V_fu_118_p2;
    sc_signal< sc_lv<1> > tmp_2_fu_92_p3;
    sc_signal< sc_lv<15> > p_s_fu_124_p3;
    sc_signal< sc_lv<15> > p_2_fu_132_p3;
    sc_signal< sc_lv<14> > tmp_4_fu_140_p1;
    sc_signal< sc_lv<15> > index_fu_144_p2;
    sc_signal< sc_lv<1> > tmp_6_fu_156_p3;
    sc_signal< sc_lv<14> > index_cast_fu_150_p2;
    sc_signal< sc_lv<14> > p_1_fu_164_p3;
    sc_signal< sc_lv<1> > icmp_fu_186_p2;
    sc_signal< sc_lv<10> > index_1_fu_191_p3;
    sc_signal< sc_lv<1> > ap_NS_fsm;
    sc_signal< sc_logic > ap_pipeline_idle_pp0;
    static const sc_logic ap_const_logic_1;
    static const sc_logic ap_const_logic_0;
    static const sc_lv<1> ap_ST_fsm_pp0_stage0;
    static const sc_lv<32> ap_const_lv32_0;
    static const sc_lv<1> ap_const_lv1_1;
    static const sc_lv<32> ap_const_lv32_4;
    static const sc_lv<32> ap_const_lv32_11;
    static const sc_lv<6> ap_const_lv6_0;
    static const sc_lv<10> ap_const_lv10_0;
    static const sc_lv<15> ap_const_lv15_1;
    static const sc_lv<15> ap_const_lv15_200;
    static const sc_lv<14> ap_const_lv14_200;
    static const sc_lv<32> ap_const_lv32_E;
    static const sc_lv<14> ap_const_lv14_0;
    static const sc_lv<32> ap_const_lv32_A;
    static const sc_lv<32> ap_const_lv32_D;
    static const sc_lv<4> ap_const_lv4_0;
    static const sc_lv<10> ap_const_lv10_3FF;
    // Thread declarations
    void thread_ap_clk_no_reset_();
    void thread_ap_CS_fsm_pp0_stage0();
    void thread_ap_done();
    void thread_ap_enable_reg_pp0_iter0();
    void thread_ap_idle();
    void thread_ap_pipeline_idle_pp0();
    void thread_ap_ready();
    void thread_ap_return();
    void thread_icmp_fu_186_p2();
    void thread_index_1_fu_191_p3();
    void thread_index_cast_fu_150_p2();
    void thread_index_fu_144_p2();
    void thread_p_1_fu_164_p3();
    void thread_p_2_fu_132_p3();
    void thread_p_Result_2_fu_104_p3();
    void thread_p_s_fu_124_p3();
    void thread_ret_V_cast_fu_88_p1();
    void thread_ret_V_fu_118_p2();
    void thread_sigmoid_table3_address0();
    void thread_sigmoid_table3_ce0();
    void thread_tmp_1_fu_78_p4();
    void thread_tmp_2_fu_92_p3();
    void thread_tmp_3_fu_100_p1();
    void thread_tmp_4_fu_140_p1();
    void thread_tmp_5_fu_112_p2();
    void thread_tmp_6_fu_156_p3();
    void thread_tmp_7_fu_172_p1();
    void thread_tmp_s_fu_198_p1();
    void thread_ap_NS_fsm();
};

}

using namespace ap_rtl;

#endif
