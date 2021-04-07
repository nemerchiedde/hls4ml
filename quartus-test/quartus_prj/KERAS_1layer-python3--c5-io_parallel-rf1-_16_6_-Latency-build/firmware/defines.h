#ifndef DEFINES_H_
#define DEFINES_H_

#ifdef __INTELFPGA_COMPILER__
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_LAYER_2 32
#define N_LAYER_4 1


//hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16, 6, true> model_default_t;
typedef ac_fixed<16, 6, true> input_t;
typedef ac_fixed<16, 6, true> layer2_t;
typedef ac_fixed<16, 6, true> layer3_t;
typedef ac_fixed<16, 6, true> layer4_t;
typedef ac_fixed<16, 6, true> result_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
