#ifndef DEFINES_H_
#define DEFINES_H_

#include <complex>
#ifndef __INTELFPGA_COMPILER__
#include "ref/ac_int.h"
#include "ref/ac_fixed.h"
#else
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_6 32
#define N_LAYER_10 32
#define N_LAYER_14 5


//hls-fpga-machine-learning insert layer-precision
typedef <16,6> model_default_t;
typedef <16,6> input_t;
typedef <16,6> layer2_t;
typedef <16,6> layer5_t;
typedef <16,6> layer6_t;
typedef <16,6> layer9_t;
typedef <16,6> layer10_t;
typedef <16,6> layer13_t;
typedef <16,6> layer14_t;
typedef <16,6> result_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif