#ifndef PARAMINCLUDED
#define PARAMINCLUDED


//#include "weights/weight.h"
#include "nnet_utils/nnet_activation.h"
//

typedef ac_fixed<16,6,true> fixed_p;

struct lstm_config : public nnet::activ_config{
 static const unsigned n_in=4;
 static const unsigned n_timestamp=10;
 static fixed_p inputs [n_timestamp];
};

fixed_p lstm_config::inputs [lstm_config::n_timestamp]= {-0.00838574,  -0.00209644, -0.00838574,  0.00209644, 0.01467505, -0.01257862, 0.02515723,  0.5660378,  0.8679246,  0.6687631 };
#endif
