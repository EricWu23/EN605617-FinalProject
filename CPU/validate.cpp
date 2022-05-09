#include "mse.h"
#include "validate.h"
#include "../utils/utils.h"
#include <iostream>

void validate_cpu(Sequential_CPU & seq, float *inp, float *targ, int bs, int n_in,int n_out,int batch_idx,float &loss,int &correct){
    float* inp_shift=inp+batch_idx*bs*n_in;
    float* targ_shft=targ+batch_idx*bs*n_out;
    int sz_out = bs*n_out;
    float *out;//dumy not updated.

    MSE_CPU mse(sz_out);

    seq.update_batchsize(bs);

    seq.forward(inp_shift,out);// there will be temporary memory allocated by this function that needs to be freed by calling seq.free()

    mse._forward(seq.layers.back()->out,targ_shft);// compute the actual loss

    loss=loss+mse.loss[0];// accumuate loss

    seq.free();
    mse.free();

}