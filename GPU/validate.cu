#include <iostream>

#include "mse.h"
#include "validate.h"
#include "../utils/utils.h"


void validate_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in,int n_out,int batch_idx,float &loss,int &correct){
    float* inp_shift=inp+batch_idx*bs*n_in;
    float* targ_shft=targ+batch_idx*bs*n_out;
    int sz_out = bs*n_out;
    MSE_GPU mse(sz_out);
    float *out;//dumy not updated.

    seq.update_batchsize(bs);
    seq.forward(inp_shift,out);
  
    mse._forward(seq.layers.back()->out,targ_shft);// compute the actual loss
    
    /* To DO: implement a GPU CUDA kenrel to find out the index of maxmum element in every 10 outputs (one-hot encoding converts back to scalar)

        If using the CPU version max_element_index(arr,10,result);, it will greatly slow down the testing process.

        The reason to implement max_element_index is to give us ability to count the correct prediction and compute the accuracy. 
    */
    loss=loss+mse.loss[0];
    seq.free();
    mse.free();
}
  