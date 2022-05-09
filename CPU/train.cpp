#include <iostream>

#include "mse.h"
#include "train.h"
#include "../utils/utils.h"

/*
    Description:
        train the neural network defined by seq using input data defined by inp, bs, n_in, and target label defined by targ. It trains on the same 
        data n_epochs times
    Params:
        Sequential_CPU seq ------ sequential model defines the neural network
        float *inp         ------ pointer to the array that contains all the inputs (bsxn_in) in size
        float *targ        ------ pointer to the array taht contains all the labels 
        int bs             ------ batch size
        int n_in           ------ number of inputs of the neural network
        int n_epochs       ------ number of times that we train by looping through the same data             


*/

void train_cpu(Sequential_CPU & seq, float *inp, float *targ, int bs, int n_in, int n_out, int batch_idx,int epoch_idx,int log_interval,int tbs){
    
    float* inp_shift=inp+batch_idx*bs*n_in;
    float* targ_shft=targ+batch_idx*bs*n_out;
    int sz_out = bs*n_out;    
    int sz_inp = bs*n_in;
    float *out;
    MSE_CPU mse(sz_out);

    seq.forward(inp_shift, out);// there will be temporary memory allocated by this function.
    mse.forward(seq.layers.back()->out, targ_shft);//dummy, needed to be called before mse.backward().
    mse.backward();//compute gradients
    seq.update();//update weights

    /* clean up temporary memory at the end of each batch*/
    seq.free();
   
   /*----------------Loss reporting----------------------*/
    if(batch_idx%log_interval==0){
      seq.forward(inp_shift, out);
      mse._forward(seq.layers.back()->out, targ_shft);// compute the actual loss
      seq.free();
      std::cout << "Training Epoch:"<< epoch_idx << "| [finished size/traing size] : ["<< (batch_idx+1)*bs<<"/"<<tbs<< "] ("<<
      (int)((batch_idx+1)*bs*100.0/tbs)<<"%) | Training Loss:"<< mse.loss[0] << std::endl;
    }
    mse.free();
}