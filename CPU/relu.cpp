#include "relu.h"

/*
    Description: 
                This is the actual kernel computes the output of relu for each input sample.
    Params:
        inp     --- pointer to the 1-D array that is sz_out in size and stores the input into the relu layer.
        out     --- pointer to the 1-D array that is sz_out in size and stores the output of the relu for each input it receives
        sz_out  --- This will be the same as the size of output of the linear layer (bs*n_out)

*/
void relu_forward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        out[i] = (0 < inp[i]) ? inp[i] : 0;
    }
}

/*
    Description:
                The actual kernel that computes the dJ/dZ =dA/dZ*dJ/dA,
                where A(Z) is the output of relu activated output
                Z is the input into the relu
                dA/dZ is the derivative of relu output against each input. Notice
                here all the operations are elementwise 

                after calling, the inp will point to an array with updated dJ/dZ of the current layer.
                which will be the dJ/dZ to the previous linear layer's backward method to compute dJ/dX 

    sz_out ---- the size of the output. For 1 sample inputs, there will be n_out outputs. 
                Thus, for bs, which is the batch size, there will be bs*n_out outputs
*/
void relu_backward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        inp[i] = (0 < inp[i]) * out[i];//dJ/dZ=dA/dZ*dJ/dA
    }
}

/*
    Description:
                The constructor of Class ReLu_CPU which save user input _sz_out

*/
ReLU_CPU::ReLU_CPU(int _bs,int _n_in){
    bs=_bs;
    n_in = _n_in;
    n_out = _n_in;
    sz_out = bs*n_out;
}

/*
    Description:
                This function calls the actual kernel relu_forward_cpu which computes
                the output of relu for each input it is given.
                After calling, we should expect the array that _out pointed to  
                to be updated with the Relu outputs 
*/
void ReLU_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    relu_forward_cpu(inp, out, sz_out);
}

/* 
    Description:
                This function just calls the actual kernel relu_backward_cpu to compute 
                back propagate the relu layer.
*/
void ReLU_CPU::backward(){
    relu_backward_cpu(inp, out, sz_out);
}

void ReLU_CPU::update_batchsize(int new_bs){
  if(new_bs!=bs){
    bs=new_bs;
    sz_out = bs*n_out;
  }
}