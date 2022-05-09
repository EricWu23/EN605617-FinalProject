#include "linear.h"
#include "../utils/utils.h"

/*
Description:
            This function updates the outputs of the linear layer based on the inputs, weights, and bias given.
            After calling, it updates the array that "out" points to.
Params:
    inp     ---- poniter to the 1-D array (bsxn_in) stores all the input for the batch 
    weights ---- pointer to the 1-D array (n_inxn_out) that stores the weight associated with the linear layer
    bias    ---- pointer to the array that stores the bias (n_out) associated with the linear layer
    out     ---- pointer to the output matrix viewed as the flat vector (bsxn_out)
    bs      ---- batch size (The number of samples that will be propagated through the Network)
    n_in    ---- the number of inputs into the layer
    n_out   ---- the number of outputs of the layer
Internal:
    ind_out     ---- "global" index of the output considering both batch size and number of output
    ind_inp     ---- "global" input index considering both number of input unit and the index in the batch 
    ind_weight  ---- "global" index of the weight matrix viewed as flat vector, considering both n_in and the n_output

*/
void linear_forward_cpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){
    int ind_inp, ind_weights, ind_out;

    for (int i=0; i<bs; i++){
        for (int k=0; k<n_out; k++){
            ind_out = i*n_out + k;
            out[ind_out] = bias[k];
            
            for (int j=0; j<n_in; j++){
                ind_inp = i*n_in + j;
                ind_weights = j*n_out + k;
                
                out[ind_out] += inp[ind_inp]*weights[ind_weights];
            }
        }
    }
}

/*
Description:
            This function computes dJ/dX given dJ/dZ
            while conceptually X is the input, Z is the output, J is the loss
Params:
    inp     ---- poniter to the array stores all the gradient of loss with respect to layer's input (bsxn_in)
    weights ---- pointer to the array that stores the weight associated with the linear layer. 
                 Notice this is the weight that was used during forward propogation before update()
    out     ---- pointer to the array that contains gradient of the loss with respect to the layer's output (bsxn_out)
    bs      ---- batch size (The number of samples that will be propagated through the Network)
    n_in    ---- the number of inputs into the layer
    n_out   ---- the number of outputs of the layer
Internal:
    ind_out     ---- "global" index of the output considering both batch size and number of output
    ind_inp     ---- "global" input index considering both number of input unit and the index in the batch 
    ind_weight  ---- "global" index of the weight matrix viewed as flat vector, considering both n_in and the n_output
Caveat:
    This function only computes dJ/dX, not the dJ/dW. 
    And after calling this function, inp stores the updated dJ/dX of this layer, which is 
    dJ/dZ of the previous layer. 
*/
void linear_backward_cpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out){
    int ind_inp, ind_weights, ind_out;

    for (int i=0; i<bs; i++){
        for (int k=0; k<n_out; k++){
            ind_out = i*n_out + k;
            
            for (int j=0; j<n_in; j++){
                ind_inp = i*n_in + j;
                ind_weights = j*n_out + k;
                
                inp[ind_inp] += weights[ind_weights]*out[ind_out];// dJ/dX=dZ/dX*dJ/dZ=W*dJ/Z
            }
        }
    }

}
/*
Description:
            This function uses the gradient descent algorithm to update bias matrix and weight matrix.    
Params:
    inp     ---- poniter to the array stores all the gradient of loss with respect to layer's input (bsxn_in)
    weights ---- pointer to the array that stores the weight associated with the linear layer
    out     ---- pointer to the array that contains gradient of the loss with respect to the layer's output (bsxn_out)
    bs      ---- batch size (The number of samples that will be propagated through the Network)
    n_in    ---- the number of inputs into the layer
    n_out   ---- the number of outputs of the layer
Internal:
    ind_out     ---- "global" index of the output considering both batch size and number of output
    ind_inp     ---- "global" input index considering both number of input unit and the index in the batch 
    ind_weight  ---- "global" index of the weight matrix viewed as flat vector, considering both n_in and the n_output
Caveat:
    This function will update the bias matrix and weights matrix after being called.
    Notice the dJ/dB and dJ/dW are computed inside this function instaed of the backward().

    Notice how conceptually, the dJ/dZ is the same as the dJ/dB. Thus, can be directly used to update the bias.

    The calling sequence is 
    1. mse.backward(); computes the dJ/dAout for the output layer.
        where J is the average of sum square error loss
              Aout is the activated outputs of the output layer 
    2.1 Then relu layer's update() will do nothing because it does not need to update any params
    2.2 Then relu layer's backward() will compute dJ/dZ using dJ/dAout, where Z is input to the relue and Aout is output of the relu
    3.1 the linear layer's update() will 
        compute dJ/dB using dJ/dZ , considering Z=X*W+B, dJ/dB=dJ/dZ*(dZ/dB)=dJ/dZ.  and update B. 
        where 
            B is the bias matrix associated with this linear layer
            Z is the outputs of this linear layer (and the input to the relu layer it is connected to)
        compute dJ/dW using dJ/dZ and X and update W. 
        where W is the weights associated with this linear layer,
              X is the inputs into this linear layer,
              
    3.2 the linear layer's backward() will compute dJ/dX using the dJ/dZ and the old W (before being updated by step 3.1)
    .....

    The mean reason that we have to call update() method of each layer before we call the backward() is because 
    update() requires the inp points to the inputs of the layer. But after backward() is called, inp will be the 
    dJ/dX of the layer.

*/
void linear_update_cpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr){
    int ind_inp, ind_weights, ind_out;
    
    for (int i=0; i<bs; i++){
        for (int k=0; k<n_out; k++){
            ind_out = i*n_out + k;
            bias[k] -= lr*out[ind_out];//Z=X*W+B. dZ/dB=1, dJ/dB=dJ/dZ *dZ/dB=dJ/dZ. So "out" contains the dJ/dZ            
            
            for (int j=0; j<n_in; j++){
                ind_inp = i*n_in + j;
                ind_weights = j*n_out + k;
                
                weights[ind_weights] -= lr*inp[ind_inp]*out[ind_out];//dJ/dW=dZ/dW*dJ/dZ=X*dJ/dZ. so the "inp" is just the input
            }
        }
    }
}

/*
    Description:
                This is the constructor for the Class Linear_CPU
                After calling this function, an object with several public data will be created
    Params:
            _bs     ---- batch size (The number of samples that will be propagated through the Network)
            _n_in   ---- the number of inputs into the layer. (The number of output units of the previous layer)
            _n_out  ---- the number of outputs of the layer (The number of inputs into the next layer if exists)
            _lr     ---- learning rate used in gradient descent algorithm
    

*/
Linear_CPU::Linear_CPU(int _bs, int _n_in, int _n_out, float _lr){
    // update class public data
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;
    
    sz_weights = n_in*n_out;
    sz_out = bs*n_out;// class Module is the parent of class Linear_CPU 
    // allocate memory for the weights matrix and bias matrix associated with this layer
    weights = new float[sz_weights];
    bias = new float[n_out]; 
    // initialize the weights matrix using Kaiming intialization
    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);// intialize bias matrix with zero
}

/*
    Description: 
                This function calls the kernel "linear_forward_cpu" which updates the outputs of linear layer
                in-place (out) based on the inp, weights,bias that passed in.
                also, it uses the public object data inp and out to store the inputs _inp and outputs passed in.
     Params:           
                _inp ---- poniter to the 1-D array (bsxn_in) stores all the inputs for the batch 
                _out ---- poniter to the 1-D array (bsxn_out) stores all the outputs for the batch 
*/
void Linear_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    linear_forward_cpu(inp, weights, bias, out, bs, n_in, n_out);//update out
}

/*
    Description:
            This function zero out the inp. Then call linear_backward_cpu() which 
            uses the cp_weight (weight matrix before updated) and out (dJ/dZ) to 
            compute inp (dJ/dX).
            Then it delete the cp_weights because we don't need it anymore after computing
            dJ/dX.

*/
void Linear_CPU::backward(){
    // Zero out "inp" because the gradients are
    // added to it; they don't replace its elements.
    init_zero(inp, bs*n_in);

    linear_backward_cpu(inp, cp_weights, out, bs, n_in, n_out);

    delete[] cp_weights;
}
/*
    Description:
        This function calls linear_update_cpu() to update weights and bias matrix based on
        the inputs inp and the gradient of loss against layer output out
    Params:
            None
    Internal:
        cp_weights  --- A deep copy of weights before it is updated by the gradient descent
    Caveat:
        for each layer, we will call update() first to figure out weights and bias that 
        will be used in forward progogation in the next iteration.
        Then call backward() to figure out dJ/dZ that will be used for update() in the previous layer.
        The backward() needs the weights before current layer's update() is called. Thus, we have to do a deep
        copy before the weight is updated. See "calling sequence" in linear_update_cpu().
*/
void Linear_CPU::update(){
    cp_weights = new float[n_in*n_out];
    set_eq(cp_weights, weights, sz_weights);

    linear_update_cpu(inp, weights, bias, out, bs, n_in, n_out, lr);
}

void Linear_CPU::update_batchsize(int new_bs){
  if(new_bs!=bs){
    bs=new_bs;
    sz_out = bs*n_out;
  }
}