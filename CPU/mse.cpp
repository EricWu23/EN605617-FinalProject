#include "mse.h"
#include <iostream>

/*
    Params:
        inp    ----- pointer to the array that stores the output layer's outputs
        out    ----- pointer to the array that stores the target labels
        sz_out ---- the size of the output. For 1 sample inputs, there will be n_out outputs. 
                    Thus, for bs, which is the batch size, there will be bs*n_out outputs
    Caveat: Notice that out is an array of size sz_out +1. out[sz_out] is used for storing the average square
            err

*/
void mse_forward_cpu(float *inp, float *out, int sz_out,float* const loss){
    for (int i=0; i<sz_out; i++){
        loss[0] += (inp[i]-out[i])*(inp[i]-out[i])/sz_out;// Average Square error
        //std::cout << " pred: " << inp[i];
        //std::cout << " label:" << out[i];
        //if (i%10==1){printf("\n");} 
    }
}

/*
    Description:
            Assuming J= sum (y_hat[i]-y[i])^2 / sz_out
            The dJ/dy_hat[i] = 2(y_hat[i]-y[i])/sz_out
            So this function computes the gradient of average square error against each output sample
    Params:
        inp    ----- pointer to the array that stores the output layer's outputs (model predictions)
        out    ----- pointer to the array that stores the target labels 
        sz_out ----- the size of the output. For 1 sample inputs, there will be n_out outputs. 
                     Thus, for bs, which is the batch size, there will be bs*n_out outputs
    Output:
        inp    ----- No explicit return values for this function. it is in-place operation. 
                     After calling this funciton, inp is the pointer to the array storing dJ/dAout
                     where J is average MSE and Aout is the matrix stores all the model predictions
    Caveat: 
            Unlike MSE itself, which returns a scalar, its backward pass gives n_out gradients 
            that will be stored in the array that contains the output. Notice how this step actually 
            does not require us to compute sum of square loss explicitly. 
*/
void mse_backward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
               //std::cout << "Yujiang: " << i << std::endl;
        inp[i] = 2*(inp[i]-out[i])/sz_out;//dJ/dY_hat
    }
}

void mse_free(float* &loss){
   delete[] loss;
}
/*  
    Description:
                This is just the class constructor that initialize sz_out based on user input.
*/
MSE_CPU::MSE_CPU(int _sz_out){
    sz_out = _sz_out;
    loss = new float[1];
    loss[0]=0.0f;
}

/*
    Description: 
                Stores the model prediction and target labels for backpropagation. 
                This function needs to be called before calling the backward propogation.
                Since we actually don't need to compute the loss for forward propogation.
                we can have a dummy function "forward" that simply stores the input and output 
                for backpropagation. We might use it if performance is very, very 
                important, and we don't want to waste time calculating the loss.

    Params:
        _inp    ----- pointer to the array that stores the output layer's outputs (model predictions)
        _out    ----- pointer to the array that stores the target labels 
    Outputs:
        No explicit return values for this function.
        After calling this function, inp stores the output layer's outputs (model predictions)
        and out stores the target labels 
*/
void MSE_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}

/*
    Description: This function is different from the dummy forward function. This one 
                 calls the kernel mse_forward_cpu which truly compute the square loss.

    Params:             
        _inp    ----- pointer to the 1-D array of size (bsxn_out) that stores the output layer's outputs
        _out    ----- pointer to the 1-D array of size (bsxn_out+1) that stores the target labels and with the last 
                      element being used to store the average square error loss.
 
*/
void MSE_CPU::_forward(float *_inp, float *_out){
    //_out[sz_out] = 0.0f;// zero out due to the way we use _out[sz_out] to store average square error
    loss[0]=0.0f;
    //std::cout << "size_out: " << sz_out << std::endl;
    mse_forward_cpu(_inp, _out, sz_out,loss);
}

/*
    Description: 
        This function calls the kernel mse_backward_cpu which computes the gradient of average square loss
        against each output

    Internal:
        Before calling:              
            inp    ----- pointer to the 1-D array of size (bsxn_out) that stores the output layer's outputs.

            out    ----- pointer to the 1-D array of size (bsxn_out+1) that stores the target labels and with the last 
                      element being used to store the average square error loss.
        After calling:
            inp    ----- pointer to the 1-D array of size (bsxn_out) that stores the gradient of average min square error against each output.

            out    ----- pointer to the 1-D array of size (bsxn_out+1) that stores the target labels and with the last 
                      element being used to store the average square error loss (not necessary, see difference between _forward and forward).
 
*/
void MSE_CPU::backward(){
    mse_backward_cpu(inp, out, sz_out);
}

void MSE_CPU::free(){
  mse_free(loss);
}