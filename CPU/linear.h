#ifndef LINEAR_CPU_H
#define LINEAR_CPU_H


#include "../utils/module.h"

/*
Data Encapsulation:
    public data that can be accessed by anyone who uses our class Linear_CPU:
        weights     --- pointer to the array that stores the weight associated with the linear layer
        cp_weights  --- A deep copy of weights before it is updated by the gradient descent
        bias        --- pointer to the array that stores the bias associated with the linear layer
        lr          --- the learning rate for the gradient decent.
        bs          --- batch size (The number of samples that will be propagated through the Network)
        n_in        --- the number of inputs into the layer
        n_out       --- the number of outputs of the layer
        sz_weights  --- the size of the weight matrix associated with the layer viewed as 1D vector
 Functionality Encapsulation:
    public accessible functions:
        Linear_CPU() --- constructor to create a linear layer object whose weight matrix is initialized by Kaiming intialization, whose bias matrix is initialized to zero
        forward ()     --- update the outputs of the linear layer
        backward()    --- compute dJ/dX assuming dJ/dZ is given. recall for a linear layer Z=W*X+B
        update()       --- compute dJ/dW and dJ/dB and updates W and B using gradient decent
*/
class Linear_CPU: public Module{
    public:
        float *weights, *cp_weights, *bias, lr;
        int bs, n_in, n_out, sz_weights;
        
        Linear_CPU(int _bs, int _n_in, int _n_out, float _lr = 0.1f);
        void forward(float *_inp, float *_out);
        void backward();
        void update();
        void update_batchsize(int new_bs);
};


#endif

