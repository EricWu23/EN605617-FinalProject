#ifndef RELU_CPU_H
#define RELU_CPU_H


#include "../utils/module.h"
/*
    Data Encapsulation:
    public data that can be accessed by anyone who uses our class MSE_CPU:

    Functionality Encapsulation:
    public accessible functions:
        ReLU_CPU(int _sz_out);                   ---- class constructor      
        void forward(float *_inp, float *_out);  ---- updates the Relu layer outputs "_out".
        void backward();                         ---- Compute dJ/dX given dJ/dA.  A=phi(Z) with phi as the relu. 
    Caveat:
        The dJ/dA will be given by the backward() method of the layer that this relu layer connect to.    
*/
class ReLU_CPU: public Module{
    public:
        int bs,n_in, n_out;
        ReLU_CPU(int _bs,int _n_in);
        void forward(float *_inp, float *_out);
        void backward();
        void update_batchsize(int new_bs);
};


#endif
