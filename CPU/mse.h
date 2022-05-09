#ifndef MSE_CPU_H
#define MSE_CPU_H


#include "../utils/module.h"

/*
    Data Encapsulation:
    public data that can be accessed by anyone who uses our class MSE_CPU:
        inp ---- pointer to the array that stores the model's predictions
        out ---- pointer to the array that stores the desired target value
    Functionality Encapsulation:
    public accessible functions:
        MSE_CPU(int _sz_out);              ---- class constructor      
        forward(float *_inp, float *_out)  ---- an internal function to store the model prediction and target labels for backpropagation
                                                and it needs to be called before backpropagation.
        _forward(float *_inp, float *_out) ---- the function that actually computes the MSE loss. Not necessarily needed for backpropagation
        void backward();                   ---- computes the gradient of average of sum of square error against each output sample. 
                                                This function needs to be called after forward() is called. 
*/
class MSE_CPU: public Module{
    public:
        float *inp, *out;
        float *loss;
        
        MSE_CPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void _forward(float *_inp, float *_out);
        void backward();
        void free();
};


#endif
