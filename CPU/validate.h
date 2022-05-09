#ifndef TEST_CPU_H
#define TEST_CPU_H


#include "sequential.h"

/*
  Descrition: 
    This function test the Neural Network defined by Sequential_GPU & seq. and print out the testing loss
    Inputs:
            Sequential_CPU & seq ----- sequential model defines the model of neural network under test
            inp                  ----- pointer to the bsxn_input array, which stores input data for a batch in the flat array format
            targ                 ----- pointer to the bsxn_out array, which stores labels for a batch in the flat array format
            bs                   ----- batch size
            n_in                 ----- number of input features into the Neural Network for per training sample
            n_out                ----- number of output units of the Neural Network
            batch_idx            ----- index for the test batch
            loss                 ----- accumuated loss. The loss computed from each test batch will be added on top of the loss passed in.
            correct              ----- accumulated correct number of estimation we made so far 
*/
void validate_cpu(Sequential_CPU & seq, float *inp, float *targ, int bs, int n_in,int n_out,int batch_idx,float &loss,int &correct);

#endif