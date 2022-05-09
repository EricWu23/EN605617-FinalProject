#ifndef TRAIN_GPU_H
#define TRAIN_GPU_H


#include "sequential.h"

/*
      inputs:
            seq         --- sequential model defining the neural network
            inp         --- pointer to the input array (bs*n_in) as a flatten format
            targ        --- pointer to the array (bs*n_out+1) which uses the first bs*n_out elements to store the labels and the last element for holding the loss for each batch
            bs          --- batch size for training (in samples)
            n_in        --- number of input features into the Neural Network
            n_out       --- number of output units in the output layer of the neural network
            batch_idx   --- the batch index in an epoch
            epoch_idx   --- the epoch index
            log_interval--- after "int log_interval" number of batches, print out training loss
            tbs         --- total training samples per epoch
*/
void train_gpu(Sequential_GPU & seq, float *inp, float *targ, int bs, int n_in,int n_out, int batch_idx,int epoch_idx,int log_interval,int tbs);


#endif
