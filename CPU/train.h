#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H


#include "sequential.h"

/*
    Description:
        after calling this, the out of last layer of seq will contain the final prediction. all the weights and bias associated with each layer will be updated by
        training on the inputs (bsxn_in) defined by inp and target label defined by targ. The training will loop through the same data n_epochs times.

    Caveat:
        The size of the array pointed by targ should be bsxn_out +1. The fist bsxnout elements are labels while targ[bsxnout] will contain the average sum of square error after taining.
*/
void train_cpu(Sequential_CPU & seq, float *inp, float *targ, int bs, int n_in,int n_out, int batch_idx,int epoch_idx,int log_interval,int tbs);

#endif
