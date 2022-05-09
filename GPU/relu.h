#ifndef RELU_GPU_H
#define RELU_GPU_H


#include "../utils/module.h"


class ReLU_GPU: public Module{
    public:
        int n_blocks;
        int bs,n_in,n_out;
        modeltype model_type;

        ReLU_GPU(int _bs,int _n_in);
        void forward(float *_inp, float *_out);
        void backward();
        void update_batchsize(int new_bs);
};


#endif
