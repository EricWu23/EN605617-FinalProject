#ifndef MODULE_H
#define MODULE_H

#define block_size 32

enum modeltype { linear, relu, other};
class Module{
    public:
        float *inp, *out;
        int sz_out;
        
        virtual void forward(float *inp, float *out){};
        virtual void backward(){};
        virtual void update(){};
        virtual void update_batchsize(int _bs){};
};


#endif
