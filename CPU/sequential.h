#ifndef SEQUENTIAL_CPU_H
#define SEQUENTIAL_CPU_H


#include <vector>

#include "../utils/module.h"
/*
    Data Encapsulation:
    public data that can be accessed by anyone who uses our class Sequential_CPU:
         std::vector<Module*> layers; ---- a vector of pointers to modules. Each module is a layer object. Module is parent class to linear, relu,mse,etc.

    Functionality Encapsulation:
    public accessible functions:
         Sequential_CPU(std::vector<Module*> _layers); ---- constructor to store user input _layer into object public data "layers"
         void forward(float *inp, float *out);         ---- make prediction using the Neural network described by the _layers
         update()                                      ---- call this to update weights and bias associated with the Neural network described by the _layers. 
                                                            Once called, it is good to do another prediction using updated weights and bias
*/
class Sequential_CPU: public Module{
    public:
        std::vector<Module*> layers; 
        Sequential_CPU(std::vector<Module*> _layers);
        void forward(float *inp, float *out);
        void update();
        void free();
        void update_batchsize(int _bs);
};


#endif
