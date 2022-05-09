#include "sequential.h"

/*
    Description:
            Starts with the 0-th layer up to the last layer in layers.
            For each layer, call its forward() method.
            After this call, the inp should be a pointer points to an array storing final outputs from
            the Network described by layers.
             
*/
void sequential_forward_cpu(float *inp, std::vector<Module*> layers, float *out){
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        curr_out = new float[sz_out];
        layer->forward(inp, curr_out);

        inp = curr_out;// current layer's output is next layer's input
    }
    
    curr_out = new float[1];// be aware how each layer reserved its pointer pointing to its input and output data array
    delete[] curr_out;// while we redirected the curr_out here and delete it such that there is no local pointer haning around.
}

/* 
    Description:
    This function starts with the final layer and works its way backwards.
    For each of layers, it will call update() to compute dJ/dW and dJ/dB and update the weigiht and bias 
    that will be used in forward propogation in next iteration, 
    then call backward() to compute the dJ/d_input given dJ/dout and the Weights before updated by update().
    See "calling sequence" in linear_update_cpu().
*/
void sequetial_update_cpu(std::vector<Module*> layers){
    for (int i=layers.size()-1; 0<=i; i--){
        Module *layer = layers[i];

        layer->update();
        layer->backward();
    }
}

void sequential_free(std::vector<Module*> layers){

  for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];
        delete[] layer->out;
    }

}

void sequential_update_bs(std::vector<Module*> layers,int bs){
    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];
        layer->update_batchsize(bs);
    }
}

/*
    Description:
        Class constructor. Initialize the object public data layers from the user input


*/
Sequential_CPU::Sequential_CPU(std::vector<Module*> _layers){
    layers = _layers;
}

/*
    Description:
            Call sequential_forward_cpu to do inference using the network defined by layers
    Params:
        inp ---- after this function call, this points to an array stores the network's outputs from output layer
        out ---- Dummy parameter, not used
    Internal:
        layers ---- layers defining the network.

*/
void Sequential_CPU::forward(float *inp, float *out){
    sequential_forward_cpu(inp, layers, out);
}

/*
    Description:
            This methods help to call update() and backward() of a specific layer starting from the last layer
            and works its way backwards. After this finised, we can assume that the weights and bias matrix associated
            with each layer has been correctly updated. And we are ready for another forward propogation using the 
            updated parameters.
*/
void Sequential_CPU::update(){
    sequetial_update_cpu(layers);
}

void Sequential_CPU::free(){
  sequential_free(layers);
}

void Sequential_CPU::update_batchsize(int _bs){
  sequential_update_bs(layers,_bs);
}