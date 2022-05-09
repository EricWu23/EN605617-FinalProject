#include "sequential.h"
#include "../utils/utils.h"


void sequential_forward_gpu(float *inp, std::vector<Module*> layers, float *out){
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        cudaMallocManaged(&curr_out, sz_out*sizeof(float));//temporary mem for storing the current layer's forward outpt. Notice that due to the implementation, these temporary memory can't be freed until we finish backward propogation.
        layer->forward(inp, curr_out);//linear.forward does not modify inp, it only stores inp inside linear.inp. Similarly, curr_out is stored in linear.out

        inp = curr_out;// the output of current layer (curr_out) is the input of next layer. Question: does modify inp changes the pointer beinng orignally pass in this function? No. It is just a copied pointer of the pointer that was passed in as argument.
    }
    // kill the curr_out pointer
    cudaMallocManaged(&curr_out, sizeof(float));
    cudaFree(curr_out);
}


void sequetial_update_gpu(std::vector<Module*> layers){
    for (int i=layers.size()-1; 0<=i; i--){
        Module *layer = layers[i];

        layer->update(); 
        layer->backward();
    }
}

void sequential_free(std::vector<Module*> layers){

  for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];
        
        cudaFree(layer->out);
    }

}

void sequential_update_bs(std::vector<Module*> layers,int bs){
    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];
        layer->update_batchsize(bs);
    }
}

Sequential_GPU::Sequential_GPU(std::vector<Module*> _layers){
    layers = _layers;
}


void Sequential_GPU::forward(float *inp, float *out){
    sequential_forward_gpu(inp, layers, out);
}


void Sequential_GPU::update(){
    sequetial_update_gpu(layers);
}

void Sequential_GPU::free(){

  sequential_free(layers);

}

void Sequential_GPU::update_batchsize(int _bs){
  sequential_update_bs(layers,_bs);
}