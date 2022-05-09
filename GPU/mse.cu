#include "mse.h"

#include <iostream>

inline void CUDAErrorCheck(cudaError_t err,const char * name){
 
    if(err!= cudaSuccess)
    {
      std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

__global__
void mse_forward_gpu(float *inp, float *out, int sz_out,float* loss){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        atomicAdd(&loss[0], fdividef(powf(inp[ind]-out[ind], 2), sz_out));
    }
}


__global__
void mse_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        inp[ind] = fdividef(2*(inp[ind]-out[ind]), sz_out);
    }
}

void mse_free(float* &loss){
  cudaFree(loss);
}

MSE_GPU::MSE_GPU(int _sz_out){
    model_type = other;
    sz_out = _sz_out;
    cudaMallocManaged(&loss, sizeof(float));
    loss[0]=0.0f;
    n_blocks = (sz_out + block_size - 1) / block_size;
}


void MSE_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}


void MSE_GPU::_forward(float *_inp, float *_out){
    //_out[sz_out] = 0.0f;
    loss[0]=0.0;
    mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out,loss);
    cudaError_t err = cudaGetLastError();
    CUDAErrorCheck(err,"mse_forward_gpu launch failed");
    cudaDeviceSynchronize();
}


void MSE_GPU::backward(){
    mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaError_t err = cudaGetLastError();
    CUDAErrorCheck(err,"mse_backward_gpu launch failed");
    cudaDeviceSynchronize();
}

void MSE_GPU::free(){
  mse_free(loss);
}