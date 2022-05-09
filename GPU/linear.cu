#include "linear.h"
#include "../utils/utils.h"
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
void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        out[ind_out] = bias[col];

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;
            
            out[ind_out] += inp[ind_inp]*weights[ind_weights];
        }
    }
}


__global__
void linear_backward_gpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&inp[ind_inp], weights[ind_weights]*out[ind_out]);
        }
    }
}


__global__
void linear_update_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        atomicAdd(&bias[col], -lr*out[ind_out]);

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&weights[ind_weights], -lr*inp[ind_inp]*out[ind_out]);
        }
    }
}


Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr){
    model_type = linear;
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in*n_out;
    sz_out = bs*n_out;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;

    cudaMallocManaged(&weights, sz_weights*sizeof(float));
    cudaMallocManaged(&bias, n_out*sizeof(float));

    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}


void Linear_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
    
    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);
 /*   
    std::cout<<"n_block_rows: "<< n_block_rows<<std::endl;
    std::cout<<"n_block_cols: "<< n_block_cols<<std::endl;
    std::cout<<"block_size: "<< block_size<<std::endl;
    std::cout<<"n_out: "<< n_out<<std::endl;
    std::cout<<"n_in: "<< n_in<<std::endl;
    std::cout<<"bs: "<< bs<<std::endl;
*/
    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);

    cudaError_t err = cudaGetLastError();
    CUDAErrorCheck(err,"linear_forward_gpu launch failed");
    cudaDeviceSynchronize();
}


void Linear_GPU::backward(){
    init_zero(inp, bs*n_in);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);
    linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
    cudaError_t err = cudaGetLastError();
    CUDAErrorCheck(err,"linear_backward_gpu launch failed");
    cudaDeviceSynchronize();

    cudaFree(cp_weights);
}


void Linear_GPU::update(){
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    set_eq(cp_weights, weights, sz_weights);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);
    
    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaError_t err = cudaGetLastError();
    CUDAErrorCheck(err,"linear_update_gpu launch failed");
    cudaDeviceSynchronize();
}

void Linear_GPU::update_batchsize(int new_bs){
  if(new_bs!=bs){
    bs=new_bs;
    sz_out = bs*n_out;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;

  }

}