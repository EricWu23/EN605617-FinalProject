#include <chrono>
#include <iostream>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "validate.h"
#include "../data/read_csv.h"

#define TOTAL_TRAINING_SAMPLE 60000
#define TRAING_BATCH_SIZE 600
#define NUM_OF_INPUT 784
#define NUM_OF_OUTPUT 10
#define NUM_OF_EPOCH 3
#define LOG_INTERVAL 20//print out training loss every LOG_INTERVAL number of batches

#define TOTAL_TESTING_SAMPLE 10000
#define TEST_BATCH_SIZE 100

inline void CUDAErrorCheck(cudaError_t err,const char * name){
 
    if(err!= cudaSuccess)
    {
      std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

int main(){

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
    std::chrono::steady_clock::time_point begin, end;

    int tbs = TOTAL_TRAINING_SAMPLE, n_in = NUM_OF_INPUT, n_epochs = NUM_OF_EPOCH;
    int n_out = NUM_OF_OUTPUT;

    float *inp, *targ;  
    cudaError_t err=cudaMallocManaged(&inp, tbs*n_in*sizeof(float));
    CUDAErrorCheck(err,"failed to allocate memory for input data");
    err=cudaMallocManaged(&targ, (tbs*n_out)*sizeof(float));
    CUDAErrorCheck(err,"failed to allocate memory for target data");

    /*------------------------- reading in training data------------------------*/
    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/train_x.csv",tbs*n_in);
    read_csv(targ, "../data/train_y.csv",tbs*n_out);
    end = std::chrono::steady_clock::now();
    std::cout << "Data (Training) reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    


    /*-------------- Define your Neural Network--------------------*/
    int bs=TRAING_BATCH_SIZE;
    int num_bs=tbs/bs;
    int n_hidden_1 = 512;
    //int n_hidden_2 = 512;
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden_1);
    ReLU_GPU* relu1 = new ReLU_GPU(bs,n_hidden_1);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden_1, n_out);
    std::vector<Module*> layers = {lin1, relu1,lin2};
    Sequential_GPU seq(layers);

    /*----------------Training:-----------------------------------*/
    int log_interval=min(LOG_INTERVAL,num_bs);
    begin = std::chrono::steady_clock::now();
    for(int epoch_idx = 0;epoch_idx<n_epochs;epoch_idx++){
                                 
      for(int batch_idx=0;batch_idx<num_bs;batch_idx++){
          train_gpu(seq,inp,targ,bs,n_in,n_out,batch_idx,epoch_idx,log_interval,tbs); 
        }

     }
    end = std::chrono::steady_clock::now();
                                     
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;



    /*----------------------Testing-------------------------*/ 
    int tbs_test = TOTAL_TESTING_SAMPLE;
    float *inp_test, *targ_test;  
    cudaMallocManaged(&inp_test, tbs_test*n_in*sizeof(float));
    cudaMallocManaged(&targ_test, (tbs_test*n_out)*sizeof(float));
/*
    // read in testing data
    begin = std::chrono::steady_clock::now();
    read_csv(inp_test, "../data/test_x.csv",tbs_test*n_in);
    read_csv(targ_test, "../data/test_y.csv",tbs_test*n_out);
    end = std::chrono::steady_clock::now();
    std::cout << "Data (Validation) reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    int bs_test = TEST_BATCH_SIZE;
    int num_batch_test = tbs_test/bs_test;
    begin = std::chrono::steady_clock::now();
    int correct=0;
    float testloss=0.0f;
     for(int batch_idx=0;batch_idx<num_batch_test;batch_idx++){
            validate_gpu(seq,inp_test,targ_test,bs_test,n_in,n_out,batch_idx,testloss,correct);
            std::cout << "Testing batch:"<< batch_idx << "| [finished size/testing size] : ["<< (batch_idx+1)*bs_test<<"/"<<tbs_test<< "]"<< std::endl;
     }
    std::cout << "Test Loss: " <<testloss/tbs_test<< std::endl;
    end = std::chrono::steady_clock::now();
    std::cout << "Validation time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
*/

    /*-------------------------Clean up---------------------------*/
    cudaFree(inp_test);
    cudaFree(targ_test);
    cudaFree(inp);
    cudaFree(targ);
    cudaDeviceReset();
    return 0;
}
