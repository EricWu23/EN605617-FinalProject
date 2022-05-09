#include <chrono>
#include <iostream>
#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"
#include "validate.h"

#define TOTAL_TRAINING_SAMPLE 60000
#define TRAING_BATCH_SIZE 600
#define NUM_OF_INPUT 784
#define NUM_OF_OUTPUT 10
#define NUM_OF_EPOCH 3
#define LOG_INTERVAL 20//print out training loss every LOG_INTERVAL number of batches

#define TOTAL_TESTING_SAMPLE 10000
#define TEST_BATCH_SIZE 100

int main(){
    std::chrono::steady_clock::time_point begin, end;

    int tbs = TOTAL_TRAINING_SAMPLE, n_in = NUM_OF_INPUT, n_epochs = NUM_OF_EPOCH;
    int n_hidden = n_in/2;
    int n_out = NUM_OF_OUTPUT;

    /*------------------------- reading in training data------------------------*/
    float *inp = new float[tbs*n_in], *targ = new float[tbs*n_out];    
    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/train_x.csv",tbs*n_in);
    read_csv(targ, "../data/train_y.csv",tbs*n_out);
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    
    /*-------------------------- Define your Neural Network--------------------*/
    int bs=std::min(TRAING_BATCH_SIZE,tbs);
    int num_bs=tbs/bs;
    int n_hidden_1 = 512;
    //int n_hidden_2 = 512;
    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden_1);
    ReLU_CPU* relu1 = new ReLU_CPU(bs,n_hidden_1);
    //Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden_1, n_hidden_2);
    //ReLU_CPU* relu2 = new ReLU_CPU(bs*n_hidden_2);
    //Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden_2, n_out);
    //std::vector<Module*> layers = {lin1, relu1, lin2, relu2, lin3};
    Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden_1, n_out);
    std::vector<Module*> layers = {lin1, relu1,lin3};
    Sequential_CPU seq(layers);

    /*----------------Training:-----------------------------------*/
    int log_interval=std::min(LOG_INTERVAL,num_bs);
    begin = std::chrono::steady_clock::now();
  
    for(int epoch_idx = 0;epoch_idx<n_epochs;epoch_idx++){
                                 
      for(int batch_idx=0;batch_idx<num_bs;batch_idx++){
          train_cpu(seq,inp,targ,bs,n_in,n_out,batch_idx,epoch_idx,log_interval,tbs);
        }

     }
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    /*----------------------Testing-------------------------*/ 
    int tbs_test = TOTAL_TESTING_SAMPLE;

    // read in testing data
    float *inp_test = new float[tbs_test*n_in], *targ_test = new float[tbs_test*n_out]; 
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
            validate_cpu(seq,inp_test,targ_test,bs_test,n_in,n_out,batch_idx,testloss,correct);
            std::cout << "Testing batch:"<< batch_idx << "| [finished size/testing size] : ["<< (batch_idx+1)*bs_test<<"/"<<tbs_test<< "]"<< std::endl;
     }
    std::cout << "Test Loss:" <<testloss/tbs_test<< std::endl;
    end = std::chrono::steady_clock::now();
    std::cout << "Validation time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;
    /*-------------------------Clean up---------------------------*/
    delete[] inp_test;
    delete[] targ_test;
    delete[] inp;
    delete[] targ;
    return 0;
}
