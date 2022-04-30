#include <torch/extension.h>
#include "grayscaleimagenormalization.h"
/*  
    Interface function
    Set the relationship between python (PyTorch) and c/cpp (cuda)
*/
void torch_launch_grayscaleimagenormalization(torch::Tensor &c,
                       const torch::Tensor &a,
                       int64_t n) {
    launch_grayscaleimagenormalization((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                n);
}

/* 
    Use pybind11 to package the Interface function
    such that it can be compiled by cmake to generate
    a dynamic library that can be called by python
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_grayscaleimagenormalization",
          &torch_launch_grayscaleimagenormalization,
          "grayscaleimagenormalization kernel warpper");
}

TORCH_LIBRARY(grayscaleimagenormalization, m) {
    m.def("torch_launch_grayscaleimagenormalization", torch_launch_grayscaleimagenormalization);
}