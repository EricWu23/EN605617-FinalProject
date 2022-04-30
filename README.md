# EN605617-FinalProject
Yujiang's final project of the course introduction to GPU programming at JHU

# Project Description
In this project, I 
1. wrote a kernel function in CUDA (grayscaleimagenormalization_kernel.cu)
2. wrote the host side caller (grayscaleimagenormalization_kernel.cu,launch_grayscaleimagenormalization.h)
3. wrote an python,C++ interfacing file (grayscaleimagenormalization_ops.cpp)
4. compiled our custom cuda_module using JIT option
5. normalized the pixel of raw gray scale image,which is between [0,255], to a number between [-1,1] by calling the custom cuda kernel from pytorch 

I also compared timing among pixel normalizer function implemented in 
1. pytorch called custom CUDA kernel, 
2. pure pytorch tensor running on CPU, 
3. pure pytorch sensor running on GPU.

Then, I trained a Neural network using pytorch to do dignit recognition based on the data preprocessed by the custom pixel normalizer defined in CUDA kernel.

## additional notes:
The data can be downloaded from http://yann.lecun.com/exdb/mnist/.

## References:
* [CUSTOM CUDA Example](https://github.com/godweiyang/NN-CUDA-Example) 
* [CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [MNIST Handwritten Digit Recognition in PyTorch](https://nextjournal.com/gkoehler/pytorch-mnist)
* [Neural network in CUDA/C++](https://github.com/BobMcDear/Neural-Network-CUDA)
* [pybind11 — Seamless operability between C++11 and Python](https://pybind11.readthedocs.io/en/stable/)
* [PyTorch C++ API](https://pytorch.org/cppdocs/api/library_root.html)

