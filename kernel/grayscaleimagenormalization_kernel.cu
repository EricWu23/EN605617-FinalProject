/* 
cuda kernel running on GPU that adds two vectors together
params:
    c - pointer to the memory location that store the nomalized results
    a - pointer to the memory location that store the original gray scale image with each pixel ranging from 0-255
    n - specify the length of the input array (the array that pointed by a and c needs to have the same length)
*/
#define SCALER 2.0/255
#define OFFSET -1.0
#define LOWER_BOUND -1.0
#define UPPER_BOUND 1.0 

__global__ void grayscaleimagenormalization_kernel(float* c,
                            const float* a,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = fmaxf(fminf(a[i]*SCALER + OFFSET,UPPER_BOUND),LOWER_BOUND);
    }
}

/*  host side kernel caller*/
void launch_grayscaleimagenormalization(float* c,
                 const float* a,
                 int n) {
    dim3 grid((n + 1023) / 1024);// for my device. n can't be larger than 2147483647
    dim3 block(1024);
    grayscaleimagenormalization_kernel<<<grid, block>>>(c, a, n);
}