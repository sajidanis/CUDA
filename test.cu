#include <stdio.h>

#include <cuda_runtime.h>

__global__ void dkernel(){
    uid_t id = threadIdx.x;
    printf("Thread id %d\n", id);
}

int main() {
    dkernel<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}