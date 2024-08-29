#ifndef MIS_CUH
#define MIS_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <iostream>
#include <vector>

#include "cudaError.cuh"

#define BLOCKSIZE 256

__global__ void assignRandomLabels(float *labels, int numNodes, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        thrust::default_random_engine rng(seed + tid);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        labels[tid] = dist(rng);
    }
}

__global__ void findMIS(int *rowPtr, int *colInd, float *labels, int *d_MIS, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes && d_MIS[tid] == -1) { // Unprocessed vertex
        bool isCandidate = true;
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        for (int i = start; i < end; ++i) {
            int neighbor = colInd[i];
            if (labels[neighbor] < labels[tid]) {
                isCandidate = false;
                break;
            }
        }
        if (isCandidate) {
            d_MIS[tid] = 1;
            for (int i = start; i < end; ++i) {
                d_MIS[colInd[i]] = 0;
            }
        }
    }
}

void Maximal_Independent_Set(thrust::device_vector<int> &rowPtr, thrust::device_vector<int> &colInd, int numNodes, thrust::device_vector<int> &d_MIS) {
    thrust::device_vector<float> d_labels(numNodes);
    
    int blocksPerGrid = (numNodes + BLOCKSIZE - 1) / BLOCKSIZE;

    // Step 4-5: Assign random labels to each vertex
    assignRandomLabels<<<blocksPerGrid, BLOCKSIZE>>>(thrust::raw_pointer_cast(d_labels.data()), numNodes, time(0));
    CUDA_KERNEL_CHECK();
    CUDA_CALL(cudaDeviceSynchronize());

    // Step 10-23: Find MIS based on labels using CSR format
    findMIS<<<blocksPerGrid, BLOCKSIZE>>>(thrust::raw_pointer_cast(rowPtr.data()), thrust::raw_pointer_cast(colInd.data()), thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_MIS.data()), numNodes);
    CUDA_KERNEL_CHECK();
    
    CUDA_CALL(cudaDeviceSynchronize());
}

// int main() {
//     // Example graph: 4 nodes, CSR representation
//     int numNodes = 4;
//     std::vector<int> h_rowPtr = {0, 2, 5, 6, 8}; // Row pointers
//     std::vector<int> h_colInd = {1, 3, 0, 2, 3, 1, 0, 1}; // Column indices

//     thrust::device_vector<int> d_rowPtr = h_rowPtr;
//     thrust::device_vector<int> d_colInd = h_colInd;
//     thrust::device_vector<int> d_MIS(numNodes, -1); // Initialize with -1 (unprocessed)

//     // Call the Maximal Independent Set function using CSR
//     Maximal_Independent_Set_CSR(d_rowPtr, d_colInd, numNodes, d_MIS);

//     // Copy result back to host
//     thrust::host_vector<int> h_MIS = d_MIS;

//     // Print result
//     std::cout << "Maximal Independent Set: ";
//     for (int i = 0; i < numNodes; ++i) {
//         if (h_MIS[i] == 1) {
//             std::cout << i << " ";
//         }
//     }
//     std::cout << std::endl;

//     return 0;
// }

#endif