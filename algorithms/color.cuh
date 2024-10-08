#ifndef COLOR_CUH
#define COLOR_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "cudaError.cuh"

__global__ void initialColoring(int* rowPtr, int* colInd, int* colors, int* U, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices && U[idx] == 1) {
        int forbiddenColors[256] = {0};  

        // Mark the colors of the neighbors as forbidden
        for (int i = rowPtr[idx]; i < rowPtr[idx + 1]; i++) {
            int neighbor = colInd[i];
            int color = colors[neighbor];
            if (color != -1) {
                forbiddenColors[color] = 1;
            }
        }

        // Assign the minimum available color
        for (int color = 0; color < 256; color++) {
            if (!forbiddenColors[color]) {
                colors[idx] = color;
                break;
            }
        }
    }
}

__global__ void resolveConflicts(int* rowPtr, int* colInd, int* colors, int* U, int* nextU, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices && U[idx] == 1) {
        bool conflict = false;
        int forbiddenColors[256] = {0};

        // Check for conflicts
        for (int i = rowPtr[idx]; i < rowPtr[idx + 1]; i++) {
            int neighbor = colInd[i];
            if (colors[idx] == colors[neighbor] && idx < neighbor) {
                conflict = true;
            }
        }

        if (conflict) {
            // Reassign color
            for (int i = rowPtr[idx]; i < rowPtr[idx + 1]; i++) {
                int neighbor = colInd[i];
                int color = colors[neighbor];
                if (color != -1) {
                    forbiddenColors[color] = 1;
                }
            }

            for (int color = 0; color < 256; color++) {
                if (!forbiddenColors[color]) {
                    colors[idx] = color;
                    break;
                }
            }

            nextU[idx] = 1;  // Mark for further conflict checking
        } else {
            nextU[idx] = 0;  // No conflict, remove from U
        }
    }
}

__global__ void updateU(int* U, int* nextU, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices) {
        U[idx] = nextU[idx];
        nextU[idx] = 0;
    }
}

void graph_coloring(std::vector<int> &offsets, std::vector<int> &colInd, int numVertices) {
    thrust::host_vector<int> h_rowPtr = offsets;
    thrust::host_vector<int> h_colInd = colInd;


    auto start = std::chrono::high_resolution_clock::now(); // for timing


    // Allocate device memory
    thrust::device_vector<int> d_rowPtr = h_rowPtr;
    thrust::device_vector<int> d_colInd = h_colInd;
    thrust::device_vector<int> d_colors(numVertices, -1);  // Initialize colors to -1 (uncolored)
    thrust::device_vector<int> d_U(numVertices, 1);        // All vertices are initially in U
    thrust::device_vector<int> d_nextU(numVertices, 0);

    auto end = std::chrono::high_resolution_clock::now();
    
    long double copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    copy_time *= 1e-6;

    int blockSize = 256;
    int numBlocks = (numVertices + blockSize - 1) / blockSize;


    start = std::chrono::high_resolution_clock::now();

    // Initial coloring
    initialColoring<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colInd.data()), thrust::raw_pointer_cast(d_colors.data()), thrust::raw_pointer_cast(d_U.data()), numVertices);
    CUDA_CALL(cudaDeviceSynchronize());

    // Resolve conflicts iteratively
    while (thrust::reduce(d_U.begin(), d_U.end()) > 0) {
        resolveConflicts<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colInd.data()), thrust::raw_pointer_cast(d_colors.data()), thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_nextU.data()), numVertices);
        CUDA_CALL(cudaDeviceSynchronize());

        // Update U
        updateU<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_nextU.data()), numVertices);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    end = std::chrono::high_resolution_clock::now();

    long double running_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    running_time *= 1e-6;

    // Copy the result back to host and print
    // thrust::host_vector<int> h_colors = d_colors;

    // int limit = min(numVertices, 40);
    // std::cout << "Vertex Colors: ";
    // for (int i = 0; i < limit; ++i) {
    //     std::cout << h_colors[i] <<  " ";
    // }

    // Print Statistics
    std::cout << "\n";
    std::cout << "Copy Time : " << copy_time << " ms\n";
    std::cout << "Running Time : " << running_time << " ms\n";
    std::cout << "\n";
}

#endif