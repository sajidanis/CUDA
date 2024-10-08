#ifndef BFS_CUH
#define BFS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#include "cudaError.cuh"


__global__ void bfs_kernel(int *adjacencyList, int *offsets, int *levels, int *frontier, int *new_frontier, int numNodes, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && frontier[idx]) {
        int start = offsets[idx];
        int end = offsets[idx + 1];
        for (int i = start; i < end; i++) {
            int neighbor = adjacencyList[i];
            if (atomicCAS(&levels[neighbor], -1, level) == -1) {
                new_frontier[neighbor] = 1;
            }
        }
    }
}

__global__ void check_empty_frontier(int *frontier, int numNodes, bool *d_done) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && frontier[idx]) {
        *d_done = false;
    }
}

int* bfs_cuda(int *adjacencyList, int *offsets, int numNodes, int startNode) {
    int *d_adjacencyList, *d_offsets, *d_levels, *d_frontier, *d_new_frontier;
    bool *d_done;
    size_t adjListSize = offsets[numNodes] * sizeof(int);
    size_t offsetsSize = (numNodes + 1) * sizeof(int);
    size_t levelsSize = numNodes * sizeof(int);

    std::vector<int> levels(numNodes, -1);
    levels[startNode] = 0;

    std::vector<int> frontier(numNodes, 0);
    frontier[startNode] = 1;

    auto start = std::chrono::high_resolution_clock::now(); // for timing

    CUDA_CALL(cudaMalloc(&d_adjacencyList, adjListSize));
    CUDA_CALL(cudaMalloc(&d_offsets, offsetsSize));
    CUDA_CALL(cudaMalloc(&d_levels, levelsSize));
    CUDA_CALL(cudaMalloc(&d_frontier, levelsSize));
    CUDA_CALL(cudaMalloc(&d_new_frontier, levelsSize));
    CUDA_CALL(cudaMalloc(&d_done, sizeof(bool)));

    CUDA_CALL(cudaMemcpy(d_adjacencyList, adjacencyList, adjListSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_offsets, offsets, offsetsSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_levels, levels.data(), levelsSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_frontier, frontier.data(), levelsSize, cudaMemcpyHostToDevice));

    auto end = std::chrono::high_resolution_clock::now();
    
    long double copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    copy_time *= 1e-6;


    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    int level = 1;
    bool done;
    start = std::chrono::high_resolution_clock::now();
    do {
        CUDA_CALL(cudaMemset(d_new_frontier, 0, levelsSize));
        bfs_kernel<<<gridSize, blockSize>>>(d_adjacencyList, d_offsets, d_levels, d_frontier, d_new_frontier, numNodes, level);
        CUDA_CALL(cudaDeviceSynchronize());

        done = true;
        CUDA_CALL(cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice));

        check_empty_frontier<<<gridSize, blockSize>>>(d_new_frontier, numNodes, d_done);
        
        CUDA_CALL(cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));

        int *temp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp;

        level++;
    } while (!done);

    end = std::chrono::high_resolution_clock::now();

    long double running_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    running_time *= 1e-6;


    // CUDA_CALL(cudaMemcpy(levels.data(), d_levels, levelsSize, cudaMemcpyDeviceToHost));

    // int limit = std::min(numNodes, 40);
    // std::cout << "\nBFS -> [ ";
    // for (int i = 0; i < limit; i++) {
    //     std::cout << ""<< i << ":" << levels[i] << " ";
    // }
    // std::cout << "]\n";

    // Print Statistics
    std::cout << "\n";
    std::cout << "Copy Time : " << copy_time << " ms\n";
    std::cout << "Running Time : " << running_time << " ms\n";

    CUDA_CALL(cudaFree(d_adjacencyList));
    CUDA_CALL(cudaFree(d_offsets));
    CUDA_CALL(cudaFree(d_levels));
    CUDA_CALL(cudaFree(d_frontier));
    CUDA_CALL(cudaFree(d_new_frontier));
    CUDA_CALL(cudaFree(d_done));

    return levels.data();
}


#endif