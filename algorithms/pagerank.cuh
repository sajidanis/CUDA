#ifndef PAGERANK_CUH
#define PAGERANK_CUH


#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>
#include <fstream>
#include <chrono>

#define BLOCK_SIZE 256
#define EPSILON 1e-6
#define DAMPING_FACTOR 0.85
#define MAX_ITER 10

__global__ void computeOutbound(int *src, int *outbound, int numEdges){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEdges) {
        int s = src[idx];
        atomicAdd(&outbound[s], 1);
    }
}

__global__ void pageRankKernel(int *src, int *dest, int *outbound, float *rank, float *new_rank, int numEdges, int numNodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEdges) {
        int s = src[idx];
        int d = dest[idx];
        if(outbound[s] > 0){
            atomicAdd(&new_rank[d], rank[s] / outbound[s]);
        } else {
            atomicAdd(&new_rank[d], rank[s] / numNodes);
        }
    }
}

__global__ void applyDampingFactorKernel(float *new_rank, float *rank, int numNodes, float *diff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numNodes) {
        new_rank[idx] = (DAMPING_FACTOR * new_rank[idx]) + (1.0f - DAMPING_FACTOR) / numNodes;
        diff[0] += fabs(new_rank[idx] - rank[idx]);
    }
}

int pagerank(int *src, int *dest, size_t numNodes, size_t numEdges) {

    // Host variables
    std::vector<float> h_rank(numNodes, 1.0f / numNodes);
    std::vector<float> h_newRank(numNodes, 0.0f);
    

    auto start = std::chrono::high_resolution_clock::now(); // for timing

    // Device variables
    int *d_src, *d_dest, *d_outbound;
    float *d_rank, *d_newRank, *d_diff;
    cudaMalloc(&d_src, numEdges * sizeof(int));
    cudaMalloc(&d_dest, numEdges * sizeof(int));
    cudaMalloc(&d_rank, numNodes * sizeof(float));
    cudaMalloc(&d_newRank, numNodes * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));
    cudaMalloc(&d_outbound, numNodes * sizeof(float));

    cudaMemcpy(d_src, src, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, dest, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, h_rank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_newRank, h_newRank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);

    auto end = std::chrono::high_resolution_clock::now();
    
    long double copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    copy_time *= 1e-6;

    int numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    start = std::chrono::high_resolution_clock::now();

    // Preprocess for outbound computation
    cudaMemset(d_outbound, 0, numNodes * sizeof(int));
    computeOutbound<<<numBlocks, BLOCK_SIZE>>>(d_src, d_outbound, numEdges);

    float *diff = (float *)malloc(sizeof(float));
    // Main loop for PageRank computation
    int iter = 0;
    while (iter < MAX_ITER) {
        numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemset(d_newRank, 0, numNodes * sizeof(float));
        pageRankKernel<<<numBlocks, BLOCK_SIZE>>>(d_src, d_dest, d_outbound, d_rank, d_newRank, numEdges, numNodes);


        numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        applyDampingFactorKernel<<<numBlocks, BLOCK_SIZE>>>(d_newRank, d_rank, numNodes, d_diff);

        cudaDeviceSynchronize();

        std::swap(d_rank, d_newRank);
        iter++;
    }

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    long double running_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    running_time *= 1e-6;

    // cudaMemcpy(h_rank.data(), d_rank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    // long long N = numNodes > 40 ? 40 : numNodes;
    // // Output the PageRank values
    // for (int i = 0; i < N; ++i) {
    //     std::cout << h_rank[i] << " ";
    // }
    // std::cout << "\n";


     // Print Statistics
    std::cout << "\n";
    std::cout << "Copy Time : " << copy_time << " ms\n";
    std::cout << "Running Time : " << running_time << " ms\n";
    std::cout << "\n";

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_rank);
    cudaFree(d_newRank);

    return 0;
}

#endif