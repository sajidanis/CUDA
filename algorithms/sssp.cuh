#ifndef SSSP_CUH
#define SSSP_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <float.h>

#define BLOCK_SIZE 256

using namespace std;

__global__ void bellman_ford(int *src, int *dest, float *weight, float *distance, int numEdges, int numNodes){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numEdges){
        int u = src[tid];
        int v = dest[tid];
        float wt = weight[tid];

        if(distance[u] != MAXFLOAT && distance[v] > distance[u] + wt){
            // printf("\ndistance[%d]:  %f wt[%d-%d]: %f\n", u, distance[u], u, v, wt);
            atomicExch(&distance[v], distance[u] + wt);
        }
    }
}

void sssp(std::vector<int> h_src, std::vector<int> h_dest, std::vector<float> h_weight, int numNodes, int numEdges) {

    std::vector<float> h_distance(numNodes, MAXFLOAT);

    h_distance[0] = 0.0;
    // Device variables
    int *d_src, *d_dest ;
    float *d_distance, *d_weight;

    auto start = std::chrono::high_resolution_clock::now(); // for timing
    cudaMalloc(&d_src, numEdges * sizeof(int));
    cudaMalloc(&d_dest, numEdges * sizeof(int));
    cudaMalloc(&d_weight, numEdges * sizeof(float));

    cudaMalloc(&d_distance, numNodes * sizeof(float));

    // Copy the data from host to device
    cudaMemcpy(d_src, h_src.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, h_distance.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    
    long double copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    copy_time *= 1e-6;


    int nblocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    start = std::chrono::high_resolution_clock::now();

    // Run Bellman-Ford algorithm
    for (int i = 0; i < numNodes - 1; ++i) {
        bellman_ford<<<nblocks, BLOCK_SIZE>>>(d_src, d_dest, d_weight, d_distance, numEdges, numNodes);
        cudaDeviceSynchronize();

    }

    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();

    long double running_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    running_time *= 1e-6;

    float dist[numNodes];

    // cudaMemcpy(dist, d_distance, numNodes * sizeof(float), cudaMemcpyDeviceToHost);


    // int N = min(numNodes, 40);
    // for(int i = 0 ; i < N; i++){
    //     std::cout << dist[i] << " ";
    // }
    // std::cout << "\n";

     // Print Statistics
    std::cout << "\n";
    std::cout << "Copy Time : " << copy_time << " ms\n";
    std::cout << "Running Time : " << running_time << " ms\n";
    std::cout << "\n";

}


#endif