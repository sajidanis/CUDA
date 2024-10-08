#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#define INF 1000000

// Kernel to initialize betweenness centrality (BC) and other BFS-related arrays
__global__ void initializeBC(int* BC, int* distance, int* sigma, int* Stack_Array, int* predecessor, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V) {
        BC[v] = 0;
        distance[v] = -1;
        sigma[v] = 0;
        predecessor[v] = -1;
        Stack_Array[v] = -1;
    }
}

// Kernel to set initial conditions for BFS from a source vertex
__global__ void initializeBFS(int* sigma, int* distance, int* Stack_Array, int source) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v == source) {
        sigma[source] = 1;
        distance[source] = 0;
        Stack_Array[0] = source;
    }
}

// Kernel to perform the BFS and calculate shortest paths
__global__ void bfsKernel(int* distance, int* sigma, int* Stack_Array, int* csrRowPtr, int* csrColIdx, int V, int level, int* count) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w < V && Stack_Array[w] != -1 && distance[Stack_Array[w]] == level) {
        int vertex = Stack_Array[w];
        for (int i = csrRowPtr[vertex]; i < csrRowPtr[vertex + 1]; i++) {
            int n = csrColIdx[i];

            // Atomic operation for updating distance and Stack_Array
            if (atomicCAS(&distance[n], -1, level + 1) == -1) {
                int pos = atomicAdd(count, 1);
                Stack_Array[pos] = n;
            }

            if (distance[n] == level + 1) {
                atomicAdd(&sigma[n], sigma[vertex]);
            }
        }
    }
}

// Kernel to accumulate dependencies and update BC
__global__ void accumulateBC(int* BC, int* sigma, int* distance, int* Stack_Array, int* predecessor, float* delta, int V, int level) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < V && Stack_Array[u] != -1 && distance[Stack_Array[u]] == level) {
        int vertex = Stack_Array[u];
        for (int i = predecessor[vertex]; i != -1; i = predecessor[i]) {
            atomicAdd(&delta[i], ((float)sigma[i] / sigma[vertex]) * (1 + delta[vertex]));
        }

        if (vertex != u) {
            atomicAdd(&BC[vertex], delta[vertex]);
        }
    }
}

// Function to calculate betweenness centrality using parallel BFS and dependency accumulation
void betweennessCentrality(int* csrRowPtr, int* csrColIdx, int V, int E) {
    // Allocate memory on device
    int *d_BC, *d_distance, *d_sigma, *d_Stack_Array, *d_predecessor, *d_csrRowPtr, *d_csrColIdx;
    float *d_delta;
    cudaMalloc(&d_BC, V * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_sigma, V * sizeof(int));
    cudaMalloc(&d_Stack_Array, V * sizeof(int));
    cudaMalloc(&d_predecessor, V * sizeof(int));
    cudaMalloc(&d_delta, V * sizeof(float));
    cudaMalloc(&d_csrColIdx, E * sizeof(int));
    cudaMalloc(&d_csrRowPtr, (V+1) * sizeof(int));

    //Transfer the host csr data to gpu device
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (V+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, (E)* sizeof(int), cudaMemcpyHostToDevice);

    // Initialize BC and other arrays
    int threadsPerBlock = 256;
    int blocksPerGrid = (V + threadsPerBlock - 1) / threadsPerBlock;
    initializeBC<<<blocksPerGrid, threadsPerBlock>>>(d_BC, d_distance, d_sigma, d_Stack_Array, d_predecessor, V);

    for (int source = 0; source < V; source++) {
        int count = 1;
        int level = 0;

        // Initialize BFS for source vertex
        initializeBFS<<<blocksPerGrid, threadsPerBlock>>>(d_sigma, d_distance, d_Stack_Array, source);

        // BFS loop
        while (count > 0) {
            count = 0;
            int* d_count;
            cudaMalloc(&d_count, sizeof(int));
            cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

            bfsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_distance, d_sigma, d_Stack_Array, d_csrRowPtr, d_csrColIdx, V, level, d_count);
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_count);

            level++;
        }

        // Accumulate BC by backtracking through levels
        level--;
        while (level > 0) {
            accumulateBC<<<blocksPerGrid, threadsPerBlock>>>(d_BC, d_sigma, d_distance, d_Stack_Array, d_predecessor, d_delta, V, level);
            level--;
        }
    }

    // Copy results back to host and clean up
    int* h_BC = (int*)malloc(V * sizeof(int));
    cudaMemcpy(h_BC, d_BC, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_BC);
    cudaFree(d_distance);
    cudaFree(d_sigma);
    cudaFree(d_Stack_Array);
    cudaFree(d_predecessor);
    cudaFree(d_delta);

    // Output results (for testing)
    for (int i = 0; i < V; i++) {
        printf("%d ", h_BC[i]);
    }
    free(h_BC);
}