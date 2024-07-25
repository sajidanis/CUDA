#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_EDGES 100
#define NUM_VERTICES 5
#define NUM_EDGES 6

#define BLOCKSIZE 256

// CUDA error checking macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to insert edges into adjacency list
__global__ void insertEdges(int* d_adjList, int* d_startIdx, int* d_edgeCount, int numEdges, int* srcVertices, int* dstVertices) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numEdges) {
        int src = srcVertices[tid];
        int dst = dstVertices[tid];
        
        // Atomic operation to safely insert the edge
        int index = atomicAdd(&d_edgeCount[src], 1);
        d_adjList[d_startIdx[src] + index] = dst;
    }
}

// Kernel to print adjacency list
__global__ void printAdjacencyList(int* d_adjList, int* d_startIdx, int* d_edgeCount, int numVertices) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < numVertices) {
        printf("Vertex %d: ", tid);
        for (int i = 0; i < d_edgeCount[tid]; ++i) {
            printf("%d ", d_adjList[d_startIdx[tid] + i]);
        }
        printf("\n");
    }
}

int main() {
    // Example edges
    int h_srcVertices[NUM_EDGES] = {0, 0, 1, 1, 2, 3};
    int h_dstVertices[NUM_EDGES] = {1, 4, 0, 2, 4, 4};

    // Host arrays for adjacency list, start indices, and edge counts
    int h_adjList[MAX_EDGES] = {0};
    int h_startIdx[NUM_VERTICES + 1];
    int h_edgeCount[NUM_VERTICES] = {0};

    // Initialize start indices
    h_startIdx[0] = 0;
    for (int i = 1; i <= NUM_VERTICES; ++i) {
        h_startIdx[i] = h_startIdx[i - 1] + NUM_EDGES; // Maximum possible edges per vertex
    }

    // Device arrays
    int *d_adjList, *d_startIdx, *d_edgeCount, *d_srcVertices, *d_dstVertices;

    // Allocate memory on the device
    cudaMalloc((void**)&d_adjList, MAX_EDGES * sizeof(int));
    cudaMalloc((void**)&d_startIdx, (NUM_VERTICES + 1) * sizeof(int));
    cudaMalloc((void**)&d_edgeCount, NUM_VERTICES * sizeof(int));
    cudaMalloc((void**)&d_srcVertices, NUM_EDGES * sizeof(int));
    cudaMalloc((void**)&d_dstVertices, NUM_EDGES * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_adjList, h_adjList, MAX_EDGES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startIdx, h_startIdx, (NUM_VERTICES + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeCount, h_edgeCount, NUM_VERTICES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcVertices, h_srcVertices, NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstVertices, h_dstVertices, NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    
    int nblocks = ceil(float(NUM_EDGES) / BLOCKSIZE);

    // Launch the kernel to insert edges
    insertEdges<<<nblocks, BLOCKSIZE>>>(d_adjList, d_startIdx, d_edgeCount, NUM_EDGES, d_srcVertices, d_dstVertices);
    cudaCheckError();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Launch the kernel to print the adjacency list
    nblocks = ceil(float(NUM_VERTICES) / BLOCKSIZE);

    printAdjacencyList<<<nblocks, BLOCKSIZE>>>(d_adjList, d_startIdx, d_edgeCount, NUM_VERTICES);
    cudaCheckError();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_adjList);
    cudaFree(d_startIdx);
    cudaFree(d_edgeCount);
    cudaFree(d_srcVertices);
    cudaFree(d_dstVertices);

    return 0;
}
