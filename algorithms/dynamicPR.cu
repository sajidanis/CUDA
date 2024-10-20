#include <curand_kernel.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#define MAX_WALK_LENGTH 100
#define EPSILON 0.15f

// Kernel to initialize random states for each thread
__global__ void initRNG(curandState *state, unsigned long seed, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// Kernel to generate random walks for edge insertion
__global__ void generateRandomWalks(curandState *state, int *csrRowPtr, int *csrColIdx, 
                                    int *walkSet, int *walkLabels, int numNodes, int u, int v, int maxWalkLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    curandState localState = state[tid];
    int degreeU = csrRowPtr[u + 1] - csrRowPtr[u];
    int degreeV = csrRowPtr[v + 1] - csrRowPtr[v];

    // Loop over time steps from 1 to the length of the longest walk
    for (int t = 1; t <= maxWalkLength; t++) {
        // Sample from Su,t with probability 1/du
        int numWalks = t % degreeU;
        for (int i = 0; i < numWalks; i++) {
            // Perform a random walk starting from u
            int currentNode = u;
            int walkLength = 0;

            while (curand_uniform(&localState) > EPSILON && walkLength < t) {
                int start = csrRowPtr[currentNode];
                int end = csrRowPtr[currentNode + 1];
                int numNeighbors = end - start;

                if (numNeighbors == 0) break;

                int nextNeighborIdx = start + curand(&localState) % numNeighbors;
                currentNode = csrColIdx[nextNeighborIdx];
                walkLength++;
            }

            // Label and store the walk
            if (walkLength == t) {
                walkSet[tid] = currentNode;  // Store walk in walkSet
                walkLabels[tid] = t;         // Label with the current step
            }
        }

        // Repeat for Sv,t
        numWalks = t % degreeV;
        for (int i = 0; i < numWalks; i++) {
            int currentNode = v;
            int walkLength = 0;

            while (curand_uniform(&localState) > EPSILON && walkLength < t) {
                int start = csrRowPtr[currentNode];
                int end = csrRowPtr[currentNode + 1];
                int numNeighbors = end - start;

                if (numNeighbors == 0) break;

                int nextNeighborIdx = start + curand(&localState) % numNeighbors;
                currentNode = csrColIdx[nextNeighborIdx];
                walkLength++;
            }

            // Label and store the walk
            if (walkLength == t) {
                walkSet[tid] = currentNode;
                walkLabels[tid] = t;
            }
        }
    }

    state[tid] = localState;
}

// Kernel to regenerate walks after insertion
__global__ void regenerateWalks(curandState *state, int *csrRowPtr, int *csrColIdx, 
                                int *walkSet, int *walkLabels, int numNodes, int u, int v, int maxWalkLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    curandState localState = state[tid];

    // Regenerate walks
    for (int i = 0; i < maxWalkLength; i++) {
        int currentWalk = walkSet[tid];
        int label = walkLabels[tid];

        // Ensure walks share the same prefix
        if (label <= maxWalkLength) {
            int currentNode = u;
            int walkLength = label;

            while (curand_uniform(&localState) > EPSILON && walkLength < maxWalkLength) {
                int start = csrRowPtr[currentNode];
                int end = csrRowPtr[currentNode + 1];
                int numNeighbors = end - start;

                if (numNeighbors == 0) break;

                int nextNeighborIdx = start + curand(&localState) % numNeighbors;
                currentNode = csrColIdx[nextNeighborIdx];
                walkLength++;
            }

            // Add to walkSet after regenerating walk
            walkSet[tid] = currentNode;
        }
    }

    state[tid] = localState;
}

int main() {
    int N = 5;  // Number of nodes
    int M = 7;  // Number of edges

    // Example graph in CSR format
    int h_csrRowPtr[] = {0, 2, 4, 5, 6, 7};  // Row pointers
    int h_csrColIdx[] = {1, 2, 0, 2, 3, 4, 3};  // Column indices (neighbors)

    // Device arrays
    int *d_csrRowPtr, *d_csrColIdx;
    int *d_walkSet, *d_walkLabels;
    curandState *d_state;

    // Allocate memory
    cudaMalloc((void**)&d_csrRowPtr, (N+1) * sizeof(int));
    cudaMalloc((void**)&d_csrColIdx, M * sizeof(int));
    cudaMalloc((void**)&d_walkSet, N * sizeof(int));
    cudaMalloc((void**)&d_walkLabels, N * sizeof(int));
    cudaMalloc((void**)&d_state, N * sizeof(curandState));

    // Copy data to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, h_csrColIdx, M * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize random states
    initRNG<<<(N + 255) / 256, 256>>>(d_state, time(NULL), N);
    cudaDeviceSynchronize();

    int u = 0, v = 3;  // Example edge insertion
    int maxWalkLength = MAX_WALK_LENGTH;

    // Generate random walks
    generateRandomWalks<<<(N + 255) / 256, 256>>>(d_state, d_csrRowPtr, d_csrColIdx, d_walkSet, d_walkLabels, N, u, v, maxWalkLength);
    cudaDeviceSynchronize();

    // Regenerate walks after edge insertion
    regenerateWalks<<<(N + 255) / 256, 256>>>(d_state, d_csrRowPtr, d_csrColIdx, d_walkSet, d_walkLabels, N, u, v, maxWalkLength);
    cudaDeviceSynchronize();

    

    // Clean up
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_walkSet);
    cudaFree(d_walkLabels);
    cudaFree(d_state);

    return 0;
}
