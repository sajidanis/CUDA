#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <algorithm>

// Constants
const float ALPHA = 0.85f;  // Damping factor
const int MAX_WALK_LENGTH = 100;  // Max steps in a random walk
const int NUM_WALKS_PER_NODE = 1000;  // Number of random walks per node

// CUDA error checking macro
#define CUDA_CHECK_ERROR(call) \
    { cudaError_t err = call; if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; exit(1); } }

// Kernel to initialize random number generators for random walks
__global__ void initializeRNG(curandState *states, int numNodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numNodes) {
        curand_init(1234, tid, 0, &states[tid]);  // Initialize RNG
    }
}

// Kernel to perform random walks for PageRank estimation
__global__ void randomWalkKernel(int *rowPtr, int *colIdx, float *pageRank, curandState *states,
                                 int numNodes, int numWalks, float alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numNodes) {
        curandState localState = states[tid];  // Load RNG state
        int walksCompleted = 0;

        while (walksCompleted < numWalks) {
            int currentNode = tid;  // Start walk at current node

            for (int step = 0; step < MAX_WALK_LENGTH; step++) {
                float randNum = curand_uniform(&localState);  // Get random float in [0, 1]

                // Teleport with probability (1 - alpha)
                if (randNum > alpha) {
                    currentNode = curand(&localState) % numNodes;  // Teleport to random node
                } else {
                    // If node has no outgoing edges, teleport to random node
                    if (rowPtr[currentNode] == rowPtr[currentNode + 1]) {
                        currentNode = curand(&localState) % numNodes;
                    } else {
                        // Choose a random neighbor
                        int numNeighbors = rowPtr[currentNode + 1] - rowPtr[currentNode];
                        int nextNodeIndex = curand(&localState) % numNeighbors;
                        currentNode = colIdx[rowPtr[currentNode] + nextNodeIndex];
                    }
                }

                // Increment visit count for currentNode
                atomicAdd(&pageRank[currentNode], 1.0f);
            }

            walksCompleted++;
        }

        states[tid] = localState;  // Store updated RNG state
    }
}

__global__ void updateAffectedWalksKernel(int *rowPtr, int *colIdx, float *pageRank, curandState *states,
                                          int numNodes, int *affectedNodes, int numAffectedNodes,
                                          int numWalks, float alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numAffectedNodes) {
        int node = affectedNodes[tid];  // Get the affected node
        curandState localState = states[node];  // Load RNG state for the node

        for (int walk = 0; walk < numWalks; walk++) {
            int currentNode = node;  // Start walk at the affected node
            for (int step = 0; step < MAX_WALK_LENGTH; step++) {
                float randNum = curand_uniform(&localState);  // Get random float in [0, 1]

                // Teleport with probability (1 - alpha)
                if (randNum > alpha) {
                    currentNode = curand(&localState) % numNodes;  // Teleport to a random node
                } else {
                    // If node has no outgoing edges, teleport to random node
                    if (rowPtr[currentNode] == rowPtr[currentNode + 1]) {
                        currentNode = curand(&localState) % numNodes;
                    } else {
                        // Choose a random neighbor
                        int numNeighbors = rowPtr[currentNode + 1] - rowPtr[currentNode];
                        int nextNodeIndex = curand(&localState) % numNeighbors;
                        currentNode = colIdx[rowPtr[currentNode] + nextNodeIndex];
                    }
                }

                // Update the visit count of the current node
                atomicAdd(&pageRank[currentNode], 1.0f);
            }
        }
        
        states[node] = localState;  // Store updated RNG state
    }
}

// Host function to handle dynamic updates
void handleDynamicUpdate(int *rowPtr, int *colIdx, thrust::device_vector<float> &pageRank, 
                         thrust::device_vector<curandState> &d_rngStates, std::vector<int> &affectedNodes, int numNodes) {
    // Transfer affected nodes to device memory
    thrust::device_vector<int> d_affectedNodes = affectedNodes;
    int numAffectedNodes = affectedNodes.size();

    // Launch kernel to recompute random walks for affected nodes only
    int blockSize = 256;
    int gridSize = (numAffectedNodes + blockSize - 1) / blockSize;
    updateAffectedWalksKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(rowPtr), thrust::raw_pointer_cast(colIdx),
                                                       thrust::raw_pointer_cast(pageRank.data()), thrust::raw_pointer_cast(d_rngStates.data()),
                                                       numNodes, thrust::raw_pointer_cast(d_affectedNodes.data()), numAffectedNodes, 
                                                       NUM_WALKS_PER_NODE, ALPHA);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// Main function
int main() {
    // Initial graph setup in CSR format
    std::vector<int> h_rowPtr = {0, 2, 4, 7};  // Row pointers
    std::vector<int> h_colIdx = {1, 2, 0, 2, 0, 1, 2};  // Column indices (edges)
    int numNodes = h_rowPtr.size() - 1;

    // Transfer graph data to device
    thrust::device_vector<int> d_rowPtr = h_rowPtr;
    thrust::device_vector<int> d_colIdx = h_colIdx;

    // PageRank array (initially zeros)
    thrust::device_vector<float> d_pageRank(numNodes, 0.0f);

    // RNG states for each node
    thrust::device_vector<curandState> d_rngStates(numNodes);

    // Initialize random number generators
    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;
    initializeRNG<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_rngStates.data()), numNodes);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Perform initial random walks for PageRank estimation
    randomWalkKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colIdx.data()), 
                                              thrust::raw_pointer_cast(d_pageRank.data()), thrust::raw_pointer_cast(d_rngStates.data()), 
                                              numNodes, NUM_WALKS_PER_NODE, ALPHA);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Simulate a dynamic update: Node 1 gains a new edge to Node 0
    h_colIdx.push_back(0);
    h_rowPtr[2]++;
    d_rowPtr = h_rowPtr;
    d_colIdx = h_colIdx;

    // Identify affected node(s)
    std::vector<int> affectedNodes = {1};  // Node 1 is the source of the new edge

    // Update affected nodes only
    handleDynamicUpdate(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colIdx.data()), d_pageRank, d_rngStates, affectedNodes, numNodes);

    // Transfer PageRank results back to host
    thrust::host_vector<float> h_pageRank = d_pageRank;

    // Output final PageRank scores
    for (int i = 0; i < numNodes; i++) {
        std::cout << "Node " << i << " PageRank: " << h_pageRank[i] << std::endl;
    }

    return 0;
}
