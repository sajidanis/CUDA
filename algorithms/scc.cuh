
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>

#include "cudaError.cuh"

// Kernel to trim vertices with only incoming or outgoing edges
__global__ void trimVertices(int *rowPtr, int *colInd, int *d_P, int *d_changed, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes && d_P[tid] == 1) {
        int outDegree = rowPtr[tid + 1] - rowPtr[tid];
        int inDegree = 0;
        for (int i = 0; i < numNodes; i++) {
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
                if (colInd[j] == tid) inDegree++;
            }
        }
        if (outDegree == 0 || inDegree == 0) {
            d_P[tid] = 0;
            *d_changed = 1;
        }
    }
}

// Kernel to perform BFS and find reachable vertices
__global__ void bfsKernel(int *rowPtr, int *colInd, int *d_visited, int *d_frontier, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes && d_frontier[tid] == 1) {
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        for (int i = start; i < end; ++i) {
            int neighbor = colInd[i];
            if (d_visited[neighbor] == 0) {
                d_visited[neighbor] = 1;
                d_frontier[neighbor] = 1;
            }
        }
        d_frontier[tid] = 0; // clear the frontier
    }
}

// Function to perform forward or backward BFS
void BFS(int *rowPtr, int *colInd, thrust::device_vector<int> &d_P, int pivot, thrust::device_vector<int> &d_reachable, int numNodes) {
    thrust::device_vector<int> d_frontier(numNodes, 0);
    thrust::device_vector<int> d_visited(numNodes, 0);

    d_frontier[pivot] = 1;
    d_visited[pivot] = 1;

    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

    while (thrust::reduce(d_frontier.begin(), d_frontier.end()) > 0) {
        bfsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(rowPtr), thrust::raw_pointer_cast(colInd),
            thrust::raw_pointer_cast(d_visited.data()), thrust::raw_pointer_cast(d_frontier.data()), numNodes);
        CUDA_KERNEL_CHECK();
    }

    thrust::copy(d_visited.begin(), d_visited.end(), d_reachable.begin());
}

// Parallel SCC using CSR
void Parallel_SCC_CSR(thrust::device_vector<int> &rowPtr, thrust::device_vector<int> &colInd, thrust::device_vector<int> &d_P, thrust::device_vector<int> &StrongCompSet, int numNodes) {
    thrust::device_vector<int> d_changed(1, 1);
    
    while (thrust::reduce(d_changed.begin(), d_changed.end()) > 0) {
        d_changed[0] = 0;
        int threadsPerBlock = 256;
        int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

        trimVertices<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(rowPtr.data()), thrust::raw_pointer_cast(colInd.data()),
            thrust::raw_pointer_cast(d_P.data()), thrust::raw_pointer_cast(d_changed.data()), numNodes);
        CUDA_KERNEL_CHECK();
    }

    if (thrust::reduce(d_P.begin(), d_P.end()) == 0) return;

    // Select a random pivot
    int pivot = thrust::find(d_P.begin(), d_P.end(), 1) - d_P.begin();

    thrust::device_vector<int> Vfwd(numNodes, 0);
    thrust::device_vector<int> Vbwd(numNodes, 0);

    BFS(thrust::raw_pointer_cast(rowPtr.data()), thrust::raw_pointer_cast(colInd.data()), d_P, pivot, Vfwd, numNodes);
    BFS(thrust::raw_pointer_cast(rowPtr.data()), thrust::raw_pointer_cast(colInd.data()), d_P, pivot, Vbwd, numNodes);

    thrust::device_vector<int> scc(numNodes, 0);
    thrust::set_intersection(Vfwd.begin(), Vfwd.end(), Vbwd.begin(), Vbwd.end(), scc.begin());

    // Temporary vector to store the result of the union operation
    thrust::device_vector<int> tempStrongCompSet(StrongCompSet.size() + scc.size());

    // Perform the set union operation
    auto endIter = thrust::set_union(
        StrongCompSet.begin(), StrongCompSet.end(), 
        scc.begin(), scc.end(), 
        tempStrongCompSet.begin());

    // Resize the StrongCompSet vector to fit the new elements
    tempStrongCompSet.resize(thrust::distance(tempStrongCompSet.begin(), endIter));

    // Copy the result back to StrongCompSet
    StrongCompSet = tempStrongCompSet;
    
    // Recursive calls
    thrust::device_vector<int> Vfwd_minus_scc(numNodes, 0);
    thrust::set_difference(Vfwd.begin(), Vfwd.end(), scc.begin(), scc.end(), Vfwd_minus_scc.begin());

    thrust::device_vector<int> Vbwd_minus_scc(numNodes, 0);
    thrust::set_difference(Vbwd.begin(), Vbwd.end(), scc.begin(), scc.end(), Vbwd_minus_scc.begin());

    thrust::device_vector<int> P_minus_Vfwd_Vbwd(numNodes, 0);
    thrust::set_difference(d_P.begin(), d_P.end(), Vfwd.begin(), Vfwd.end(), P_minus_Vfwd_Vbwd.begin());
    thrust::set_difference(P_minus_Vfwd_Vbwd.begin(), P_minus_Vfwd_Vbwd.end(), Vbwd.begin(), Vbwd.end(), P_minus_Vfwd_Vbwd.begin());

    Parallel_SCC_CSR(rowPtr, colInd, Vfwd_minus_scc, StrongCompSet, numNodes);
    Parallel_SCC_CSR(rowPtr, colInd, Vbwd_minus_scc, StrongCompSet, numNodes);
    Parallel_SCC_CSR(rowPtr, colInd, P_minus_Vfwd_Vbwd, StrongCompSet, numNodes);
}

// int main() {
//     // Example graph
//     int numNodes = 5;
//     std::vector<int> h_rowPtr = {0, 2, 3, 5, 6, 8}; // CSR Row pointers
//     std::vector<int> h_colInd = {1, 2, 3, 1, 4, 2, 0, 3}; // CSR Column indices

//     thrust::device_vector<int> d_rowPtr = h_rowPtr;
//     thrust::device_vector<int> d_colInd = h_colInd;
//     thrust::device_vector<int> d_P(numNodes, 1); // Initial vertex set P is the entire graph
//     thrust::device_vector<int> StrongCompSet(numNodes, 0); // Store the strongly connected components

//     Parallel_SCC_CSR(d_rowPtr, d_colInd, d_P, StrongCompSet, numNodes);

//     thrust::host_vector<int> h_StrongCompSet = StrongCompSet;

//     std::cout << "Strongly Connected Components: ";
//     for (int i = 0; i < numNodes; ++i) {
//         if (h_StrongCompSet[i] == 1) {
//             std::cout << i << " ";
//         }
//     }
//     std::cout << std::endl;

//     return 0;
// }
