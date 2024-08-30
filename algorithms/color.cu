#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/remove.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}

__global__ void initialColoring(int* rowPtr, int* colInd, int* colors, int* U, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices && U[idx] == 1) {
        int forbiddenColors[256] = {0};  // Assuming max degree is less than 256

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

int main() {
    // Example graph in CSR format
    int numVertices = 6;
    thrust::host_vector<int> h_rowPtr = {0, 2, 5, 7, 9, 12, 14};
    thrust::host_vector<int> h_colInd = {1, 3, 0, 2, 3, 1, 4, 0, 1, 3, 2, 5, 4, 5};

    // Allocate device memory
    thrust::device_vector<int> d_rowPtr = h_rowPtr;
    thrust::device_vector<int> d_colInd = h_colInd;
    thrust::device_vector<int> d_colors(numVertices, -1);  // Initialize colors to -1 (uncolored)
    thrust::device_vector<int> d_U(numVertices, 1);        // All vertices are initially in U
    thrust::device_vector<int> d_nextU(numVertices, 0);

    int blockSize = 256;
    int numBlocks = (numVertices + blockSize - 1) / blockSize;

    // Initial coloring
    initialColoring<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colInd.data()), thrust::raw_pointer_cast(d_colors.data()), thrust::raw_pointer_cast(d_U.data()), numVertices);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Resolve conflicts iteratively
    while (thrust::reduce(d_U.begin(), d_U.end()) > 0) {
        resolveConflicts<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_rowPtr.data()), thrust::raw_pointer_cast(d_colInd.data()), thrust::raw_pointer_cast(d_colors.data()), thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_nextU.data()), numVertices);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update U
        updateU<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_U.data()), thrust::raw_pointer_cast(d_nextU.data()), numVertices);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy the result back to host and print
    thrust::host_vector<int> h_colors = d_colors;
    std::cout << "Vertex Colors: ";
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << " -> Color " << h_colors[i] << std::endl;
    }

    return 0;
}
