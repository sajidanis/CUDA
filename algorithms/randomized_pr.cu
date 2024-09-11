#include <cuda.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>

// Kernel to initialize random number generator for each thread
__global__ void init_rng(curandState *state, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Kernel to perform random walks on the graph (CSR format)
__global__ void random_walks_kernel(int* row_ptr, int* col_idx, int* visit_count, curandState* state, int n, int R, float epsilon, int max_length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * R) {
        int start_vertex = id / R;
        int current_vertex = start_vertex;
        curandState localState = state[id];

        int walk_length = 0;

        // Perform a random walk with length sampled from geometric distribution
        while (curand_uniform(&localState) > epsilon && walk_length < max_length) {
            int row_start = row_ptr[current_vertex];
            int row_end = row_ptr[current_vertex + 1];

            if (row_start == row_end) break; // No neighbors, end the walk

            // Randomly pick a neighbor
            int next_vertex_index = row_start + (curand(&localState) % (row_end - row_start));
            current_vertex = col_idx[next_vertex_index];

            walk_length++;
        }

        // Update the number of visits to the final vertex
        if (walk_length <= max_length) {
            atomicAdd(&visit_count[current_vertex], 1);
        }
    }
}

// Host function to initialize CSR graph and call the random walk kernel
void random_walks_cuda(int* h_row_ptr, int* h_col_idx, int n, int R, float epsilon, int max_length) {
    int total_walks = n * R;

    // Allocate device memory for CSR graph and result arrays
    int* d_row_ptr, *d_col_idx, *d_visit_count;
    curandState* d_state;
    
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, h_row_ptr[n] * sizeof(int));  // Number of edges
    cudaMalloc(&d_visit_count, n * sizeof(int));
    cudaMalloc(&d_state, total_walks * sizeof(curandState));

    // Copy CSR data from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, h_row_ptr[n] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visit_count, 0, n * sizeof(int));

    // Initialize random number generator
    init_rng<<<(total_walks + 255) / 256, 256>>>(d_state, time(NULL), total_walks);
    
    // Launch the random walk kernel
    random_walks_kernel<<<(total_walks + 255) / 256, 256>>>(d_row_ptr, d_col_idx, d_visit_count, d_state, n, R, epsilon, max_length);

    // Copy result from device to host
    int* visit_count = new int[n];
    cudaMemcpy(visit_count, d_visit_count, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Normalize the counts to estimate π(v)
    float* pi = new float[n];
    for (int i = 0; i < n; i++) {
        pi[i] = visit_count[i] * epsilon / total_walks;
        std::cout << "pi[" << i << "] = " << pi[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_visit_count);
    cudaFree(d_state);

    delete[] visit_count;
    delete[] pi;
}

int main(int argc, char **argv) {
    int n = 39;  // number of vertices in the graph
    int R = ceil(9 * log(n) / 0.01);  // number of random walks per vertex
    float epsilon = 0.1;  // geometric distribution parameter
    int max_length = 10;  // max length for random walks

    // Example CSR representation for a graph
    int h_row_ptr[] = {0, 11, 22, 29, 33, 37, 41, 51, 64, 71, 76, 83, 92, 99, 107, 116, 121, 125, 134, 143, 147, 152, 165, 175, 179, 185, 192, 201, 207, 213, 219, 224, 231, 238, 241, 256, 285, 289, 307, 340};  // row pointer for 5 vertices
    int h_col_idx[] = {6,7,10,11,12,21,22,33,34,36,38,6,7,8,10,11,12,21,22,34,35,38,13,14,15,16,17,35,38,16,35,36,38,5,33,34,38,4,6,34,38,0,1,5,7,8,10,11,12,34,38,0,1,6,8,9,19,20,21,22,23,34,37,38,1,6,7,9,34,35,38,7,8,34,35,38,0,1,6,18,34,35,38,0,1,6,18,25,34,35,37,38,0,1,6,34,35,37,38,2,24,25,26,27,28,35,38,2,18,24,25,26,27,28,35,38,2,18,26,35,38,2,3,35,38,2,18,24,25,26,28,31,35,38,10,11,14,15,17,32,35,37,38,7,35,37,38,7,32,35,37,38,0,1,7,26,27,29,30,31,32,34,35,37,38,0,1,7,29,31,32,34,35,37,38,7,35,37,38,13,14,17,35,37,38,11,13,14,17,35,37,38,13,14,15,17,21,29,35,37,38,13,14,21,35,37,38,13,14,17,35,37,38,21,22,26,35,37,38,21,31,35,37,38,17,21,22,30,35,37,38,18,20,21,22,35,37,38,0,4,36,0,1,4,5,6,7,8,9,10,11,12,21,22,35,36,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,0,3,33,34,7,11,12,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};  // column indices of neighbors

    random_walks_cuda(h_row_ptr, h_col_idx, n, R, epsilon, max_length);

    return 0;
}
