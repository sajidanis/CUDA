#include <iostream>
#include <cuda_runtime.h>

// const int MAX_NEIGHBORS = 1024; // Maximum number of neighbors for each vertex

// CUDA kernel to count triangles in increasing order
__global__ void countTrianglesIncreasingOrder(int* d_csr_row_ptr, int* d_csr_col_ind, int* d_triangle_count, int num_vertices) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < num_vertices) {
        int count = 0;
        int start_u = d_csr_row_ptr[vertex];
        int end_u = d_csr_row_ptr[vertex + 1];

        // Iterate over the neighbors of the current vertex
        for (int i = start_u; i < end_u; i++) {
            int neighbor_u = d_csr_col_ind[i];
            // Only consider triangles where vertex < neighbor_u
            if (vertex < neighbor_u) {
                int start_v = d_csr_row_ptr[neighbor_u];
                int end_v = d_csr_row_ptr[neighbor_u + 1];

                // Check for a common neighbor to form a triangle
                for (int j = start_v; j < end_v; j++) {
                    int neighbor_w = d_csr_col_ind[j];
                    if (neighbor_w > neighbor_u){
                        int start_w = d_csr_row_ptr[neighbor_w];
                        int end_w = d_csr_row_ptr[neighbor_w + 1];

                        for(int k = start_w ; k < end_w ; k++){
                            if(d_csr_col_ind[k] == vertex){
                                count++;
                            }
                        }
                    }
                }
            }
        }
        d_triangle_count[vertex] = count;
    }
}

// Host function to count triangles
void triangleCount(int* h_csr_row_ptr, int* h_csr_col_ind, int num_vertices) {
    // Device memory allocation
    int *d_csr_row_ptr, *d_csr_col_ind, *d_triangle_count;
    cudaMalloc(&d_csr_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, sizeof(int) * (h_csr_row_ptr[num_vertices])); // total edges
    cudaMalloc(&d_triangle_count, num_vertices * sizeof(int));

    // Copy CSR arrays to device
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, h_csr_col_ind, sizeof(int) * (h_csr_row_ptr[num_vertices]), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_vertices + blockSize - 1) / blockSize;
    countTrianglesIncreasingOrder<<<numBlocks, blockSize>>>(d_csr_row_ptr, d_csr_col_ind, d_triangle_count, num_vertices);
    
    // Copy results back to host
    int* h_triangle_count = (int*)malloc(num_vertices * sizeof(int));
    cudaMemcpy(h_triangle_count, d_triangle_count, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Sum up counts to get total triangles
    int totalTriangles = 0;
    for (int i = 0; i < num_vertices; i++) {
        // std::cout << h_triangle_count[i] << " ";
        totalTriangles += h_triangle_count[i];
    }

    // Print the result
    std::cout << "\nTotal number of triangles: " << totalTriangles << std::endl;

    // Free device memory
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_ind);
    cudaFree(d_triangle_count);
    free(h_triangle_count);
}

// Example main function
// int main() {
//     // Example graph in CSR format (edges)
//     // Graph: 0 -- 1
//     //          | \
//     //          |  2
//     //          | /
//     //          3
//     // CSR representation
//     int h_csr_row_ptr[] = {0, 2, 4, 6, 8}; // Row pointers (4 vertices, 5 edges)
//     int h_csr_col_ind[] = {1, 2, 0, 2, 0, 1, 1, 2}; // Column indices (edges)

//     int num_vertices = sizeof(h_csr_row_ptr) / sizeof(int) - 1; // Number of vertices

//     triangleCount(h_csr_row_ptr, h_csr_col_ind, num_vertices);

//     return 0;
// }
