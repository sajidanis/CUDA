#include <iostream>
#include <vector>
#include <cuda_runtime.h>

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
                    if (neighbor_w > neighbor_u) {
                        int start_w = d_csr_row_ptr[neighbor_w];
                        int end_w = d_csr_row_ptr[neighbor_w + 1];

                        for (int k = start_w; k < end_w; k++) {
                            if (d_csr_col_ind[k] == vertex) {
                                count++;
                            }
                        }
                    }
                }
            }
        }
        atomicAdd(&d_triangle_count[vertex], count);
    }
}

// CUDA kernel to update triangles when an edge is added
__global__ void addEdgeAndUpdateTriangles(int* u_edges, int* v_edges, int num_updates, int* d_csr_row_ptr, int* d_csr_col_ind, int* d_triangle_count, int num_vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_updates) {
        int u = u_edges[idx];
        int v = v_edges[idx];

        int start_u = d_csr_row_ptr[u];
        int end_u = d_csr_row_ptr[u + 1];
        int start_v = d_csr_row_ptr[v];
        int end_v = d_csr_row_ptr[v + 1];

        // For each common neighbor, increment triangle count
        for (int i = start_u; i < end_u; i++) {
            int neighbor_u = d_csr_col_ind[i];
            if (neighbor_u != v) {
                for (int j = start_v; j < end_v; j++) {
                    int neighbor_v = d_csr_col_ind[j];
                    if (neighbor_v == neighbor_u) {
                        atomicAdd(&d_triangle_count[u], 1);
                        atomicAdd(&d_triangle_count[v], 1);
                        atomicAdd(&d_triangle_count[neighbor_u], 1);
                    }
                }
            }
        }
    }
}

// CUDA kernel to update triangles when an edge is removed
__global__ void removeEdgeAndUpdateTriangles(int* u_edges, int* v_edges, int num_updates, int* d_csr_row_ptr, int* d_csr_col_ind, int* d_triangle_count, int num_vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_updates) {
        int u = u_edges[idx];
        int v = v_edges[idx];

        int start_u = d_csr_row_ptr[u];
        int end_u = d_csr_row_ptr[u + 1];
        int start_v = d_csr_row_ptr[v];
        int end_v = d_csr_row_ptr[v + 1];

        // For each common neighbor, decrement triangle count
        for (int i = start_u; i < end_u; i++) {
            int neighbor_u = d_csr_col_ind[i];
            if (neighbor_u != v) {
                for (int j = start_v; j < end_v; j++) {
                    int neighbor_v = d_csr_col_ind[j];
                    if (neighbor_v == neighbor_u) {
                        atomicSub(&d_triangle_count[u], 1);
                        atomicSub(&d_triangle_count[v], 1);
                        atomicSub(&d_triangle_count[neighbor_u], 1);
                    }
                }
            }
        }
    }
}

// Host function to initialize graph and count triangles
void triangleCount(int* h_csr_row_ptr, int* h_csr_col_ind, int num_vertices, int initial = 1) {
    // Device memory allocation
    int *d_csr_row_ptr, *d_csr_col_ind, *d_triangle_count;
    cudaMalloc(&d_csr_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, sizeof(int) * (h_csr_row_ptr[num_vertices])); // total edges
    cudaMalloc(&d_triangle_count, num_vertices * sizeof(int));

    // Copy CSR arrays to device
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, h_csr_col_ind, sizeof(int) * (h_csr_row_ptr[num_vertices]), cudaMemcpyHostToDevice);

    // Initially, count triangles if required
    if (initial) {
        cudaMemset(d_triangle_count, 0, num_vertices * sizeof(int));
        int blockSize = 256;
        int numBlocks = (num_vertices + blockSize - 1) / blockSize;
        countTrianglesIncreasingOrder<<<numBlocks, blockSize>>>(d_csr_row_ptr, d_csr_col_ind, d_triangle_count, num_vertices);
    }

    // Copy results back to host
    int* h_triangle_count = (int*)malloc(num_vertices * sizeof(int));
    cudaMemcpy(h_triangle_count, d_triangle_count, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Sum up counts to get total triangles
    int totalTriangles = 0;
    for (int i = 0; i < num_vertices; i++) {
        totalTriangles += h_triangle_count[i];
    }

    // Print the result
    std::cout << "\nTotal number of triangles: " << totalTriangles / 3 << std::endl; // Divided by 3 for each triangle counted thrice

    // Free device memory
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_ind);
    cudaFree(d_triangle_count);
    free(h_triangle_count);
}

// Host function to handle batch updates for edge additions
void batchAddEdges(int* u_edges, int* v_edges, int num_updates, int* d_csr_row_ptr, int* d_csr_col_ind, int* d_triangle_count, int num_vertices) {
    int *d_u_edges, *d_v_edges;
    cudaMalloc(&d_u_edges, num_updates * sizeof(int));
    cudaMalloc(&d_v_edges, num_updates * sizeof(int));

    cudaMemcpy(d_u_edges, u_edges, num_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_edges, v_edges, num_updates * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_updates + blockSize - 1) / blockSize;
    addEdgeAndUpdateTriangles<<<numBlocks, blockSize>>>(d_u_edges, d_v_edges, num_updates, d_csr_row_ptr, d_csr_col_ind, d_triangle_count, num_vertices);

    cudaFree(d_u_edges);
    cudaFree(d_v_edges);
}

// Host function to handle batch updates for edge removals
void batchRemoveEdges(int* u_edges, int* v_edges, int num_updates, int* d_csr_row_ptr, int* d_csr_col_ind, int* d_triangle_count, int num_vertices) {
    int *d_u_edges, *d_v_edges;
    cudaMalloc(&d_u_edges, num_updates * sizeof(int));
    cudaMalloc(&d_v_edges, num_updates * sizeof(int));

    cudaMemcpy(d_u_edges, u_edges, num_updates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_edges, v_edges, num_updates * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_updates + blockSize - 1) / blockSize;
    removeEdgeAndUpdateTriangles<<<numBlocks, blockSize>>>(d_u_edges, d_v_edges, num_updates, d_csr_row_ptr, d_csr_col_ind, d_triangle_count, num_vertices);

    cudaFree(d_u_edges);
    cudaFree(d_v_edges);
}

// Example main function
int main() {
    // Example graph in CSR format (edges)
    // Graph: 0 -- 1
    //          | \
    //          |  2
    //          | /
    //          3
    // CSR representation
    int h_csr_row_ptr[] = {0, 2, 4, 6, 8}; // Row pointers (4 vertices, 5 edges)
    int h_csr_col_ind[] = {1, 2, 0, 2, 0, 1, 1, 2}; // Column indices (edges)
    int num_vertices = sizeof(h_csr_row_ptr) / sizeof(int) - 1; // Number of vertices

    // Initial triangle count
    triangleCount(h_csr_row_ptr, h_csr_col_ind, num_vertices);

    // Dynamic updates
    int u_edges_add[] = {0, 1};
    int v_edges_add[] = {3, 3};
    batchAddEdges(u_edges_add, v_edges_add, 2, h_csr_row_ptr, h_csr_col_ind, nullptr, num_vertices);

    int u_edges_remove[] = {1, 0};
    int v_edges_remove[] = {2, 3};
    batchRemoveEdges(u_edges_remove, v_edges_remove, 2, h_csr_row_ptr, h_csr_col_ind, nullptr, num_vertices);

    return 0;
}
