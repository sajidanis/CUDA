#ifndef DYNAMIC_CSR_CUH
#define DYNAMIC_CSR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <stdio.h>

#define BLOCKSIZE 256

struct DynamicCsr{
    int numEdges;
    int numNodes;
    int *row_offset;
    int *col_indices;
    float *edge_val;
};

void print_device_vector(const thrust::device_vector<int>& d_vec, const char* name) {
    // Allocate host memory to copy data
    thrust::host_vector<int> h_vec = d_vec;

    // Print the contents
    std::cout << name << " : ";
    for (int i = 0; i < h_vec.size(); ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

void print_device_array(int* d_array, int size, const char *name) {
    // Step 1: Allocate host memory to hold the copied data
    int* h_array = (int*)malloc((size) * sizeof(int));

    // Step 2: Copy data from device memory (new_offsets) to host memory (h_new_offsets)
    cudaMemcpy(h_array, d_array, (size) * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 3: Print the contents of array
    std::cout << name << " : ";
    for (int i = 0; i < size; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Step 4: Free the host memory after printing
    free(h_array);
}

__global__ void countRowPerVertex(int *h_src, int numEdges, int *row_offsets){
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < numEdges){
        atomicAdd(&row_offsets[h_src[tid]], 1);
    }
}

void preprocess_insert(int *h_src, int *h_dest, int numNodes, int numEdges, int *new_offsets, int *new_col_indices){
    thrust::device_vector<int> d_src(h_src, h_src + numEdges);
    thrust::device_vector<int> d_dest(h_dest, h_dest + numEdges);

    thrust::device_vector<int> row_offsets(numNodes + 1, 0);
    thrust::sort_by_key(d_src.begin(), d_src.end(), d_dest.begin());

    int nblocks = (numEdges + BLOCKSIZE - 1) / BLOCKSIZE;

    countRowPerVertex<<< nblocks, BLOCKSIZE >>>(thrust::raw_pointer_cast(d_src.data()), numEdges, thrust::raw_pointer_cast(row_offsets.data()));
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, row_offsets.begin(), row_offsets.end(), row_offsets.begin());

    // cudaMemcpy(new_offsets, thrust::raw_pointer_cast(row_offsets.data()), (numNodes + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    thrust::copy(row_offsets.begin(), row_offsets.end(), new_offsets);

    cudaMemcpy(new_col_indices, thrust::raw_pointer_cast(d_dest.data()), numEdges * sizeof(int), cudaMemcpyDeviceToDevice);
}

__global__ void preprocess_insert_updateOffset_kernel(DynamicCsr *dyn_csr, int *new_offset, int numNodes, int *updated_row_offsets){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= numNodes) {
        // Update each row's offset by adding the new edges being inserted
        updated_row_offsets[tid] = dyn_csr->row_offset[tid] + new_offset[tid];

        if(tid == numNodes){
            (dyn_csr->numEdges) = updated_row_offsets[tid];
        }
    }
}

__global__ void batch_insert_kernel(DynamicCsr *dyn_csr, int *new_offset, int *new_col_indices, int numNodes, int *updated_row_offset, int *updated_col_indices){
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < numNodes){
        int start = dyn_csr->row_offset[tid];
        int end = dyn_csr->row_offset[tid+1];
        int k = updated_row_offset[tid];
        for(int i = start ; i < end ; i++){
            updated_col_indices[k] = dyn_csr->col_indices[i];
            k++;
        }

        start = new_offset[tid];
        end = new_offset[tid+1];
        for(int i = start ; i < end ; i++){
            updated_col_indices[k] = new_col_indices[i];
            k++;
        }
    }
}

void batch_insert(DynamicCsr *d_dyn_csr, int *h_src, int *h_dest, int numEdges, int numNodes){

    int *d_new_offset;
    int *d_new_col_indices;

    cudaMalloc(&d_new_offset, sizeof(int) * (numNodes + 1));
    cudaMalloc(&d_new_col_indices , sizeof(int) * numEdges);

    preprocess_insert(h_src, h_dest, numNodes, numEdges, d_new_offset, d_new_col_indices);

    int *d_updated_row_offsets, *d_updated_col_indices;
    cudaMalloc(&d_updated_row_offsets, sizeof(int) * (numNodes + 1));

    int nblocks = (numNodes + BLOCKSIZE - 1) / BLOCKSIZE;

    preprocess_insert_updateOffset_kernel<<< nblocks, BLOCKSIZE >>>(d_dyn_csr, d_new_offset, numNodes, d_updated_row_offsets);
    cudaDeviceSynchronize();

    int *updated_edges = (int *)malloc(sizeof(int));

    cudaMemcpy(updated_edges, &d_dyn_csr->numEdges, sizeof(int)* 1, cudaMemcpyDeviceToHost);

    cudaMalloc(&d_updated_col_indices, sizeof(int) * (*updated_edges));

    batch_insert_kernel<<< nblocks, BLOCKSIZE >>>(d_dyn_csr, d_new_offset, d_new_col_indices, numNodes, d_updated_row_offsets, d_updated_col_indices);

    cudaDeviceSynchronize();

    // Update the newly to dyn_csr;

    DynamicCsr h_csr;

    cudaMemcpy(&h_csr, d_dyn_csr, sizeof(DynamicCsr), cudaMemcpyDeviceToHost);
    cudaFree(h_csr.row_offset);
    cudaFree(h_csr.col_indices);

    h_csr.numNodes = numNodes;
    h_csr.numEdges = *updated_edges;
    h_csr.row_offset = d_updated_row_offsets;
    h_csr.col_indices = d_updated_col_indices;

    cudaMemcpy(d_dyn_csr, &h_csr, sizeof(DynamicCsr), cudaMemcpyHostToDevice);

}

// int main(){
//     DynamicCsr *d_csr;
//     DynamicCsr h_csr;

//     cudaMalloc(&d_csr, sizeof(DynamicCsr));
   
//     int *d_row_offset, *d_col_indices;

//     cudaMalloc(&d_row_offset, sizeof(int) * 5);
//     cudaMalloc(&d_col_indices, sizeof(int) * 8);
    
//     h_csr.row_offset = d_row_offset;
//     h_csr.col_indices = d_col_indices;

//     h_csr.numNodes = 4;
//     h_csr.numEdges = 8;

//     cudaMemcpy(d_csr, &h_csr, sizeof(DynamicCsr), cudaMemcpyHostToDevice);

//     int row[] = {0, 2, 4, 6, 8};
//     int col[] = {1, 2, 0, 2, 1, 2, 1, 2};

//     int h_src[] = {0, 1, 3};
//     int h_dest[] = {3, 3, 0};

//     cudaMemcpy(d_row_offset, row, sizeof(int) * 5, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_col_indices, col, sizeof(int) * 8, cudaMemcpyHostToDevice);

//     batch_insert(d_csr, h_src, h_dest, 3, 4);

//     cudaMemcpy(&h_csr, d_csr, sizeof(DynamicCsr), cudaMemcpyDeviceToHost);

//     print_device_array(h_csr.row_offset, 5, "Updated Row Offset");

//     // int *row_off = (int *) malloc(sizeof(int) * 11);
//     // cudaMemcpy(row_off, d_csr->row_offset, sizeof(int) * 11, cudaMemcpyDeviceToHost);

//     // for(int i = 0 ; i < 11 ; i++){
//     //     std::cout << row_off[i] << " ";
//     // }
//     // cout << "\n";
// }

#endif