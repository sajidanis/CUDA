#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "marketReader.cuh"
#include "mis.cuh"
#include "scc.cuh"

using namespace std;

int main(int argc, char **argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << " <GPU_ID>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int gpuId = atoi(argv[2]);

    COO_t coo;     
    CSR_t csr;
    ReadMarketStream(filename, coo);
    COO_to_CSR(coo, csr);

    size_t numNodes = csr.nodes;
    // size_t numEdges = csr.edges;

    thrust::device_vector<int> d_offsets(csr.offsets);
    thrust::device_vector<int> d_colIndices(csr.columnIndices);

    for(auto &el : csr.offsets){
        cout << el << ", ";
    }
    cout << "\n";

    for(auto &el : csr.columnIndices){
        cout << el << ",";
    }
    cout << "\n";

    // thrust::device_vector<int> d_MIS(numNodes, -1);
    // Maximal_Independent_Set(d_offsets, d_colIndices, numNodes, d_MIS);
    // thrust::host_vector<int> h_MIS = d_MIS; // copy the vector to host

    // // Print result
    // std::cout << "Maximal Independent Set: ";
    // for (int i = 0; i < numNodes; ++i) {
    //     if (h_MIS[i] == 1) {
    //         std::cout << i << " ";
    //     }
    // }
    // std::cout << std::endl;

    thrust::device_vector<int> d_P(numNodes, 1); // Initial vertex set P is the entire graph
    thrust::device_vector<int> StrongCompSet(numNodes, 0); // Store the strongly connected components

    Parallel_SCC_CSR(d_offsets, d_colIndices, d_P, StrongCompSet, numNodes);

    thrust::host_vector<int> h_StrongCompSet = StrongCompSet;

    std::cout << "Strongly Connected Components: ";
    for (int i = 0; i < numNodes; ++i) {
        if (h_StrongCompSet[i] == 1) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    cudaDeviceSynchronize();
    return 0;
}

