#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "marketReader.cuh"
// #include "mis.cuh"
// #include "scc.cuh"
#include "gpu.cuh"
// #include "bfs.cuh"
// #include "color.cuh"
// #include "pagerank.cuh"
// #include "sssp.cuh"
// #include "aggregate_pr.cuh"
#include "tc.cuh"

using namespace std;

int main(int argc, char **argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << " <GPU_ID>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int gpuId = atoi(argv[2]);
    if(gpuId == -1){
        gpuId = selectLeastLoadedGPU();
    }

    // set the device to gpuId
    std::cout << "\n[+] Setting the GPU " << gpuId << "\n";
    CUDA_CALL(cudaSetDevice(gpuId));
    // Print statistics regarding GPU DEVICE
    printDeviceStatistics(gpuId);

    COO_t coo;     
    CSR_t csr;
    ReadMarketStream(filename, coo);
    COO_to_CSR(coo, csr); // Get the Csr Dataset

    size_t numNodes = csr.nodes;

    triangleCount(csr.offsets.data(), csr.columnIndices.data(), numNodes);

    // // Betweenness Centrality
    // betweennessCentrality(csr.offsets.data(), csr.columnIndices.data(), csr.nodes, csr.edges);

    // cout << "\n";
    // cout << "\n[+] BFS -> \n";
    // bfs_cuda(csr.columnIndices.data(), csr.offsets.data(), csr.nodes, 0);

    // cout << "\n";

    // cout << "[+] Color -> \n";
    // graph_coloring(csr.offsets, csr.columnIndices, csr.nodes);

    // cout << "\n";
    
    // cout << "[+] MIS -> \n";
    // Maximal_Independent_Set(csr.offsets, csr.columnIndices, numNodes);

    // cout << "\n";

    // cout << "[+] PR -> \n";
    // pagerank(coo.src.data(), coo.dest.data(), coo.nodes, coo.edges);

    // cout << "\n";

    // cout << "[+] SCC -> \n";
    // strongly_connected(csr.offsets, csr.columnIndices, numNodes);

    // cout << "\n";

    // cout << "[+] SSSP -> \n";
    // sssp(coo.src, coo.dest, coo.wt, coo.nodes, coo.edges);

    // cout << "\n";

    // // size_t numNodes = csr.nodes;
    // size_t numEdges = csr.edges;

    // Dynamic Page rank
    // CSRGraph h_graph;
    // h_graph.row_ptr = csr.offsets;
    // h_graph.col_idx = csr.columnIndices;
    // h_graph.num_edges = csr.edges;
    // h_graph.num_nodes = csr.nodes;

    // int n = 0.1 * csr.edges; // 1% of csr edges
    // n = min(n, 1000);

    // test_pr_dynamic(h_graph, n);

    // thrust::device_vector<int> d_offsets(csr.offsets);
    // thrust::device_vector<int> d_colIndices(csr.columnIndices);

    cudaDeviceSynchronize();
    return 0;
}

