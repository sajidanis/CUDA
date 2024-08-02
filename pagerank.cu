#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>
#include <fstream>

#define NUM_NODES 5
#define BLOCK_SIZE 256
#define EPSILON 1e-6
#define DAMPING_FACTOR 0.85

void read_market_file(const std::string &filename, std::vector<int> &adjacencyList, std::vector<int> &offsets, int &numNodes) {
    std::ifstream infile(filename);
    std::string line;
    bool isDirected = true;

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Parse the header
    std::getline(infile, line);  // Skip MatrixMarket header line
    std::getline(infile, line);  // Skip kind: undirected graph line
    
    if(line.find("undirected") != std::string::npos){
        isDirected = false;
    }

    // Read the dimensions of the graph
    std::getline(infile, line);
    int numEdges;
    std::stringstream ss(line);
    ss >> numNodes >> numNodes >> numEdges;

    // Reserve space for adjacency list and offsets
    if(!isDirected){
        adjacencyList.resize(2 * numEdges);
    } else {
        adjacencyList.resize(numEdges);
    }
    offsets.resize(numNodes + 1, 0);

    // Read all edges
    std::vector<std::pair<int, int>> edges;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        int u, v;
        ss >> u >> v;
        
        // Convert 1-based to 0-based index
        u--;
        v--;

        edges.emplace_back(u, v);
        if (!isDirected and u != v) {  // Avoid adding self-loops twice
            edges.emplace_back(v, u);
        }

        offsets[u + 1]++;
        offsets[v + 1]++;
    }
    infile.close();

    // Convert counts to offsets
    for (int i = 1; i <= numNodes; ++i) {
        offsets[i] += offsets[i - 1];
    }

    // Fill adjacency list using offsets
    std::vector<int> tempOffsets = offsets;
    for (const auto& edge : edges) {
        adjacencyList[tempOffsets[edge.first]++] = edge.second;
    }
}

__global__ void pageRankKernel(int *adjacencyList, int *offsets, int *new_rank, int *rank, int numEdges, int numNodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEdges) {
        int src = d_src[idx];
        int dest = d_dest[idx];
        atomicAdd(&d_newRank[dest], d_rank[src] / numNodes);
    }
}

int main(int argc, char **argv) {
    // Graph initialization
     if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];

    std::vector<int> adjacencyList;
    std::vector<int> offsets;
    int numNodes = 0;
    int numEdges = *(offsets.end() - 1);

    read_market_file(filename, adjacencyList, offsets, numNodes);



    // Host variables
    std::vector<int> h_src(numEdges);
    std::vector<int> h_dest(numEdges);
    std::vector<float> h_rank(numNodes, 1.0f / numNodes);
    std::vector<float> h_newRank(numNodes, 0.0f);

    // Device variables
    int *d_src, *d_dest;
    float *d_rank, *d_newRank;
    cudaMalloc((void**)&d_src, numEdges * sizeof(int));
    cudaMalloc((void**)&d_dest, numEdges * sizeof(int));
    cudaMalloc((void**)&d_rank, numNodes * sizeof(float));
    cudaMalloc((void**)&d_newRank, numNodes * sizeof(float));

    cudaMemcpy(d_src, h_src.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, h_rank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newRank, h_newRank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Main loop for PageRank computation
    while (true) {
        cudaMemset(d_newRank, 0, numNodes * sizeof(float));
        pageRankKernel<<<numBlocks, BLOCK_SIZE>>>(d_src, d_dest, d_rank, d_newRank, numEdges, numNodes);

        cudaMemcpy(h_newRank.data(), d_newRank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

        float diff = 0.0f;
        for (int i = 0; i < numNodes; ++i) {
            h_newRank[i] = h_newRank[i] * DAMPING_FACTOR + (1.0f - DAMPING_FACTOR) / numNodes;
            diff += fabs(h_newRank[i] - h_rank[i]);
        }

        if (diff < EPSILON) {
            break;
        }

        cudaMemcpy(d_rank, h_newRank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
        std::swap(h_rank, h_newRank);
    }

    cudaMemcpy(h_rank.data(), d_rank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the PageRank values
    for (int i = 0; i < numNodes; ++i) {
        std::cout << "Node " << i << " PageRank: " << h_rank[i] << std::endl;
    }

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_rank);
    cudaFree(d_newRank);

    return 0;
}
