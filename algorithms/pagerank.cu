#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>
#include <fstream>

#define BLOCK_SIZE 256
#define EPSILON 1e-6
#define DAMPING_FACTOR 0.85
#define MAX_ITER 10

void read_market_file(const std::string &filename, std::vector<int> &src, std::vector<int> &dst, int &numNodes) {
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

    // Reserve space for edges
    if(!isDirected){
        src.reserve(2*numEdges);
        dst.reserve(2*numEdges);
    }

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        int u, v;
        ss >> u >> v;
        
        // Convert 1-based to 0-based index
        u--;
        v--;

        src.push_back(u);
        dst.push_back(v);
        if (!isDirected and u != v) {  // Avoid adding self-loops twice
            src.push_back(v);
            dst.push_back(u);
        }
    }
    infile.close();
}

__global__ void computeOutbound(int *src, int *outbound, int numEdges){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEdges) {
        int s = src[idx];
        atomicAdd(&outbound[s], 1);
    }
}

__global__ void pageRankKernel(int *src, int *dest, int *outbound, float *rank, float *new_rank, int numEdges, int numNodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEdges) {
        int s = src[idx];
        int d = dest[idx];
        if(outbound[s] > 0){
            atomicAdd(&new_rank[d], rank[s] / outbound[s]);
        } else {
            atomicAdd(&new_rank[d], rank[s] / numNodes);
        }
    }
}

__global__ void applyDampingFactorKernel(float *new_rank, float *rank, int numNodes, float *diff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numNodes) {
        new_rank[idx] = (DAMPING_FACTOR * new_rank[idx]) + (1.0f - DAMPING_FACTOR) / numNodes;
        diff[0] += fabs(new_rank[idx] - rank[idx]);
    }
}

int main(int argc, char **argv) {
    // Graph initialization
     if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];

    std::vector<int> h_src;
    std::vector<int> h_dest;
    
    int numNodes = 0;

    read_market_file(filename, h_src, h_dest, numNodes);
    int numEdges = h_src.size();

    // Host variables
    std::vector<float> h_rank(numNodes, 1.0f / numNodes);
    std::vector<float> h_newRank(numNodes, 0.0f);
    

    // Device variables
    int *d_src, *d_dest, *d_outbound;
    float *d_rank, *d_newRank, *d_diff;
    cudaMalloc(&d_src, numEdges * sizeof(int));
    cudaMalloc(&d_dest, numEdges * sizeof(int));
    cudaMalloc(&d_rank, numNodes * sizeof(float));
    cudaMalloc(&d_newRank, numNodes * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));
    cudaMalloc(&d_outbound, numNodes * sizeof(float));

    cudaMemcpy(d_src, h_src.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank, h_rank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_newRank, h_newRank.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Preprocess for outbound computation
    cudaMemset(d_outbound, 0, numNodes * sizeof(int));
    computeOutbound<<<numBlocks, BLOCK_SIZE>>>(d_src, d_outbound, numEdges);

    float *diff = (float *)malloc(sizeof(float));
    // Main loop for PageRank computation
    int iter = 0;
    while (iter < MAX_ITER) {
        numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemset(d_newRank, 0, numNodes * sizeof(float));
        pageRankKernel<<<numBlocks, BLOCK_SIZE>>>(d_src, d_dest, d_outbound, d_rank, d_newRank, numEdges, numNodes);


        numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        applyDampingFactorKernel<<<numBlocks, BLOCK_SIZE>>>(d_newRank, d_rank, numNodes, d_diff);

        cudaDeviceSynchronize();

        std::swap(d_rank, d_newRank);
        iter++;
    }

    cudaMemcpy(h_rank.data(), d_rank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    long long N = numNodes > 40 ? 40 : numNodes;
    // Output the PageRank values
    for (int i = 0; i < N; ++i) {
        std::cout << "Node " << i << " PageRank: " << h_rank[i] << std::endl;
    }

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_rank);
    cudaFree(d_newRank);

    return 0;
}
