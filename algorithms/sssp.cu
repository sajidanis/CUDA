#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <float.h>

#define BLOCK_SIZE 256

using namespace std;

void read_market_file(const std::string &filename, std::vector<int> &src, std::vector<int> &dst, std::vector<float> &wt, int &numNodes) {
    std::ifstream infile(filename);
    std::string line;
    bool isDirected = true;
    bool isWeighted = false;

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
    if(line.find("weighted") != std::string::npos){
        isWeighted = true;
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
        wt.reserve(2*numEdges);
    } else {
        src.reserve(numEdges);
        dst.reserve(numEdges);
        wt.reserve(numEdges);
    }

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        int u, v, w=1;
        ss >> u >> v;
        if(isWeighted){
            ss >> w;
        }
        // Convert 1-based to 0-based index
        u--;
        v--;
        
        src.push_back(u);
        dst.push_back(v);
        wt.push_back(w);
        if (!isDirected and u != v) {  // Avoid adding self-loops twice
            src.push_back(v);
            dst.push_back(u);
            wt.push_back(w);
        }
    }
    infile.close();
}

__global__ void bellman_ford(int *src, int *dest, float *weight, float *distance, int numEdges, int numNodes){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numEdges){
        int u = src[tid];
        int v = dest[tid];
        float wt = weight[tid];

        if(distance[u] != MAXFLOAT && distance[v] > distance[u] + wt){
            // printf("\ndistance[%d]:  %f wt[%d-%d]: %f\n", u, distance[u], u, v, wt);
            atomicExch(&distance[v], distance[u] + wt);
        }
    }
}

int main(int argc, char **argv) {
    // Graph initialization
     if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];


    // Host variables
    std::vector<int> h_src;
    std::vector<int> h_dest;
    std::vector<float> h_weight;
    
    int numNodes = 0;

    read_market_file(filename, h_src, h_dest, h_weight, numNodes);
    int numEdges = h_src.size();
    std::vector<float> h_distance(numNodes, MAXFLOAT);
    h_distance[0] = 0.0;
    // Device variables
    int *d_src, *d_dest ;
    float *d_distance, *d_weight;
    cudaMalloc(&d_src, numEdges * sizeof(int));
    cudaMalloc(&d_dest, numEdges * sizeof(int));
    cudaMalloc(&d_weight, numEdges * sizeof(float));

    cudaMalloc(&d_distance, numNodes * sizeof(float));

    // Copy the data from host to device
    cudaMemcpy(d_src, h_src.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, h_dest.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, h_distance.data(), numNodes * sizeof(float), cudaMemcpyHostToDevice);
    

    int nblocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run Bellman-Ford algorithm
    for (int i = 0; i < numNodes - 1; ++i) {
        bellman_ford<<<nblocks, BLOCK_SIZE>>>(d_src, d_dest, d_weight, d_distance, numEdges, numNodes);
        cudaDeviceSynchronize();

    }
    float dist[numNodes];

    cudaMemcpy(dist, d_distance, numNodes * sizeof(float), cudaMemcpyDeviceToHost);


    int N = numNodes > 40 ? 40 : numNodes;
    for(int i = 0 ; i < N; i++){
        cout << "Node " << i << " : " << dist[i] << "\n";
    }

    return 0;
}