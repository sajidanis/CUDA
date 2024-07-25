#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

void read_market_file(const std::string &filename, std::vector<int> &adjacencyList, std::vector<int> &offsets, int &numNodes) {
    std::ifstream infile(filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    while (std::getline(infile, line)) {
        if (line[0] == '%') continue;
        std::stringstream ss(line);
        int u, v;
        ss >> u >> v;

        // Assuming 0-based index for nodes
        if (u >= numNodes) numNodes = u + 1;
        if (v >= numNodes) numNodes = v + 1;

        while (offsets.size() <= u) offsets.push_back(adjacencyList.size());
        adjacencyList.push_back(v);
    }
    offsets.push_back(adjacencyList.size());

    infile.close();
}

#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void bfs_kernel(int *adjacencyList, int *offsets, int *levels, int *frontier, int *new_frontier, int numNodes, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && frontier[idx]) {
        int start = offsets[idx];
        int end = offsets[idx + 1];
        for (int i = start; i < end; i++) {
            int neighbor = adjacencyList[i];
            if (atomicCAS(&levels[neighbor], -1, level) == -1) {
                new_frontier[neighbor] = 1;
            }
        }
    }
}

__global__ void check_empty_frontier(int *frontier, int numNodes, bool *d_done) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && frontier[idx]) {
        *d_done = false;
    }
}

void bfs_cuda(int *adjacencyList, int *offsets, int numNodes, int startNode) {
    int *d_adjacencyList, *d_offsets, *d_levels, *d_frontier, *d_new_frontier;
    bool *d_done;
    size_t adjListSize = offsets[numNodes] * sizeof(int);
    size_t offsetsSize = (numNodes + 1) * sizeof(int);
    size_t levelsSize = numNodes * sizeof(int);

    std::vector<int> levels(numNodes, -1);
    levels[startNode] = 0;

    std::vector<int> frontier(numNodes, 0);
    frontier[startNode] = 1;

    CUDA_CALL(cudaMalloc(&d_adjacencyList, adjListSize));
    CUDA_CALL(cudaMalloc(&d_offsets, offsetsSize));
    CUDA_CALL(cudaMalloc(&d_levels, levelsSize));
    CUDA_CALL(cudaMalloc(&d_frontier, levelsSize));
    CUDA_CALL(cudaMalloc(&d_new_frontier, levelsSize));
    CUDA_CALL(cudaMalloc(&d_done, sizeof(bool)));

    CUDA_CALL(cudaMemcpy(d_adjacencyList, adjacencyList, adjListSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_offsets, offsets, offsetsSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_levels, levels.data(), levelsSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_frontier, frontier.data(), levelsSize, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    int level = 1;
    bool done;
    do {
        CUDA_CALL(cudaMemset(d_new_frontier, 0, levelsSize));
        bfs_kernel<<<gridSize, blockSize>>>(d_adjacencyList, d_offsets, d_levels, d_frontier, d_new_frontier, numNodes, level);
        CUDA_CALL(cudaDeviceSynchronize());

        done = true;
        CUDA_CALL(cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice));

        check_empty_frontier<<<gridSize, blockSize>>>(d_new_frontier, numNodes, d_done);
        
        CUDA_CALL(cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));

        int *temp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp;

        level++;
    } while (!done);

    CUDA_CALL(cudaMemcpy(levels.data(), d_levels, levelsSize, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_adjacencyList));
    CUDA_CALL(cudaFree(d_offsets));
    CUDA_CALL(cudaFree(d_levels));
    CUDA_CALL(cudaFree(d_frontier));
    CUDA_CALL(cudaFree(d_new_frontier));
    CUDA_CALL(cudaFree(d_done));
}

int main(int argc, char **argv) {
    // Example graph
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];

    std::vector<int> adjacencyList;
    std::vector<int> offsets;
    int numNodes = 0;

    read_market_file(filename, adjacencyList, offsets, numNodes);

    int startNode = 0;

    const auto start = std::chrono::high_resolution_clock::now();

    bfs_cuda(adjacencyList.data(), offsets.data(), numNodes, startNode);

    const auto end = std::chrono::high_resolution_clock::now();

    long double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    diff *= 1e-9;

    std::cout << "BFS Time: " << diff << '\n';

    return 0;
}
