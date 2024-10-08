#ifndef AGGREGATE_PR_CUH
#define AGGREGATE_PR_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 256
#define MAX_ITER 32
#define DAMPING_FACTOR 0.85f

#define cudaCheckError() {                                 \
    cudaError_t e = cudaGetLastError();                    \
    if (e != cudaSuccess) {                                \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                \
    }                                                      \
}

#define CUDA_GUARD(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Constants


// const float EPSILON = 1e-6f;

// CSR format structure
struct d_CSRGraph {
    int *row_ptr;  // CSR row pointer
    int *col_idx;  // CSR column indices
    int num_nodes;
    int num_edges;
};

struct CSRGraph {
    std::vector<int> row_ptr;  // CSR row pointer
    std::vector<int> col_idx;  // CSR column indices
    int num_nodes;
    int num_edges;
};

// Utility function to copy the CSR graph to the device
void copy_graph_to_device(CSRGraph& h_graph, d_CSRGraph& d_graph) {
    // Allocate memory for row_ptr, col_idx, and values on the GPU

    CUDA_GUARD(cudaMalloc(&d_graph.row_ptr, (h_graph.num_nodes + 1) * sizeof(int)));
    
    CUDA_GUARD(cudaMalloc(&d_graph.col_idx, h_graph.num_edges * sizeof(int)));

    // Copy the CSR data from the host to the device
    CUDA_GUARD(cudaMemcpy(d_graph.row_ptr, h_graph.row_ptr.data(), (h_graph.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_GUARD(cudaMemcpy(d_graph.col_idx, h_graph.col_idx.data(), h_graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));

    // Copy the num_nodes and num_edges from the host to the device
    d_graph.num_nodes = h_graph.num_nodes;
    d_graph.num_edges = h_graph.num_edges;
}

// Function to sync host graph changes with the device graph
void sync_graph_to_device(CSRGraph& h_graph, d_CSRGraph& d_graph) {
    // Free previous device memory
    CUDA_GUARD(cudaFree(d_graph.row_ptr));
    CUDA_GUARD(cudaFree(d_graph.col_idx));

    // Copy the updated graph from host to device
    copy_graph_to_device(h_graph, d_graph);
}

// Function to add an edge to the CSR graph
void add_edge(CSRGraph& h_graph, int src, int dst, std::vector<std::pair<int, int>> &changes) {

    int pos = h_graph.row_ptr[src + 1];
    // Check for duplications if that edge is already present or not
    for (int i = h_graph.row_ptr[src]; i < h_graph.row_ptr[src + 1]; ++i) {
        if (h_graph.col_idx[i] == dst) {
            return;
        }
    }
    changes.emplace_back(src, dst);
    h_graph.num_edges++;

    for (int i = src + 1; i <= h_graph.num_nodes; ++i) {
        h_graph.row_ptr[i]++;
    }
    
    h_graph.col_idx.insert(h_graph.col_idx.begin() + pos, dst);
}

// Function to remove an edge from the CSR graph
void remove_edge(CSRGraph& h_graph, int src, int dst, std::vector<std::pair<int, int>> &changes) {
    int pos = -1;
    for (int i = h_graph.row_ptr[src]; i < h_graph.row_ptr[src + 1]; ++i) {
        if (h_graph.col_idx[i] == dst) {
            pos = i;
            break;
        }
    }

    if (pos == -1) {
        std::cout << "Edge not found!\n";
        return;
    }

    changes.emplace_back(src, dst);
    h_graph.num_edges--;

    h_graph.col_idx.erase(h_graph.col_idx.begin() + pos);

    for (int i = src + 1; i <= h_graph.num_nodes; ++i) {
        h_graph.row_ptr[i]--;
    }
}

// Function to capture changes (return affected nodes)
std::vector<int> capture_changes(const std::vector<std::pair<int, int>>& changes) {
    std::vector<int> affected_nodes;

    for (const auto& change : changes) {
        affected_nodes.push_back(change.first);
        affected_nodes.push_back(change.second);
    }

    // Remove duplicates and sort the affected nodes
    std::sort(affected_nodes.begin(), affected_nodes.end());
    affected_nodes.erase(std::unique(affected_nodes.begin(), affected_nodes.end()), affected_nodes.end());

    return affected_nodes;
}

// Kernel to initialize PageRank values
__global__ void init_pagerank(float *pagerank, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        pagerank[idx] = 1.0f / num_nodes;
    }
}

// Kernel to perform PageRank update for the entire graph
__global__ void pagerank_update(const d_CSRGraph graph, float *pagerank, float *new_pagerank, float damping) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < graph.num_nodes) {
        float sum = 0.0f;
        
        // Iterate over the neighbors of the node in CSR format
        for (int neighbor_idx = graph.row_ptr[node]; neighbor_idx < graph.row_ptr[node + 1]; ++neighbor_idx) {
            int neighbor = graph.col_idx[neighbor_idx];
            int out_degree = graph.row_ptr[neighbor + 1] - graph.row_ptr[neighbor];
            if (out_degree > 0) {
                sum += pagerank[neighbor] / out_degree;
            }
        }
        
        // Apply PageRank formula
        new_pagerank[node] = (1.0f - damping) / graph.num_nodes + damping * sum;
    }
}

// Kernel to handle dynamic updates to the PageRank (only affected nodes)
__global__ void pagerank_update_dynamic(const d_CSRGraph graph, float *pagerank, float *new_pagerank, int *affected_nodes, int num_affected, float damping) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_affected) {
        int node = affected_nodes[idx];
        float sum = 0.0f;
        
        // Iterate over the neighbors of the node in CSR format
        for (int neighbor_idx = graph.row_ptr[node]; neighbor_idx < graph.row_ptr[node + 1]; ++neighbor_idx) {
            int neighbor = graph.col_idx[neighbor_idx];
            int out_degree = graph.row_ptr[neighbor + 1] - graph.row_ptr[neighbor];
            if (out_degree > 0) {
                sum += pagerank[neighbor] / out_degree;
            }
        }
        
        // Apply PageRank formula
        new_pagerank[node] = (1.0f - damping) / graph.num_nodes + damping * sum;
    }
}

// Host function to run PageRank with dynamic updates
thrust::host_vector<float> run_init_pr(CSRGraph &h_graph, d_CSRGraph &d_graph, thrust::device_vector<float> &d_pagerank) {

    // Allocate device memory for PageRank arrays
    // thrust::device_vector<float> d_pagerank(h_graph.num_nodes);
    thrust::device_vector<float> d_new_pagerank(h_graph.num_nodes);

    // Initialize PageRank values on the device
    int numBlocks = (d_graph.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_pagerank<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_pagerank.data()), h_graph.num_nodes);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Iterate over PageRank updates (initial global update)
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        pagerank_update<<<numBlocks, BLOCK_SIZE>>>(d_graph, 
            thrust::raw_pointer_cast(d_pagerank.data()), 
            thrust::raw_pointer_cast(d_new_pagerank.data()), 
            DAMPING_FACTOR);
        cudaDeviceSynchronize();
        cudaCheckError();
        // Swap PageRank arrays
        d_pagerank.swap(d_new_pagerank);
        // Optionally: check for convergence (not implemented here)
    }

    // Copy final PageRank values back to host
    thrust::host_vector<float> h_pagerank = d_pagerank;

    std::cout << "\n Page rank static version\n";
    // Print the final PageRank values (for demonstration)
    for (int i = 0; i < (h_graph.num_nodes < 40 ? h_graph.num_nodes : 40); ++i) {
        std::cout << h_pagerank[i] << " ";
    }
    std::cout << "\n\n";
    return h_pagerank;
}

thrust::host_vector<float> run_pr_update(CSRGraph &h_graph, d_CSRGraph &d_graph, std::vector<std::pair<int, int>>& changes, thrust::device_vector<float> &d_pagerank){
    // Process dynamic changes (node/edge addition/removal)
    if (!changes.empty()) {
        auto affected_nodes = capture_changes(changes);

        // Copy affected nodes to the device
        thrust::device_vector<int> d_affected_nodes(affected_nodes.begin(), affected_nodes.end());

        // New page rank vector
        thrust::device_vector<float> d_new_pagerank(h_graph.num_nodes);
        
        // Perform PageRank update only for affected nodes
        int num_affected = affected_nodes.size();
        int affectedBlocks = (num_affected + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int iter = 0; iter < MAX_ITER / 4; ++iter) {

            pagerank_update_dynamic<<<affectedBlocks, BLOCK_SIZE>>>(d_graph, 
                thrust::raw_pointer_cast(d_pagerank.data()), 
                thrust::raw_pointer_cast(d_new_pagerank.data()), 
                thrust::raw_pointer_cast(d_affected_nodes.data()), 
                num_affected, DAMPING_FACTOR);
            cudaDeviceSynchronize();
            cudaCheckError();
            // Swap PageRank arrays after the dynamic update
            d_pagerank.swap(d_new_pagerank);
        }
    }

    // Copy final PageRank values back to host
    thrust::host_vector<float> h_pagerank = d_pagerank;

    // Print the final PageRank values (for demonstration)
    std::cout << "\n\n Page rank dynamically updated\n";
    for (int i = 0; i < (h_graph.num_nodes < 40 ? h_graph.num_nodes : 40); ++i) {
        std::cout << h_pagerank[i] << " ";
    }
    std::cout << "\n\n";
    return h_pagerank;
}

void generate_random_updations(CSRGraph& h_graph, int n, int num_nodes, std::vector<std::pair<int, int>> &changes){
    srand(time(0));  // Seed for random number generation

    int num_additions = n * 0.8;  // 80% of updations are additions
    int num_deletions = n - num_additions;  // The other half will be deletions

    // Apply random additions
    for (int i = 0; i < num_additions; ++i) {
        int src = rand() % num_nodes;  // Random source node
        int dst = rand() % num_nodes;  // Random destination node

        // Ensure we're not adding a self-loop
        while (dst == src) {
            dst = rand() % num_nodes;
        }

        // Add edge with a random weight (or default weight)
        add_edge(h_graph, src, dst, changes);
    }

    // Apply random deletions
    for (int i = 0; i < num_deletions; ++i) {
        int src = rand() % num_nodes;  // Random source node

        // Get the range of the current node's neighbors from the CSR graph
        int start = h_graph.row_ptr[src];
        int end = h_graph.row_ptr[src + 1];

        // If the node has no edges, skip to the next iteration
        if (start == end) {
            continue;
        }

        // Choose a random neighbor (destination node) to remove the edge
        int idx = start + rand() % (end - start);
        int dst = h_graph.col_idx[idx];

        remove_edge(h_graph, src, dst, changes);
    }
}

// Functor to compute the absolute difference
struct abs_diff : public thrust::binary_function<float, float, float> {
    __host__ __device__ float operator()(const float& a, const float& b) const {
        return fabs(a - b);
    }
};
float calcErrorMargin(thrust::host_vector<float> dyn_pr, thrust::host_vector<float> static_pr){
    if (dyn_pr.size() != static_pr.size()) {
        std::cerr << "Error: Vectors must be of the same size." << std::endl;
        return -1.0;
    }

    // Convert to device vectors for parallel computation
    thrust::device_vector<float> d_dyn_pr = dyn_pr;
    thrust::device_vector<float> d_static_pr = static_pr;

    // Calculate the absolute difference between the two vectors
    thrust::device_vector<float> d_abs_diff(d_dyn_pr.size());

    thrust::transform(d_dyn_pr.begin(), d_dyn_pr.end(), d_static_pr.begin(), 
                      d_abs_diff.begin(), abs_diff());

    // Compute the mean absolute error
    float mae = thrust::reduce(d_abs_diff.begin(), d_abs_diff.end(), 0.0f, thrust::plus<float>()) / d_abs_diff.size();

    return mae;
}

void test_pr_dynamic(CSRGraph &graph, int n){
// changes vector
    std::vector<std::pair<int, int>> changes;

    //GPU-variables 
    d_CSRGraph d_graph;
    thrust::device_vector<float> d_pagerank(graph.num_nodes); 

    // Transfer the CSRgraph from host to device
    auto start = std::chrono::high_resolution_clock::now(); // for timing
    copy_graph_to_device(graph, d_graph);
    auto end = std::chrono::high_resolution_clock::now();

    long double copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    copy_time *= 1e-6;


    // Run Initialization for the page rank
    start = std::chrono::high_resolution_clock::now();
    run_init_pr(graph, d_graph, d_pagerank);
    end = std::chrono::high_resolution_clock::now();

    long double init_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    init_time *= 1e-6;

    // Example dynamic changes (edge addition/removal)
    generate_random_updations(graph, n, graph.num_nodes, changes);

    // Sync the device graphs
    start = std::chrono::high_resolution_clock::now();
    sync_graph_to_device(graph, d_graph);
    end = std::chrono::high_resolution_clock::now();

    long double sync_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    sync_time *= 1e-6;


    // Call for the dynamic pr algo
    start = std::chrono::high_resolution_clock::now();
    auto dyn_pr = run_pr_update(graph, d_graph, changes, d_pagerank);
    end = std::chrono::high_resolution_clock::now();

    long double dyn_upd_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dyn_upd_time *= 1e-6;

    start = std::chrono::high_resolution_clock::now();
    auto static_pr = run_init_pr(graph, d_graph, d_pagerank); // For comparison with the dynamic updates
    end = std::chrono::high_resolution_clock::now();

    long double static_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    static_time *= 1e-6;

    // Calculate error margin
    auto mae = calcErrorMargin(dyn_pr, static_pr);

    std::cout << "\n";
    std::cout << "Copy Time : " << copy_time << " ms\n";
    std::cout << "Init Time : " << init_time << " ms\n";
    std::cout << "Sync Time : " << sync_time << " ms\n";
    std::cout << "Dyn Update Time : " << dyn_upd_time << " ms\n";
    std::cout << "Static Update Time : " << static_time << " ms\n";
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
}

// // Main function
// int main() {
//     // Create the graph

//     CSRGraph graph;
//     graph.row_ptr= {0, 2, 3, 6, 8};
//     graph.col_idx = {1, 3, 0, 1, 2, 3, 0, 2};
//     graph.num_nodes = 4;
//     graph.num_edges = 8;

//     run(graph, 4);

//     return 0;
// }


#endif