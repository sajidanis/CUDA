//
// Created by sajid on 8/26/24.
//

#ifndef MARKETREADER_CUH
#define MARKETREADER_CUH

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

typedef std::map<std::string, std::string> MetaData;

typedef struct COO_t{
    size_t nodes;
    size_t edges;
	std::vector<int> src;
	std::vector<int> dest;
	std::vector<float> wt;
    std::vector<int> indegree;
    std::vector<int> outdegree;
} COO_t;

typedef struct CSR_t{
    size_t nodes;
    size_t edges;
    std::vector<int> offsets;
    std::vector<int> columnIndices;
    std::vector<float> edgeValues;
} CSR_t;

/**
* @brief Reads a MARKET graph from an input-stream into a CSR sparse format
*
* Here is an example of the matrix market format
* +----------------------------------------------+
* |%%MatrixMarket matrix coordinate real general | <--- header line
* |%                                             | <--+
* |% comments                                    |    |-- 0 or more comment
* lines
* |%                                             | <--+
* |  M N L                                       | <--- rows, columns, entries
* |  I1 J1 A(I1, J1)                             | <--+
* |  I2 J2 A(I2, J2)                             |    |
* |  I3 J3 A(I3, J3)                             |    |-- L lines
* |     . . .                                    |    |
* |  IL JL A(IL, JL)                             | <--+
* +----------------------------------------------+
*/
void ReadMarketStream(std::string fileName, COO_t &cooT){
	std::ifstream infile(fileName);

	if (!infile.is_open()) {
        std::cerr << "[-] Error opening file: " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

  	size_t nodes = 0;
  	size_t edges = 0;
  	bool got_edge_values = false;
  	bool undirected = false;  // whether the graph is undirected
    std::cout << "\n[+] Reading the market file :" << fileName << "\n";
    const auto start = std::chrono::high_resolution_clock::now();

	std::string line;
  	while (true) {
    	std::getline(infile, line);
    	if (line[0] != '%') {
      		break;
    	} else {
      		if (strlen(line.c_str()) >= 2) {
				got_edge_values = (strstr(line.c_str(), "weighted") != NULL);
        		undirected = (strstr(line.c_str(), "undirected") != NULL);
      		}
    	}
	}

	long long nodes_x, nodes_y, ll_edges;
	std::stringstream ss(line);
	ss >> nodes_x >> nodes_y >> ll_edges;

	nodes = nodes_x;
	edges = ll_edges;

	if(undirected) edges *= 2;

    cooT.nodes = nodes;
    cooT.edges = edges;
	cooT.src.reserve(edges);
	cooT.dest.reserve(edges);
    cooT.indegree.assign(nodes+1, 0);
    cooT.outdegree.assign(nodes+1, 0);
	cooT.wt.reserve(edges);

	while(std::getline(infile, line)) {
		std::stringstream ss(line);
		long long u, v;
        float w = 1.0;
		ss >> u >> v;

		// convert 1-based to 0-based
		u--; v--;
		cooT.src.push_back(u);
		cooT.dest.push_back(v);
        cooT.indegree[v]++;
        cooT.outdegree[u]++;
		if(got_edge_values){
			ss >> w;
		}
        cooT.wt.push_back(w);

		if(undirected and u != v){
			cooT.src.push_back(v);
			cooT.src.push_back(u);
            cooT.wt.push_back(w);
            cooT.indegree[u]++;
            cooT.outdegree[v]++;
		}
    }
    infile.close();
    const auto end = std::chrono::high_resolution_clock::now();
    long double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    diff *= 1e-6;

    std::cout << "[+] Time taken to read the market file : " << diff << " ms\n";
}

void COO_to_CSR(COO_t &cooT, CSR_t &csrT){
    const auto start = std::chrono::high_resolution_clock::now(); // for timing

    auto offsets = cooT.outdegree;
    size_t numNodes = cooT.nodes;
    size_t numEdges = cooT.edges;
    csrT.nodes = numNodes;

    // Creating the offset from the outdegree gained at the reading time
    auto data = offsets.data();

    thrust::device_vector<int> d_offset(offsets); // copy the 

    thrust::exclusive_scan(d_offset.begin(), d_offset.end(), d_offset.begin());

    csrT.offsets.resize(numNodes+1);
    thrust::copy(d_offset.begin(), d_offset.end(), csrT.offsets.begin());

    // csrT.d_offsets = thrust::raw_pointer_cast(d_offset.data());
    

    // Create the adjcency list by sorting the datas as we can have edgevalues so need to get the keys and then sort it accordingly

    thrust::device_vector<int> d_src(cooT.src);
    thrust::device_vector<int> d_dest(cooT.dest);
    thrust::device_vector<float> d_edgeValues(cooT.wt);
    
    // Step 1: Create an index vector and initialize it using thrust::sequence
    thrust::device_vector<int> d_indices(d_src.size());
    thrust::sequence(thrust::cuda::par_nosync, d_indices.begin(), d_indices.end());

    // Step 2: Sort the indices based on the src vector using thrust::sort_by_key
    thrust::sort_by_key(thrust::cuda::par_nosync, d_src.begin(), d_src.end(), d_indices.begin());

    // Step 3: Use thrust::gather to reorder each iterator asynchronously based on the sorted indices
    thrust::device_vector<int> d_sorted_src(d_src.size());
    thrust::device_vector<int> d_sorted_dest(d_dest.size());
    thrust::device_vector<float> d_sorted_edgeValues(d_edgeValues.size());

    // Reorder src, dest, and edgeValues using the sorted indices
    thrust::gather(thrust::cuda::par_nosync, d_indices.begin(), d_indices.end(), d_src.begin(), d_sorted_src.begin());
    thrust::gather(thrust::cuda::par_nosync, d_indices.begin(), d_indices.end(), d_dest.begin(), d_sorted_dest.begin());
    thrust::gather(thrust::cuda::par_nosync, d_indices.begin(), d_indices.end(), d_edgeValues.begin(), d_sorted_edgeValues.begin());

    // Copy the device vector to host vector
    // thrust::host_vector<int> h_columnIndices = d_sorted_dest;
    // thrust::host_vector<float> h_edgeValues = d_sorted_edgeValues;

    // Copy the device vector to std::vector
    csrT.columnIndices.resize(numEdges);
    csrT.edgeValues.resize(numEdges);

    thrust::copy(d_sorted_dest.begin(), d_sorted_dest.end(), csrT.columnIndices.begin());
    thrust::copy(d_sorted_edgeValues.begin(), d_sorted_edgeValues.end(), csrT.edgeValues.begin());

    // csrT.d_columnIndices = thrust::raw_pointer_cast(d_sorted_dest.data());
    // csrT.d_edgeValues = thrust::raw_pointer_cast(d_edgeValues.data());

    const auto end = std::chrono::high_resolution_clock::now();
    long double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    diff *= 1e-6;

    std::cout << "[+] Time taken to convert the coo format to csr : " << diff << " ms\n";

}
#endif //MARKETREADER_CUH
