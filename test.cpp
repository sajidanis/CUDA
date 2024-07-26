#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

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
        adjacencyList.resize(2 * numEdges, -1);
    } else {
        adjacencyList.resize(numEdges, -1);
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

int main(){

    std::string filename = "/home/sajid/Documents/graphs/chesapeake.mtx";

    std::vector<int> adjacencyList;
    std::vector<int> offsets;
    int numNodes = 0;

    read_market_file(filename, adjacencyList, offsets, numNodes);

    std::cout << "adj list : ";
    for(auto &el : adjacencyList){
        std::cout << el << " ";
    }
    std::cout << "\n";

    std::cout << "offsets : ";
    for(auto &el : offsets){
        std::cout << el << " ";
    }
    std::cout << "\n";
}