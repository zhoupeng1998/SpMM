#include <unistd.h>

#include <iostream>

#include "test_generate_graph.h"
#include "graph_generator.h"

// Will produce seg fault by the end
void test_generate_graph() {
    GraphGenerator generator(50, 20);
    generator.generate();
    generator.store_graph("../../graph/test-graph-B.edges");
    
    AdjMatrixDense dense1 = generator.get_graph_dense();
    std::cout << "dense1: " << std::endl;

    AdjEdges edges("../../graph/test-graph-B.edges");
    std::cout << "edges: " << std::endl;
    AdjMatrixDense dense2(edges);
    std::cout << "dense2: " << std::endl;

    std::cout << dense1.num_vertices() << " " << dense2.num_vertices() << std::endl;
    //dense1.dump();
    std::cout << std::endl;
    dense2.dump();
    std::cout << "END" << std::endl;
}

void produce_graph(int size, int nnz) {
    GraphGenerator generator(size, nnz);
    generator.generate();
    generator.store_graph("../../graph/test-graph-A.edges");
    sleep(2);
    generator.generate();
    generator.store_graph("../../graph/test-graph-B.edges");
}