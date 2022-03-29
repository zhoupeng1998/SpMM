#include <iostream>

#include "test.h"
#include "adj_edges.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"

int main(void) {
    std::cout << "hello world" << std::endl;
    printInfo();
    AdjEdges edges("graph500-scale18-ef16_adj.edges");
    std::cout << "data load complete" << std::endl;
    std::cout << edges.num_vertices() << std::endl;
    AdjMatrixDense denseMatrix(edges);
    std::cout << denseMatrix.num_edges() << std::endl;
    AdjMatrixCSR csr(denseMatrix);
    return 0;
}