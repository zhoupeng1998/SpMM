#include <iostream>

#include "test.h"
#include "adj_edges.h"
#include "adj_list.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"

int main(void) {
    std::cout << "hello world" << std::endl;
    printInfo();
    AdjMatrixCSR matrix;
    std::cout << "size " << matrix.num_rows() << std::endl;
    {
        AdjList adjList("graph500-scale18-ef16_adj.edges");
        std::cout << adjList.num_vertices() << std::endl;
        std::cout << adjList.num_edges() << std::endl;
        /*
        AdjEdges edges("graph500-scale18-ef16_adj.edges");
        std::cout << "data load complete" << std::endl;
        std::cout << edges.num_vertices() << std::endl;
        AdjMatrixDense denseMatrix(edges);
        std::cout << denseMatrix.num_edges() << std::endl;
        AdjMatrixCSR csr(denseMatrix);
        matrix = std::move(csr);
        */
    }
    std::cout << "size " << matrix.num_rows() << std::endl;
    return 0;
}