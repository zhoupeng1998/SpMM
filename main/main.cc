#include <iostream>

#include "adj_edges.h"
#include "adj_list.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "test_gpu.h"
#include "test_simple.h"

int main(void) {


    AdjEdges test= AdjEdges("../../graph/graph500-scale19-ef16_adj.edges");

    AdjMatrixCSR test_csr(test);
    std::cout << "size " << test_csr.num_rows() << std::endl;


//    AdjMatrixCSR matrix;
//    std::cout << "size " << matrix.num_rows() << std::endl;
//    {
//        AdjList adjList("graph500-scale18-ef16_adj.edges");
//        std::cout << adjList.num_vertices() << std::endl;
//        std::cout << adjList.num_edges() << std::endl;
        /*
        AdjEdges edges("graph500-scale18-ef16_adj.edges");
        std::cout << "data load complete" << std::endl;
        std::cout << edges.num_vertices() << std::endl;
        AdjMatrixDense denseMatrix(edges);
        std::cout << denseMatrix.num_edges() << std::endl;
        AdjMatrixCSR csr(denseMatrix);
        matrix = std::move(csr);
        */
    //}
    //std::cout << "size " << matrix.num_rows() << std::endl;
    return 0;
}