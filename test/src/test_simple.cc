#include <iostream>

#include "test_simple.h"
#include "adj_matrix_dense.h"
#include "adj_list.h"
#include "adj_matrix_csr.h"
#include "spmm_serial.h"
#include "dense_mm_serial.h"
#include "spmm_cuda.h"

void test_simple_spmm() {
    int sample_A[] = {0,2,0,1,0,
                      1,0,3,0,0,
                      0,0,0,0,1,
                      0,0,0,0,0,
                      1,1,0,0,0};

    int sample_B[] = {0,1,0,0,2,
                      2,0,0,1,1,
                      1,0,0,3,0,
                      0,0,2,0,0,
                      0,3,0,1,0};
    
    AdjMatrixDense dense_A(5, sample_A);
    AdjMatrixDense dense_B(5, sample_B);

    AdjMatrixCSR csr_A(dense_A);
    AdjMatrixCSR csr_B(dense_B);

    AdjMatrixDense dense_C = dense_mm_cpu(dense_A, dense_B);
    AdjMatrixDense sparse_C = csr_spmm_dense_cpu(csr_A, csr_B);

    std::cout << "dense_C: " << std::endl;
    dense_C.dump();
    std::cout << "sparse_C: " << std::endl;
    sparse_C.dump();
}

void test_testgraph_spmm_nogpu() {
    AdjEdges edges_A("../../graph/test-graph-A.edges");
    AdjEdges edges_B("../../graph/test-graph-B.edges");

    AdjMatrixDense dense_A(edges_A);
    AdjMatrixDense dense_B(edges_B);

    AdjMatrixCSR csr_A(dense_A);
    AdjMatrixCSR csr_B(dense_B);

    AdjMatrixDense dense_C = dense_mm_cpu(dense_A, dense_B);
    AdjMatrixDense sparse_C = csr_spmm_dense_cpu(csr_A, csr_B);

    std::cout << "dense_C: " << std::endl;
    dense_C.dump();
    std::cout << "sparse_C: " << std::endl;
    sparse_C.dump();
}

void test_testgraph_spmm_gpu() {
    AdjEdges edges_A("../../graph/test-graph-A.edges");
    AdjEdges edges_B("../../graph/test-graph-B.edges");

    AdjMatrixDense dense_A(edges_A);
    AdjMatrixDense dense_B(edges_B);

    AdjMatrixCSR csr_A(dense_A);
    AdjMatrixCSR csr_B(dense_B);

    AdjMatrixDense dense_C = dense_mm_cpu(dense_A, dense_B);
    AdjMatrixDense sparse_C = csr_spmm_dense_cpu(csr_A, csr_B);
    AdjMatrixDense sparse_C_gpu = csr_spmm_dense_cuda(csr_A, csr_B);

    std::cout << "dense_C: " << std::endl;
    dense_C.dump();
    std::cout << "sparse_C: " << std::endl;
    sparse_C.dump();
    std::cout << "sparse_C_gpu: " << std::endl;
    sparse_C_gpu.dump();
}

void test_A() {
    int sample_A[] = {0,2,0,1,0,
                      1,0,3,0,0,
                      0,0,0,0,1,
                      0,0,0,0,0,
                      1,1,0,0,0};

    int sample_B[] = {0,1,0,0,2,
                      2,0,0,1,1,
                      1,0,0,3,0,
                      0,0,2,0,0,
                      0,3,0,1,0};
    
    AdjMatrixDense dense_A(5, sample_A);
    AdjMatrixDense dense_B(5, sample_B);

    AdjMatrixCSR csr_A(dense_A);
    AdjMatrixCSR csr_B(dense_B);

    INT* A_rows = csr_A.get_rows();
    INT* A_cols = csr_A.get_cols();
    INT* A_vals = csr_A.get_vals();

    INT* B_rows = csr_B.get_rows();
    INT* B_cols = csr_B.get_cols();
    INT* B_vals = csr_B.get_vals();

    std::cout << "A - csr" << std::endl;
    std::cout << csr_A.num_rows() << " " << csr_A.num_size() << std::endl;
    for (INT i = 0; i <= csr_A.num_rows(); i++) {
        std::cout << A_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_A.num_size(); i++) {
        std::cout << A_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_A.num_size(); i++) {
        std::cout << A_vals[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "B - csr" << std::endl;
    std::cout << csr_B.num_rows() << " " << csr_B.num_size() << std::endl;
    for (INT i = 0; i <= csr_B.num_rows(); i++) {
        std::cout << B_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_B.num_size(); i++) {
        std::cout << B_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_B.num_size(); i++) {
        std::cout << B_vals[i] << " ";
    }
    std::cout << '\n' << std::endl;

    csr_spmm_cpu(&csr_A, &csr_B);

    std::cout << std::endl;

    AdjMatrixDense dense_C = dense_mm_cpu(dense_A, dense_B);
    std::cout << "Result (dense):" << std::endl;
    for (INT i = 0; i < dense_C.size(); i++) {
        for (INT j = 0; j < dense_C.size(); j++) {
            std::cout << dense_C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    AdjMatrixCSR csr_C(dense_C);

    std::cout << "C - csr" << std::endl;
    INT* C_rows = csr_C.get_rows();
    INT* C_cols = csr_C.get_cols();
    INT* C_vals = csr_C.get_vals();
    std::cout << csr_C.num_rows() << " " << csr_C.num_size() << std::endl;
    for (INT i = 0; i <= csr_C.num_rows(); i++) {
        std::cout << C_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_C.num_size(); i++) {
        std::cout << C_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_C.num_size(); i++) {
        std::cout << C_vals[i] << " ";
    }
    std::cout << '\n' << std::endl;
}

void test_dense() {
    AdjMatrixCSR matrix;
    std::cout << "size " << matrix.num_rows() << std::endl;
    {
        AdjList adjList("../../graph/graph500-scale18-ef16_adj.edges");
        std::cout << adjList.num_vertices() << std::endl;
        std::cout << adjList.num_edges() << std::endl;

        AdjEdges edges("../../graph/graph500-scale18-ef16_adj.edges");
        std::cout << "data load complete" << std::endl;
        std::cout << edges.num_vertices() << std::endl;
        AdjMatrixDense denseMatrix(edges);
        std::cout << denseMatrix.num_edges() << std::endl;
        AdjMatrixCSR csr(denseMatrix);
        matrix = std::move(csr);
    }
    std::cout << "size " << matrix.num_rows() << std::endl;
}