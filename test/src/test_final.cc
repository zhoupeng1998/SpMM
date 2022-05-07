#include <iostream>

#include "test_final.h"
#include "test_generate_graph.h"

#include "adj_edges.h"
#include "adj_matrix_dense_linear.h"
#include "adj_matrix_csr.h"
#include "spmm_serial.h"
#include "spmm_cuda.h"
#include "timer.h"
#include <sys/time.h>

#define GENGRAPHSIZE 8192

// does not work for > 30 capacity!
void test_var_nnz_spmm_dense_cpu(int percentage) {
    int nnz = GENGRAPHSIZE * GENGRAPHSIZE * percentage / 100;
    produce_graph(GENGRAPHSIZE, nnz);

    AdjEdges edges("../../graph/test-graph-A.edges");
    AdjMatrixCSR A(edges);

    clock_start_cpu();
    AdjMatrixDense C = csr_spmm_dense_cpu(A, A);
    clock_stop_cpu();

    std::cout << percentage << "% Time - Dense: " << get_time_cpu() << " ms" << std::endl;
}

void test_var_nnz_spmm_ge_cpu(int percentage) {
    int nnz = GENGRAPHSIZE * GENGRAPHSIZE * percentage / 100;
    produce_graph(GENGRAPHSIZE, nnz);

    AdjEdges edges("../../graph/test-graph-A.edges");
    AdjMatrixCSR A(edges);

    std::cout << percentage << "% Time - GE:" << std::endl;
    AdjMatrixCSR* C = csr_spmm_cpu(&A, &A);
}

void test_var_nnz_spmm_dense_gpu(int percentage) {
    int nnz = GENGRAPHSIZE * GENGRAPHSIZE * percentage / 100;
    produce_graph(GENGRAPHSIZE, nnz);

    AdjEdges edges("../../graph/test-graph-A.edges");
    AdjMatrixCSR csr_A(edges);
    AdjMatrixDense C = csr_spmm_dense_cuda(csr_A, csr_A);

    std::cout << percentage << "% Time - Dense: " << get_time_cuda() << " ms" << std::endl;
}

void test_var_nnz_spmm_ge_gpu(int percentage) {
    int nnz = GENGRAPHSIZE * GENGRAPHSIZE * percentage / 100;
    produce_graph(GENGRAPHSIZE, nnz);

    AdjEdges edges("../../graph/test-graph-A.edges");
    AdjMatrixCSR A(edges);

    INT *work = (INT *) calloc(A.rows, sizeof(INT));
    AdjMatrixCSR* C_symb = csr_spmm_cpu_symbolic(&A, &A, work);
    std::cout<<A.num_size()<<std::endl;
    free(work);

    std::cout << percentage << "% Time - GE:" << std::endl;


    struct timeval t1, t2;

    gettimeofday(&t1, 0);
    INT NNZ=A.num_size();

    AdjMatrixCSR C = csr_spmm_cuda_v0(A, A, C_symb->rowPtr,NNZ);

    //HANDLE_ERROR(cudaThreadSynchronize();)

    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to generate:  %3.2f ms \n", time);

    // std::cout << " " << get_time_cpu() << " ms" << std::endl;



}