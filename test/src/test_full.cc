#include <assert.h>

#include <iostream>

#include "timer.h"
#include "data.h"
#include "adj_edges.h"
#include "adj_list.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "test_gpu.h"
#include "test_simple.h"
#include "spmm_serial.h"
#include "spmm_cuda.h"
#include "time.h"

#include "test_full.h"

#define PATH_GRAPH16 "../../graph/graph500-scale18-ef16_adj.edges"

void test_cpu_full_v1() {
    AdjEdges dataset = AdjEdges(PATH_GRAPH16);
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);

    AdjMatrixCSR *A= &matrix_A;
    AdjMatrixCSR *B= &matrix_A;

    std::cout << "A: " << std::endl;
    A->dump_front();
    std::cout << std::endl;

    clock_start_cpu();
    AdjMatrixCSR *C = csr_spmm_cpu(A, B);
    clock_stop_cpu();

    std::cout << "C: " << std::endl;
    C->dump_front();

    std::cout << "time: " << get_time_cpu() << "ns" << std::endl;
}

void test_cpu_full_v1(int limit) {
    AdjEdges dataset = AdjEdges(PATH_GRAPH16, limit);
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);

    AdjMatrixCSR *A= &matrix_A;
    AdjMatrixCSR *B= &matrix_A;

    std::cout << "A: " << std::endl;
    A->dump_front();
    std::cout << std::endl;

    clock_start_cpu();
    AdjMatrixCSR *C = csr_spmm_cpu(A, B);
    clock_stop_cpu();

    std::cout << "C: " << std::endl;
    C->dump_front();

    std::cout << "time: " << get_time_cpu() << "ns" << std::endl;
}

void test_cuda_full_v1() {
    AdjEdges dataset = AdjEdges(PATH_GRAPH16);
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);

    AdjMatrixCSR& A= matrix_A;
    AdjMatrixCSR& B= matrix_A;

    std::cout << "A: " << std::endl;
    A.dump_front();
    std::cout << std::endl;

    clock_start_cuda();
    AdjMatrixCSR C = csr_spmm_cuda(A, B);
    clock_stop_cuda();

    std::cout << "C: " << std::endl;
    C.dump_front();

    std::cout << "time: " << get_time_cuda() << "ns" << std::endl;
}

void test_cuda_full_v1(int limit) {
    AdjEdges dataset = AdjEdges(PATH_GRAPH16, limit);
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);

    AdjMatrixCSR& A= matrix_A;
    AdjMatrixCSR& B= matrix_A;

    std::cout << "A: " << std::endl;
    A.dump_front();
    std::cout << std::endl;

    clock_start_cuda();
    AdjMatrixCSR C = csr_spmm_cuda(A, B);
    clock_stop_cuda();

    std::cout << "C: " << std::endl;
    C.dump_front();

    std::cout << "time: " << get_time_cuda() << "ns" << std::endl;
}