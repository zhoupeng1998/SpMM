#include <iostream>

#include "adj_edges.h"
#include "adj_list.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "test_gpu.h"
#include "test_simple.h"
#include "serial_spmm.h"
#include "time.h"
int main(void) {
    struct timespec start, end;
    double time;

    //AdjEdges dataset= AdjEdges("../../graph/graph500-scale18-ef16_adj.edges");
    AdjEdges dataset= AdjEdges("../../graph/test.edges");

    std::cout<< " A nnz "<<dataset.data.size()<<std::endl;
    std::cout<< " A rows "<<dataset.CountRows()<<std::endl;
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);
    AdjMatrixCSR matrix_B = AdjMatrixCSR(dataset);


    AdjMatrixCSR matrix_C;
    AdjMatrixCSR *A= &matrix_A;
    AdjMatrixCSR *B= &matrix_B;
    AdjMatrixCSR *C= &matrix_C;
    if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) {perror("time error");}
    csr_spmm_cpu(A,B,C);
    if(clock_gettime(CLOCK_REALTIME, &end) == -1 ) {perror("time error");}

    std::cout << "size " << matrix_A.num_size()<< std::endl;



    std::cout<<" rows C" << C->num_rows() << std::endl;
    time = (end.tv_sec - start.tv_sec)+ (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    printf("time : %f ns\n", time * 1e9);




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
