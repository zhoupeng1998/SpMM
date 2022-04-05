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

    AdjEdges dataset= AdjEdges("../../graph/graph500-scale18-ef16_adj.edges");
    //AdjEdges dataset= AdjEdges("../../graph/test.edges");

    // std::cout<< " A nnz "<<dataset.data.size()<<std::endl;
    // std::cout<< " A rows "<<dataset.CountRows()<<std::endl;
    AdjMatrixCSR matrix_A = AdjMatrixCSR(dataset);
    AdjMatrixCSR matrix_B = AdjMatrixCSR(dataset);


    AdjMatrixCSR matrix_C;
    AdjMatrixCSR *A= &matrix_A;
    AdjMatrixCSR *B= &matrix_B;
    AdjMatrixCSR *C= &matrix_C;

    
    //std::cout<<" ind A " << A->colInd[0]<<" "<<A->colInd[1]<<" "<<A->colInd[2]<<" "<<A->colInd[3]<<" "<<A->colInd[4]<<" "<<A->colInd[5]<<" "<<A->colInd[6]<<" "<<A->colInd[7]<<" "<<A->colInd[9]<<" "<< std::endl;
    std::cout<<" ind A " << A->val[0]<<" "<<A->val[1]<<" "<<A->val[2]<<" "<<A->val[3]<<" "<<A->val[4]<<" "<<A->val[5]<<" "<<A->val[6]<<" "<<A->val[7]<<" "<<A->val[9]<<" "<< std::endl;

    if(clock_gettime(CLOCK_REALTIME, &start) == -1 ) {perror("time error");}
    csr_spmm_cpu(A,B,C);
    if(clock_gettime(CLOCK_REALTIME, &end) == -1 ) {perror("time error");}

    //std::cout << "size " << matrix_A.num_size()<< std::endl;

    std::cout<<" rows C " << C->num_rows() << std::endl;
    
    std::cout<<" size C " << C->num_size() << std::endl;

    std::cout<<" ind C " << C->rowPtr[0]<<" "<<C->rowPtr[1]<<" "<<C->rowPtr[2]<<" "<<C->rowPtr[3]<<" "<<C->rowPtr[4]<<" "<<C->rowPtr[5]<<" "<<std::endl;

    std::cout<<" ind C " << C->colInd[0]<<" "<<C->colInd[1]<<" "<<C->colInd[2]<<" "<<C->colInd[3]<<" "<<C->colInd[4]<<" "<<C->colInd[5]<<" "<<C->colInd[6]<<" "<<C->colInd[7]<<" "<<C->colInd[15]<<" "<< std::endl;
    std::cout<<" ind C " << C->val[0]<<" "<<C->val[1]<<" "<<C->val[2]<<" "<<C->val[3]<<" "<<C->val[4]<<" "<<C->val[5]<<" "<<C->val[6]<<" "<<C->val[7]<<" "<<C->val[14]<<" "<< std::endl;

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
