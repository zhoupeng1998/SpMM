#ifndef _SPMM_CUDA_H_
#define _SPMM_CUDA_H_

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

/**
 * example: 
 * 1d grid, 1d block
 * blocks = 128?
 * threads per block = 1024 (maximum)
 */
#define GRIDSIZE 128
#define BLOCKSIZE 1024
#define SIZE GRIDSIZE*BLOCKSIZE

AdjMatrixCSR csr_spmm_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixDense csr_spmm_dense_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixDense csr_spmm_dense_cuda_v2(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif