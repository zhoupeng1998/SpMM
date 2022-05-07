#ifndef _SPMM_CUDA_H_
#define _SPMM_CUDA_H_

#include "data.h"
#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

/**
 * example: 
 * 1d grid, 1d block
 * blocks = 128?
 * threads per block = 1024 (maximum)
 */

#define TEST_DENSE 1

#ifdef TEST_DENSE
#define GRIDSIZE 8
#define BLOCKSIZE 512
#else
#define GRIDSIZE 128
#define BLOCKSIZE 1024
#endif
#define SIZE GRIDSIZE*BLOCKSIZE

AdjMatrixCSR csr_spmm_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixCSR csr_spmm_cuda_v0(AdjMatrixCSR& A, AdjMatrixCSR& B, INT* C_row,INT nnz);

AdjMatrixDense csr_spmm_dense_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixDense csr_spmm_dense_cuda_v2(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif