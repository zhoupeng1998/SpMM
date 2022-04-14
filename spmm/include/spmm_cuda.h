#ifndef _SPMM_CUDA_H_
#define _SPMM_CUDA_H_

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR csr_spmm_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixDense csr_spmm_dense_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif