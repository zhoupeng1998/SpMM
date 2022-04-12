#ifndef _SPMM_CUDA_H_
#define _SPMM_CUDA_H_

#include "adj_matrix_csr.h"

AdjMatrixCSR csr_spmm_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif