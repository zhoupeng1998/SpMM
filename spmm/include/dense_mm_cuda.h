#ifndef _DENSE_MM_CUDA_H_
#define _DENSE_MM_CUDA_H_

#include "adj_matrix_dense.h"
#include "adj_matrix_dense_linear.h"

/**
 * example: 
 * 1d grid, 1d block
 * blocks = 128?
 * threads per block = 1024 (maximum)
 */

AdjMatrixDenseLinear dense_mm_cuda(AdjMatrixDenseLinear& A, AdjMatrixDenseLinear& B);
AdjMatrixDenseLinear dense_mm_cuda_v2(AdjMatrixDenseLinear& A, AdjMatrixDenseLinear& B);

#endif