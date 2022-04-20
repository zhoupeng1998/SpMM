#ifndef _HADAMARD_PRODUCT_H_
#define _HADAMARD_PRODUCT_H_

#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"

AdjMatrixDense hadamard_product(AdjMatrixDense &A, AdjMatrixDense &B);
AdjMatrixCSR hadamard_product(AdjMatrixCSR &A, AdjMatrixCSR &B);

#endif