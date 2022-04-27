#ifndef _SPMM_SERIAL_H_
#define _SPMM_SERIAL_H_

#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"

//AdjMatrixCSR serial_spmm_csr(AdjMatrixCSR& A, AdjMatrixCSR& B);
AdjMatrixCSR* csr_spmm_cpu_symbolic(AdjMatrixCSR *A, AdjMatrixCSR *B, INT *work);
void csr_spmm_cpu_numeric(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, INT *work);
AdjMatrixCSR* csr_spmm_cpu(AdjMatrixCSR *A, AdjMatrixCSR *B);

AdjMatrixDense csr_spmm_dense_cpu(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif