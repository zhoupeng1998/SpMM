#ifndef _SERIAL_SPMM_H_
#define _SERIAL_SPMM_H_

#include "adj_matrix_csr.h"

AdjMatrixCSR serial_spmm_csr(AdjMatrixCSR& A, AdjMatrixCSR& B);
void csr_spmm_symbolic(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, int *work);
void csr_spmm_numeric(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, int *work);
void csr_spmm_cpu(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C);

#endif