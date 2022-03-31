#ifndef _SERIAL_SPMM_H_
#define _SERIAL_SPMM_H_

#include "adj_matrix_csr.h"

AdjMatrixCSR serial_spmm_csr(AdjMatrixCSR& A, AdjMatrixCSR& B);

#endif