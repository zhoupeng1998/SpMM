#include <stdio.h>
#include <stdlib.h>

#include "hadamard_product.h"
#include "adj_matrix_dense.h"

AdjMatrixDense hadamard_product(AdjMatrixDense &A, AdjMatrixDense &B) {
    AdjMatrixDense C(A.size());
    for (INT i = 0; i < A.size(); i++) {
        for (INT j = 0; j < A.size(); j++) {
            C[i][j] = A[i][j] * B[i][j];
            if (C[i][j] != 0) {
                C.add_edge();
            }
        }
    }
    return C;
}

// TODO: test
AdjMatrixCSR hadamard_product(AdjMatrixCSR &A, AdjMatrixCSR &B) {
    INT* C_rowptr = (INT*)malloc((A.num_rows() + 1) * sizeof(INT));
    INT count = 0;
    INT row_count = 0;
    // symbolic
    for (INT row = 0; row < A.num_rows(); row++) {
        row_count = 0;
        INT ia = A.rowPtr[row];
        INT ib = B.rowPtr[row];
        while (ia < A.rowPtr[row+1] && ib < B.rowPtr[row+1]) {
            INT ca = A.colInd[ia];
            INT cb = B.colInd[ib];
            if (ca < cb) {
                ia++;
            } else if (ca > cb) {
                ib++;
            } else {
                // ca == cb
                ia++;
                ib++;
                row_count++;
            }
        }
        C_rowptr[row+1] = row_count;
    }
    C_rowptr[0] = 0;
    for (INT i = 0; i < A.num_rows(); i++) {
        C_rowptr[i+1] += C_rowptr[i];
    }
    INT* C_colInd = (INT*)malloc(C_rowptr[A.num_rows()] * sizeof(INT));
    INT* C_val = (INT*)malloc(C_rowptr[A.num_rows()] * sizeof(INT));
    // numeric
    for (INT row = 0; row < A.num_rows(); row++) {
        row_count = 0;
        INT ia = A.rowPtr[row];
        INT ib = B.rowPtr[row];
        while (ia < A.rowPtr[row+1] && ib < B.rowPtr[row+1]) {
            INT ca = A.colInd[ia];
            INT cb = B.colInd[ib];
            if (ca < cb) {
                ia++;
            } else if (ca > cb) {
                ib++;
            } else {
                // ca == cb
                C_colInd[count] = ca;
                C_val[count] = A.val[ia] * A.val[ib];
                count++;
                ia++;
                ib++;
            }
        }
    }
    return AdjMatrixCSR(A.num_rows(), C_rowptr[A.num_rows()], C_rowptr, C_colInd, C_val);
}