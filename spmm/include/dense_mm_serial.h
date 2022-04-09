#ifndef _SERIAL_DENSE_MM_H_
#define _SERIAL_DENSE_MM_H_

#include "adj_matrix_dense.h"

AdjMatrixDense serial_dense_mm(AdjMatrixDense& A, AdjMatrixDense& B) {
    int size = A.size();
    int edges = 0;
    AdjMatrixDense C(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
            if (C[i][j] > 0) {
                edges++;
            }
        }
    }
    C.set_edges(edges);
    return C;
}

#endif