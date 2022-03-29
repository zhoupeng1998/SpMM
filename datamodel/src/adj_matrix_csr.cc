#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
    rowPtr = (int*)malloc(sizeof(int) * (matrixDense.size() + 1));
    colInd = (int*)malloc(sizeof(int) * matrixDense.num_edges());
    val = (int*)malloc(sizeof(int) * matrixDense.num_edges());
    int ind = 0;
    rowPtr[0] = 0;
    for (int row = 0; row < matrixDense.size(); row++) {
        for (int col = 0; col < matrixDense.size(); col++) {
            if (matrixDense[row][col] != 0) {
                val[ind] = matrixDense[row][col];
                colInd[ind] = col;
                ind++;
            }
        }
        rowPtr[row+1] = ind;
    }
}

AdjMatrixCSR::~AdjMatrixCSR() {
    free(rowPtr);
    free(colInd);
    free(val);
}