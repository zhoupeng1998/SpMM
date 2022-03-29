#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR() {
    rows = 0;
    size = 0;
    rowPtr = NULL;
    colInd = NULL;
    val = NULL;
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
    rows = matrixDense.size() + 1;
    size = matrixDense.num_edges();
    rowPtr = (int*)malloc(sizeof(int) * rows);
    colInd = (int*)malloc(sizeof(int) * size);
    val = (int*)malloc(sizeof(int) * size);
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

int AdjMatrixCSR::num_rows() const {
    return rows;
}

int AdjMatrixCSR::num_size() const {
    return size;
}

AdjMatrixCSR& AdjMatrixCSR::operator=(AdjMatrixCSR&& other) {
    free(rowPtr);
    free(colInd);
    free(val);
    rows = other.rows;
    size = other.size;
    rowPtr = other.rowPtr;
    colInd = other.colInd;
    val = other.val;
    other.rowPtr = NULL;
    other.colInd = NULL;
    other.val = NULL;
    return *this;
}