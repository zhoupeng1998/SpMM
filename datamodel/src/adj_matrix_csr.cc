#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR()
    :rows(0), size(0), rowPtr(NULL), colInd(NULL), val(NULL) 
{
}

AdjMatrixCSR::AdjMatrixCSR(int rows, int size)
    :rows(rows), size(size), rowPtr(NULL), colInd(NULL), val(NULL)
{
    rowPtr = (int*)malloc(sizeof(int) * (rows + 1));
    colInd = (int*)malloc(sizeof(int) * size);
    val = (int*)malloc(sizeof(int) * size);
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixCSR& other)
    :rows(other.rows), size(other.size), rowPtr(other.rowPtr), colInd(other.colInd), val(other.val)
{
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
    rows = matrixDense.size();
    size = matrixDense.num_edges();
    rowPtr = (int*)malloc(sizeof(int) * (rows + 1));
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

int* AdjMatrixCSR::get_rows() const {
    return rowPtr;
}

int* AdjMatrixCSR::get_cols() const {
    return colInd;
}

int* AdjMatrixCSR::get_vals() const {
    return val;
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