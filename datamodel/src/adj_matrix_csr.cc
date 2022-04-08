#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR()
    :rows(0), cols(0), size(0), rowPtr(NULL), colInd(NULL), val(NULL) 
{
}

AdjMatrixCSR::AdjMatrixCSR(long rows, long size)
    :rows(rows), cols(rows), size(size), rowPtr(NULL), colInd(NULL), val(NULL)
{
    rowPtr = (long*)malloc(sizeof(long) * (rows + 1));
    colInd = (long*)malloc(sizeof(long) * size);
    val = (long*)malloc(sizeof(long) * size);
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixCSR& other)
    :rows(other.rows), cols(other.rows), size(other.size), rowPtr(other.rowPtr), colInd(other.colInd), val(other.val)
{
}

AdjMatrixCSR::AdjMatrixCSR( AdjEdges& AdjEdges) {
    rows = AdjEdges.num_vertices();

    cols = AdjEdges.num_vertices();
    size = AdjEdges.num_entries();
    //std::cout<<"rows "<<rows<<"   nnzs: "<<size<<std::endl;
    rowPtr = (long*)malloc(sizeof(long) * (rows + 1));
    colInd = (long*)malloc(sizeof(long) * size);
    val = (long*)malloc(sizeof(long) * size);
    for (long i = 0; i < size; i++) {
        val[i] = 1;
        colInd[i] = AdjEdges[i][0];
        rowPtr[AdjEdges[i][1] + 1]++;
    }
    for (long i = 1; i <= rows; i++) {
        rowPtr[i] += rowPtr[i-1];
    }
}


AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
    rows = matrixDense.size();
    cols = matrixDense.size();
    size = matrixDense.num_edges();
    rowPtr = (long*)malloc(sizeof(long) * rows);
    colInd = (long*)malloc(sizeof(long) * size);
    val = (long*)malloc(sizeof(long) * size);
    long ind = 0;
    rowPtr[0] = 0;
    for (long row = 0; row < matrixDense.size(); row++) {
        for (long col = 0; col < matrixDense.size(); col++) {
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

long AdjMatrixCSR::num_rows() const {
    return rows;
}

long AdjMatrixCSR::num_size() const {
    return size;
}

long* AdjMatrixCSR::get_rows() const {
    return rowPtr;
}

long* AdjMatrixCSR::get_cols() const {
    return colInd;
}

long* AdjMatrixCSR::get_vals() const {
    return val;
}

void AdjMatrixCSR::dump() const {
    std::cout << rows << " " << size << std::endl;
    for (int i = 0; i <= rows; i++) {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << colInd[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
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