#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR()
    :rows(0), size(0), rowPtr(NULL), colInd(NULL), val(NULL) 
{
}

AdjMatrixCSR::AdjMatrixCSR(long rows, long size)
    :rows(rows), size(size), rowPtr(NULL), colInd(NULL), val(NULL)
{
    rowPtr = (long*)malloc(sizeof(long) * (rows + 1));
    colInd = (long*)malloc(sizeof(long) * size);
    val = (long*)malloc(sizeof(long) * size);
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixCSR& other)
    :rows(other.rows), size(other.size), rowPtr(other.rowPtr), colInd(other.colInd), val(other.val)
{
}

AdjMatrixCSR::AdjMatrixCSR( AdjEdges& AdjEdges) {
    rows = AdjEdges.CountRows();

    cols = AdjEdges.CountRows();
    size = AdjEdges.CountNNZ();
    //std::cout<<"rows "<<rows<<"   nnzs: "<<size<<std::endl;
    rowPtr = (long*)malloc(sizeof(long) * (rows + 1));
    colInd = (long*)malloc(sizeof(long) * size);
    val = (long*)malloc(sizeof(long) * size);
    long count=0;
    long previous=0;
    rowPtr[0]=0;
    for (long i = 0; i <size; i++) {
        val[i] = 1;
        colInd[i] = AdjEdges.data[i][0];
        if(previous != AdjEdges.data[i][1]) {
            rowPtr[previous+1] = count;
            previous++;
        }
        count++;
    }
    rowPtr[previous+1] = count;
}


// AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
//     rows = matrixDense.size();

//     size = matrixDense.num_edges();
//     rowPtr = (int*)malloc(sizeof(int) * (rows + 1));
//     colInd = (int*)malloc(sizeof(int) * size);
//     val = (int*)malloc(sizeof(int) * size);
//     int ind = 0;
//     rowPtr[0] = 0;
//     for (int row = 0; row < matrixDense.size(); row++) {
//         for (int col = 0; col < matrixDense.size(); col++) {
//             if (matrixDense[row][col] != 0) {
//                 val[ind] = matrixDense[row][col];
//                 colInd[ind] = col;
//                 ind++;
//             }
//         }
//         rowPtr[row+1] = ind;
//     }
// }



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