#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

AdjMatrixCSR::AdjMatrixCSR()
    :rows(0), cols(0), size(0), rowPtr(NULL), colInd(NULL), val(NULL) 
{
}

AdjMatrixCSR::AdjMatrixCSR(INT rows, INT size)
    :rows(rows), cols(rows), size(size), rowPtr(NULL), colInd(NULL), val(NULL)
{
    rowPtr = (INT*)malloc(sizeof(INT) * (rows + 1));
    colInd = (INT*)malloc(sizeof(INT) * size);
    val = (INT*)malloc(sizeof(INT) * size);
}

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixCSR& other)
    :rows(other.rows), cols(other.rows), size(other.size), rowPtr(other.rowPtr), colInd(other.colInd), val(other.val)
{
}

AdjMatrixCSR::AdjMatrixCSR(AdjEdges& AdjEdges) {
    rows = AdjEdges.num_vertices();
    cols = AdjEdges.num_vertices();
    size = AdjEdges.num_entries();
    //std::cout<<"rows "<<rows<<"   nnzs: "<<size<<std::endl;
    rowPtr = (INT*)malloc(sizeof(INT) * (rows + 1));
    colInd = (INT*)malloc(sizeof(INT) * size);
    val = (INT*)malloc(sizeof(INT) * size);
    INT count=0;
    INT previous=0;
    rowPtr[0]=0;
    for (INT i = 0; i <size; i++) {
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

AdjMatrixCSR::AdjMatrixCSR(const AdjMatrixDense& matrixDense) {
    rows = matrixDense.size();
    cols = matrixDense.size();
    size = matrixDense.num_edges();
    rowPtr = (INT*)malloc(sizeof(INT) * rows);
    colInd = (INT*)malloc(sizeof(INT) * size);
    val = (INT*)malloc(sizeof(INT) * size);
    INT ind = 0;
    rowPtr[0] = 0;
    for (INT row = 0; row < matrixDense.size(); row++) {
        for (INT col = 0; col < matrixDense.size(); col++) {
            if (matrixDense[row][col] != 0) {
                val[ind] = matrixDense[row][col];
                colInd[ind] = col;
                ind++;
            }
        }
        rowPtr[row+1] = ind;
    }
}

AdjMatrixCSR::AdjMatrixCSR(INT rows, INT size, INT* rowPtr, INT* colInd, INT* val)
    :rows(rows), cols(rows), size(size), rowPtr(rowPtr), colInd(colInd), val(val)
{
}

AdjMatrixCSR::AdjMatrixCSR(const char* symbolicfile)
    :rows(0), cols(0), size(0), rowPtr(NULL), colInd(NULL), val(NULL)
{
    if (symbolicfile == NULL) {
        symbolicfile = "../../graph/graph-rowptr.txt";
    }
    FILE* file = fopen(symbolicfile, "r");
    if (file == NULL) {
        perror("Output file open error!");
        exit(-1);
    }
    fscanf(file, "%ld", &rows);
    cols = rows;
    rowPtr = (INT*)malloc(sizeof(INT) * (rows + 1));
    for (INT i = 0; i <= rows; i++) {
        fscanf(file, "%ld", &rowPtr[i]);
    }
    size = rowPtr[rows];
    colInd = (INT*)malloc(sizeof(INT) * size);
    val = (INT*)malloc(sizeof(INT) * size);
}

AdjMatrixCSR::~AdjMatrixCSR() {
    free(rowPtr);
    free(colInd);
    free(val);
}

INT AdjMatrixCSR::num_rows() const {
    return rows;
}

INT AdjMatrixCSR::num_size() const {
    return size;
}

INT* AdjMatrixCSR::get_rows() const {
    return rowPtr;
}

INT* AdjMatrixCSR::get_cols() const {
    return colInd;
}

INT* AdjMatrixCSR::get_vals() const {
    return val;
}

void AdjMatrixCSR::dump() const {
    std::cout << "rows: " << rows << "; size:" << size << std::endl;
    std::cout << "rowPtr: ";
    for (int i = 0; i <= rows; i++) {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "colInd: ";
    for (int i = 0; i < size; i++) {
        std::cout << colInd[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "val: ";
    for (int i = 0; i < size; i++) {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
}

void AdjMatrixCSR::dump_front() const {
    std::cout << "rows: " << rows << "; size:" << size << std::endl;
    std::cout << "rowPtr: ";
    for (int i = 0; i <= 5 && i <= rows; i++) {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "colInd: ";
    for (int i = 0; i < 15 && i < size; i++) {
        std::cout << colInd[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "val: ";
    for (int i = 0; i <= 15 && i < size; i++) {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
}

void AdjMatrixCSR::store_symbolic(const char* filename) {
    if (filename == NULL) {
        filename = "../../graph/graph-rowptr.txt";
    }
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Output file open error!");
        exit(-1);
    }
    fprintf(file, "%d\n", rows);
    for (INT i = 0; i <= rows; i++) {
        fprintf(file, "%d\n", rowPtr[i]);
    }
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