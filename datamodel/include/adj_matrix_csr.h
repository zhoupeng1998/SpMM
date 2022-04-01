#ifndef _ADJ_MATRIX_CSR_H_
#define _ADJ_MATRIX_CSR_H_

#include "adj_matrix_dense.h"

class AdjMatrixCSR
{
private:
    int rows;
    int size;

    int* rowPtr;
    int* colInd;
    int* val;
public:
    AdjMatrixCSR();
    AdjMatrixCSR(const AdjMatrixCSR& other);
    
    AdjMatrixCSR(int rows, int size);
    AdjMatrixCSR(const AdjMatrixDense& matrixDense);
    AdjMatrixCSR( AdjEdges& AdjEdges);
    ~AdjMatrixCSR();

    int num_rows() const;
    int num_size() const;

    int* get_rows() const;
    int* get_cols() const;
    int* get_vals() const;

    AdjMatrixCSR& operator=(AdjMatrixCSR&& other);
};

#endif