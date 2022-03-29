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
    AdjMatrixCSR(const AdjMatrixDense& matrixDense);
    ~AdjMatrixCSR();

    int num_rows() const;
    int num_size() const;
    AdjMatrixCSR& operator=(AdjMatrixCSR&& other);
};

#endif