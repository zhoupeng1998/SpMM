#ifndef _ADJ_MATRIX_CSR_H_
#define _ADJ_MATRIX_CSR_H_

#include "adj_matrix_dense.h"

class AdjMatrixCSR
{
private:
    int* rowPtr;
    int* colInd;
    int* val;
public:
    AdjMatrixCSR(const AdjMatrixDense& matrixDense);
    ~AdjMatrixCSR();
};

#endif