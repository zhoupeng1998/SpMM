#ifndef _ADJ_MATRIX_CSR_H_
#define _ADJ_MATRIX_CSR_H_

#include "adj_matrix_dense.h"

class AdjMatrixCSR
{
private:




public:
    long* rowPtr;
    long* colInd;
    long* val;
    long rows;
    long cols;
    long size;

    AdjMatrixCSR();
    AdjMatrixCSR(const AdjMatrixCSR& other);
    
    AdjMatrixCSR(long rows, long size);
    AdjMatrixCSR(const AdjMatrixDense& matrixDense);
    AdjMatrixCSR( AdjEdges& AdjEdges);
    ~AdjMatrixCSR();

    long num_rows() const;
    long num_size() const;

    long* get_rows() const;
    long* get_cols() const;
    long* get_vals() const;

    AdjMatrixCSR& operator=(AdjMatrixCSR&& other);
};

#endif