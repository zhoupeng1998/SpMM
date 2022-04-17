#ifndef _ADJ_MATRIX_CSR_H_
#define _ADJ_MATRIX_CSR_H_

#include "data.h"
#include "adj_matrix_dense.h"

class AdjMatrixCSR
{
private:

public:
    INT* rowPtr;
    INT* colInd;
    INT* val;
    INT rows;
    INT cols;
    INT size;

    AdjMatrixCSR();
    AdjMatrixCSR(const AdjMatrixCSR& other);
    
    AdjMatrixCSR(INT rows, INT size);
    AdjMatrixCSR(const AdjMatrixDense& matrixDense);
    AdjMatrixCSR(AdjEdges& AdjEdges);
    AdjMatrixCSR(INT rows, INT size, INT* rowPtr, INT* colInd, INT* val);
    AdjMatrixCSR(const char* symbolicfile);
    ~AdjMatrixCSR();

    INT num_rows() const;
    INT num_size() const;

    INT* get_rows() const;
    INT* get_cols() const;
    INT* get_vals() const;

    void dump() const;
    void dump_front() const;
    void store_symbolic(const char* filename);

    AdjMatrixCSR& operator=(AdjMatrixCSR&& other);
};

#endif