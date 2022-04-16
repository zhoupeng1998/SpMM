#ifndef _ADJ_MATRIX_DENSE_LINEAR_H_
#define _ADJ_MATRIX_DENSE_LINEAR_H_

#include "data.h"
#include "adj_matrix_dense.h"

class AdjMatrixDenseLinear
{
private:
    int vertices;
    int edges;
    int* matrix;
public:
    AdjMatrixDenseLinear(int size);
    AdjMatrixDenseLinear(int size, int* arr);
    AdjMatrixDenseLinear(const AdjMatrixDense& dense);
    ~AdjMatrixDenseLinear();

    int* data();
    int num_vertices() const;
    int num_edges() const;
    void set_edges(int edges);
    int size() const;
    int* operator[](int index);
    const int* operator[](int index) const;

    void dump() const;
    void dump_back() const;
};

#endif