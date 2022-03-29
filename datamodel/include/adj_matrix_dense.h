#ifndef _ADJ_MATRIX_DENSE_H_
#define _ADJ_MATRIX_DENSE_H_

#include "adj_edges.h"

class AdjMatrixDense
{
private:
    int vertices;
    int edges;
    int** matrix;
public:
    AdjMatrixDense(int size);
    AdjMatrixDense(const AdjEdges& edges);
    ~AdjMatrixDense();

    int num_vertices() const;
    int num_edges() const;
    int size() const;
    const int* operator[](int index) const;
};

#endif