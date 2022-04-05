#ifndef _ADJ_MATRIX_DENSE_H_
#define _ADJ_MATRIX_DENSE_H_

#include "adj_edges.h"

class AdjMatrixDense
{
private:
    long vertices;
    long edges;
    long** matrix;
public:
    AdjMatrixDense(long size);
    AdjMatrixDense(long size, long* arr);
    AdjMatrixDense(const AdjEdges& edges);
    ~AdjMatrixDense();

    long num_vertices() const;
    long num_edges() const;
    void set_edges(long edges);
    long size() const;
    long* operator[](long index);
    const long* operator[](long index) const;
};

#endif