#ifndef _ADJ_MATRIX_DENSE_H_
#define _ADJ_MATRIX_DENSE_H_

#include "data.h"
#include "adj_edges.h"

class AdjMatrixDense
{
private:
    INT vertices;
    INT edges;
    INT** matrix;
public:
    AdjMatrixDense(INT size);
    AdjMatrixDense(INT size, INT* arr);
    AdjMatrixDense(INT size, INT** matrix);
    AdjMatrixDense(const AdjEdges& edges);
    ~AdjMatrixDense();

    INT num_vertices() const;
    INT num_edges() const;
    void set_edges(INT edges);
    INT size() const;
    INT* operator[](INT index);
    const INT* operator[](INT index) const;

    void dump() const;
    void dump_back() const;
};

#endif