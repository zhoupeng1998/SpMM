#ifndef _GRAPH_GENERATOR_H_
#define _GRAPH_GENERATOR_H_

#include "adj_matrix_dense.h"

class GraphGenerator {
    INT size;
    INT nnz;
    INT **matrix;

    void alloc();
    void dealloc();
    void clear();
public:
    GraphGenerator(INT size, INT nnz);
    ~GraphGenerator();

    void resize(INT size, INT nnz);
    void generate();
    AdjMatrixDense get_graph_dense();
    void store_graph(const char *filename);
};

#endif