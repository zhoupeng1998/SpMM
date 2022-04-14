#ifndef _GRAPH_GENERATOR_H_
#define _GRAPH_GENERATOR_H_

#include "adj_matrix_dense.h"

class GraphGenerator {
    int size;
    int nnz;
    int **matrix;

    void alloc();
    void dealloc();
    void clear();
public:
    GraphGenerator(int size, int nnz);
    ~GraphGenerator();

    void resize(int size, int nnz);
    void generate();
    AdjMatrixDense get_graph_dense();
    void store_graph(const char *filename);
};

#endif