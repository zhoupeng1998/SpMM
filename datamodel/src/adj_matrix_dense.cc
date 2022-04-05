#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "adj_matrix_dense.h"
#include "adj_edges.h"

AdjMatrixDense::AdjMatrixDense(long size) {
    vertices = size;
    edges = 0;
    matrix = (long**)malloc(sizeof(long*) * vertices);
    for (long i = 0; i < vertices; i++) {
        matrix[i] = (long*)malloc(sizeof(long) * vertices);
    }
}

AdjMatrixDense::AdjMatrixDense(long size, long* arr) 
    :vertices(size), edges(0), matrix(NULL)
{
    matrix = (long**)malloc(sizeof(long*) * vertices);
    for (long i = 0; i < vertices; i++) {
        matrix[i] = (long*)malloc(sizeof(long) * vertices);
    }
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < size; j++) {
            matrix[i][j] = arr[i*size+j];
            if (arr[i*size+j] != 0) {
                edges++;
            }
        }
    }
}

AdjMatrixDense::AdjMatrixDense(const AdjEdges& edges) {
    vertices = edges.num_vertices();
    this->edges = 0;
    matrix = (long**)malloc(sizeof(long*) * vertices);
    for (long i = 0; i < vertices; i++) {
        matrix[i] = (long*)malloc(sizeof(long) * vertices);
    }
    for (long i = 0; i < edges.num_entries(); i++) {
        matrix[edges[i][0]][edges[i][1]] = 1;
        matrix[edges[i][1]][edges[i][0]] = 1;
    }
    // TODO: edge count?
    for (long i = 0; i < vertices; i++) {
        for (long j = 0; j < vertices; j++) {
            if (matrix[i][j] > 0) {
                this->edges++;
            }
        }
    }
}

AdjMatrixDense::~AdjMatrixDense() {
    for (long i = 0; i < vertices; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

long AdjMatrixDense::num_vertices() const {
    return vertices;
}

long AdjMatrixDense::num_edges() const {
    return edges;
}

void AdjMatrixDense::set_edges(long edges) {
    this->edges = edges;
}

long AdjMatrixDense::size() const {
    return vertices;
}

long* AdjMatrixDense::operator[](long index) {
    return matrix[index];
}

const long* AdjMatrixDense::operator[](long index) const {
    return matrix[index];
}