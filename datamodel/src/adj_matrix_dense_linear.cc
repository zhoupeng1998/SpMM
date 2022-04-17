#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "adj_matrix_dense.h"
#include "adj_matrix_dense_linear.h"

AdjMatrixDenseLinear::AdjMatrixDenseLinear(INT size)
:vertices(size), edges(0), matrix(NULL) {
    matrix = (int*)malloc(sizeof(int) * size * size);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = 0;
    }
}

AdjMatrixDenseLinear::AdjMatrixDenseLinear(INT size, INT* arr)
:vertices(size), edges(0), matrix(NULL) {
    matrix = (int*)malloc(sizeof(int) * size * size);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = arr[i];
        if (arr[i] != 0) {
            edges++;
        }
    }
}

AdjMatrixDenseLinear::AdjMatrixDenseLinear(const AdjMatrixDense& dense)
:vertices(dense.num_vertices()), edges(0), matrix(NULL) {
    matrix = (int*)malloc(sizeof(int) * vertices * vertices);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            matrix[i * vertices + j] = dense[i][j];
            if (dense[i][j] != 0) {
                edges++;
            }
        }
    }
}

AdjMatrixDenseLinear::~AdjMatrixDenseLinear() {
    if (matrix != NULL) {
        free(matrix);
    }
}

int* AdjMatrixDenseLinear::data() {
    return matrix;
}

int AdjMatrixDenseLinear::num_vertices() const {
    return vertices;
}

int AdjMatrixDenseLinear::num_edges() const {
    return edges;
}

void AdjMatrixDenseLinear::set_edges(int edges) {
    this->edges = edges;
}

int AdjMatrixDenseLinear::size() const {
    return vertices;
}

int* AdjMatrixDenseLinear::operator[](int i) {
    return matrix + i;
}

const int* AdjMatrixDenseLinear::operator[](int i) const {
    return matrix + i;
}

void AdjMatrixDenseLinear::dump() const {
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            std::cout << matrix[i * vertices + j] << " ";
        }
        std::cout << std::endl;
    }
}

void AdjMatrixDenseLinear::dump_back() const {
    for (int i = vertices * vertices - 10; i < vertices * vertices; i++) {
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
}