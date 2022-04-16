#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "adj_matrix_dense.h"
#include "adj_edges.h"

AdjMatrixDense::AdjMatrixDense(int size) {
    vertices = size;
    edges = 0;
    matrix = (int**)malloc(sizeof(int*) * vertices);
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * vertices);
        for (int j = 0; j < vertices; j++) {
            matrix[i][j] = 0;
        }
    }
}

AdjMatrixDense::AdjMatrixDense(int size, int* arr) 
    :vertices(size), edges(0), matrix(NULL)
{
    matrix = (int**)malloc(sizeof(int*) * vertices);
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * vertices);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = arr[i*size+j];
            if (arr[i*size+j] != 0) {
                edges++;
            }
        }
    }
}

AdjMatrixDense::AdjMatrixDense(int size, int** matrix)
    :vertices(size), edges(0), matrix(NULL)
{
    matrix = (int**)malloc(sizeof(int*) * vertices);
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * vertices);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = matrix[i][j];
            if (matrix[i][j] != 0) {
                edges++;
            }
        }
    }
}

AdjMatrixDense::AdjMatrixDense(const AdjEdges& edges) {
    vertices = edges.num_vertices();
    this->edges = 0;
    matrix = (int**)malloc(sizeof(int*) * vertices);
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (int*)calloc(vertices, sizeof(int));
    }
    for (int i = 0; i < edges.num_entries(); i++) {
        matrix[edges[i][0]][edges[i][1]] = 1;
        //matrix[edges[i][1]][edges[i][0]] = 1;
    }
    // TODO: edge count?
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
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

int AdjMatrixDense::num_vertices() const {
    return vertices;
}

int AdjMatrixDense::num_edges() const {
    return edges;
}

void AdjMatrixDense::set_edges(int edges) {
    this->edges = edges;
}

int AdjMatrixDense::size() const {
    return vertices;
}

int* AdjMatrixDense::operator[](int index) {
    return matrix[index];
}

const int* AdjMatrixDense::operator[](int index) const {
    return matrix[index];
}

void AdjMatrixDense::dump() const {
    std::cout << "V" << vertices << std::endl;
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            //std::cout << i << " " << j << std::endl;
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void AdjMatrixDense::dump_back() const {
    for (int i = vertices - 10; i < vertices; i++) {
        std::cout << matrix[vertices - 1][i] << " ";
    }
    std::cout << std::endl;
}