#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "adj_matrix_dense.h"
#include "adj_edges.h"

AdjMatrixDense::AdjMatrixDense(INT size) {
    vertices = size;
    edges = 0;
    matrix = (INT**)malloc(sizeof(INT*) * vertices);
    for (INT i = 0; i < vertices; i++) {
        matrix[i] = (INT*)malloc(sizeof(INT) * vertices);
        for (INT j = 0; j < vertices; j++) {
            matrix[i][j] = 0;
        }
    }
}

AdjMatrixDense::AdjMatrixDense(INT size, INT* arr) 
    :vertices(size), edges(0), matrix(NULL)
{
    matrix = (INT**)malloc(sizeof(INT*) * vertices);
    for (INT i = 0; i < vertices; i++) {
        matrix[i] = (INT*)malloc(sizeof(INT) * vertices);
    }
    for (INT i = 0; i < size; i++) {
        for (INT j = 0; j < size; j++) {
            matrix[i][j] = arr[i*size+j];
            if (arr[i*size+j] != 0) {
                edges++;
            }
        }
    }
}

AdjMatrixDense::AdjMatrixDense(INT size, INT** matrix)
    :vertices(size), edges(0), matrix(NULL)
{
    matrix = (INT**)malloc(sizeof(INT*) * vertices);
    for (INT i = 0; i < vertices; i++) {
        matrix[i] = (INT*)malloc(sizeof(int) * vertices);
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
    matrix = (INT**)malloc(sizeof(INT*) * vertices);
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (INT*)calloc(vertices, sizeof(INT));
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

INT AdjMatrixDense::num_vertices() const {
    return vertices;
}

INT AdjMatrixDense::num_edges() const {
    return edges;
}

void AdjMatrixDense::set_edges(INT edges) {
    this->edges = edges;
}

void AdjMatrixDense::add_edge() {
    edges++;
}

INT AdjMatrixDense::size() const {
    return vertices;
}

INT* AdjMatrixDense::operator[](INT index) {
    return matrix[index];
}

const INT* AdjMatrixDense::operator[](INT index) const {
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