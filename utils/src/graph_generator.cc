#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "graph_generator.h"

GraphGenerator::GraphGenerator(int size, int nnz)
: size(size), nnz(nnz) {
    alloc();
}

GraphGenerator::~GraphGenerator() {
    dealloc();
}

void GraphGenerator::alloc() {
    matrix = (int**)malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * size);
        memset(matrix[i], 0, sizeof(int) * size);
    }
}

void GraphGenerator::dealloc() {
    if (matrix != NULL) {
        for (int i = 0; i < size; i++) {
            if (matrix[i] != NULL) {
                free(matrix[i]);
            }
        }
        free(matrix);
    }
}

void GraphGenerator::clear() {
    for (int i = 0; i < size; i++) {
        memset(matrix[i], 0, sizeof(int) * size);
    }
}

void GraphGenerator::resize(int size, int nnz) {
    this->size = size;
    this->nnz = nnz;
    dealloc();
    alloc();
}

void GraphGenerator::generate() {
    clear();
    srand((unsigned)time(NULL));

    // a "must", to ensure graph size when it reads from file
    matrix[size-1][size-2] = 1;
    matrix[size-2][size-1] = 1;

    int count = 1;
    int edges = nnz / 2;
    while (count < edges) {
        int i = rand() % size;
        int j = rand() % size;
        if (matrix[i][j] == 0 && i != j) {
            matrix[i][j] = 1;
            matrix[j][i] = 1;
            count++;
        }
    }

}

AdjMatrixDense GraphGenerator::get_graph_dense() {
    AdjMatrixDense graph(size, matrix);
    return graph;
}

void GraphGenerator::store_graph(const char* filename) {
    time_t timestamp = time(NULL);
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Output file open error!");
        exit(-1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (matrix[i][j] == 1) {
                /*
                if (i > j) {
                    fprintf(file, "%d %d\n", i+1, j+1);
                } else {
                    fprintf(file, "%d %d\n", j+1, i+1);
                }
                */
                 fprintf(file, "%d %d\n", j+1,i+1);
            }
        }
    }
    fclose(file);
}