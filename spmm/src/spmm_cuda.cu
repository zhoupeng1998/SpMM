#include "data.h"
#include "adj_matrix_csr.h"
#include "spmm_cuda.h"

/**
 * example: 
 * 1d grid, 1d block
 * blocks = 128?
 * threads per block = 1024 (maximum)
 */
#define GRIDSIZE 128
#define BLOCKSIZE 1024
#define SIZE GRIDSIZE*BLOCKSIZE

__managed__ int numrows;

__global__ void csr_spmm_symbolic(INT* A_row, INT* A_col, INT* A_val, INT* B_row, INT* B_col, INT* B_val, INT* C_row, INT* work) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //__shared__ INT work[BLOCKSIZE];
    for (int i = 0; i < numrows; i += SIZE) {
        INT i1 = i * SIZE + tid;
        INT MARK = i1 + 1;
        INT count = 0;
        for (INT i2 = A_row[i1]; i2 < A_row[i1+1]; i2++) {
            INT j = A_col[i2];
            // assert(j >= 0 && j < numrows);
            for (INT i3 = B_row[j]; i3 < B_row[j+1]; i3++) {
                INT col = B_col[i3];
                // assert(col >= 0 && col < numrows);
                if (work[col] != MARK) {
                    count++;
                    work[col] = MARK;
                }
            }
        }
        C_row[i1+1] = count;
    }
    // prefix sum at host
}

AdjMatrixCSR csr_spmm(AdjMatrixCSR& A, AdjMatrixCSR& B) {
    INT* A_row;
    INT* A_col;
    INT* A_val;
    INT* B_row;
    INT* B_col;
    INT* B_val;
    
    numrows = A.num_rows();

    cudaMalloc(&A_row, (A.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&A_col, A.num_size() * sizeof(INT));
    cudaMalloc(&A_val, A.num_size() * sizeof(INT));
    cudaMalloc(&B_row, (B.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&B_col, B.num_size() * sizeof(INT));
    cudaMalloc(&B_val, B.num_size() * sizeof(INT));

    cudaMemcpy(A_row, A.get_rows(), (A.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_col, A.get_cols(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_val, A.get_vals(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_row, B.get_rows(), (B.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_col, B.get_cols(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_val, B.get_vals(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);

    // code start

    // call kernel
    // prefix sum

    // code end

    // cudaMemcpy to host
    AdjMatrixCSR result(0,0);
    return result;
}