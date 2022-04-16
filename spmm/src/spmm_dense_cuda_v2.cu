#include <stdio.h>
#include <stdlib.h>

#include "spmm_cuda.h"
#include "data.h"
#include "adj_matrix_csr.h"
#include "adj_matrix_dense.h"

__managed__ int numrows;

__global__ void csr_spmm_dense_kernel_v2(INT* A_row, INT* A_col, INT* A_val, INT* B_row, INT* B_col, INT* B_val, INT* C_gpu) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i * SIZE < numrows; i++) {
        INT i1 = i * SIZE + tid;
        if (i1 < numrows) {
            for (int i2 = A_row[i1]; i2 < A_row[i1 + 1]; i2++) {
                INT colA = A_col[i2];
                for (int i3 = B_row[colA]; i3 < B_row[colA + 1]; i3++) {
                    INT colB = B_col[i3];
                    C_gpu[i1 * numrows + colB] += A_val[i2] * B_val[i3];
                }
            }
        }
    }
}

AdjMatrixDense csr_spmm_dense_cuda_v2(AdjMatrixCSR& A, AdjMatrixCSR& B) {
    INT* A_row;
    INT* A_col;
    INT* A_val;
    INT* B_row;
    INT* B_col;
    INT* B_val;

    INT* C_cpu;
    INT* C_gpu;

    numrows = A.num_rows();
    C_cpu = (INT*)malloc(numrows * numrows * sizeof(INT));

    cudaMalloc(&A_row, (A.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&A_col, A.num_size() * sizeof(INT));
    cudaMalloc(&A_val, A.num_size() * sizeof(INT));
    cudaMalloc(&B_row, (B.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&B_col, B.num_size() * sizeof(INT));
    cudaMalloc(&B_val, B.num_size() * sizeof(INT));

    cudaMalloc(&C_gpu, numrows * numrows * sizeof(INT));
    cudaMemset(C_gpu, 0, numrows * numrows * sizeof(INT));

    cudaMemcpy(A_row, A.get_rows(), (A.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_col, A.get_cols(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_val, A.get_vals(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_row, B.get_rows(), (B.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_col, B.get_cols(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_val, B.get_vals(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);

    // call kernel
    csr_spmm_dense_kernel_v2<<<GRIDSIZE, BLOCKSIZE>>>(A_row, A_col, A_val, B_row, B_col, B_val, C_gpu);

    cudaMemcpy(C_cpu, C_gpu, numrows * numrows * sizeof(INT), cudaMemcpyDeviceToHost);
    AdjMatrixDense C(numrows, C_cpu);

    cudaFree(A_row);
    cudaFree(A_col);
    cudaFree(A_val);
    cudaFree(B_row);
    cudaFree(B_col);
    cudaFree(B_val);
    cudaFree(C_gpu);
    free(C_cpu);

    return C;
}