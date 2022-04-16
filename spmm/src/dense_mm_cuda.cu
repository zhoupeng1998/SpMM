#include <stdlib.h>

#include "timer.h"
#include "dense_mm_cuda.h"
#include "adj_matrix_dense.h"
#include "spmm_cuda.h"

__managed__ int numrows;

__global__ void dense_mm_kernel(INT* A_gpu, INT* B_gpu, INT* C_gpu) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i * SIZE < numrows; i++) {
        int i1 = i * SIZE + tid;
        if (i1 < numrows) {
            for (int j = 0; j < numrows; j++) {
                for (int k = 0; k < numrows; k++) {
                    C_gpu[i1 * numrows + j] += A_gpu[i1 * numrows + k] * B_gpu[k * numrows + j];
                }
            }
        }
    }
}

AdjMatrixDenseLinear dense_mm_cuda(AdjMatrixDenseLinear& A, AdjMatrixDenseLinear& B) {
    INT* A_gpu;
    INT* B_gpu;
    INT* C_gpu;
    INT* C_cpu;

    numrows = A.size();
    C_cpu = (INT*)malloc(A.size() * A.size() * sizeof(INT));

    cudaMalloc(&A_gpu, A.size() * A.size() * sizeof(INT));
    cudaMalloc(&B_gpu, B.size() * B.size() * sizeof(INT));
    cudaMalloc(&C_gpu, A.size() * B.size() * sizeof(INT));

    cudaMemset(C_gpu, 0, A.size() * A.size() * sizeof(INT));

    cudaMemcpy(A_gpu, A.data(), A.size() * A.size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B.data(), B.size() * B.size() * sizeof(INT), cudaMemcpyHostToDevice);

    // call kernel
    clock_start_cuda();
    dense_mm_kernel<<<GRIDSIZE, BLOCKSIZE>>>(A_gpu, B_gpu, C_gpu);
    clock_stop_cuda();

    cudaMemcpy(C_cpu, C_gpu, A.size() * B.size() * sizeof(INT), cudaMemcpyDeviceToHost);
    AdjMatrixDenseLinear C(A.size(), C_cpu);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    free(C_cpu);

    return C;
}