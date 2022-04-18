#include "data.h"
#include "adj_matrix_csr.h"
#include "spmm_cuda.h"
#include "stdio.h"
#include <iostream>


// #define GRIDSIZE 128
// #define BLOCKSIZE 1024
#define SIZE GRIDSIZE*BLOCKSIZE

__managed__ int numrows;


__global__ void GetNNZ(INT* A_row, INT* A_col, INT* A_val, INT* B_row, INT* B_col, INT* B_val, INT* C_row, INT* work,INT rows) 
{
	const int laneId = threadIdx.x;
	const int warpId = blockIdx.x;
	
	INT* nonzeros;
	INT rowAStart, rowAEnd, rowBStart, rowBEnd;
	INT nnz;
	INT colC;
	
	extern __shared__ int nzCount[];
	
	nonzeros = &work[warpId * rows];
	
	// Iterate through each assigned row in A.
	for(INT rowA = warpId; rowA < rows; rowA += gridDim.x)
	{
    //printf("%ld, %d\n",rowA,gridDim.x);
		rowAStart = A_row[rowA];

    //printf("%ld, %d\n",A_row[3],gridDim.x);

		rowAEnd = A_row[rowA + 1];
		// There are no non-zeros in this row so continue
		if(rowAStart == rowAEnd)
		{
			if (laneId == 0)
				C_row[rowA] = 0;
			__syncthreads();
			continue;
		}

		// Reset the nz counts
		nzCount[laneId] = 0;
		
		// reset the nonzeros table
		for (INT i=laneId; i<rows; i+= warpSize)
		{
			nonzeros[i] = 0;
		}
		__syncthreads();
		
		for(INT i = rowAStart; i < rowAEnd; ++i)
		{
			rowBStart = B_row[A_col[i]];
			rowBEnd = B_row[A_col[i]+1];

			for (INT j = rowBStart + laneId; j < rowBEnd; j += warpSize)
			{
				colC = B_col[j];
				nzCount[laneId] += nonzeros[colC] == 0;
				nonzeros[colC] = 1;
			}
      
			__syncthreads();
		}

		if(laneId == 0)
		{
			nnz = nzCount[0];
			for(INT i = 1; i < BLOCKSIZE; ++i)
			{
				nnz += nzCount[i];
			}
			C_row[rowA] = nnz;

		}
		
		__syncthreads();
	}
}

AdjMatrixCSR csr_spmm_cuda(AdjMatrixCSR& A, AdjMatrixCSR& B) {
    INT* A_row;
    INT* A_col;
    INT* A_val;
    INT* B_row;
    INT* B_col;
    INT* B_val;
    INT* C_row;
    INT* C_row_gpu;
    INT* work;
    
    numrows = A.num_rows();
    C_row = (INT*)malloc(sizeof(INT) * (numrows+1));

    cudaMalloc(&A_row, (A.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&A_col, A.num_size() * sizeof(INT));
    cudaMalloc(&A_val, A.num_size() * sizeof(INT));
    cudaMalloc(&B_row, (B.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&B_col, B.num_size() * sizeof(INT));
    cudaMalloc(&B_val, B.num_size() * sizeof(INT));
    cudaMalloc(&C_row_gpu, (A.num_rows() + 1) * sizeof(INT));
    cudaMalloc(&work, 1024*B.num_rows() * sizeof(INT));

    cudaMemcpy(A_row, A.get_rows(), (A.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_col, A.get_cols(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(A_val, A.get_vals(), A.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_row, B.get_rows(), (B.num_rows() + 1) * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_col, B.get_cols(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);
    cudaMemcpy(B_val, B.get_vals(), B.num_size() * sizeof(INT), cudaMemcpyHostToDevice);

    // call kernel

    GetNNZ<<<GRIDSIZE, BLOCKSIZE,numrows>>>(A_row, A_col, A_val, B_row, B_col, B_val, C_row_gpu, work,numrows);
    cudaMemcpy(C_row, C_row_gpu, (A.num_rows() + 1) * sizeof(INT), cudaMemcpyDeviceToHost);

    // prefix sum
    C_row[0] = 0;
    for (INT i = 0; i < numrows; i++) {
        C_row[i+1] += C_row[i];
    }


    // cudaMemcpy to host
    AdjMatrixCSR result(A.num_rows(), 0, C_row, NULL, NULL);
    return result;
}