#include "data.h"
#include "adj_matrix_csr.h"
#include "spmm_cuda.h"
#include "stdio.h"
#include <iostream>


// #define GRIDSIZE 128
// #define BLOCKSIZE 1024
#define SIZE GRIDSIZE*BLOCKSIZE

__managed__ INT numrows;


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

__global__ void GetVals(INT* A_row, INT* A_col, INT* A_val, INT* B_row, INT* B_col, INT* B_val, 
INT* C_row, INT* C_col, INT* C_val, int* indexTable)
{
	const int laneId = threadIdx.x;
	const int bloackId = blockIdx.x;
	
	__shared__ unsigned int back;
	
	int rowAStart; // The index into A.jc and A.val
	int rowAEnd; // The boundary index for A
	float valA; // The value of the current A nonzero
	int rowBStart; // The index into B.jc and B.val
	int rowBEnd; // The boundary index for B
	int colB; // The current column in B being used
	int rowCStart; // The index into C.jc and C.val
	int rowCEnd; // The boundary index for C
	int hash; // The calculated hash value
	int i, j; // Loop iterators

	// Set the global hash table to point to the space
	// used by this warp
	int* gColHashTable;
	float* gValHashTable;
	int globalEntries;
	
	indexTable = &indexTable[C.cols * blockId];
	
	if(laneId == 0)
		back = 0;
	
	for(int rowA = blockId; rowA < numrows; rowA += gridDim.x)
	{
		rowAStart = A_row[rowA];
		rowAEnd = A_row[rowA + 1];
		for(i = laneId; i < numrows; ++i)
		{
			indexTable[i] = -1;
		}
		__syncthreads();

		// Set the location of the global hash table
		rowCStart = C_row[rowA];
		rowCEnd = C_row[rowA + 1];
		globalEntries = rowCEnd - rowCStart;
		gColHashTable = &C_col[rowCStart];
		gValHashTable = &C_val[rowCStart];
		for(i = rowAStart; i < rowAEnd; ++i)
		{
			valA = A_val[i];
			rowBStart = B_row[A_col[i]];
			rowBEnd = B_row[A_col[i] + 1];
			int curIdx;
			int* storeInt;
			float* storeFloat;
			float valB;
			for(j = rowBStart + laneId; __any(j < rowBEnd); j += warpSize)
			{
				colB = j < rowBEnd ? B_col[j] : -1;
				curIdx = colB == -1 ? -1 : indexTable[colB];
				hash = colB != -1 && curIdx == -1 ? atomicInc(&back, globalEntries - 1) : curIdx;
				storeInt = hash == -1 ? &hash : &indexTable[colB];
				*storeInt = hash;
				storeInt = hash == -1 ? &colB : &gColHashTable[hash];
				*storeInt = colB;
				valB = colB == -1 ? 1 : B_val[j];
				storeFloat = hash == -1 ? &valA : &gValHashTable[hash];
				*storeFloat += valB * valA;
			}
		} // For each nonzero in the A row
	} // For each assigned row in A
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
	INT* C_col;
	INT* C_col_gpu;
	INT* C_val;
	INT* C_val_gpu;
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

	cudaMalloc(&C_col_gpu, C_row[numrows+1] * sizeof(INT));
	cudaMalloc(&C_val_gpu, C_row[numrows+1] * sizeof(INT));
	


	GetVals<<<GRIDSIZE, BLOCKSIZE>>>(A_row, A_col, A_val, B_row, B_col, B_val, C_row_gpu,C_col_gpu,C_val_gpu, work,numrows);
	cudaMemcpy(C_col, C_col_gpu, (C_row[numrows+1] * sizeof(INT), cudaMemcpyDeviceToHost);
	cudaMemcpy(C_val, C_val_gpu, (C_row[numrows+1] * sizeof(INT), cudaMemcpyDeviceToHost);

    // cudaMemcpy to host
    AdjMatrixCSR result(A.num_rows(), 0, C_row, NULL, NULL);
    
    result.size=C_row[numrows+1];
    return result;
}