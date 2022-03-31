#include <assert.h>
#include <stdlib.h>

#include <iostream>

#include "serial_spmm.h"
#include "adj_matrix_csr.h"

AdjMatrixCSR serial_spmm_csr(AdjMatrixCSR& A, AdjMatrixCSR& B) {
    int A_nrow = A.num_rows(), A_nnz = A.num_size(), B_nrow = B.num_rows(), B_nnz = B.num_size();
    int* A_rows = A.get_rows();
    int* A_cols = A.get_cols();
    int* A_vals = A.get_vals();
    int* B_rows = B.get_rows();
    int* B_cols = B.get_cols();
    int* B_vals = B.get_vals();

    AdjMatrixCSR C;
    int* work = (int*)calloc(B_nrow, sizeof(int));
    int* C_rows = (int*)malloc(sizeof(int) * (A_nrow + 1));
    //int pos = 0;
    for (int i1 = 0; i1 < A_nrow; i1++) {
        int count = 0;
        int mark = i1 + 1;
        //int ipos = pos;
        for (int i2 = A_rows[i1]; i2 < A_rows[i1+1]; i2++) {
            int j = A_cols[i2];
            //int va = A_vals[i2];
            assert(j >= 0 && j < B_nrow);
            for (int i3 = B_rows[j]; i3 < B_rows[j+1]; i3++) {
                int col = B_cols[i3];
                //int vb = B_vals[i3];
                assert(col >= 0 && col < B_nrow);
                if (work[col] != mark) {
                    count++;
                    work[col] = mark;
                }
            }
        }
        C_rows[i1+1] = count + C_rows[i1];
    }
    // TODO: complete
    for (int i = 0; i < B_nrow; i++) {
        std::cout << work[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i <= A_nrow; i++) {
        std::cout << C_rows[i] << " ";
    }
    std::cout << std::endl;
    return C;
}