#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "serial_spmm.h"
#include "adj_matrix_csr.h"


void csr_spmm_symbolic(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, int *work)
{
    int i1, i2, i3;
    int m = A->num_rows(), k = A->num_rows(), n = B->num_rows();

    C->rowPtr = (int *) malloc((m+1)*sizeof(int));
    
    for (i1 = 0; i1 < m; i1++)
    {
        int MARK=i1+1;
        int count = 0;
        for (i2 = A->rowPtr[i1]; i2 < A->rowPtr[i1+1]; i2++)
        {
            //std::cout<<"i2 is  "<<i2<<std::endl;
            int j = A->colInd[i2]; //某行
            if(j < 0|| j>=k){
                std::cout<<j<<std::endl;
            }
            assert(j >= 0 && j < k);
            for (i3 = B->rowPtr[j]; i3 < B->rowPtr[j+1]; i3++)
            {
                int col = B->colInd[i3];
                assert(col >= 0 && col < n);
                if (work[col] != MARK)
                {
                    count++;
                    work[col] = MARK;
                }
            }
        }
        //printf("%d\n",count);
        C->rowPtr[i1+1] = count;
    }
    for (i1=0, C->rowPtr[0]=0; i1 < m; i1++)
    {
        C->rowPtr[i1+1] += C->rowPtr[i1];

    }
    C->rows = m;
    C->cols = n;
    C->size = C->rowPtr[m];

    C->colInd = (int *) malloc(C->size*sizeof(int));
    C->val =  (int *) malloc(C->size*sizeof(int));
    std::cout<<"finished symbolic"<<std::endl;

}


void csr_spmm_numeric(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, int *work)
{
    int i1, i2, i3;
    int m = A->num_rows(), k = A->num_rows(), n = B->num_rows(), pos = 0;

    std::cout<<"start loop1"<<std::endl;
    for (i1 = 0; i1 < m; i1++)
    {
        int ipos=pos;
        std::cout<<"start loop2"<<std::endl;
        for (i2 = A->rowPtr[i1]; i2 < A->rowPtr[i1+1]; i2++)
        {
            int j = A->colInd[i2];
            int va = A->val[i2];
            assert(j >= 0 && j < k);
            std::cout<<"start loop3"<<std::endl;
            for (i3 = B->rowPtr[j] ; i3 < B->rowPtr[j+1] ; i3++)
            {
                int q, col = B->colInd[i3];
                int vb = B->val[i3];
                assert(col >= 0 && col < n);
                std::cout<<"enter if"<<std::endl;
                if ((q = work[col]) <= ipos)
                {   std::cout<<"1"<<std::endl;
                    std::cout<<"col[0]"<<C->colInd[0]  <<std::endl;
                    C->colInd[pos] = col;
                    std::cout<<"2"<<std::endl;
                    C->val[pos] = va*vb;
                    std::cout<<"3"<<std::endl;
                    work[col] = ++pos;
                }
                else
                {
                    assert(C->colInd[q-1] == col);
                    C->val[q-1] += va*vb;

                }
            }
            std::cout<<"fuck"<<std::endl;
        }
        assert(C->rowPtr[i1+1] == pos);
    }
}

void csr_spmm_cpu(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C)
{
    int *work = (int *) calloc(B->rows, sizeof(int));
    csr_spmm_symbolic(A, B, C, work);
    memset(work, 0, B->cols*sizeof(int));
    std::cout<<"start numeric"<<std::endl;
    csr_spmm_numeric(A, B, C, work);
    free(work);
}

//AdjMatrixCSR serial_spmm_Symbolic(AdjMatrixCSR& A, AdjMatrixCSR& B) {
//    int A_nrow = A.num_rows(), A_nnz = A.num_size(), B_nrow = B.num_rows(), B_nnz = B.num_size();
//    int* A_rows = A.get_rows();
//    int* A_cols = A.get_cols();
//    int* A_vals = A.get_vals();
//    int* B_rows = B.get_rows();
//    int* B_cols = B.get_cols();
//    int* B_vals = B.get_vals();
//
//    AdjMatrixCSR C;
//    int* work = (int*)calloc(B_nrow, sizeof(int));
//    int* C_rows = (int*)malloc(sizeof(int) * (A_nrow + 1));
//    //int pos = 0;
//    for (int i1 = 0; i1 < A_nrow; i1++) {
//        int count = 0;
//        int mark = i1 + 1;
//        //int ipos = pos;
//        for (int i2 = A_rows[i1]; i2 < A_rows[i1+1]; i2++) {
//            int j = A_cols[i2];
//            //int va = A_vals[i2];
//            assert(j >= 0 && j < B_nrow);
//            for (int i3 = B_rows[j]; i3 < B_rows[j+1]; i3++) {
//                int col = B_cols[i3];
//                //int vb = B_vals[i3];
//                assert(col >= 0 && col < B_nrow);
//                if (work[col] != mark) {
//                    count++;
//                    work[col] = mark;
//                }
//            }
//        }
//        C_rows[i1+1] = count + C_rows[i1];
//    }
//    // TODO: complete
//    for (int i = 0; i < B_nrow; i++) {
//        std::cout << work[i] << " ";
//    }
//    std::cout << std::endl;
//    for (int i = 0; i <= A_nrow; i++) {
//        std::cout << C_rows[i] << " ";
//    }
//    std::cout << std::endl;
//    return C;
//}