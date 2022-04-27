#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <omp.h>
#include <time.h>

#include "data.h"
#include "spmm_serial.h"
#include "adj_matrix_csr.h"
#include "timer.h"

AdjMatrixCSR *csr_spmm_cpu_symbolic(AdjMatrixCSR *A, AdjMatrixCSR *B, INT *work)
{
    INT i1, i2, i3;
    INT m = A->num_rows(), k = A->num_rows(), n = B->num_rows();

    AdjMatrixCSR*C=new AdjMatrixCSR(m,n);


    C->rowPtr = (INT*) malloc((m+1)*sizeof(INT));
    for (i1 = 0; i1 < m; i1++)
    {
        INT MARK=i1+1;
        INT count = 0;
        for (i2 = A->rowPtr[i1]; i2 < A->rowPtr[i1+1]; i2++)
        {
            INT j = A->colInd[i2]; 
            /*
            if(j < 0|| j>=k){
                std::cout<<j<<std::endl;
            }
            */
            assert(j >= 0 && j < k);
            for (i3 = B->rowPtr[j]; i3 < B->rowPtr[j+1]; i3++)
            {
                INT col = B->colInd[i3];
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
    C->colInd = (INT *) malloc(C->size*sizeof(INT));
    // if(C->colInd){
    //     std::cout<<"malloc C->colInd success the required size  colInd"<< C->size<<std::endl;
    // }else{
    //     std::cout<<"malloc C->colInd failed, the required size  colInd"<< C->size<<std::endl;
    // }
    C->val =  (INT *) malloc(C->size*sizeof(INT));

    
    // if(C->val){
    //     std::cout<<"malloc C->colInd success the required size val"<< C->size<<std::endl;
    // }else{
    //     std::cout<<"malloc C->colInd failed, the required size val"<< C->size<<std::endl;
    // }

    //std::cout<<"finished symbolic"<<std::endl;

    return C;
}

void csr_spmm_cpu_numeric(AdjMatrixCSR *A, AdjMatrixCSR *B, AdjMatrixCSR *C, INT *work)
{
    INT i1, i2, i3;
    INT m = A->num_rows(), k = A->num_rows(), n = B->num_rows(), pos = 0;


    for (i1 = 0; i1 < m; i1++)
    {
        INT ipos=pos;

        for (i2 = A->rowPtr[i1]; i2 < A->rowPtr[i1+1]; i2++)
        {
            INT j = A->colInd[i2];
            INT va = A->val[i2];
            assert(j >= 0 && j < k);

            for (i3 = B->rowPtr[j] ; i3 < B->rowPtr[j+1] ; i3++)
            {
                INT q, col = B->colInd[i3];
                INT vb = B->val[i3];
                assert(col >= 0 && col < n);
 
                if ((q = work[col]) <= ipos)
                {   
                    C->colInd[pos] = col;

                    C->val[pos] = va*vb;
  
                    work[col] = ++pos;
                }
                else
                {
                    assert(C->colInd[q-1] == col);
                    C->val[q-1] += va*vb;

                }
            }
            
        }
        assert(C->rowPtr[i1+1] == pos);
    }
}

AdjMatrixCSR * csr_spmm_cpu(AdjMatrixCSR *A, AdjMatrixCSR *B)
{
    struct timespec start,end;
    double time ;

    INT *work = (INT *) calloc(B->rows, sizeof(INT));
    
    clock_start_cpu();
    AdjMatrixCSR *C = csr_spmm_cpu_symbolic(A, B, work);
    clock_stop_cpu();
    std::cout << "Symbolic: " << get_time_cpu() << " ms" << std::endl;

    memset(work, 0, B->cols*sizeof(INT));
   
    clock_start_cpu();
    csr_spmm_cpu_numeric(A, B, C, work);
    clock_stop_cpu();
    std::cout << "Numeric: " << get_time_cpu() << " ms" << std::endl;

    free(work);
    return C;
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

AdjMatrixDense csr_spmm_dense_cpu(AdjMatrixCSR& A, AdjMatrixCSR& B) {
    AdjMatrixDense result(A.num_rows());
    for (int i1 = 0; i1 < A.num_rows(); i1++) {
        for (int i2 = A.rowPtr[i1]; i2 < A.rowPtr[i1+1]; i2++) {
            int colA = A.colInd[i2];
            int va = A.val[i2];
            assert(colA >= 0 && colA < B.num_rows());
            for (int i3 = B.rowPtr[colA]; i3 < B.rowPtr[colA+1]; i3++) {
                int colB = B.colInd[i3];
                int vb = B.val[i3];
                assert(colB >= 0 && colB < B.num_rows());
                result[i1][colB] += va * vb;
            }
        }
    }
    // set result edges
    return result;
}