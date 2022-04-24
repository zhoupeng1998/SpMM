#include "test_hadamard.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "dense_mm_serial.h"
#include "spmm_serial.h"
#include "hadamard_product.h"

void test_hadamard() {
    INT sample_A[] = {0,2,0,1,0,
                     1,0,3,0,0,
                     0,0,0,0,1,
                     0,0,0,0,0,
                     1,1,0,0,0};

    AdjMatrixDense dense_A(5, sample_A);
    AdjMatrixCSR csr_A(dense_A); // will cause invalid free here

    AdjMatrixDense dense_AA = dense_mm_cpu(dense_A, dense_A);
    AdjMatrixCSR* csr_AA = csr_spmm_cpu(&csr_A, &csr_A);

    AdjMatrixDense hadamard_dense_A = hadamard_product(dense_AA, dense_A);
    dense_AA.dump();
    dense_A.dump();
    hadamard_dense_A.dump();

    AdjMatrixCSR B(dense_AA);
    AdjMatrixCSR hadamard_csr_A = hadamard_product(B, csr_A);

    AdjMatrixCSR hadamard_dense_A_csr(hadamard_dense_A);

    hadamard_dense_A_csr.dump();
    hadamard_csr_A.dump();
}