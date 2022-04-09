#include <iostream>

#include "data.h"
#include "test_simple.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "serial_spmm.h"
#include "serial_dense_mm.h"

void test_A() {
    int sample_A[] = {0,2,0,1,0,
                      1,0,3,0,0,
                      0,0,0,0,1,
                      0,0,0,0,0,
                      1,1,0,0,0};

    int sample_B[] = {0,1,0,0,2,
                      2,0,0,1,1,
                      1,0,0,3,0,
                      0,0,2,0,0,
                      0,3,0,1,0};
    
    AdjMatrixDense dense_A(5, sample_A);
    AdjMatrixDense dense_B(5, sample_B);

    AdjMatrixCSR csr_A(dense_A);
    AdjMatrixCSR csr_B(dense_B);

    INT* A_rows = csr_A.get_rows();
    int* A_cols = csr_A.get_cols();
    int* A_vals = csr_A.get_vals();

    INT* B_rows = csr_B.get_rows();
    int* B_cols = csr_B.get_cols();
    int* B_vals = csr_B.get_vals();

    std::cout << "A - csr" << std::endl;
    std::cout << csr_A.num_rows() << " " << csr_A.num_size() << std::endl;
    for (INT i = 0; i <= csr_A.num_rows(); i++) {
        std::cout << A_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_A.num_size(); i++) {
        std::cout << A_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_A.num_size(); i++) {
        std::cout << A_vals[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "B - csr" << std::endl;
    std::cout << csr_B.num_rows() << " " << csr_B.num_size() << std::endl;
    for (INT i = 0; i <= csr_B.num_rows(); i++) {
        std::cout << B_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_B.num_size(); i++) {
        std::cout << B_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_B.num_size(); i++) {
        std::cout << B_vals[i] << " ";
    }
    std::cout << '\n' << std::endl;

    serial_spmm_csr(csr_A, csr_B);

    std::cout << std::endl;

    AdjMatrixDense dense_C = serial_dense_mm(dense_A, dense_B);
    std::cout << "Result (dense):" << std::endl;
    for (INT i = 0; i < dense_C.size(); i++) {
        for (INT j = 0; j < dense_C.size(); j++) {
            std::cout << dense_C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    AdjMatrixCSR csr_C(dense_C);

    std::cout << "C - csr" << std::endl;
    INT* C_rows = csr_C.get_rows();
    int* C_cols = csr_C.get_cols();
    int* C_vals = csr_C.get_vals();
    std::cout << csr_C.num_rows() << " " << csr_C.num_size() << std::endl;
    for (INT i = 0; i <= csr_C.num_rows(); i++) {
        std::cout << C_rows[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_C.num_size(); i++) {
        std::cout << C_cols[i] << " ";
    }
    std::cout << std::endl;
    for (INT i = 0; i < csr_C.num_size(); i++) {
        std::cout << C_vals[i] << " ";
    }
    std::cout << '\n' << std::endl;
}