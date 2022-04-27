#ifndef _TEST_FINAL_H_
#define _TEST_FINAL_H_

void test_var_nnz_spmm_dense_cpu(int percentage);
void test_var_nnz_spmm_ge_cpu(int percentage);

void test_var_nnz_spmm_dense_gpu(int percentage);
void test_var_nnz_spmm_ge_gpu(int percentage);

#endif