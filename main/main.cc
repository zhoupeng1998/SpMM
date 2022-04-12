#include <assert.h>

#include <iostream>

#include "data.h"
#include "adj_edges.h"
#include "adj_list.h"
#include "adj_matrix_dense.h"
#include "adj_matrix_csr.h"
#include "test_gpu.h"
#include "test_simple.h"
#include "spmm_serial.h"
#include "time.h"

#include "test_full.h"

int main(void) {
    // full graph
    //test_cpu_full_v1();

    // partial graph
    test_cpu_full_v1(10000);

    return 0;
}
