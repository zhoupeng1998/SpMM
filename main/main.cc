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
#include "test_generate_graph.h"
#include "test_hadamard.h"

int main(void) {
    // full graph
    //test_cpu_full_v1();

    // partial graph
    //test_cpu_full_v1(10000);
    //test_cuda_full_v1();

    // comment out this line if you don't want to re-generate graph
    /*
    produce_graph(8192, 1000000);
    test_testgraph_spmm_gpu();
    test_testgraph_spmm_cpu();
    */
    //test_testgraph_spmm_dense_gpu();
    
    //test_hadamard();
    test_cpu_full_v1(80000);
    test_cuda_full_v1(80000);
    
    return 0;
}
