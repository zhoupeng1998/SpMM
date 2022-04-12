#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <cublas.h>

#include "timer.h"

struct timespec timer_start_cpu, timer_stop_cpu;
cudaEvent_t timer_start_cuda, timer_stop_cuda;
float time_cuda;

void clock_start_cpu() {
    if (clock_gettime(CLOCK_REALTIME, &timer_start_cpu) == -1) {
        perror("clock gettime");
        exit(EXIT_FAILURE);
    }
}

void clock_stop_cpu() {
    if (clock_gettime(CLOCK_REALTIME, &timer_stop_cpu) == -1) {
        perror("clock gettime");
        exit(EXIT_FAILURE);
    }
}

double get_time_cpu() {
    double time = (timer_stop_cpu.tv_sec - timer_start_cpu.tv_sec)+ (double)(timer_stop_cpu.tv_nsec - timer_start_cpu.tv_nsec)/1e9;
    return time * 1e9;
}

void clock_start_cuda() {
    cudaEventCreate(&timer_start_cuda);
    cudaEventCreate(&timer_stop_cuda);
    cudaEventRecord(timer_start_cuda, 0);
}

void clock_stop_cuda() {
    cudaEventRecord(timer_stop_cuda, 0);
    cudaEventSynchronize(timer_stop_cuda);
    cudaEventElapsedTime(&time_cuda, timer_start_cuda, timer_stop_cuda);
}

double get_time_cuda() {
    return time_cuda * 1e6;
}