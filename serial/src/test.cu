#include <iostream>
#include <device_launch_parameters.h>

#include "test.h"

void printInfo() {
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    for (int i = 0; i < devcnt; i++) {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, i);
        std::cout << "Using GPU device " << i << ": " << devprop.name << std::endl;
        std::cout << "Total memory: " << devprop.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "Shared memory per block: " << devprop.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
        std::cout << "Max thread per block: " << devprop.maxThreadsPerBlock << std::endl;
        std::cout << "32-bit registers per block: " << devprop.regsPerBlock << std::endl;
        std::cout << "Total SM: " << devprop.multiProcessorCount << std::endl;
        std::cout << "Max threads per SM: " << devprop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max warps per SM: " << devprop.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "======================================================" << std::endl;
    }
}