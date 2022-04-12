cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

project(spmm LANGUAGES CXX CUDA)
add_definitions(-std=c++11)
add_subdirectory(utils)
add_subdirectory(datamodel)
add_subdirectory(spmm)
add_subdirectory(test)
add_subdirectory(main)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
