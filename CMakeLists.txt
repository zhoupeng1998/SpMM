cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project(spmm LANGUAGES CXX CUDA)

add_subdirectory(datamodel)
add_subdirectory(serial)
add_subdirectory(main)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)