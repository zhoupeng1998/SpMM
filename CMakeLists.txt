
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project(spmm LANGUAGES CXX )
add_definitions(-std=c++11)
add_subdirectory(datamodel)
add_subdirectory(spmm)
add_subdirectory(test)
add_subdirectory(main)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
