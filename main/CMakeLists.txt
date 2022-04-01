project(main)

add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME}
        spmm::serial
        spmm::datamodel
        spmm::test
        )