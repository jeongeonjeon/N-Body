cmake_minimum_required(VERSION 3.26)
project(N_body)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(MPI)

find_package(Python COMPONENTS Development NumPy)
include_directories(matplotlib-cpp)

add_executable(N_body main.cpp)

target_compile_options(N_body PRIVATE -Wall)
if(OPENMP_FOUND)
    target_compile_options(N_body PRIVATE -fopenmp -fopenmp-simd)
    target_link_libraries(N_body PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
endif()
if(MPI_FOUND)
    target_link_libraries(N_body PRIVATE MPI::MPI_CXX)
endif()
if(PYTHON_FOUND)
    target_link_libraries(N_body PRIVATE Python::Python Python::NumPy)
endif ()