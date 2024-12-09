cmake_minimum_required(VERSION 3.30)
project(N_body_Code_Skeleton)

set(CMAKE_CXX_STANDARD 14)

include_directories("Code Skeleton")

add_executable(N_body_Code_Skeleton
    "Code Skeleton/build/Debug/main.o"
    "Code Skeleton/cmake-build-debug/Testing/Temporary/LastTest.log"
    "Code Skeleton/cmake-build-debug/cmake_install.cmake"
    "Code Skeleton/cmake-build-debug/CMakeCache.txt"
    "Code Skeleton/cmake-build-debug/Makefile"
    "Code Skeleton/cmake-build-debug/N_body"
    "Code Skeleton/CMakeLists.txt"
    "Code Skeleton/main.cpp"
    "Code Skeleton/matplotlibcpp.h")
