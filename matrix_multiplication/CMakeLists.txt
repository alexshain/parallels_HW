cmake_minimum_required(VERSION 3.29)

project(MPI_MATRIX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)

add_executable(MPI_MATRIX 
    src/main.cpp
    src/MatrixMultiplier.cpp
)

target_include_directories(MPI_MATRIX PUBLIC include)

target_link_libraries(MPI_MATRIX PUBLIC MPI::MPI_CXX)