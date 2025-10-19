cmake_minimum_required(VERSION 3.29)

project(MPI_SLAU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)

add_executable(MPI_SLAU 
    src/main.cpp
    src/JacobiSolver.cpp
)

target_include_directories(MPI_SLAU PUBLIC include)

target_link_libraries(MPI_SLAU PUBLIC MPI::MPI_CXX)