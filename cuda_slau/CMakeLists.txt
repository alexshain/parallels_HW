cmake_minimum_required(VERSION 3.18)
project(CUDA_SLAU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

add_executable(CUDA_SLAU 
    src/main.cpp
    src/JacobiSolver.cu
)

target_include_directories(CUDA_SLAU PUBLIC include)

target_link_libraries(CUDA_SLAU PUBLIC CUDA::cudart)