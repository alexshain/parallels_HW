cmake_minimum_required(VERSION 3.29)

project(OMP_SLAU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(OMP_SLAU 
    src/main.cpp
    src/GaussSolver.cpp
)

target_include_directories(OMP_SLAU PUBLIC include)

target_link_libraries(OMP_SLAU PUBLIC "-fopenmp")
