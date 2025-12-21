cmake_minimum_required(VERSION 3.29)

project(LOAD_BALANCING LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)

add_executable(LOAD_BALANCING 
    main.cpp
)

target_include_directories(LOAD_BALANCING PUBLIC include)

target_link_libraries(LOAD_BALANCING PUBLIC MPI::MPI_CXX)