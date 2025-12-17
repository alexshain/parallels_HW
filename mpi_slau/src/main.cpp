#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "JacobiSolver.h"

std::vector<double> createKnownSolutionMatrix(int n) {
    std::vector<double> A(n * n, 0.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = n + 1.0;
            } else {
                A[i * n + j] = 1.0;
            }
        }
    }
    return A;
}

std::vector<double> createKnownSolutionVector(int n) {
    std::vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = 2.0 * n;
    }
    return b;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1024; 
    
    auto A = createKnownSolutionMatrix(n);
    auto b = createKnownSolutionVector(n);
    
    JacobiSolver solver(A, b);
    bool success = solver.solve(1e-6, 10000, rank, size);
    
    if(rank == 0) {
        if (success) {
            std::cout << "Решение найдено" << std::endl;
            auto solution = solver.getSolution();
            std::cout << "Первые 10 компонент: " << std::endl;
            for (int i = 0; i < 10; i++) {
                std::cout << "x[" << i << "] = " << solution[i] << std::endl;
            }
        } else {
            std::cout << "Что то не так!" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}