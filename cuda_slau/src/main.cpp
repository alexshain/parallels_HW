#include <iostream>
#include <vector>
#include <cmath>
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

int main() {
    int n = 1000; 
    
    auto A = createKnownSolutionMatrix(n);
    auto b = createKnownSolutionVector(n);
    
    JacobiSolver solver(A, b);
    
    bool success = solver.solve(1e-6, 10000);
    
    if (success) {
        std::cout << "Решение найдено!" << std::endl;
        solver.printSolution();
    } else {
        std::cout << "Не удалось найти решение!" << std::endl;
    }
    
    return 0;
}