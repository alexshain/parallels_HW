#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "GaussSolver.h"

std::vector<std::vector<double>> createKnownSolutionMatrix(int n) {
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = n + 1.0;
            } else {
                A[i][j] = 1.0;
            }
        }
    }
    return A;
}

std::vector<double> createKnownSolutionVector(int n) {
    std::vector<double> b(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        b[i] = 2.0 * n;
    }
    return b;
}

int main() {
    int n = 1000; 
    
    auto A = createKnownSolutionMatrix(n);
    auto b = createKnownSolutionVector(n);
    
    GaussSolver solver(A, b);
    
    if (solver.solve(1e-6, 1000)) {
        std::cout << "Решение найдено" << std::endl;
        
        auto solution = solver.getSolution();
        std::cout << "Первые 10 компонент решения: " << std::endl;
        for (int i = 0; i < std::min(10, n); i++) {
            std::cout << "x[" << i << "] = " << solution[i] << std::endl;
        }
    } else {
        std::cout << "Какая то ошибка!" << std::endl;
    }
    
    return 0;
}