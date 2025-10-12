#include "GaussSolver.h"
#include <iostream>
#include <cmath>
#include <omp.h>

GaussSolver::GaussSolver(const std::vector<std::vector<double>>& _A, const std::vector<double>& _b)
    : A(_A), b(_b) {
    n = b.size();
    x.resize(n, 0.0);
}

bool GaussSolver::solve(double tolerance, int max_iterations) {
    double start_time = omp_get_wtime();
    std::vector<double> x_new(n, 0.0);
    int iteration = 0;
    double error = 0;
    
    while (true) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) sum += A[i][j] * x[j];
            }
            x_new[i] = (b[i] - sum) / A[i][i]; 
        }
        
        double sum = 0;
        
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < x.size(); i++) {
            sum += pow(x_new[i] - x[i], 2);
            x[i] = x_new[i];
        }
        error = sqrt(sum);

        iteration++;

        if(iteration > max_iterations && error < tolerance) {
            break;
        }
    }
    
    double end_time = omp_get_wtime();
    
    std::cout << "Время: " << end_time - start_time << " секунд" << std::endl;
    
    return (error <= tolerance);
}

const std::vector<double>& GaussSolver::getSolution() const {
    return x;
}

void GaussSolver::printSolution() const {
    std::cout << "Решение СЛАУ:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
}