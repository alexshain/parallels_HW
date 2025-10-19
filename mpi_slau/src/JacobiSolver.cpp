#include "JacobiSolver.h"
#include <iostream>
#include <cmath>
#include <mpi.h>

JacobiSolver::JacobiSolver(const std::vector<double>& _A, const std::vector<double>& _b)
    : A(_A), b(_b) {
    n = b.size();
    x.resize(n, 0.0);
}

bool JacobiSolver::solve(double tolerance, int max_iterations, int rank, int size) {
    double start_time = MPI_Wtime();
    std::vector<double> x_old(n, 0.0);
    int iteration = 0;
    double global_error = tolerance + 1.0;
    
    int local_rows = n / size;
    int start_row = rank * local_rows;
    int end_row = start_row + local_rows;

    while (iteration < max_iterations && global_error > tolerance) {
        x_old = x;
        std::vector<double> x_new(n, 0.0); 

        for (int i = start_row; i < end_row; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) sum += A[i * n + j] * x_old[j];  
            }
            x_new[i] = (b[i] - sum) / A[i * n + i];  
        }

        MPI_Allgather(
            x_new.data() + start_row,  
            local_rows,            
            MPI_DOUBLE,              
            x.data(),              
            local_rows,             
            MPI_DOUBLE,             
            MPI_COMM_WORLD
        );

        double local_error = 0.0;
        for (int i = start_row; i < end_row; i++) {
            double diff = x_new[i] - x_old[i]; 
            local_error += diff * diff;
        }

        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_error = sqrt(global_error);

        iteration++;
    }
    
    double end_time = MPI_Wtime();
    
    if(rank == 0) {
        std::cout << "global err: " << global_error << std::endl;
        std::cout << "iteration: " << iteration << std::endl;
        std::cout << "Время: " << end_time - start_time << " секунд" << std::endl;
    }
    
    return (global_error <= tolerance);
}

const std::vector<double>& JacobiSolver::getSolution() const {
    return x;
}

void JacobiSolver::printSolution() const {
    std::cout << "Решение СЛАУ:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
}