// JacobiSolver.cu
#include "JacobiSolver.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void jacobiKernel(const double* A, const double* b, const double* x_old, 
                            double* x_new, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                sum += A[i * n + j] * x_old[j];
            }
        }
        x_new[i] = (b[i] - sum) / A[i * n + i];
    }
}

__global__ void errorKernel(const double* x_new, const double* x_old, 
                           double* error, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double diff = x_new[i] - x_old[i];
        error[i] = diff * diff;
    }
}

JacobiSolver::JacobiSolver(const std::vector<double>& _A, const std::vector<double>& _b)
    : A(_A), b(_b) {
    n = b.size();
    x.resize(n, 0.0);
    initializeCUDA();
}

JacobiSolver::~JacobiSolver() {
    cleanupCUDA();
}

void JacobiSolver::initializeCUDA() {
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_x_new, n * sizeof(double));
    cudaMalloc(&d_x_old, n * sizeof(double));
    
    cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
}

void JacobiSolver::cleanupCUDA() {
    if (d_A) cudaFree(d_A);
    if (d_b) cudaFree(d_b);
    if (d_x) cudaFree(d_x);
    if (d_x_new) cudaFree(d_x_new);
    if (d_x_old) cudaFree(d_x_old);
}

bool JacobiSolver::solve(double tolerance, int max_iterations) {
    double start_time = clock();
    
    int iteration = 0;
    double global_error = tolerance + 1.0;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    double* d_error;
    cudaMalloc(&d_error, n * sizeof(double));
    std::vector<double> h_error(n);

    while (iteration < max_iterations && global_error > tolerance) {
        jacobiKernel<<<num_blocks, block_size>>>(d_A, d_b, d_x, d_x_new, n);
        cudaDeviceSynchronize();

        errorKernel<<<num_blocks, block_size>>>(d_x_new, d_x, d_error, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_error.data(), d_error, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        global_error = 0.0;
        for (int i = 0; i < n; i++) {
            global_error += h_error[i];
        }
        global_error = sqrt(global_error);

        iteration++;
    }
    
    cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_error);
    
    double end_time = clock();
    std::cout << "Время: " << (end_time - start_time) / CLOCKS_PER_SEC << " секунд" << std::endl;
    
    return (global_error <= tolerance);
}

const std::vector<double>& JacobiSolver::getSolution() const {
    return x;
}

void JacobiSolver::printSolution() const {
    std::cout << "Решение СЛАУ:" << std::endl;
    for (int i = 0; i < std::min(n, 10); i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
}