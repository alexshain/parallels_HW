#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>


class GaussSolver {
    private:
        std::vector<std::vector<double>> A;  
        std::vector<double> b;               
        std::vector<double> x;           
        int n;                             
    
    public:
        GaussSolver(const std::vector<std::vector<double>>& _A, const std::vector<double>& _b);
        bool solve(double tolerance, int max_iterations);
        const std::vector<double>& getSolution() const;
        void printSolution() const;
    };