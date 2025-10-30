#include <vector>

class JacobiSolver {
private:
    std::vector<double> A;  
    std::vector<double> b;               
    std::vector<double> x;           
    int n;                             

    double* d_A = nullptr;
    double* d_b = nullptr;
    double* d_x = nullptr;
    double* d_x_new = nullptr;
    double* d_x_old = nullptr;

public:
    JacobiSolver(const std::vector<double>& _A, const std::vector<double>& _b);
    ~JacobiSolver();
    bool solve(double tolerance, int max_iterations, int rank, int size);
    const std::vector<double>& getSolution() const;
    void printSolution() const;

private:
    void initializeCUDA();
    void cleanupCUDA();
};