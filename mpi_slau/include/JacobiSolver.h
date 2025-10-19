#include <vector>

class JacobiSolver {
    private:
        std::vector<double> A;  
        std::vector<double> b;               
        std::vector<double> x;           
        int n;                             
    
    public:
        JacobiSolver(const std::vector<double>& _A, const std::vector<double>& _b);
        bool solve(double tolerance, int max_iterations, int rank, int size);
        const std::vector<double>& getSolution() const;
        void printSolution() const;
    };