#include <vector>

struct Matrix {
    std::vector<double> data;
    int rows;
    int columns;

    Matrix(int r, int c, double value = 0.0) : 
        rows(r),
        columns(c),
        data(r * c, value) {}
};

class MatrixMultiplier {
private:
    Matrix solution_;

public:
    MatrixMultiplier() = default;
    bool multiply(int rank, int size, const Matrix& _A, const Matrix& _B);
    const Matrix& getSolution() const;
    void printSolution() const;

private:
    
};