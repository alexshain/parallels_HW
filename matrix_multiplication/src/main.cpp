#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "MatrixMultiplier.h"

Matrix createOnesMatrix(int rows, int columns) {
    return Matrix(rows, columns, 1.0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // возвращает номер данного процесса в коммуникаторе
    MPI_Comm_size(MPI_COMM_WORLD, &size); // возвращает количество процессов в коммуникаторе

    int rowsA = 10;
    int colsA = 20;
    int rowsB = 20;
    int colsB = 10;
    
    auto A = createOnesMatrix(rowsA, colsA);
    auto B = createOnesMatrix(rowsB, colsB);
    
    MatrixMultiplier multiplier;
    bool success = multiplier.multiply(rank, size, A, B);
    
    if(rank == 0) {
        if (success) {
            multiplier.printSolution();
        } else {
            std::cout << "Что то не так!" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}