#include "MatrixMultiplier.h"
#include <iostream>
#include <cmath>
#include <mpi.h>

bool MatrixMultiplier::multiply(int rank, int size, const Matrix& _A, const Matrix& _B) {
    double start_time = MPI_Wtime();

    if(_A.columns != _B.rows) {
        return false;
    }
    
    int dims[2] = {0, 0}, periods[2] = {1, 1};
    int coords[2], reorder = 1;
    int size, rank, p1, p2, rank_y, rank_x;
    int prev_y, prev_x, next_y, next_x;

    MPI_Comm comm2d; // дескриптор коммуникатора
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // определение размеров решетки: dims
    MPI_Dims_create(size, 2, dims); // вычисляет размер двумерной решетки процессов – заполняет равные нулю элементы массива
                                    // dims таким образом, чтобы число процессов в решетке было равно size.

    p1 = dims[0]; p2 = dims[1];

    int a_rows = _A.rows / p1;
    int a_cols = _A.columns / p;
    int b_rows = _B.rows / p;
    int b_cols = _B.columns / p;

    std::vector<double> A_block(a_rows * a_cols);
    std::vector<double> B_block(b_rows * b_cols);
    std::vector<double> C_block(a_rows * b_cols, 0.0);

    // создание коммуникатора: comm2d
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d); // создает на основе коммуникатора MPI_COMM_WORLD 
                                                                         //новый коммуникатор comm2d с двумерной декартовой 
                                                                         //топологией и размерами, заданными в массиве dims

    // получение своего номера в коммуникаторе comm2d: rank
    MPI_Comm_rank(comm2d, &rank);

    // получение своих координат в 2D решетке: coords
    MPI_Cart_get(comm2d, 2, dims, periods, coords); // позволяет текущему процессу получить параметры
                                                    //коммуникатора comm2d, включая свои координаты в ней
    rank_y = coords[0]; rank_x = coords[1];

    // определение соседей: prev_y, next_y, prev_x, next_x
    //позволяет получить номера соседних процессов, отстоящих в коммуникаторе comm2d от текущего на заданном 
    //расстоянии (1) по заданной оси(0 или 1).
    MPI_Cart_shift(comm2d, 0, 1, &prev_y, &next_y); //по оси 0
    MPI_Cart_shift(comm2d, 1, 1, &prev_x, &next_x); //по оси 1

    double end_time = MPI_Wtime();
    
    return true;
}

const std::vector<double>& MatrixMultiplier::getSolution() const {
    return solution_;
}

//fixme
void MatrixMultiplier::printSolution() const {
}
