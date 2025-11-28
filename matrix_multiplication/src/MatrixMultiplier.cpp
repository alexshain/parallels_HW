#include "MatrixMultiplier.h"
#include <iostream>
#include <cmath>
#include <mpi.h>

MatrixMultiplier::MatrixMultiplier() : solution_(0, 0) {}

bool MatrixMultiplier::multiply(int rank, int size, const Matrix& _A, const Matrix& _B) {
    double start_time = MPI_Wtime();

    if(_A.columns != _B.rows) {
        std::cout << "Число столбцов матрицы A не совпадает с числом строк матрицы B" << std::endl;
        return false;
    }
    
    int dims[2] = {0, 0}, periods[2] = {0, 0};
    int coords[2], reorder = 1;
    int p1, p2, rank_y, rank_x;
    int prev_y, prev_x, next_y, next_x;

    MPI_Comm comm2d; // дескриптор коммуникатора
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // определение размеров решетки: dims
    MPI_Dims_create(size, 2, dims); // вычисляет размер двумерной решетки процессов – заполняет равные нулю элементы массива
                                    // dims таким образом, чтобы число процессов в решетке было равно size.

    p1 = dims[0]; p2 = dims[1];

    if (_A.rows % p1 != 0 || _B.columns % p2 != 0) {
        if (rank == 0) {
            std::cout << "Размеры матриц не делятся нацело на размеры решетки процессов" << std::endl;
        }
        return false;
    }

    int a_rows = _A.rows / p1;
    int a_cols = _A.columns;
    int b_rows = _B.rows;
    int b_cols = _B.columns / p2;
    int c_rows = a_rows;
    int c_cols = b_cols;

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

    MPI_Comm comm_row, comm_col;
    int slice_dims_row[2] = {0, 1};
    int slice_dims_col[2] = {1, 0};
    MPI_Cart_sub(comm2d, slice_dims_row, &comm_row);
    MPI_Cart_sub(comm2d, slice_dims_col, &comm_col);

    if (rank_x == 0) {
        MPI_Scatter(_A.data.data(), a_rows * a_cols, MPI_DOUBLE,
                   A_block.data(), a_rows * a_cols, MPI_DOUBLE,
                   0, comm_col);
    }

    MPI_Bcast(A_block.data(), a_rows * a_cols, MPI_DOUBLE, 0, comm_row);

    MPI_Datatype column_type, resized_column_type;
    MPI_Type_vector(b_rows, b_cols, _B.columns, MPI_DOUBLE, &column_type);
    MPI_Type_create_resized(column_type, 0, sizeof(double), &resized_column_type);
    MPI_Type_commit(&resized_column_type);

    if (rank_y == 0) {
        MPI_Scatter(_B.data.data(), 1, resized_column_type,
                   B_block.data(), b_rows * b_cols, MPI_DOUBLE,
                   0, comm_row);
    }
    MPI_Bcast(B_block.data(), b_rows * b_cols, MPI_DOUBLE, 0, comm_col);

        //for(int i = 0; i < A_block.size(); i++)
          //  std::cout << B_block[i] << std::endl;

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            for (int k = 0; k < a_cols; k++) {
                C_block[i * c_cols + j] += A_block[i * a_cols + k] * B_block[k * b_cols + j];
            }
        }
    }


    if (rank == 0) {
        solution_ = Matrix(_A.rows, _B.columns, 0.0);
    }

    MPI_Datatype block_type, resized_block_type;
    MPI_Type_vector(c_rows, c_cols, _B.columns, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    std::vector<int> recvcounts(size, 1);
    std::vector<int> displs(size, 0);
    
    for (int i = 0; i < p1; i++) {
        for (int j = 0; j < p2; j++) {
            int process_rank;
            int coords_proc[2] = {i, j};
            MPI_Cart_rank(comm2d, coords_proc, &process_rank);
            displs[process_rank] = i * c_rows * _B.columns + j * c_cols;
            //std::cout << i * c_rows * _B.columns + j * c_cols << std::endl;
        }
    }

    MPI_Gatherv(C_block.data(), c_cols * c_rows, MPI_DOUBLE,
               solution_.data.data(), recvcounts.data(), displs.data(), resized_block_type,
               0, comm2d);
    
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Время выполнения: " << end_time - start_time << " секунд" << std::endl;
    }
    
    return true;
}

const Matrix& MatrixMultiplier::getSolution() const {
    return solution_;
}

void MatrixMultiplier::printSolution() const {
    std::cout << "Результат умножения матриц:" << std::endl;
    for (int i = 0; i < solution_.rows; i++) {
        for (int j = 0; j < solution_.columns; j++) {
            std::cout << solution_.data[i * solution_.columns + j] << " ";
        }
        std::cout << std::endl;
    }
}
