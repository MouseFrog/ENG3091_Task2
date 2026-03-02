#include <iostream>
#include <vector>


std::vector<std::vector<double>> transformMatrix(const std::vector<std::vector<double>>& matrix_in){
    int rows = matrix_in.size;   // Number of inner vectors
    int cols = rows > 0? matrix[0].size : 0;    // Size of inner vectors
    std::vector<std::vector<double> results(cols, std::vector<double>(rows,0));

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            transposed[j][i] = matrix_in[i][j];
        }
    }

    return transposed;
}


int main(){


    return 0;
}