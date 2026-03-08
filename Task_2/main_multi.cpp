#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation
#include "MultiRegression.hpp"

int main() {

    // Create Dataset instance and define parameters, otherwise uses default
    Dataset custom;
    custom.bed_max = 10;
    custom.area_min = 15;
    custom.area_max = 1400;

    DataGenerator gen(custom); // Create instance with custom settings
    Matrices myMatrix = gen.make_matrix(100,2) ;

    const std::vector<std::vector<double>>& variableMatrix = myMatrix.X;
    const std::vector<double>& priceMatrix = myMatrix.Y;


    saveFile(variableMatrix,"og file" );
    
    return 0;
}    

