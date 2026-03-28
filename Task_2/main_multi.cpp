// Multi-Variable Regression 
// g++ main_multi.cpp DataGenerator.cpp GradientDescent.cpp Utilities.cpp -o"multi.exe"
// Copy and paste above command into terminal to compile

#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation
#include "MultiRegression.hpp"

int main() {

    std::random_device device_num;
    std::mt19937 global_num(device_num());

    // Create Dataset instance and define parameters, otherwise uses default
    Dataset custom;
    custom.w_bedroom = 20000.0;

    // Generate Matrix with custom settings
    DataGenerator gen(custom,global_num); 
    int data_points{100};
    int num_features{2};

    Matrices multiMatrix = gen.make_matrix(data_points,num_features);  // 100x3 matrix

    // Normalise data
    NormResult norm_X =  normaliseData(multiMatrix.X);

    // Train model
    GradientDescent grad_model(num_features,0.1);  // Learning rate = 0.1
    int num_iterations{100000}; 
    grad_model.train(norm_X.matrix,multiMatrix.Y,num_iterations);

    // Normalised weights
    // Weights[0] = Intercept
    // Weights[1] = Bedroom
    // Weights[2] = Area    
    std::vector<double> model_weights = grad_model.getWeights();   


    // De-normalise results
    // real weight = normalised weight/standard deviation
    double real_bedroom = model_weights[1]/norm_X.std_devs[1];  
    double real_area = model_weights[2]/norm_X.std_devs[2];

    // real intercept = normalised intercept - sum of (variable real weights * variable mean values)
    double real_intercept = model_weights[0]-(real_bedroom * norm_X.means[1])-(real_area * norm_X.means[2]);

    std::cout <<
    "Multi Intercept: "<<real_intercept<<"\n"<<
    "Multi Bedroom Weight: "<<real_bedroom<<"\n"<<
    "Multi Area Weight: "<<real_area<<"\n";

    /* 
    // Debugging
    //Normalised weight
    std::cout<<
    "Model Intercept: "<<model_weights[0]<<"\n"<<
    "Model Bedroom Weight: "<<model_weights[1]<<"\n"<<
    "Model Area Weight: "<<model_weights[2]<<"\n";

    // Standard Deviations Calculated
    std::cout<<
    "Intercept std_dev: "<<norm_X.std_devs[0]<<"\n"<<
    "Bedroom std_dev: "<<norm_X.std_devs[1]<<"\n"<<
    "Area std_dev: "<<norm_X.std_devs[2]<<"\n";
    */

    saveFile(multiMatrix.X,multiMatrix.Y,"MultiVariableData");

    return 0;
}    

