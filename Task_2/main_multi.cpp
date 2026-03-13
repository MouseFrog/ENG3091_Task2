#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation
#include "MultiRegression.hpp"

int main() {

    // Create Dataset instance and define parameters, otherwise uses default
    Dataset custom;
    custom.w_bedroom = 20000.0;

    // Generate Matrix with custom settings
    DataGenerator gen(custom); 
    int num_features{2};
    Matrices myMatrix = gen.make_matrix(100,num_features);  // 100x3 matrix

    // Normalise data
    NormResult norm_X =  normaliseData(myMatrix.X);

    // Train model
    GradientDescent mlr(num_features,0.1);  // Learning rate = 0.1
    int num_iterations{100000}; 
    mlr.train(norm_X.matrix,myMatrix.Y,num_iterations);
    std::vector<double> model_weights = mlr.getWeights();   // Weights (normalised)

    // Recover de-normalised weights 
    // real weight = normalised weight/standard deviation
    double real_bedroom = model_weights[1]/norm_X.std_devs[1];  
    double real_area = model_weights[2]/norm_X.std_devs[2];
    // real intercept = normalised intercept - sum of (variable real weights * variable mean values)
    double real_intercept = model_weights[0]-(real_bedroom * norm_X.means[1])-(real_area * norm_X.means[2]);

    std::cout<<
    "Model Intercept: "<<model_weights[0]<<"\n"<<
    "Model Bedroom Weight: "<<model_weights[1]<<"\n"<<
    "Model Area Weight: "<<model_weights[2]<<"\n";

    std::cout <<
    "Real Intercept: "<<real_intercept<<"\n"<<
    "Real Bedroom Weight: "<<real_bedroom<<"\n"<<
    "Real Area Weight: "<<real_area<<"\n";

    std::cout<<
    "Intercept std_dev: "<<norm_X.std_devs[0]<<"\n"<<
    "Bedroom std_dev: "<<norm_X.std_devs[1]<<"\n"<<
    "Area std_dev: "<<norm_X.std_devs[2]<<"\n";

    saveFile(myMatrix.X,"variables" );
    saveFile(myMatrix.Y,"prices");
    
    return 0;
}    

