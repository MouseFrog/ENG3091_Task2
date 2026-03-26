// Single-Variable Regression: Normal Gradient method and Gradient Descent method
// g++ main_single.cpp DataGenerator.cpp GradientDescent.cpp Normal.cpp Utilities.cpp -o"single.exe"
// Copy and paste above command into terminal to compile

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include "MultiRegression.hpp"

int main() {

    // Instantiate object gen using instance of Dataset custom with default values
    Dataset custom;
    DataGenerator gen(custom);

    // Generate variables matrix
    int data_points{100};   // Sample size 100
    int num_features{1};    // 1 independent variable
    Matrices singleMatrix = gen.make_matrix(data_points,num_features);

    // NORMAL GRADIENT MODEL
    Normal norm_model;  
    norm_model.train(singleMatrix.X, singleMatrix.Y);

    std::pair<double,double> norm_weights = norm_model.getWeights();

    std::cout<<"Normal Gradient inctercept: "
             <<norm_weights.first<< "\n"
             <<"Normal Gradient gradient: "
             <<norm_weights.second<<std::endl;
    double interpolate_x{5.0};
    double predict_price = norm_model.predict(interpolate_x);
    std::cout<<"Predicted price of " <<interpolate_x<< " bedrooms based on Normal method:"<<"$"<<predict_price<<std::endl;



    // GRADIENT DESCENT MODEL
    // Normalise data to prep for calculations
    NormResult norm_data = normaliseData(singleMatrix.X);

    GradientDescent grad_model(num_features,0.1);  // Learning rate = 0.1
    int num_iterations{100000}; 
    grad_model.train(norm_data.matrix,singleMatrix.Y,num_iterations);

    // Normalised weights
    std::vector<double> grad_weights = grad_model.getWeights();   

    // Recover de-normalised results
    // real weight = normalised weight/standard deviation
    double real_gradient = grad_weights[1]/norm_data.std_devs[1];  
    // real intercept = normalised intercept - sum of (variable real weights * variable mean values)
    double real_intercept = grad_weights[0]-(real_gradient * norm_data.means[1]);


    std::cout <<
    "Gradient Descent Intercept: "<<real_intercept<<"\n"<<
    "Gradient Descent Gradient: "<<real_gradient<<"\n";

    std::cout<<
    "Intercept std_dev: "<<norm_data.std_devs[0]<<"\n"<<
    "Bedroom std_dev: "<<norm_data.std_devs[1]<<"\n";

    saveFile(singleMatrix.X,singleMatrix.Y,"SingleVariableData" );

    return 0;
}
