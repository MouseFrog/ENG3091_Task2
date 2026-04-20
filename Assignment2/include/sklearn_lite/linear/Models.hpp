// Linear and Logistic models inherit from base model Models
// Binary and Multi-class inherit from Logistic model --> Binary is reduced version of Multi-class
#ifndef SKLEARN_LITE_MODELS_HPP
#define SKLEARN_LITE_MODELS_HPP

#include <iostream>
#include <vector>   
#include <fstream>  // File operations


// Function to read csv file here

// Normalise input variables
// mean and standard deviation
struct NormResult {
    std::vector<std::vector<double>> matrix;
    std::vector<double> means;
    std::vector<double> std_devs;
};

// BASE MODEL for all models
namespace sklearn_lite::linear {
class Models {

    protected:
    
    double lr {}; // Update rule parameter required for all models, learning rate
    int num_features {};
    std::vector<std::vector<double>> weights {}; // Store calculated weights and biases

    public:
    Models(double lr, int num_features, const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y) 
    : lr{lr}, num_features{num_features}, X{X}, Y{Y} 
    {
        weights.resize(num_features+1, 0.0);
    }


    // Function to access weights and biases common across all models
    const std::vector<std::vector<double>>& get_weights() const{ //return by reference (large matrices), const to protect object
        return weights;
    }

    // Common Update Rule 
    for (int j = 0; j < n; j++) {
            weights[j] -= learningRate * gradients[j] * (1.0/m);    // 1.0 to force floating pt. calculations
        }


   
};
} 

// Linear Regression Model
namespace sklearn_lite::linear {
class LinearRegression : public Models {

    private:

    public:
    LinearRegression():{}

   
};
} 


// Logistic Regression class 
// Base for Binary and Multi-class
namespace sklearn_lite::linear {
class Logistic : public Models {

    private:
    double reg_strength{};

    public:
    BinaryLogistic(double reg_strength):reg_strength{reg_strength}{
    }
    // fit() function to train model


    // predict() function for a given variable set
    // Softmax function to wrap probability between 0 and 1
    // Binary version sigmoid function in class Binary

   
};
} 

// Binary Logistic Regression
namespace sklearn_lite::linear {
class BinaryLogistic : public Logistic {

    private:
    double reg_strength{};

    public:
    BinaryLogistic():{}{
    }
    // fit() function to train model
        // Sigmoid
        // Regularised loss function

    for (int i = 0; i < m; i++) {

            double y_prediction{0};

            for (int j = 0; j < n; j++) {
                y_prediction += weights[j] * X[i][j]; // Summation of weight * x-value for all variables 
            }

            // Discrepancy in y value prediction
            double error = y_prediction - Y[i][0];  

            for (int j = 0; j < n; j++) {
                gradients[j] += error * X[i][j]; // Calculate errors proportional to each data point
            }
        }

    // predict() function for a given variable set
    // Sigmoid function = Softmax when K=2, z0=0
        // Softmax function simplifies to sigmoid function at K=1: 
        // denominator e^-z term: z = z1-z0, relative difference in probability between y=0 and y=1
        // z0 is reference, assumed to be 0
        // If z0 > z1, sign flips --> e^z

    
   
};
} 

// Multi-class Logistic Regression
namespace sklearn_lite::linear {
class MultiLogistic : public Logistic {

    private:

    public:
    MultiLogistic():{}{
    }

    // fit() function to train model
        // Softmax function 
        // Regularised Loss function


    // predict() function for a given variable set
        // Apply softmax function to calculate class K probability
        // Select highest value probability


   
};
} 



#endif 