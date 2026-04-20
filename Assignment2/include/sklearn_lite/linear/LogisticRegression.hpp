#ifndef SKLEARN_LITE_LOSGISTICREGRESSION_HPP
#define SKLEARN_LITE_LOSGISTICREGRESSION_HPP

#include <iostream>
#include <vector>   // for std::vector
#include "Models.hpp"


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