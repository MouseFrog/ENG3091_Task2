#include "MultiRegression.hpp"

// This file consists Gradient Descent class 
GradientDescent::GradientDescent(int numFeatures, double lr ) : learningRate(lr) {    
        // Vector size: # of variables + Intercept 
        // Resized within constructor so it isn't altered accidentally, causing size mismatch 
        weights.resize(numFeatures + 1, 0.0); 
    }


void GradientDescent::train(const std::vector<std::vector<double>>& X, 
                            const std::vector<std::vector<double>>& Y, 
                            int iteration) {

    int m = X.size();    // Number of samples, row
    int n = X[0].size(); // # of weights including intercept
    hi;

    for (int e = 0; e < iteration; e++) {  // # of iteration
        std::vector<double> gradients(n, 0.0);  

        for (int i = 0; i < m; i++) {
            double y_prediction{0};

            for (int j = 0; j < n; j++) {
                y_prediction += weights[j] * X[i][j]; // addition of all weight*x-val in row i
            }
            // Error in predicted y
            double error = y_prediction - Y[i][0];  

            for (int j = 0; j < n; j++) {
                gradients[j] += error * X[i][j]; // Calculate errors proportional to each data point
            }
        }
        // Update weights using the average gradient
        // Take one step size away from calculated error
        for (int j = 0; j < n; j++) {
            weights[j] -= (learningRate * gradients[j]) * (2.0/m);    // 2.0 to force floating pt calculations
        }
    }
}

std::vector<double> GradientDescent::getWeights() const { 
    return weights; 
}