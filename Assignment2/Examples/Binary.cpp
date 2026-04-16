// This file contains the implementation of the Gradient Descent method
/*
Experiment with different learning rates 𝛼 and regularization strengths 𝜆 to see how they
affect the training process and the final performance of the model.
*/

#include "MultiRegression.hpp"

GradientDescent::GradientDescent(int numFeatures, double lr ) : learningRate(lr) {    
        // Vector size: # of variables + Intercept 
        // Resized within constructor so it isn't altered accidentally, causing size mismatch 
        weights.resize(numFeatures + 1, 0.0); 
    }

void GradientDescent::train(const std::vector<std::vector<double>>& X, 
                            const std::vector<std::vector<double>>& Y, 
                            int iteration) {

    int m = X.size();    // Number of samples, row
    int n = X[0].size(); // # of weights including intercept, col

    for (int e = 0; e < iteration; e++) {  // # of iterations

        // Value beyond which change is negligible
        double converge_threshold = 1*10e-12;

        // Gradient vector to store error in each data point
        // Reset every iteration
        std::vector<double> gradients(n, 0.0);  

        // Store current weights to compare change in weights for this iteration
        std::vector<double> old_weights = weights;

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

        // Update weights using average gradient
        // Takes one step size away from calculated error
        for (int j = 0; j < n; j++) {
            weights[j] -= learningRate * gradients[j] * (1.0/m);    // 1.0 to force floating pt. calculations
        }

        // Track magnitude of change between old and new weights
        double overall_change = 0.0;
        for (int j = 0; j < n; j++){
            double weight_change = old_weights[j]-weights[j];
            overall_change +=  weight_change * weight_change; // Total sum of square error
        }
        
        overall_change = std::sqrt(overall_change);     // Final value used as measure of change
        if (overall_change < converge_threshold){   // Value stabilised
            std::cout<<"Iteration complete: "<< e <<"\n";
            break;
        }
    }
}

std::vector<double> GradientDescent::getWeights() const { 
    return weights; 
}