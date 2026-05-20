// Linear Regression model using gradient descent
// Inherits shared variables and functions from Models base class 
// Adapted from Assignment 1 GradientDescent class 


#ifndef SKLEARN_LITE_LINEARREGRESSION_HPP
#define SKLEARN_LITE_LINEARREGRESSION_HPP

#include <iostream>
#include <vector>
#include <cmath>      // std::sqrt, std::pow
#include "BaseModel.hpp"

namespace sklearn_lite::linear {

class LinearRegression : public BaseModel {

    private: 
    std::vector<double> weights{}; // one per festure + first column for bias
    double lr{};                   // Learning rate/ step size
    int iterations{};              // Maximum number of training cycles

    public:

        // Constructor passes the learning rate and iteration count to the base class Models
        // lr controls step size of each gradient descent
        // iterations sets maximum number of training cycles
        LinearRegression(double lr, int iterations)
            : lr{lr}, iterations{iterations} {}

        // Trains the model using gradient descent
        // X is the normalised feature matrix with intercept column
        // y contains the actual concrete strength values
        // The goal is to adjust the weights so the prediction error gets smaller each iteration
        void fit(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y) override
        {
            int num_samples = X.size();    // rows
            int num_features = X[0].size(); // Number of weights including intercept, cols

            // Resize weights to correspond with number of features+1, weights[0] helps calculate intercept 
            weights.resize(num_features, 0.0);

            // Threshold value beyond which change is negligible, stop early
            double converge_threshold = 1e-12;

            // Vector to save weights from previous iteration to track how much it changed
            std::vector<double> old_weights(num_features, 0.0);

            // Repeat the gradient descent update until we either converge or hit the maximum number of iterations
            for (int e{0}; e < iterations; e++) {

                // Save previous iteration's weights into old_weights before updating weights vector 
                old_weights = weights;

                // Store gradient (rate of change necessary) for each weight during this iteration
                // Reset to 0 at the start of every pass through the data
                std::vector<double> gradients(num_features, 0.0);

                // Loop through each training sample
                for (int i{0}; i < num_samples; i++) {

                    // Calculate the predicted strength for this sample
                    // Sum of weight * feature value across all features
                    double y_prediction {0};
                    for (int j{0}; j < num_features; j++) {
                        y_prediction += weights[j] * X[i][j]; // weight * feature value
                    }

                    // Work out how far the prediction is from the actual value
                    double error{y_prediction - y[i]};

                    // Add this sample's contribution to each gradient
                    // Gradient for each weight = error * its corresponding feature value
                    for (int j{0}; j < num_features; j++) {
                        gradients[j] += error * X[i][j];
                    }
                }

                // Update each weight using the average gradient across all samples
                for (int j{0}; j < num_features; j++) {
                    weights[j] -= lr * gradients[j] * (1.0 / num_samples);
                }

                // Measure how much the weights changed overall in this iteration
                double overall_change{0};
                for (int j{0}; j < num_features; j++) {
                    double weight_change {old_weights[j] - weights[j]};
                    overall_change += weight_change * weight_change;  // Sum the squared change for each weight
                }
                overall_change = std::sqrt(overall_change); // Magnitude of total change, taking sqrt

                // If weights have stabilised, stop training early
                if (overall_change < converge_threshold) {
                    std::cout << "Converged at iteration: " << e << "\n";
                    break;
                }
            }

            std::cout << "Training complete\n";
        }

        // Uses the learned weights to make predictions for new input data
        std::vector<double> predict(const std::vector<std::vector<double>>& X) const override{

            // If fit() has not been called yet, there are no learned weights to use
            if (weights.empty()) {
                std::cerr << "Model not trained. Call fit() first\n";
                return {};
            }

            std::vector<double> predictions;

            // Calculate prediction per sample (row)
            for (int i{0}; i < (int)X.size(); i++) {

                // Same dot product as in fit(): sum of weight * feature for all features
                double y_hat{0};
                for (int j{0}; j < (int)weights.size(); j++) {
                    y_hat += weights[j] * X[i][j]; 
                }

                predictions.push_back(y_hat);
            }

            return predictions;
        }

        // Checks how well the model predicts compared to just guessing the mean
        // R2 close to 1 means the model is doing well
        // R2 close to 0 means it is not really improving on a basic average guess
        double r2_score(const std::vector<double>& y_true,
                        const std::vector<double>& y_pred) const {

            int num_samples {static_cast<int>(y_true.size())};

            // Find the mean of the actual concrete strength values
            double mean_y{0};
            for (int i{0}; i < num_samples; i++) {
                mean_y += y_true[i];
            }
            mean_y /= num_samples; // Divide by number of samples to get mean

            // SS_res: total squared error made by our model
            double ss_res{0.0};
            for (int i{0}; i < num_samples; i++) {
                ss_res += std::pow(y_true[i] - y_pred[i], 2);
            }

            // SS_tot: total squared error if we always guessed the mean strength
            // This represents the dumbest possible model as a baseline
            double ss_tot{0};
            for (int i{0}; i < num_samples; i++) {
                ss_tot += std::pow(y_true[i] - mean_y, 2); 
            }

            // Compare the model against the mean baseline
            return 1.0 - (ss_res / ss_tot);
        }

}; // end LinearRegression

} // namespace sklearn_lite::linear

#endif