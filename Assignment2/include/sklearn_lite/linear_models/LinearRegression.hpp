// Linear Regression model using gradient descent
// Inherits shared variables and functions from Models base class 
// Adapted from Assignment 1 GradientDescent class 


#ifndef SKLEARN_LITE_LINEARREGRESSION_HPP
#define SKLEARN_LITE_LINEARREGRESSION_HPP

#include <iostream>
#include <vector>
#include <cmath>      // std::sqrt, std::pow
#include "BaseModel.hpp"

namespace sklearn_lite::linear_models {

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
            int num_samples {static_cast<int>(X.size())};    // rows
            int num_features {static_cast<int>(X[0].size())}; // Number of weights including intercept, cols

            // Resize weights to correspond with number of features+1, weights[0] holds intercept values
            weights.resize(num_features, 0.0);

            double converge_threshold {1e-12}; // Threshold value beyond which change is negligible, stop early

            std::vector<double> old_weights(num_features, 0.0); // Help track change in weights

            for (int e{0}; e < iterations; e++) { // Repeat until convergence or max iteration value

                old_weights = weights; // Save weights from previous iteration

                // Store gradient (rate of change necessary) 
                std::vector<double> gradients(num_features, 0.0); // Resets to 0 at every pass

                for (int i{0}; i < num_samples; i++) { // Loop through each training sample

                    // 1. Calculate the predicted strength for this sample
                    double y_prediction {0.0}; 
                    for (int j{0}; j < num_features; j++) {
                        y_prediction += weights[j] * X[i][j]; // weight * feature value
                    }
                    double error{y_prediction - y[i]}; // Error in prediction vs actual value

                    // 2. Calculate rate of change per sample necessary based on error
                    for (int j{0}; j < num_features; j++) {
                        gradients[j] += error * X[i][j]; // Error atttributed to each feature
                    }
                }

                // 3. Update rule
                for (int j{0}; j < num_features; j++) {
                    weights[j] -= lr * gradients[j] * (1.0 / num_samples);
                }

                // 4. Track net change in weights in this iteration
                double overall_change{0.0};
                for (int j{0}; j < num_features; j++) {
                    double weight_change {old_weights[j] - weights[j]};
                    overall_change += weight_change * weight_change;  // Sum the squared change for each weight
                }
                overall_change = std::sqrt(overall_change); // Magnitude of total change, taking sqrt
                if (overall_change < converge_threshold) { // If change in weights has stabilised, stop training early
                    std::cout << "Converged at iteration: " << e << "\n"; // save energy, save time
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

            int num_samples {static_cast<int>(X.size())};
            int num_classes {static_cast<int>(weights.size())};
            std::vector<double> predictions {}; // Final class predictions

            // Calculate prediction per sample (row)
            for (int i{0}; i < num_samples ; i++) {

                // Same dot product as in fit(): sum of weight * feature for all features
                double y_hat{0.0};
                for (int j{0}; j < num_classes; j++) {
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