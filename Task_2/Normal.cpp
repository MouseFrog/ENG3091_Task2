// This file contains the implementation of the Normal Equation 

#include "MultiRegression.hpp"
#include <iostream>

// Train the model using the dataset of bedrooms and prices
// The goal is to compute the line that best fits the data
void Normal::train(const std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<double>>& Y)
{
    int num_points = X.size();

    // Sums needed to calculate final weights
    // Variables updated every loop
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x_squared = 0.0;

    // Loop through the dataset and update running sums for each value
    for (int i = 0; i < num_points; i++) {

        double X_val = X[i][1];
        double Y_val = Y[i][0];

        sum_x += X_val;
        sum_y += Y_val;

        // Tracks how bedrooms and prices vary together
        sum_xy += X_val * Y_val;

        // Tracks how spread out the bedroom values are
        sum_x_squared += X_val * X_val;
    }

    // Slope calculation
    w = (num_points * sum_xy - sum_x * sum_y)
        /
        (num_points * sum_x_squared - sum_x * sum_x);

    // Intercept calculation
    b = (sum_y - w * sum_x) / num_points;
}

std::pair<double,double> Normal::getWeights() const{
    return {b,w};
}

// Predicts price for given x value from the model
double Normal::predict(double bedrooms) {
    return w * bedrooms + b;
}
