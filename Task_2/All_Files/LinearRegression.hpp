// Declares the Linear Regression model used in this project
// This file defines the class structure but not the function implementations
#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <vector>

// Class that stores the parameters of the regression line
// After training, w represents the price increase per bedroom
// and b represents the base price when bedrooms = 0
class SimpleLinearRegression {
public:

    double w = 0.0;   // slope of the line
    double b = 0.0;   // y-intercept

    void train(const std::vector<double>& bedroom_values,
               const std::vector<double>& price_values);

   
    // Gradient descent training
    void train_gradient(const std::vector<double>& bedroom_values,
                        const std::vector<double>& price_values,
                        double learning_rate,
                        int num_iterations);
                    
    // Predicts price 
    double predict(double bedrooms);
};

#endif
