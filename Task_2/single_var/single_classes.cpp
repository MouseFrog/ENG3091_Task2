#include <iostream>
#include <vector>
#include <numeric>


// Create a class that holds the learnable parameters
// Allows for storage of different states as the model improves

#include <iostream>
#include <vector>

class LinearRegression {
public: // Classes are private by default, make public

    double w = 0.0;     // Weight (slope)
    double b = 0.0;     // Bias (y-intercept)
    double alpha;       // Learning rate

    // Store lr value into alpha, ramains in LinearRegression object throughout iteration
    LinearRegression(double lr) : alpha(lr) {}

    // Prediction function: y_hat = wx + b
    double predict(double x) {
        return w * x + b;
    }
};



class SimpleLinearRegression {

    // Learning parameters
    double slope = 0;
    double intercept = 0;

public:
    void train(const std::vector<double>& x, const std::vector<double>& y) {

        int n = x.size();   // Scaling factor for taking average 
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

        // Summation of all data points per variable manipulatoin
        for (int i = 0; i < n; i++) {
            double var_col = x[i][1]    // Extract variable column, ignore intercept column
            sumX += var_col;
            sumY += y[i];
            sumXY += var_col * y[i];
            sumX2 += var_col * var_col;
            
        }

        // Formula for Least Squares
        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX); // 
        intercept = (sumY - slope * sumX) / n;
    }

    // Substitute x-val of interest into prediction eqn for y-val 
    double predict(double x) {
        return intercept + slope * x;
    }
};