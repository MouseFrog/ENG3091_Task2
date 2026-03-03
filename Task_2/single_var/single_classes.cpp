#include <iostream>
#include <vector>
#include <numeric>


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
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            
        }

        // Formula for Least Squares
        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX); // 
        intercept = (sumY - slope * sumX) / n;
    }

    // Function to return line 
    double predict(double x) {
        return intercept + slope * x;
    }
};