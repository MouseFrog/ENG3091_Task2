#include <iostream>
#include <vector>
#include <numeric>

//Simple Linear Regression

int main() {
    // Data: {Square Footage} -> {Price in $1000s}
    std::vector<double> x = {1500, 2000, 1200, 2500, 1800};
    std::vector<double> y = {300, 400, 250, 500, 360};

    SimpleLinearRegression slr;
    slr.train(x, y);

    double newHouse = 2200;
    std::cout << "Predicted price for 2200 sqft: $" << slr.predict(newHouse) << "k" << std::endl;
    
    return 0;
}

//Multiple Linear Regression
class MultipleLinearRegression {
    std::vector<double> weights; // Vector holding weights and intercept
    double learningRate;    // Step-size

public:
    MultipleLinearRegression(int numFeatures, double lr = 0.01) : learningRate(lr) {    
        // Vector size: # of variables + Intercept 
        // Resized within constructor so it isn't altered accidentally, causing size mismatch 
        weights.resize(numFeatures + 1, 0.0); 
    }

    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, int iteration) {
        int m = X.size();    // Number of samples, row
        int n = X[0].size(); // # of weights including intercept

        for (int e = 0; e < iteration; e++) {  // # of iteration
            std::vector<double> gradients(n, 0.0);  

            for (int i = 0; i < m; i++) {
                double y_prediction{0};

                for (int j = 0; j < n; j++) {
                    y_prediction += weights[j] * X[i][j]; // addition of all weight*x-val in row i
                }
                // Error in predicted y
                double error = y_prediction - Y[i];  

                for (int j = 0; j < n; j++) {
                    gradients[j] += error * X[i][j]; // Calculate errors proportional to each data point
                }
            }
            // Update weights using the average gradient
            for (int j = 0; j < n; j++) {
                weights[j] -= (learningRate * gradients[j]) / m;    // Take one step size away from calculated error
            }
        }
    }
    std::vector<double> getWeights() const { 
        return weights; 
    }
};