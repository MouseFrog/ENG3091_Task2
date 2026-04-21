// Compile: g++ -std=c++17 -I ../include main.cpp readCSV.cpp -o main.out
// Run: ./main.out

// Runs the full linear regression pipeline on the concrete dataset
// Loads the data, normalises it, trains the model, and checks performance
// Expected R2 is around 0.61 (same as Python sklearn)

#include <iostream>
#include <vector>
#include "../include/sklearn_lite/linear/LinearRegression.hpp"

// Forward declaration of readCSV - defined in readCSV.cpp
void readCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<double>& y,
             bool has_header = true);


int main() {

    // Set up containers for the data
    // X will store the input features, y will store the target values
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // Load the dataset from file into X and y
    // concrete.csv has a header so we skip the first row
    readCSV("../data/concrete.csv", X, y, true);

    // Normalise the feature values before training
    // This keeps all columns on a similar scale so gradient descent behaves properly
    // Also adds the intercept column at the front
    NormResult norm = normaliseData(X);

    // Create the model with chosen settings
    // lr controls step size, iterations controls how long we train for
    sklearn_lite::linear::LinearRegression model(0.1, 10000);

    // Train the model using the normalised data
    model.fit(norm.matrix, y);

    // Use the trained model to make predictions
    std::vector<double> predictions = model.predict(norm.matrix);

    // Check how good the predictions are
    double r2 = model.r2_score(y, predictions);

    // Print the result and compare to expected value
    std::cout << "R2 Score: " << r2 << "\n";
    std::cout << "Python sklearn benchmark: ~0.61\n";

    return 0;
}