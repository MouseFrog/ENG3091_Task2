// Compile: Linear.cpp -o linear.out
// Run: ./linear.out

// Runs the full linear regression method on provided concrete dataset
// Loads and normalises data, trains model, and checks performance
// Expected R2 is around 0.61 (same as Python sklearn)

#include <iostream>
#include <vector>
#include "../include/sklearn_lite/linear_models/LinearRegression.hpp"
#include "../include/sklearn_lite/Utils.hpp"

int main() {

    // Set up containers for the data
    // X will store the input features, y will store the target values
    std::vector<std::vector<double>> X; 
    std::vector<double> y;

    // Load the dataset from file into X and y, skip header
    sklearn_lite::readCSV("../data/concrete.csv", X, y, true);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm = sklearn_lite::normaliseData(X);

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear::LinearRegression model(0.1, 10000);

    // Train the model using the normalised data
    model.fit(norm, y);

    // Use the trained model to make predictions
    std::vector<double> predictions = model.predict(norm);

    // Check how good the predictions are
    double r2 = model.r2_score(y, predictions);
    sklearn_lite::saveCSV(y, predictions, "Linear_Predictions.csv");

    // Print the result and compare to expected value
    std::cout << "R2 Score: " << r2 << "\n";
    std::cout << "Python sklearn benchmark: ~0.61\n";

    return 0;
}