// Compile: g++ main.cpp -o main.out
// Run: ./main.out

// Runs the full linear regression pipeline on the concrete dataset
// Loads the data, normalises it, trains the model, and checks performance
// Expected R2 is around 0.61 (same as Python sklearn)

#include <iostream>
#include <vector>
#include "../include/sklearn_lite/linear_models/LinearRegression.hpp"
#include "../include/sklearn_lite/Utils.hpp"

int main() {

    std::vector<std::vector<double>> features {}; 
    std::vector<double> labels {};

    // Load the dataset from file into X and y, skip header
    sklearn_lite::readCSV("../data/concrete.csv", features, labels, true);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear::LinearRegression model(0.1, 10000);

    // Train the model using the normalised data
    model.fit(norm_features, labels);

    // Use the trained model to make predictions
    std::vector<double> predictions {model.predict(norm_features)};

    // Check how good the predictions are
    double r2 {model.r2_score(labels, predictions)};

    // Print the result and compare to expected value
    std::cout << "R2 Score: " << r2 << "\n";
    std::cout << "Python sklearn benchmark: ~0.61\n";

    return 0;
}