// Compile: g++ Binary.cpp -o binary.out
// Run: ./binary.out

// Runs the full Binary Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance
// Average Accuracy ~98%

#include <iostream>
#include <vector>
#include "../include/sklearn_lite/linear_models/LogisticRegression.hpp"
#include "../include/sklearn_lite/Utils.hpp"


int main() {

    // Set up containers for the data
    // X will store the input features, y will store the target values
    std::vector<std::vector<double>> features {}; 
    std::vector<double> labels {};

    // Load the dataset from file into X and y, skip header
    sklearn_lite::readCSV("../data/ecg.csv", features, labels, false);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear_models::LogisticRegression Binary(0.1, 10000, 0.1);

    // Train the model using the normalised data
    Binary.fit(norm_features, labels);

    // Use the trained model to make predictions
    std::vector<double> predictions {Binary.predict(norm_features)};

    sklearn_lite::saveCSV(labels, predictions, "Binary_Predictions.csv");
    sklearn_lite::check_accuracy(labels, predictions);

    return 0;
}