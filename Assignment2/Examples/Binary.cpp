// Compile:     g++ Binary.cpp -o binary.out
// Run:         ./binary.out

// Runs the full Binary Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance
// Average Accuracy ~98%

#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"


int main() {

    // Setup containers for separating dataset into features and true labels
    std::vector<std::vector<double>> features {}; 
    std::vector<double> labels {};

    // Load the dataset into containers, do not skip header
    sklearn_lite::readCSV("../data/ecg.csv", features, labels, false);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear_models::LogisticRegression Binary(0.1, 1000, 0.1);

    // Train the model using the normalised data
    Binary.fit(norm_features, labels);

    // Use the trained model to make predictions
    std::vector<double> predictions {Binary.predict(norm_features)};

    sklearn_lite::saveCSV(labels, predictions, "Binary_Predictions.csv");
    sklearn_lite::check_accuracy(labels, predictions); // check accurary of predictions with actual labels 

    return 0;
}