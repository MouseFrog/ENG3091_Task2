// Compile: g++ Multiclass.cpp -o multiclass.out
// Run: ./multiclass.out

// Runs the full Multi-class Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance
// Average Accuracy ~94%

#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"


int main() {

    // Set up containers for the data
    std::vector<std::vector<double>> features {}; // stores feature + bias in first column
    std::vector<double> labels {};

    // Load the dataset from file into X and y, skip header
    sklearn_lite::readCSV("../data/mnist_micro.csv", features, labels, false);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear::LogisticRegression Multiclass(0.1, 200, 0.01);

    // Train the model using the normalised data
    Multiclass.fit(norm_features, labels);

    // Use the trained model to make predictions
    std::vector<double> predictions {Multiclass.predict(norm_features)};

    sklearn_lite::saveCSV(labels, predictions, "Multi_Predictions.csv");
    sklearn_lite::check_accuracy(labels, predictions);

    return 0;
}