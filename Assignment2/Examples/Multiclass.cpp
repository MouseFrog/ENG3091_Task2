// Compile:     g++ Multiclass.cpp -o multiclass.out
// Run:         ./multiclass.out

// Runs the full Multi-class Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance

#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"


int main() {

    // Setup containers for separating dataset into features and true labels
    std::vector<std::vector<double>> features {}; // stores feature + bias in first column
    std::vector<double> labels {};

    // Load TRAINING dataset into container, do not skip header:
    sklearn_lite::readCSV("../data/mnist_micro.csv", features, labels, false);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear_models::LogisticRegression Multiclass(0.1, 1000, 0.01); 
    
    // Train the model using the normalised data
    Multiclass.fit(norm_features, labels);

    // Prep and normalise TESTING dataset
    std::vector<std::vector<double>> test_features {}; 
    std::vector<double> test_labels {};
    sklearn_lite::readCSV("../data/mnist_mini.csv", test_features, test_labels, false);
    std::vector<std::vector<double>> norm_test_features {sklearn_lite::normaliseData(features)};

    // Use the trained model to make predictions
    std::vector<double> predictions {Multiclass.predict(norm_test_features)};

    sklearn_lite::saveCSV(test_labels, predictions, "Multi_Predictions.csv");
    sklearn_lite::check_accuracy(test_labels, predictions); // check accurary of predictions with actual labels 

    return 0;
}

//git add -- . ':!*.csv' (upload without large csv files)
