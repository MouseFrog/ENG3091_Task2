// Compile: Multiclass.cpp -o multiclass.out
// Run: ./multiclass.out

// Runs the full Multi-class Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance
// Average Accuracy ~94%

#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"


int main() {

    // Set up containers for the data
    // X will store the input features, y will store the target values
    std::vector<std::vector<double>> X; 
    std::vector<double> y;

    // Load the dataset from file into X and y, skip header
    sklearn_lite::readCSV("../data/mnist_micro.csv", X, y, true);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm = sklearn_lite::normaliseData(X);

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear::LogisticRegression Multiclass(0.1, 200, 0.01);

    // Train the model using the normalised data
    Multiclass.fit(norm, y);

    // Use the trained model to make predictions
    std::vector<double> predictions = Multiclass.predict(norm);

    sklearn_lite::saveCSV(y, predictions, "Multi_Predictions.csv");
    sklearn_lite::check_accuracy(y, predictions);

    return 0;
}