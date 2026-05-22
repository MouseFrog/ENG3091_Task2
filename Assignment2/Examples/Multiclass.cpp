// Compile:     g++ Multiclass.cpp -o multiclass.out
// Run:         ./multiclass.out

// Runs the full Multi-class Logistic Regression method on Handwritten Numbers dataset
// Loads and normalises data, trains model, and checks performance
// Accuracy ~94% at max.iteration 200
// Accuracy ~95% at max.iteratoion 300, but takes longer so value is set at 200


#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"


int main() {

    // Setup containers for separating dataset into features and true labels
    std::vector<std::vector<double>> features {}; // stores feature + bias in first column
    std::vector<double> labels {};

    // LOAD MINI DATASET into container, do not skip header:
    sklearn_lite::readCSV("../data/mnist_mini.csv", features, labels, false);
    // LOAD MICRO DATASET into container, do not skip header:
    // sklearn_lite::readCSV("../data/mnist_micro.csv", features, labels, false);
    // LOAD FULL DATASET into container, do not skip header: Run at your own risk: 
    //sklearn_lite::readCSV("../data/mnist_full.csv", features, labels, false);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear_models::LogisticRegression Multiclass(0.1, 1000, 0.01); 
    
    // Train the model using the normalised data
    Multiclass.fit(norm_features, labels);

    // Use the trained model to make predictions
    std::vector<double> predictions {Multiclass.predict(norm_features)};

    sklearn_lite::saveCSV(labels, predictions, "Multi_Predictions.csv");
    sklearn_lite::check_accuracy(labels, predictions); // check accurary of predictions with actual labels 

    return 0;
}

//git add -- . ':!*.csv' (upload without large csv files)
