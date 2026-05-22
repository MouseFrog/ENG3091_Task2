/* 
REPO LINK: https://github.com/MouseFrog/ENG3091.git

GENERATION AI ACKNOWLEDGEMENT
Claude and Gemini were used as educational resource for this assignment. 
Such usage included:
  - Reviewing variable naming conventions 
  - General Coding & Debugging: resolving compilation errors, structuring class architecture
  - Verifying file path structure adjustments for library organization.

Design choices and core algorithm logic were manually tested and verified by us.
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////

// Compile:     g++ Linear.cpp -o linear.out
// Run:         ./linear.out

// Runs the full linear regression pipeline on the concrete dataset
// Loads the data, normalises it, trains the model, and checks performance
// Expected R2 is around 0.61 (same as Python sklearn)

#include <iostream>
#include <vector>
#include "sklearn_lite.hpp"``

int main() {

    // Setup containers for separating dataset into features and true values
    std::vector<std::vector<double>> features {}; 
    std::vector<double> strength {};

    // Load the dataset from file into containers, skip header
    sklearn_lite::readCSV("../data/concrete.csv", features, strength, true);

    // Prep & normalise feature values for training
    std::vector<std::vector<double>> norm_features {sklearn_lite::normaliseData(features)};

    // Instantiate the model with chosen learning parameters
    sklearn_lite::linear_models::LinearRegression model(0.1, 10000);

    // Train the model using the normalised data
    model.fit(norm_features, strength);

    // Use the trained model to make predictions
    std::vector<double> predictions {model.predict(norm_features)};

    // Check how good the predictions are
    double r2 {model.r2_score(strength, predictions)};

    // Print the result and compare to expected value
    std::cout << "R2 Score: " << r2 << "\n";
    std::cout << "Python sklearn benchmark: ~0.61\n";

    return 0;
}