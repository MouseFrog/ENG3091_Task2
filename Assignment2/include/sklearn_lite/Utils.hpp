#ifndef SKLEARN_LITE_UTILS_HPP
#define SKLEARN_LITE_UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

namespace sklearn_lite {

// Normalisation function for all three datasets 
// User can pass means and std_dev vector by reference for debugging 
inline std::vector<std::vector<double>> normaliseData (const std::vector<std::vector<double>>& X){

    int rows = X.size(); // Number of rows
    int cols = X[0].size() + 1; // Number of columns (including bias term)

    // Initialise matrix
    std::vector<std::vector<double>> result (rows, std::vector<double>(cols,0.0));
    // Set first column to 1 for bias term
    for (int i{0}; i < rows; i++) {
        result[i][0] = 1.0; 
    }

    // z-normal: stddev =1, mean=0
    std::vector<double> means{};
    std::vector<double> std_devs{};
    means.push_back(0.0); 
    std_devs.push_back(1.0);

    // Loop through each column (starting 1 to skip bias term)
       for (int j{1}; j < cols; j++) {

        // Add up all values in this feature column
        double value_sum = 0.0;
        for (int i{0}; i < rows; i++) {
            value_sum += X[i][j - 1]; // j-1 to account for skipped index
        }

        // Divide by number of samples to get mean 
        double mean {value_sum / rows};
        means.push_back(mean);

        // Squared erorr sum to measure spread around mean
        double error_sum{0.0};
        for (int i{0}; i < rows; i++) {
            error_sum += std::pow((X[i][j - 1] - mean), 2); 
        } 
    
        double std_dev {std::sqrt(error_sum / rows)};
        std_devs.push_back(std_dev);

        // If std_dev = 0, all values in this column are identical
        // Bypass equation and set values to 0
        if (std_dev == 0) {
            for (int i{0}; i < rows; i++) {
                result[i][j] = 0.0;
            }
        }
        // Shifts values so mean becomes 0, scales so spread becomes 1
        else {
            for (int i{0}; i < rows; i++) {
                result[i][j] = (X[i][j - 1] - mean) / std_dev;
            }
        }
    }
    return result; // scaled data values
};


// has_header defaults to true since concrete.csv is used first
// For ECG and MNIST, we pass false so we don't skip the first row
inline void readCSV(const std::string& filename,
                    std::vector<std::vector<double>>& X,
                    std::vector<double>& y,
                    bool has_header = true) // skip first line if has_header
{
    // Try to open the file
    std::ifstream file(filename);

    // If it didn't open, raise error
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        return;
    }
    std::string line{};

    // Skip first line only if it's a header
    if (has_header) {
        std::getline(file, line);
    }

    // Go through each line in the file
    while (std::getline(file, line)) {

        // Wrap string in object stream to extract data
        std::stringstream stream(line);
        std::string value;
        std::vector<double> row;

        // Read comma separated values
        // Convert each value from string to double using std::stod
        while (std::getline(stream, value, ',')) {
            row.push_back(std::stod(value));
        }

        // Skip empty rows just in case
        if (row.empty()) continue;

        // Appaend last value (label(y)) to vector
        y.push_back(row.back());

        // Remove the label(y) so only features are left
        row.pop_back();

        // Store the feature row in X
        X.push_back(row);
    }

    file.close();

    // Just to confirm how many rows we loaded
    std::cout << "Loaded " << X.size() << " samples from " << filename << "\n";
}

inline void saveCSV(const std::vector<double>& actual, const std::vector<double>& predictions, const std::string& filename) {
    
    // Try to open the file
    std::ofstream file(filename);
    
    // Raise error if files is unable to open
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    // Input column headers separated by comma
    file << "Actual,Prediction\n";

    // Input given labels and predicted labels
    for (int i{0}; i < predictions.size(); ++i) {
        file << actual[i] << "," << predictions[i] << "\n";
    }

    file.close();
    std::cout << "Successfully saved results to " << filename << "\n";
}

inline void check_accuracy(const std::vector<double>& actual, const std::vector<double>& predictions) {

    // Dimension mismatch in input data vectors
    if (actual.size() != predictions.size()) {
        std::cerr << "Error: Vector sizes do not match. Cannot check for accuracy.\n";
        return;
    }

    // Count number of matches/ correct predictions
    int total_samples {static_cast<int>(actual.size())};
    int correct_counts{0};
    for (int i{0}; i < total_samples; i++) {
        if (actual[i] == predictions[i]) {
            correct_counts++;
        }
    }
    
    // Accuracy in percentage
    double accuracy {(static_cast<double>(correct_counts) * 100.0) / static_cast<double>(total_samples)};

    std::cout << "Correct Predictions: " << correct_counts << " / " << total_samples << "\n";
    std::cout << "Accuracy Rate:       " << accuracy << "%\n";
}


} // namespace sklearn_lite

#endif // SKLEARN_LITE_UTILS_HPP