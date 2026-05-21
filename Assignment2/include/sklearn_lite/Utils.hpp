#ifndef SKLEARN_LITE_UTILS_HPP
#define SKLEARN_LITE_UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

namespace sklearn_lite {

// NORMALISATION Function : ensure features are on a similar scale/ magnitude,
//                          allows error to be fairly "distributed" amongst features 
// IMPORTANT: adds a bias column to the feature matrix to simplify future matrix calculations 
inline std::vector<std::vector<double>> normaliseData (const std::vector<std::vector<double>>& dataset){

    int num_samples {static_cast<int>(dataset.size())}; // Number of rows
    int num_features {static_cast<int>(dataset[0].size() + 1)}; // Number of columns (including bias term)

    // Initialise matrix and set entire first column to 1 for bias calculations
    std::vector<std::vector<double>> result (num_samples, std::vector<double>(num_features,0.0));
    for (int i{0}; i < num_samples; i++) {
        result[i][0] = 1.0; 
    }

    // z-normalisation: stddev =1, mean=0
    // Means and Standard Deviations are returned by function, but is kept for good practice/ incase of bugs
    std::vector<double> means{};
    std::vector<double> std_devs{};
    means.push_back(0.0); 
    std_devs.push_back(1.0);

    // Loop through each column (starting 1 to skip bias term)
    for (int j{1}; j < num_features; j++) {

        // 1. Add up all values in current feature column
        double value_sum {0.0};
        for (int i{0}; i < num_samples; i++) {
            value_sum += dataset[i][j - 1]; // j-1 to account for skipped index
        }

        // 2. Calculate mean and append to means vector
        double mean {value_sum / num_samples}; // mean = sum of current feature values / # of samples
        means.push_back(mean);

        // 3. Squared erorr sum to measure spread around mean
        double error_sum{0.0};
        for (int i{0}; i < num_samples; i++) {
            error_sum += std::pow((dataset[i][j - 1] - mean), 2); 
        } 

        // 4. Calculate standard deviatiom and append to std_dev vector
        double std_dev {std::sqrt(error_sum / num_samples)}; // std_dev = magnitude of error/ # of samples
        std_devs.push_back(std_dev);

        // 5. Calculate normalised values
        // Special case: std_dev = 0 means values are all identical, by pass and set value to 0
        // Eqn involves dividing by std_dev, cannot divide by 0
        if (std_dev == 0) { 
            for (int i{0}; i < num_samples; i++) {
                result[i][j] = 0.0;
            }
        }
        else {
            for (int i{0}; i < num_samples; i++) {
                result[i][j] = (dataset[i][j - 1] - mean) / std_dev;
            }
        }
    }
    return result; // scaled data values
};

// CSV READIND function
// For concrete.csv, has_header = true
// For ECG and MNIST, has_header = false
inline void readCSV(const std::string& filename,
                    std::vector<std::vector<double>>& features,
                    std::vector<double>& labels,
                    bool has_header = true) // skip first line if has_header
{
    std::ifstream file(filename); //Try open file

    if (!file.is_open()) { // If it didn't open, raise error
        std::cerr << "Could not open file: " << filename << "\n";
        return;
    }
    std::string line{};

    if (has_header) { // Skip first line if there is a header
        std::getline(file, line);
    }

    // Go through each line in the file
    while (std::getline(file, line)) {

        // Wrap string in object stream to extract data
        std::stringstream stream(line);
        std::string value;
        std::vector<double> row {};

        while (std::getline(stream, value, ',')) { // Read comma separated values
            row.push_back(std::stod(value)); // Convert each value from string to double using std::stod
        }

        if (row.empty()) continue; // Skip empty rows just in case

        labels.push_back(row.back()); // Appaend last value, y label, to label vector

        row.pop_back(); // Remove the label(y) so only features are left

        features.push_back(row); // Store the feature row in X
    }

    file.close();

    // Check how many rows we loaded
    std::cout << "Loaded " << features.size() << " samples from " << filename << std::endl;
}

// CSV SAVING function
// Side-by-side comparison of prediction and actual data
inline void saveCSV(const std::vector<double>& actual, const std::vector<double>& predictions, const std::string& filename) {
    
    std::ofstream file(filename);
    
    if (!file.is_open()) { // Raise error if files is unable to open
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    file << "Actual,Prediction\n"; // Headers for clarity

    // Input given labels and predicted labels
    for (int i{0}; i < predictions.size(); ++i) {
        file << actual[i] << "," << predictions[i] << "\n";
    }

    file.close();
    std::cout << "Successfully saved results to " << filename << std::endl;
}

// ACCURACY CHECK function
// Provides ratio and percentage of correct predictions to test best parameters
inline void check_accuracy(const std::vector<double>& actual, const std::vector<double>& predictions) {

    if (actual.size() != predictions.size()) { // Dimension mismatch in input data vectors, cannot proceed
        std::cerr << "Error: Vector sizes do not match. Cannot check for accuracy."<< std::endl;
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
    std::cout << "Accuracy Rate: " << accuracy << "%"<< std::endl;
}


} // namespace sklearn_lite

#endif // SKLEARN_LITE_UTILS_HPP