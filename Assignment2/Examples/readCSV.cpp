// readCSV.cpp
// Loads a CSV file into X (features) and y (labels)
// Works for all datasets in this assignment
// If there's a header row (like concrete.csv), we skip it
// Otherwise (ECG, MNIST), we just read from the start
// Last value in each row is always the label, rest are features

#include <iostream>
#include <vector>
#include <fstream>    
#include <sstream>    
#include <string>     

// has_header defaults to true since concrete.csv is used first
// For ECG and MNIST, we pass false so we don't skip the first row
void readCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<double>& y,
             bool has_header = true)
{
    // Try to open the file
    std::ifstream file(filename);

    // If it didn't open, raise error
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        return;
    }

    std::string line;

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

        // Remove the label so only features are left
        row.pop_back();

        // Store the feature row in X
        X.push_back(row);
    }

    file.close();

    // Just to confirm how many rows we loaded
    std::cout << "Loaded " << X.size() << " samples from " << filename << "\n";
}
