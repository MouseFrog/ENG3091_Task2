#include <iostream>
#include <vector>
#include <fstream>  // Required for file operations

int main() {
    // Initialise vector "house_prices" with data type double 
    std::vector<double> house_prices = {250.5, 300.2, 150.7, 420.0, 310.3};

    // Open a file named "prices.csv"
    std::ofstream myFile("prices.csv");

    // 3. Check if the file opened correctly
    if (myFile.is_open()) {
        // Add a header (optional, but good for Python/Excel)
        myFile << "Price" << std::endl;

        // Range based loop: 1st iteration: val = first value in house_price, 2nd iteration: val = second value in house_price
        for (double val : house_prices) {
            myFile << val << std::endl; 
        }

        // 5. Close the file
        myFile.close();
        std::cout << "Successfully wrote to prices.csv" << std::endl;
    } else {
        std::cerr << "Unable to open file!" << std::endl; // Raise error
    }

    return 0;
}