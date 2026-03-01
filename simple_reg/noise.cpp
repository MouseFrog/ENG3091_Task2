#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation

    std::random_device device_num; // Random number generated from hardware 
    std::mt19937 mt_num(device_num()); // mt19937: Mersenne Twister pseudo-random generator 
    std::uniform_int_distribution<> distribute(1, 5);  // Value range 1 to 5

    // std::uniform_real_distribution<type> name(min,max) // for price --> double and no integer

int main() {

    int data_points = 100;  // rows: # of data points
    int num_variables = 3; // columns: land size, # of bedrooms, y-intercept

    // 100x2 matrix, initialise all points with value of 0.0
    std::vector<std::vector<double>> raw_data(data_points, std::vector<double>(num_variables, 0.0)); 
/*
    // Initialise vectors with data type double, with 100 points
    std::vector<double> house_prices(100); 
    std::vector <double> bedroom(100); 


    // Generate bedroom data points with equation form y = mx + c
    for (int i=0; i<100; i++) {
        bedroom[i]=distribute(mt_num);
        house_prices[i] = 50*bedroom[i]+100; // m = 50, c = 100
    }
*/
    // Open a file named "prices.csv"
    std::ofstream myFile("prices.csv");

    // Check if file opened correctly
    if (myFile.is_open()) { 
        // Add headers for clarity 
        // .csv file stands for "comma separated values" --> "Number of Bedrooms" will print in next column
        myFile << "Price" << "," << "Number of Bedrooms" <<std::endl; // Use comma as separator

        // Write house prices and bedroom # vectors into file
        for (int i=0; i<100; i++) {
            myFile << house_prices[i]<< "," <<bedroom[i] << std::endl; // Comma as separator to print in respective column
        }

        // Close file
        myFile.close();
        std::cout << "Successfully wrote to prices.csv" << std::endl; // Success prompt
    } else {
        std::cerr << "Unable to open file!" << std::endl; // Raise error if unable to open file
    }

    return 0;
}