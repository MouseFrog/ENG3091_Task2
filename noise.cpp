#include <iostream>
#include <vector>
#include <fstream>  // Required for file operations
#include <random>

    std::random_device device_num; // Random number generated from hardware 
    std::mt19937 mt_num(device_num()); // mt19937: Mersenne Twister pseudo-random generator 
    std::uniform_int_distribution<> distribute(1, 5);  // Value range 1 to 5

    // std::uniform_real_distribution<double> distr(1,900) // for price --> double and no integer

int main() {
    // Initialise vector "house_prices" with data type double 
    std::vector<double> house_prices(100); // 100 points
    // Initialise bedroom number vector 
    std::vector <double> bedroom(100); // 100 points

    // Generate bedroom data points
    for (int i=0; i<100; i++) {
        bedroom[i]=distribute(mt_num);
        house_prices[i] = 50*bedroom[i]+100;
    }
    

    // Open a file named "prices.csv"
    std::ofstream myFile("prices.csv");

    // Check if file opened correctly
    if (myFile.is_open()) {
        // Add header 
        // Comma used as separator, "Number of Bedrooms" prints in next column
        // .csv file stands for "comma separated values"
        myFile << "Price" << "," << "Number of Bedrooms" <<std::endl;

        // Write house prices and bedroom # vectors into file
        for (int i=0; i<100; i++) {
            myFile << house_prices[i]<< "," <<bedroom[i] << std::endl;
        }

        // Close file
        myFile.close();
        std::cout << "Successfully wrote to prices.csv" << std::endl; // Success prompt
    } else {
        std::cerr << "Unable to open file!" << std::endl; // Raise error if unable to open file
    }

    return 0;
}