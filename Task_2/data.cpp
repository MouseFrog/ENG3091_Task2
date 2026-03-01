#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation


    
    // Calculate appropriate noise distribution level based on average house price
    void calc_noise(std::vector <double>& data_in, std::mt19937& mt_num){ // Pass by reference alters original dataset, return nothing
        double current_sum = 0;
        for (int i=0; i<data_in.size(); i++){
            current_sum = current_sum + data_in[i];
        }
        double average_val = current_sum/data_in.size();
        double standard_dev= average_val*0.05;
        std::cout<<"average price: "<<average_val<<std::endl;
        // Gaussian White Noise
        std::normal_distribution<double> noise_dist(0.0, standard_dev); // Mean = 0.0, standar dev to be calculated

        for (int i=0; i<data_in.size(); i++){
            double noise = noise_dist(mt_num);
            data_in[i]=data_in[i]+noise;
        }
    }


int main() {

    // Set up random data generation
    std::random_device device_num; // Random number generated from hardware 
    std::mt19937 mt_num(device_num()); // mt19937: Mersenne Twister pseudo-random generator 
    // Generate data for variables
    std::uniform_int_distribution<> room_dist(1, 5);  // Bedroom values range 1 to 5
    std::uniform_int_distribution<> area_dist(25, 1000);  // Land area range 25m^2 to 1500m^2

    // Setup matrix for storing value of variables
    int data_points = 100;  // row: # of data points
    int num_variables = 3; // column: y-intercept, land size, # of bedrooms
    // 100x3 matrix, initialise all points with value of 0.0
    std::vector<std::vector<double>> raw_data(data_points, std::vector<double>(num_variables, 0.0)); 

    // Initialise vector for calculating house prices
    std::vector <double> single_price(100); // Simple Linear Regression
    std::vector <double> multi_price(100); // Multi Linear Regression


    // Constants in linear regression equations
    const double intercept = 50000; // Intercept/ Starting price of houses, unchanged
    // Single variable regression slope
    //const double m = 10000;
    // Multi-variable regression weights
    const double W_bedroom = 5000;    // 5000 pounds per bedroom added
    const double W_area = 1600; // 1600 pounds per square meter 


    for (int i=0; i<100; i++) {

        double bedroom = room_dist(mt_num);   // Generate bedroom value
        double area = area_dist(mt_num);    // Generate land area 

        raw_data[i][0] = intercept; // Store intercept value in first column
        raw_data[i][1] = bedroom; // Store bedroom value in second column
        raw_data[i][2] = area;    // Store land size in third column

        // Single Variable
        // House price data points with equation form y = mx + c
        single_price[i] = W_bedroom*raw_data[i][1]+ raw_data[i][0]; // m = 5000, c = 50000

        // Multi Variable
        // House price data points with equation form y = w1*x1 + w2*x2 + c
        multi_price[i] = W_bedroom*raw_data[i][1] + W_area*raw_data[i][2] + raw_data[i][0];

    }

    // Add noise to house prices
    calc_noise(single_price, mt_num);
    calc_noise(multi_price, mt_num);

    // Open file to store data
    std::ofstream myFile("prices.csv");

    // Check if file opened correctly
    if (myFile.is_open()) { 
        // Add headers for clarity 
        // .csv file stands for "comma separated values"
        myFile << "Intercept" << "," // Comma as separator
        << "Number of Bedrooms" << ","
        << "Land Size" << ","
        << "Single Variable Price" << ","
        << "Multi Variable Price" << "\n";


        // Write house prices and variables into file
        for (int i=0; i<100; i++) {
            myFile <<raw_data[i][0] << ","
            <<raw_data[i][1] << ","
            <<raw_data[i][2] << ","
            <<single_price[i]<< ","
            <<multi_price[i]<< "\n";
        }

        // Close file
        myFile.close();
        std::cout << "Successfully wrote to prices.csv" << std::endl; // Success prompt
    } else {
        std::cerr << "Unable to open file!" << std::endl; // Raise error if unable to open file
    }

    return 0;
}    
    
    
    