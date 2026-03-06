// Main program for testing single-variable linear regression
// Generates a dataset, trains the model, and prints the results
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

#include "LinearRegression.hpp"

int main() {

    // Create the random generator used for producing bedroom values and noise
    std::random_device rd;
    std::mt19937 gen(rd());

    // Random bedroom values between 1 and 5
    std::uniform_int_distribution<int> bedroom_dist(1, 5);

    // Small noise applied to bedroom
    std::uniform_real_distribution<double> bedroom_noise(-0.1, 0.1);

    // Noise added to price
    std::uniform_real_distribution<double> price_noise(-4000.0, 4000.0);


    // True price parameters
    double true_price_per_bedroom = 50000.0;
    double true_base_price = 50000.0;

    int num_houses = 100;


    // Store the generated bedroom and price values
    std::vector<double> bedroom_values;
    std::vector<double> price_values;

    // Open CSV file to store the generated dataset
    std::ofstream myFile("prices.csv");

    // If the file cannot be opened, print an error and stop the program
    if (!myFile.is_open()) {
        std::cout << "Error: could not open prices.csv for writing." << std::endl;
        return 1;
    }

    // Column headers for the CSV file
    myFile << "Bedrooms,Price\n";


    // Generate the dataset used for training the regression model
    // Each iteration creates one house entry
    for (int i = 0; i < num_houses; i++) {

        int base_bedrooms = bedroom_dist(gen);

        // Add small noise to the bedroom value
        double bedrooms = base_bedrooms + bedroom_noise(gen);

        // Compute the price using the true model with added noise
        double price =
            (true_price_per_bedroom * base_bedrooms)
            + true_base_price
            + price_noise(gen);

        // Store values so the regression model can train on them
        bedroom_values.push_back(bedrooms);
        price_values.push_back(price);

        // Also write the same data to the CSV file
        myFile << bedrooms << "," << price << "\n";
    }

    // Close the dataset file once writing is finished
    myFile.close();

    std::cout << "Data written to prices.csv" << std::endl;


    // Create the regression model and train it using the generated dataset
    SimpleLinearRegression model;
    model.train(bedroom_values, price_values);


    // Print the estimated parameters and compare them with the true values
    std::cout << "\n--- Normal Equation Results ---" << std::endl;

    std::cout << "Estimated price per bedroom (w): "
              << model.w
              << " | True: "
              << true_price_per_bedroom
              << std::endl;

    std::cout << "Estimated base price (b): "
              << model.b
              << " | True: "
              << true_base_price
              << std::endl;

    return 0;
}
