#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // Required for file operations
#include <random>   // For data generation
    
// Default data values for equation
// Optimise arguments for generate_data()
struct ValuesConfig {
    // Range for number of bedrooms
    double bed_min = 1.0;   
    double bed_max = 5.0;
    // Range for land area
    double area_min = 25.0;
    double area_max = 1000.0;
    // Weights of variables
    double w_intercept = 50000.0; 
    double w_bedroom = 50000.0;
    double w_area = 1600.0;
};

// Generate random values for variables
double getRandomValue(std::mt19937& mt_num, double min, double max){
    std::uniform_int_distribution<> variable_dist(min,max);
    double variable_val = variable_dist(mt_num);
    return variable_val;
}

// Adds noise to each individual data point
void calc_noise(std::vector <double>& data_in, std::mt19937& mt_num){ // Pass by reference alters original dataset, return nothing
    double current_sum = 0;
    for (int i=0; i<data_in.size(); i++){
        double indiv_dev = data_in[i] * 0.05;   // Calc standard deviation for each point to apply noise
        // Gaussian White Noise
        std::normal_distribution<double> noise_dist(0.0, indiv_dev);
        double noise = noise_dist(mt_num);
        data_in[i]=data_in[i]+noise;
    }
}

// Fill up matrices based on random generated data and equations
// Access ValuesConfig for pre-set equation parameters
void generate_data(std::vector<std::vector<double>>& var_matrix, 
                           std::vector<double>& price_matrix, 
                           const ValuesConfig& config, // Access values from ValuesConfig
                           std::mt19937& mt_num) {  // Engine 

    for (int i = 0; i < var_matrix.size(); i++) {

        // Generate random variable values
        double bedroom = getRandomValue(mt_num, config.bed_min, config.bed_max);
        double area = getRandomValue(mt_num, config.area_min, config.area_max);

        // Fill up matrix column with respective values
        const double intercept = 1.0;

        var_matrix[i][0] = intercept; // Reserved for intercept and math manipulation later on
        var_matrix[i][1] = bedroom;

        // Multi-Varible Calculations
        if (var_matrix[i].size() >1){   //Check if matrix is multi-variable
        var_matrix[i][2] = area;
        price_matrix[i] = config.w_intercept + (config.w_bedroom * bedroom) + (config.w_area * area);

        }
        else {//Single Variable Calculations
        price_matrix[i] = config.w_intercept + (config.w_bedroom * bedroom);  }
        
    }
}


int main() {

    // Set up random data generation
    std::random_device device_num; // Random number generated from hardware 
    std::mt19937 mt_num(device_num()); // Mersenne Twister pseudo-random generator 

    // Setup matrix for storing value of variables
    //int data_points = 100;  // row: # of data points
    //int num_variables = 3; // column: y-intercept, land size, # of bedrooms
    std::vector<std::vector<double>>  multi_data(100, std::vector<double>(3, 0.0)); // 100x3 matrix, initialise all points with value of 0.0
    std::vector<std::vector<double>>  single_data(100, std::vector<double>(2, 0.0)); // 100x2 matrix, initialise all points with value of 0.0

    // Initialise vector for holding house prices
    std::vector <double> single_price(100); // Simple Linear Regression
    std::vector <double> multi_price(100); // Multi Linear Regression

    // Set up ValuesConfig Object for value parameters
    ValuesConfig data_config;


    generate_data(single_data, single_price, data_config,mt_num);
    generate_data(multi_data, multi_price, data_config,mt_num);

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
            <<multi_data[i][1] << ","
            <<multi_data[i][2] << ","
            <<single_price[i]<< ","
            <<multi_price[i]<< "\n";
        }

        // Close file
        myFile.close();
        std::cout << "Successfully wrote to prices.csv" << std::endl; // Success prompt
    } 
    
    else {
        std::cerr << "Unable to open file!" << std::endl; // Raise error if unable to open file
    }

    return 0;
}    
    
    
    