#include "MultiRegression.hpp"
DataGenerator::DataGenerator(Dataset c) : data(c) {    
    std::random_device device_num;
    mt_num.seed(device_num()); 
}

    // Create instance from Matrices struct and fills it up to output to user 
    // Generate X (intercept+variables) and Y (price) matrices based on user input 
Matrices DataGenerator::make_matrix(int sample_size, int num_features){    // Input # features and # data points
        Matrices matrix; 

        // Set matrix size based on desired data sizes
        matrix.X.resize(sample_size,std::vector<double>(num_features+1));
        matrix.Y.resize(sample_size);

        generate_data(matrix.X,matrix.Y); // Generate random variable data points
        calc_noise(matrix.Y);   // Add noise to price matrix
        return matrix;
    }

// Generate random value for variable
double DataGenerator::getRandomValue(double min, double max){  // Specify value range
    std::uniform_real_distribution<> variable_dist(min,max);
    double variable_val = variable_dist(mt_num);    
    return variable_val;
}
int DataGenerator::getRandomValue(int min, int max){      // Overload function for generating bedroom values (integer)
    std::uniform_int_distribution<> variable_dist(min,max);
    int variable_val = variable_dist(mt_num);    
    return variable_val;
}

// Fill matrices up with generated values
void DataGenerator::generate_data(std::vector<std::vector<double>>& var_matrix, 
                        std::vector<double>& price_matrix) {  

    for (int i = 0; i < var_matrix.size(); i++) {

        // Generate random variable values
        int bedroom = getRandomValue(data.bed_min, data.bed_max);
        double area = getRandomValue(data.area_min, data.area_max);

        // Fill up matrix column with respective values
        var_matrix[i][0] = 1.0; // Reserved for intercept and math manipulation later on
        var_matrix[i][1] = bedroom; // Bedroom # values stored in second column

        // Multi-Varible Calculations
        if (var_matrix[i].size() >2){   //Check if matrix is multi-variable
        var_matrix[i][2] = area;
        price_matrix[i] = data.w_intercept + (data.w_bedroom * bedroom) + (data.w_area * area);

        }
        //Single Variable Calculations
        else {
        price_matrix[i] = data.w_intercept + (data.w_bedroom * bedroom);  }
        
    }
}

    // Adds noise to each individual data point in the price matrix
void DataGenerator::calc_noise(std::vector <double>& price_matrix){ // Pass by reference alters original dataset, return nothing
    double current_sum = 0;
    for (int i=0; i<price_matrix.size(); i++){
        double indiv_dev = price_matrix[i] * 0.05;   // Calc standard deviation for each point to apply noise
        // Gaussian White Noise
        std::normal_distribution<double> noise_dist(0.0, indiv_dev);
        double noise = noise_dist(mt_num);  
        price_matrix[i]+= noise;
    }
}

// Iterate through matrix and save into .csv file
// Matrix is read only to preserve original
void saveFile(const std::vector<std::vector<double>>& data_in, std::string file_name){  
// Open file to store data
    std::ofstream myFile(file_name +".csv");

    // Check if file opened correctly
    if (myFile.is_open()) { 
        
        // Iterate through row and column
        for (int i=0; i<data_in.size(); i++) {
            for (int j=0; j<data_in[i].size(); j++){
                myFile << data_in[i][j];

                // Inserts comma only between values --> if value is not in last column, print ","
                if (j < (data_in[i].size()-1) ){
                    myFile << ",";
                }
            }
            // Value in last column, begin new row
            myFile << "\n";
        }
        // Close file
        myFile.close();
        std::cout << "File successfully saved" << std::endl; // Success prompt
    }
    
    else {
    std::cerr << "Unable to open file!" << std::endl; // Raise error if unable to open file
    } 
} 