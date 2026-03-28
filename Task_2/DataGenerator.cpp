#include "MultiRegression.hpp"

// This file holds the Data Generator Class

DataGenerator::DataGenerator(Dataset c, std::mt19937& global_num) : data(c), mt_num(global_num) {    
    }

// MAIN MATRIX GENERATOR 
// Resize and populate matrix with values
Matrices DataGenerator::make_matrix(int sample_size, int num_features){    // Input # features and # data points
        Matrices matrix; 

        // Set matrix size based on input values
        matrix.X.resize(sample_size,std::vector<double>(num_features+1));
        matrix.Y.resize(sample_size,std::vector<double>(1));

        generate_data(matrix.X,matrix.Y); // Generate data points
        calc_noise(matrix.Y);   // Add noise to price matrix
        return matrix;
    }

// RANDOM VALUE GENERATOR
double DataGenerator::getRandomValue(double min, double max){  // Specify value range
    std::uniform_real_distribution<> variable_dist(min,max);
    double variable_val = variable_dist(mt_num);    // generated from same seed
    return variable_val;
}
int DataGenerator::getRandomValue(int min, int max){      // Overload function to generate bedroom values (integer)
    std::uniform_int_distribution<> variable_dist(min,max);
    int variable_val = variable_dist(mt_num);    
    return variable_val;
}

// FILL MATRICES
void DataGenerator::generate_data(std::vector<std::vector<double>>& X, 
                                  std::vector<std::vector<double>>& Y) {  

    for (int i = 0; i < X.size(); i++) {

        // Generate random variable values
        int bedroom = getRandomValue(data.bed_min, data.bed_max);
        double area = getRandomValue(data.area_min, data.area_max);

        // Fill up matrix column with respective values
        X[i][0] = 1.0; // Reserved for intercept and math manipulation later on
        X[i][1] = bedroom; // Bedroom # values stored in second column

        // Multi-Varible Calculations
        if (X[i].size() >2){   //Check if matrix is multi-variable
        X[i][2] = area;
        Y[i][0] = data.w_intercept + (data.w_bedroom * bedroom) + (data.w_area * area);

        }
        //Single Variable Calculations
        else {
        Y[i][0] = data.w_intercept + (data.w_bedroom * bedroom);  }
        
    }
}

// ADD NOISE TO INDIVIDUAL POINT
void DataGenerator::calc_noise(std::vector<std::vector<double>>& Y){ // Pass by reference alters original dataset
    double current_sum = 0;
    for (int i=0; i<Y.size(); i++){
        double indiv_dev = Y[i][0] * data.noise_dev;   // Calc percentage of indiv point as standard deviation for noise
        // Gaussian White Noise
        std::normal_distribution<double> noise_dist(0.0, indiv_dev);
        double noise = noise_dist(mt_num);  
        Y[i][0]+= noise;
    }
}

// Allow new dataset to be used without needing to seed DataGenerator again
void DataGenerator::updateSettings(Dataset new_c) {
    data = new_c; // Swap the private 'data' struct
}
 
