#include "MultiRegression.hpp"

// This file consists standalone functions
// saveFile() and normaliseData()

// Save data into .csv file for plotting
void saveFile(const std::vector<std::vector<double>>& X, 
              const std::vector<std::vector<double>>& Y,
              std::string file_name){  
// Open file to store data
    std::ofstream myFile(file_name +".csv");

    // Check if file opened correctly
    if (myFile.is_open()) { 
        
        // Iterate through row and column of variable and price matrices
        for (int i=0; i<X.size(); i++) {

            // Save each variable value in row i 
            for (int j=0; j<X[i].size(); j++){  
                myFile << X[i][j] // 
                       <<","; // Insert comma between values
            }
            
            // Corresponding price value in row i
            for (int k=0; k < Y[i].size(); k++){  
                myFile << Y[i][k];
                
                // Insert comma only if not final value
                if (k < (Y[i].size()-1) ){
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

// Z-VALUE NORMALISATION
// Mean = 0.0
// Standard Deviation = 1.0
NormResult normaliseData(const std::vector<std::vector<double>>& X){

    int m = X.size();    // rows
    int n = X[0].size(); // cols

    NormResult result;
    result.matrix.resize(m,std::vector<double>(n)); // Matrix to hold normalised values

    // Reserve first set of data point for intercept
    // Fill default std_dev and mean for Z-Value Normalisation
    for(int i = 0; i < m ; i++){
        result.matrix[i][0]=1.0;    
    }
    result.means.push_back(0.0); 
    result.std_devs.push_back(1.0);
    
    // Loop through each variable j (col)
    for (int j = 1; j < n; j++){  

        // Summation of variable values
        double value_sum{0};
        for(int i = 0; i < m ; i++){
            value_sum += X[i][j];
        }

        // Mean of variable 
        double mean = value_sum/m;  
        result.means.push_back(mean);

        // Summation of square error
        double error_sum{0};
        for (int i = 0; i < m; i++) {
            error_sum += std::pow((X[i][j]-mean), 2);  // sqaure error = (x_val - mean)^2

        }

        // Standard Deviation
        double std_dev = std::sqrt(error_sum/m);   // stddev = sqrt(sum of square errors/ # data points)
        result.std_devs.push_back(std_dev);

        // If stddev = 0, variable values remain constant
        // Variable is assigned no weight, 0
        if (std_dev == 0){
            for (int i = 0; i < m; i++) {
            result.matrix[i][j] = 0.0;
        }
        }

        //x_normalised = (x - mean)/std_dev
        else {
            for (int i = 0; i < m; i++){
            result.matrix[i][j] = (X[i][j]-mean)/std_dev;
            }
        }
        
    }
    return result;
}