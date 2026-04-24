// Base class for all regression models in the sklearn-lite library
// LinearRegression, Logistic, BinaryLogistic, and MultiLogistic all inherit from Model
// Provides shared Member variables and functions common to all models 
// Binary and Multi-class inherit from Logistic model --> Binary is reduced version of Multi-class

#ifndef SKLEARN_LITE_MODELS_HPP
#define SKLEARN_LITE_MODELS_HPP

#include <iostream>
#include <vector>   
#include <fstream>  // File operations
#include <cmath>


// Function to read csv file here

// Holds result of normalising a dataset
// Carries scaled matrix plus mean and std_dev for each column
// Taken directly from Assignment 1 Utilities.cpp
struct NormResult {
    std::vector<std::vector<double>> matrix;
    std::vector<double> means;
    std::vector<double> std_devs;
};

// Normalisation function for all three datasets 
inline NormResult normaliseData (const std::vector<std::vector<double>>& X){

    int m = X.size(); // Number of rows
    int n = X[0].size() + 1; // Number of columns (including bias term)

    NormResult result;

    // Creating an empty matrix
    result.matrix.resize(m, std::vector<double>(n));

    // Setting first column of every row to 1 for bias term
    for (int i = 0; i < m; i++) {
        result.matrix[i][0] = 1.0; 
    }

    result.means.push_back(0.0); 
    result.std_devs.push_back(1.0);

    // Loop through each column (starting from 1 to skip bias term)
       for (int j = 1; j < n; j++) {

        // Add up all values in this feature column
        double value_sum = 0.0;
        for (int i = 0; i < m; i++) {
            value_sum += X[i][j - 1]; 
        }

        // Divide by number of samples to get mean of this column
        double mean = value_sum / m;
        result.means.push_back(mean);

        // Measures how spread out the values are around the mean
        double error_sum = 0.0;
        for (int i = 0; i < m; i++) {
            error_sum += std::pow((X[i][j - 1] - mean), 2); 
        } 

        // Taken from Assignment 1
        double std_dev = std::sqrt(error_sum / m);
        result.std_devs.push_back(std_dev);

        // If std_dev is 0 every value in this column is identical
        // Set to 0 to avoid dividing by zero
        if (std_dev == 0) {
            for (int i = 0; i < m; i++) {
                result.matrix[i][j] = 0.0;
            }
        }

        // Shifts values so mean becomes 0, scales so spread becomes 1
        else {
            for (int i = 0; i < m; i++) {
                result.matrix[i][j] = (X[i][j - 1] - mean) / std_dev;
            }
        }
    }

    return result; // Return scaled matrix with means and std_devs stored
}



// BASE CLASS for all models
namespace sklearn_lite::linear {
class Models {

    protected:
        std::vector<double> weights; // Learned parameters, one per feature + bias 
        double lr; // Learning rate
        int iterations; // Max number of gradient descent steps 

    public:
    // Constructor sets learning rate and iterations
    Models(double lr, int iterations) 
         : lr{lr}, iterations{iterations} {}
    // Returns the learned weights after training 
    const std::vector<double>& get_weights() const {
        return weights;
    }
};



// LOGISTIC base class 
// Binary and Multi-class inherit from Logistic
// Contains softmax which works for both binary and multi-class cases
// Binary ECG uses it with k = 2, Multi-class MNIST uses it with k = 10
class Logistic : public Models{

    protected:
        double reg_strength; // Regularisation strength lamda 
                            // Used to stop wieghts from gettiing too large 
        
        // Converts raw scores into probabilities between 0 and 1
        std::vector<double> softmax(const std::vector<double>& z) const {
                     
            std::vector<double> probs(z.size());

            // Substract the largest score first so exp() doesnt become too large 
            double max_z = *std::max_element(z.begin(), z.end());

            // Apply e^(score - max) to every score
            double sum = 0.0;
            for (int i = 0; i < (int)z.size(); i++) {
                probs[i] = std::exp(z[i] - max_z);
                sum += probs[i];
            }

            // Divide each value by the total to get probabilities
            for (int i = 0; i < (int)z.size(); i++) {
                probs[i] /= sum;
            }

            return probs;
        }

    public:

        // Constructor passes learning rate and iterations to the base class
        // Also stores the regularisation strength for logistic regression
        Logistic(double lr, int iterations, double reg_strength)
            : Models{lr, iterations}, reg_strength{reg_strength} {}

 };

 
 
 // BINARY Logistic Regression
 // Used for ECG classification (sick = 0, healthy = 1)
 // Calls softmax with k = 2, which simplifies to sigmoid function
class BinaryLogistic : public Logistic {
    
    public:
        BinaryLogistic(double lr, int iterations, double reg_strength)
            : Logistic{lr, iterations, reg_strength} {}

            // fit() function to train model
            // predict()



};           


// MULTI-class Logistic Regression
// Used for MNIST classification (digits 0-9)
// Calls softmax with k = 10 
class MultiLogistic : public Logistic {
    
    public:
        MultiLogistic(double lr, int iterations, double reg_strength)
            : Logistic{lr, iterations, reg_strength} {}

            // fit() function to train model
            // predict()


};

} 

#endif
        




    // Function to access weights and biases common across all models
   // const std::vector<std::vector<double>>& get_weights() const{ //return by reference (large matrices), const to protect object
        //return weights;
   // }

    // Common Update Rule 
   // for (int j = 0; j < n; j++) {
     //      weights[j] -= learningRate * gradients[j] * (1.0/m);    // 1.0 to force floating pt. calculations
      //  }


   
//};
//} 


// Logistic Regression class 
// Base for Binary and Multi-class
//namespace sklearn_lite::linear {
//class Logistic : public Models {

    //private:
   // double reg_strength{};

    //public:
    //BinaryLogistic(double reg_strength):reg_strength{reg_strength}{
    //}
    // fit() function to train model


    // predict() function for a given variable set
    // Softmax function to wrap probability between 0 and 1
    // Binary version sigmoid function in class Binary

   
//};
//} 

// Binary Logistic Regression
//namespace sklearn_lite::linear {
//class BinaryLogistic : public Logistic {

    //private:
    //double reg_strength{};

    //public:
    //BinaryLogistic():{}{
    //}
    // fit() function to train model
        // Sigmoid
        // Regularised loss function

    //for (int i = 0; i < m; i++) {

            //double y_prediction{0};

            //for (int j = 0; j < n; j++) {
            //    y_prediction += weights[j] * X[i][j]; // Summation of weight * x-value for all variables 
            //}

            // Discrepancy in y value prediction
            //double error = y_prediction - Y[i][0];  

            //for (int j = 0; j < n; j++) {
                //gradients[j] += error * X[i][j]; // Calculate errors proportional to each data point
            //}
        //}

    // predict() function for a given variable set
    // Sigmoid function = Softmax when K=2, z0=0
        // Softmax function simplifies to sigmoid function at K=1: 
        // denominator e^-z term: z = z1-z0, relative difference in probability between y=0 and y=1
        // z0 is reference, assumed to be 0
        // If z0 > z1, sign flips --> e^z

    
   
//};
//} 

// Multi-class Logistic Regression
//namespace sklearn_lite::linear {
//class MultiLogistic : public Logistic {

   // private:

    //public:
    //MultiLogistic():{}{
    //}

    // fit() function to train model
        // Softmax function 
        // Regularised Loss function


    // predict() function for a given variable set
        // Apply softmax function to calculate class K probability
        // Select highest value probability


   
//};
//} 



