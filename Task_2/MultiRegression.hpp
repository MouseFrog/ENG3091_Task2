#ifndef MULTIREGRESSION_HPP
#define MULTIREGRESSION_HPP

#include <iostream>
#include <vector>   // for std::vector
#include <fstream>  // File operations
#include <random>   // For data generation


// Default parameter values for equation
struct Dataset {
    // Range for number of bedrooms
    int bed_min{1};   
    int bed_max{5};
    // Range for land area
    double area_min{25.0};
    double area_max{1000.0};
    // Weights of variables
    double w_intercept{50000.0}; 
    double w_bedroom {50000.0};
    double w_area {1600.0};

    // Percentage of price used as standard deviation for adding Gaussian noise
    // Change to 0 to verify math is working
    double noise_dev {0.1};
};

// Used to create matrix instances
struct Matrices { 
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;
};

// Normalised matrix and associated characteristics
// mean and standard deviation
struct NormResult {
    std::vector<std::vector<double>> matrix;
    std::vector<double> means;
    std::vector<double> std_devs;
};

// DataGenerator Class
// Generates flexible sized matrices
class DataGenerator {
private:
    std::mt19937& mt_num;
    Dataset data;

public:
    DataGenerator(Dataset c, std::mt19937& global_num);
    Matrices make_matrix(int sample_size, int num_features);
    
    // Function Overloading
    double getRandomValue(double min, double max);
    int getRandomValue(int min, int max);

    void generate_data(std::vector<std::vector<double>>& var_matrix, std::vector<std::vector<double>>& price_matrix);
    void calc_noise(std::vector<std::vector<double>>& price_matrix);

    void updateSettings(Dataset new_c);
};


// Normal Gradient Calculation class
class Normal {
public:

    double w = 0.0;   // slope of the line
    double b = 0.0;   // y-intercept

    // Normal method calculations
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& Y);

    std::pair<double,double> getWeights() const;
                    
    // Predicts price 
    double predict(double bedrooms);
};

// Gradient Descent Calculation class
class GradientDescent {
    private: 
    std::vector<double> weights; // Vector holding weights and intercept
    double learningRate;    // Step-size

    public:
    GradientDescent(int numFeatures, double lr = 0.01);

    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int iteration);
    std::vector<double> getWeights() const;
};

// Save file function
void saveFile(const std::vector<std::vector<double>>& X, 
              const std::vector<std::vector<double>>& Y,
              std::string file_name);

// Normalises data for gradient descent 
NormResult normaliseData(const std::vector<std::vector<double>>& var_matrix);

#endif