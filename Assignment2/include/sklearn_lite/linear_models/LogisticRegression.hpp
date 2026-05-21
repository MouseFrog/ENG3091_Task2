// LogisticRegression is instantiated by user in main, holds pointer type LogisticModel to Binary and Multiclass, instantiating either by dynamic binding
// LogisticModel is Abstract parent class of Binary and Multiclass: virtual fitting() and predicting(), and learning parameters 

// Alternative 1: Binary and Multiclass inherti directly from LogisticRegression
// Problem: Slicing because user instantiates LogisticRegression object in main
// Alternative 2: MEGA LogisticRegression fit() and predict() incorporting logic from both Binary and Multiclass
// Problem: Lots of if-else logic, waste of computational power when conducting Binary computations 
//          can become code spaghetti and hard to debug for amateurs like us
// Solution: LogisticRegression is access point class for user, uses dynamic binding to offload computation 
//           to children of LogisticModel, Binary and Multiclass. 


#ifndef SKLEARN_LITE_LOGISTICREGRESSION_HPP
#define SKLEARN_LITE_LOGISTICREGRESSION_HPP

#include <iostream>
#include <vector>   // for std::vector
#include <unordered_set> // to detect number of unique classes in label vector y
#include <memory>        // for std::unique_ptr and std::make_unique
#include "BaseModel.hpp"

namespace sklearn_lite::linear_models {


// LOGISTIC MODEL
// Parent of Binary and Multiclass
// Abstract class, defines virtual fitting() and predicting()
class LogisticModel {

    protected:
        double lr {}; // learning rate/ step size
        int iterations {}; // max number of iterations/ learning cycles
        double reg_strength {}; // Regularisation strength lamda 
    
    public:
        LogisticModel(double lr, int iterations, double reg_strength)
            : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {}

        // actual brains behind fit() and predict() functions, named fitting() and predicting() for clarity
        virtual void fitting(const std::vector<std::vector<double>>& X,const std::vector<double>& y) = 0;  
        virtual std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& X) = 0;

        virtual ~LogisticModel() = default; // virtual destructor so the proper derived destructor is called     
};
 
// BINARY Logistic Regression
// ECG classification (sick = 0, healthy = 1)
// While Binary class has almost identical logic to Multiclass (mathematically multiclass with k=2)
// But separating classes allows simplified 1D vector calculation, saving computational overhead 
// (we have to make it distinct somehow because this feels like a nice design)
class Binary : public LogisticModel {

    private: 

    std::vector<double> weights {}; // Weights per feature for class 1 is stored in 1D vector

    // Function to calculate raw probability score, used in both fitting() and predicting() sections
    std::vector<double> calc_logits(const std::vector<std::vector<double>>& features) const {
        int num_samples = static_cast<int>(features.size()); // rows
        int num_features = static_cast<int>(features[0].size()); // cols

        std::vector<double> logits(num_samples,0.0); // Store raw calculated probabilities
        for (int i{0}; i < num_samples; i++) { 
            double indiv_logit{0.0}; // Used to accumulate probability per sample
            for (int j{0}; j < num_features; j++) {
                indiv_logit += weights[j] * features[i][j]; // Accumulate probability per sample
            }
            logits[i]= indiv_logit; // Append raw value of current sample
        }
        return logits;
    } 
    // SIGMOID FUNCTION to wrap raw probability
    // Uses two versions of sigmoid function to prevent explosion of values
    // 1D vector for only class 1 probability as class 0 probability is complimentary to class 1
    void sigmoid(std::vector<double>& logits) {
        for (size_t i{0}; i < logits.size(); i++) {
            if (logits[i] >= 0.0) {
                logits[i] = 1.0 / (1.0 + std::exp(-logits[i])); // Standard formula for positive z values 
            } 
            else {
                double exp_z = std::exp(logits[i]); // Stable formula for negative z values
                logits[i] = exp_z / (1.0 + exp_z);
            }
        }
}

    public:
        Binary(double lr, int iterations, double reg_strength)
            : LogisticModel{lr, iterations, reg_strength} {}
        
        // Train using prepped features and labels
        void fitting(const std::vector<std::vector<double>>& features,
                 const std::vector<double>& labels) override{

            int num_samples {static_cast<int>(features.size())};
            int num_features {static_cast<int>(features[0].size())};

            weights.resize(num_features, 0.0); // Prep weights vector for values

            double converge_threshold {1e-12}; // Threshold value for meaningful change in weights between each interation
            std::vector<double> old_weights(num_features,0.0); // Store previous iteration's weights to track change per iteration

            for (int e{0}; e < iterations; e++) { // Repeat until convergence or max iteration value

                old_weights = weights; // Save current weights (from previous iteration)

                // 1. Calculate raw probability scores and convert to 0s and 1s
                std::vector<double> probabilities {calc_logits(features)}; //Initialise probability matrix here
                sigmoid(probabilities);

                // 2. Calculate rate of change per sample necessary based on error
                std::vector<double> gradients(num_features, 0.0); // resets to 0.0 every iteration
                for (int i{0}; i < num_samples; i++) {
                    double error {probabilities[i] - static_cast<double>(labels[i])}; // error = predicted label (0 or 1) - actual class label
                    for (int j{0}; j < num_features; j++) {
                        gradients[j] += error * features[i][j]; // Error attributed to each feature
                    }
                }

                // 3. UPDATE RULE: subtract gradients from current weights which are scaled via regularisation term
                for (int j{0}; j < num_features; j++) {
                    double raw_gradient {gradients[j] / num_samples}; // Average gradient across all samples
                    double reg_gradient {raw_gradient + (2.0 * reg_strength * weights[j])}; // Apply regularisation term
                    weights[j] -= lr * reg_gradient; // Apply update to current weights
                }

                // 4. Track net change in weights in this iteration
                double overall_change {0.0}; // To be compared to conversion threshold
                for (int j{0}; j < num_features; j++) {
                    double weight_change {old_weights[j] - weights[j]};
                    overall_change += weight_change * weight_change; // Sum of all individual squared change
                }
                overall_change = std::sqrt(overall_change); // Magnitude of total change via sqrt: meths
                if (overall_change < converge_threshold) { // If change in weights has stabilised, stop training early
                    std::cout << "Converged at iteration: " << e << "\n"; // save energy, save the earth
                    break;
                }
            } 
            std::cout << "Training complete\n";
        } 

        std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& features) override{

            if (weights.empty()) { // Make sure fitting() had been called prior
                std::cerr << "Model not trained. Call fit() first.\n";
                return {};
            }
            int num_samples {static_cast<int>(features.size())};
            int num_features {static_cast<int>(features[0].size())};         

            // 1. Calculate and wrap probabilities
            std::vector<double> probabilities {calc_logits(features)};
            sigmoid(probabilities);

            // 2. return 2D probability matrix for compatibility with predict() function in LogisticRegression
            // Include class 0 probability
            // It's ok because we already saved memory and energy in the heavy work above, necessary for compatibility and design
            std::vector<std::vector<double>> probability_matrix(num_samples, std::vector<double>(2, 0.0));            
            for (int i = 0; i < num_samples; i++) {
                double class1_prob {probabilities[i]};
                probability_matrix[i][0] = 1.0 - class1_prob;  // Class 0 probability complimentary to class 1
                probability_matrix[i][1] = class1_prob;        // Class 1
            }           
            return probability_matrix;
        }            
};

// MULTI-CLASS Logistic Regression
// Used for MNIST classification (digits 0-9)
class Multiclass : public LogisticModel {

private:

    std::vector<std::vector<double>> weights {}; // 2D matrix here because [# of classes][# of features]
    int num_classes {};

    // Function to calculate raw probability score, used in both fitting() and predicting() sections
    std::vector<std::vector<double>> calc_logits(const std::vector<std::vector<double>>& features) const {
        int num_samples {static_cast<int>(features.size())}; // rows
        int num_features {static_cast<int>(features[0].size())}; // cols
        int num_classes_here {static_cast<int>(weights.size())}; // avoid name clash with member variable num_classes

        // Prep logits matrix with appropriate size
        std::vector<std::vector<double>> logits(num_samples, std::vector<double>(num_classes_here, 0.0)); // Prep logits matrix with appropriate size
        for (int i{0}; i < num_samples; i++) {
            for (int j{0}; j < num_classes_here; j++) {
                double indiv_logit{0.0};
                for (int k{0}; k < num_features; k++) {
                    indiv_logit += weights[j][k] * features[i][k];
                }
                logits[i][j] = indiv_logit;
            }
        }
        return logits;
    } 
    //SOFTMAX FUNCTION wraps raw probabilities between 0 and 1; per sample per class; inaccessible from main
    // Modification of usual softmax function:
    // e^z --> e^{z - max}: subtraction of max term is included in sigmoid function for computational stabilitiy
    // since negative high value exponents can explode result
    void softmax(std::vector<std::vector<double>>& logits) {

        int num_samples {static_cast<int>(logits.size())}; 
        int num_classes {static_cast<int>(logits[0].size())}; 
        
        // 1. Find highest probability value in every sample 
        for (int i{0}; i < num_samples; i++){
            double max_logit = logits[i][0];
            for (int j{0}; j < num_classes; j++) {
                if (logits[i][j] > max_logit) {
                    max_logit = logits[i][j];
                }
            }
            double sum_probability {0.0}; // Track sum of probabilities, denominator in eqn

            // 2. Apply softmax equation
            for (int j{0}; j < num_classes; j++) {
                logits[i][j] = std::exp(logits[i][j] - max_logit); //e^{z - max}
                sum_probability += logits[i][j];
            }
            for (int j{0}; j < num_classes; j++) {
            logits[i][j] /= sum_probability;
            }
        }
    }
        
public:
    Multiclass(int num_classes, double lr, int iterations, double reg_strength)
    : num_classes {num_classes}, LogisticModel {lr, iterations, reg_strength} {}

    // Train using prepped features and labels
    void fitting(const std::vector<std::vector<double>>& features, const std::vector<double>& labels) override {
        int num_samples {static_cast<int>(features.size())}; // rows
        int num_features {static_cast<int>(features[0].size())}; // cols
        
        weights.resize(num_classes, std::vector<double>(num_features, 0.0)); // Weights allocated to every feature for every class [num_classes][num_features]

        double converge_threshold {1e-12}; // Threshold value for meaningful change in weights between each interation
        std::vector<std::vector<double>> old_weights(num_classes, std::vector<double>(num_features, 0.0)); // Weights from previous iteration to track change
        
        for (int e{0}; e < iterations; e++) { // Repeat until convergence or max iteration value
            old_weights = weights; // Save current weights (from previous iteration)
            
            // 1. Calculate raw probability scores and convert to 0s and 1s
            std::vector<std::vector<double>> probabilities{calc_logits(features)}; //Initialise probability matrix here
            softmax(probabilities);// Softmax wrap raw probabilities between 0 and 1

            // 2. Calculate rate of change per sample necessary based on error
            std::vector<std::vector<double>> gradients(num_classes, std::vector<double>(num_features, 0.0));   
            for (int i{0}; i < num_samples; i++) { 
                int target_class {static_cast<int>(labels[i])}; // Current class label of focus e.g. class 3

                for (int j{0}; j < num_classes; j++) { // Sample's probability for every class 
                    double class_label {(j == target_class) ? 1.0 : 0.0};// if current class is the target class: 1.0 (true) label.
                    double error {probabilities[i][j] - class_label}; // Error in probability for each class label

                    for (int k{0}; k < num_features; k++) { // Accumulated error attributed to each feature 
                        gradients[j][k] += error * features[i][k]; // = necessary magnitude of change
                    }
                }
            }
            // 3. UPDATE RULE: subtract gradients from current weights which are scaled via regularisation term
            for (int i{0}; i < num_classes; i++) { 
                for (int j{0}; j < num_features; j++) {
                    double regless_grad {gradients[i][j] / num_samples};  // Average gradient across all samples
                    double reg_gradient {regless_grad + (2.0 * reg_strength * weights[i][j])}; // Gradient with applied regularisation
                    
                    weights[i][j] -= lr * reg_gradient;
                }
            }
            // 4. Track net change in weights in this iteration
            double overall_change {0.0}; 
            for (int i{0}; i < num_classes; i++) {
                for (int j{0}; j < num_features; j++) { 
                    double weight_change {old_weights[i][j] - weights[i][j]}; 
                    overall_change += weight_change * weight_change; // Sum of all individual squared change
                } 
            }
            overall_change = std::sqrt(overall_change); // Magnitude of total change via sqrt: meths
            if (overall_change < converge_threshold) { // If change in weights has stabilised, stop training early
                std::cout << "Converged at iteration: " << e << "\n"; // save energy, save your wallet
                break;
            }
        } 
        std::cout << "Multiclass Engine training complete.\n";
    } // end of fitting()


    // predicting(): brains behind predict()
    std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& features) override{

            if (weights.empty()) { // Check for existing weights, if model went through fitting()
                std::cerr << "Model not trained. Call fit() first.\n";
                return {};
            }

            int num_samples {static_cast<int>(features.size())}; // rows
            int num_features {static_cast<int>(features[0].size())}; // cols

            // Calculate and wrap probabilities
            std::vector<std::vector<double>> probabilities {calc_logits(features)}; 
            softmax(probabilities);

            return probabilities;
        }

        
};

// User-facing LogisticRegression Class
// Utilises dynamic binding to offload computation to Binary/Multiclass 
class LogisticRegression : public BaseModel {
protected:

    double lr {}; // learning rate
    int iterations {}; // max number of iterations/ learning cycles
    double reg_strength {}; // Regularisation strength lamda 

    // LogisticRegression owns pointer to the abstract math class LogisticModel
    std::unique_ptr<LogisticModel> classptr; 

    // Function to count number of unique items in y label vector: determines which child class function is called
    int detect_classes(const std::vector<double>& labels) const {
        std::unordered_set<int> unique_labels(labels.begin(), labels.end()); // std::unordered_set chosen for fast access > sorted order
        return static_cast<int>(unique_labels.size());
    }

public:

    LogisticRegression(double lr, int iterations, double reg_strength)
        : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {} 

    // FIT FUNCTION
    // Allocates pointer to address of Binary/ Multiclass object
    // Allows child class function to override virtual function in parent class LogisticModel
    void fit(const std::vector<std::vector<double>>& features, const std::vector<double>& labels) override {

        int classes {detect_classes(labels)};
        
        // Instantiate either Binary or Multiclass object depending on number of classes
        // std::make_unique is function, pass arguments to constructor via ()
        if (classes == 2) classptr = std::make_unique<Binary>(lr, iterations, reg_strength);
        else classptr = std::make_unique<Multiclass>(classes, lr, iterations, reg_strength);
        
        // Call train() in Binary/Multi
        classptr->fitting(features, labels);
    }

    // PREDICT
    // Utilises dynamic binding to access child class predict() function
    // Find class with highest probability match
    std::vector<double> predict(const std::vector<std::vector<double>>& sample) const override {
        // Check if there is existing LogisticModel object
        if (classptr==nullptr) {
            throw std::runtime_error("Model must be fitted before predicting!");
        }

        // predicting() from Binary/Multiclass called
        std::vector<std::vector<double>> probability {classptr->predicting(sample)};

        int num_samples {static_cast<int>(probability.size())};
        int num_classes {static_cast<int>(probability[0].size())};

        std::vector<double> predicted_class(num_samples, 0.0);// Initialise final class predictions 
        
        // Loop through each sample to find the class with highest probability value
        for (int i{0}; i < num_samples; i++) {
            double highest_prob {probability[i][0]}; // Set first value as default highest probability
            predicted_class[i] =0.0 ; // Set first class (class 0) as default label

            for (int j{1}; j < num_classes; j++) { // Check remaining classes and their respective probabilities
                if (probability[i][j] > highest_prob) {
                    highest_prob = probability[i][j];
                    predicted_class[i] = static_cast<double>(j); 
                }
            }
        }

        return predicted_class;
    }
};

};


#endif 