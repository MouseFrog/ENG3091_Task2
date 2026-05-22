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
        double reg_strength {}; // Regularisation strength:increases "dampening" of weights. 
                                // Too high : weights will not reach optimal values
                                // Too low: higher accuracy on training data but not  new data
                                // Sweet spot: highest possible without lowered accuracy?
    
    public:
        LogisticModel(double lr, int iterations, double reg_strength)
            : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {}

        // actual brains behind fit() and predict() functions, named fitting() and predicting() for clarity
        virtual void fitting(const std::vector<std::vector<double>>& X,const std::vector<double>& y) = 0;  
        virtual std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& X) const = 0;

        virtual ~LogisticModel() = default; // virtual destructor so the proper derived destructor is called     
};
 
////////////////////////////////////////////////////////////////////////////////////////////////////////
// BINARY LOGISTIC REGRESSION //////////////////////////////////////////////////////////////////////////
// ECG classification (sick = 0, healthy = 1)
// While Binary class has almost identical logic to Multiclass
// separating classes allows simplified 1D vector calculation, saving computational overhead 

class Binary : public LogisticModel {

    private: 

    std::vector<double> weights {}; // Weights per feature for class 1 is stored in 1D vector

    // Function to calculate raw probability score, used in both fitting() and predicting() functions
    std::vector<double> calc_logits(const std::vector<std::vector<double>>& features) const {
        int num_samples = static_cast<int>(features.size()); // rows
        int num_features = static_cast<int>(features[0].size()); // cols

        std::vector<double> logits(num_samples,0.0); // Store raw calculated probabilities
        // Scalar-matrix multiplication
        for (int i{0}; i < num_samples; i++) { 
            double indiv_logit{0.0}; // Used to accumulate probability per sample
            for (int j{0}; j < num_features; j++) {
                indiv_logit += weights[j] * features[i][j]; 
            }
            logits[i]= indiv_logit; // Append raw value of current sample
        }
        return logits;
    } 
    // SIGMOID FUNCTION to wrap raw probability
    // Two versions of sigmoid function to prevent explosion of values in using exponent
    // 1D vector for only class 1 probability as class 0 probability is complimentary to class 1
    void sigmoid(std::vector<double>& logits) const {
        int num_samples {static_cast<int>(logits.size())};
        for (int i{0}; i < num_samples; i++) {
            if (logits[i] >= 0.0) {                             // Standard formula for positive z values
                logits[i] = 1.0 / (1.0 + std::exp(-logits[i])); // 1/(1+e^z)
            } 
            else {                                                             // Alternate formula for negative z values
                logits[i] = std::exp(logits[i]) / (1.0 + std::exp(logits[i])); // e^{-z}/(1+e^{-z})
            }
        }
}

    public:
        Binary(double lr, int iterations, double reg_strength)
            : LogisticModel{lr, iterations, reg_strength} {}
        
        // Train using prepped features and labels
        void fitting(const std::vector<std::vector<double>>& features,
                 const std::vector<double>& labels)  override{

            int num_samples {static_cast<int>(features.size())}; // rows
            int num_features {static_cast<int>(features[0].size())}; // cols

            weights.resize(num_features, 0.0); // Prep weights vector element for every feature

            double converge_threshold {1e-12}; // Threshold value for meaningful change in weights between each interation
            std::vector<double> old_weights(num_features,0.0); // Comtainer for previous iteration's weights to track change per iteration

            for (int e{0}; e < iterations; e++) { // Repeat until convergence or max iteration value

                old_weights = weights; // Save current weights (from previous iteration)

                // 1. Calculate and wrap raw probability scores 
                std::vector<double> probabilities {calc_logits(features)}; // Initialised probability matrix here
                sigmoid(probabilities);

                // 2. Calculate rate of change per sample per feature necessary based on error
                std::vector<double> gradients(num_features, 0.0); // resets to 0.0 every iteration
                for (int i{0}; i < num_samples; i++) {
                    double error {probabilities[i] - static_cast<double>(labels[i])}; // error = predicted label - actual class label
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
                double num_weights{static_cast<double>(num_features)};
                overall_change = std::sqrt(overall_change) / num_weights; // Average change per weight

                if (overall_change < converge_threshold) { // If change in weights has stabilised, stop training early
                    std::cout << "Converged at iteration: " << e << std::endl; // save energy, save your wallet
                    break;
                }
                if (e == (iterations - 1)) { // Track overall change at max.iteration to adjust for good convergence threshold
                    std::cout << "Final iteration reached. Last overall_change was: " << overall_change << std::endl;
                }
            } 
            std::cout << "Training complete\n";
        } 

        // Actual predicting behind predict() function
        std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& features) const override{

            if (weights.empty()) { // Make sure fitting() has been called prior
                std::cerr << "Model not trained. Call fit() first.\n";
                return {};
            }
            int num_samples {static_cast<int>(features.size())};
            int num_features {static_cast<int>(features[0].size())};         

            // 1. Calculate and wrap probabilities
            std::vector<double> probabilities {calc_logits(features)};
            sigmoid(probabilities);

            // 2. return 2D probability matrix for compatibility with predict() function in LogisticRegression
            // column 0: class 0; column 1: class 1
            // It's ok because we already saved memory and energy in the heavy work above, necessary for compatibility and design
            std::vector<std::vector<double>> probability_matrix(num_samples, std::vector<double>(2, 0.0));            
            for (int i{0}; i < num_samples; i++) {
                double class1_prob {probabilities[i]};
                probability_matrix[i][0] = 1.0 - class1_prob;  // class 0 probability complimentary to class 1
                probability_matrix[i][1] = class1_prob;        // class 1
            }           
            return probability_matrix;
        }            
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
// MULTICLASS LOGISTIC REGRESSION //////////////////////////////////////////////////////////////////////
// Used for MNIST classification (digits 0-9)

class Multiclass : public LogisticModel {

private:

    std::vector<std::vector<double>> weights {}; // 2D matrix here because each class has own weights for each feature
    int num_classes {};

    // Function to calculate raw probability score, used in both fitting() and predicting() functions
    std::vector<std::vector<double>> calc_logits(const std::vector<std::vector<double>>& features) const {
        int num_samples {static_cast<int>(features.size())}; // rows
        int num_features {static_cast<int>(features[0].size())}; // cols
        int num_classes_here {static_cast<int>(weights.size())}; // avoid name clash with member variable num_classes

        // Prep logits matrix with appropriate size
        std::vector<std::vector<double>> logits(num_samples, std::vector<double>(num_classes_here, 0.0)); // Prep logits matrix with appropriate size
        // Matrix multiplication 
        for (int i{0}; i < num_samples; i++) {
            for (int j{0}; j < num_classes_here; j++) {
                double indiv_logit{0.0};    // Accumulate weights per sample per class
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
    // e^z -> e^{z - max}: subtraction of max term is included in sigmoid function for computational stabilitiy
    void softmax(std::vector<std::vector<double>>& logits) const {

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

        // Convergence Threshold: change in weights beyond which is negligible
        // MINI/MICRO dataset setting:
        double converge_threshold {1e-6};   //set to >=1e-6, @ 1e-6: ~92% accuracy, @ 1e-7: ~96% accuracy; >= 1e-8 takes more than 1000 iterations, too long
        // FULL dataset setting:
        // double converge_threshold {1e-5};    // @ 1e-5 ~83% accuracy. Try >1e-5 if you have faith in computer
                                        
        std::vector<std::vector<double>> old_weights(num_classes, std::vector<double>(num_features, 0.0)); // Weights from previous iteration to track change
        
        for (int e{0}; e < iterations; e++) { // Repeat until convergence or max iteration value
            old_weights = weights; // Save current weights (from previous iteration)
            
            // 1. Calculate and wrap raw probability scores 
            std::vector<std::vector<double>> probabilities{calc_logits(features)}; //Initialised probability matrix here
            softmax(probabilities);

            // 2. Calculate rate of change per sample necessary based on error
            std::vector<std::vector<double>> gradients(num_classes, std::vector<double>(num_features, 0.0));   
            for (int i{0}; i < num_samples; i++) { 
                int target_class {static_cast<int>(labels[i])}; // Current class label of focus e.g. class 3

                for (int j{0}; j < num_classes; j++) { 
                    double class_label {(j == target_class) ? 1.0 : 0.0};// if current class is the target class: 1 (true) label. else 0 (false)
                    double error {probabilities[i][j] - class_label}; // Error in probabilities for each class label

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
            double num_weights{static_cast<double>(num_classes*num_features)};
            overall_change = std::sqrt(overall_change) / num_weights; // Average change per weight

            if (overall_change < converge_threshold) { // If change in weights has stabilised, stop training early
                std::cout << "Converged at iteration: " << e << std::endl; // save energy, save your wallet
                break;
            }
            if (e == (iterations - 1)) { // Track overall change at max.iteration to adjust for appropriate convergence threshold
                std::cout << "Final iteration reached. Last overall_change was: " << overall_change << std::endl;
            }
        }
        std::cout << "Multiclass Engine training complete.\n";
    } 

    // predicting(): brains behind predict()
    std::vector<std::vector<double>> predicting(const std::vector<std::vector<double>>& features) const override{

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

////////////////////////////////////////////////////////////////////////////////////////////////////////
// USER-FACING LOGISTIC REGRESSION CLASS
// Utilises dynamic binding to offload computation to Binary/Multiclass 

class LogisticRegression : public BaseModel {
protected:

    double lr {}; // learning rate
    int iterations {}; // max number of iterations/ learning cycles
    double reg_strength {}; // Regularisation strength lamda 

    // LogisticRegression owns pointer to abstract class LogisticModel
    std::unique_ptr<LogisticModel> class_ptr; 

    // Function to count number of unique class labeks to determines which child class function is called
    int detect_classes(const std::vector<double>& labels) const {
        std::unordered_set<int> unique_labels(labels.begin(), labels.end()); // std::unordered_set used for fast access
        return static_cast<int>(unique_labels.size());                       
    }

public:

    LogisticRegression(double lr, int iterations, double reg_strength)
        : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {} 

    // FIT FUNCTION:
    // Allocates pointer to address of Binary/ Multiclass object
    // Allows child class function to override virtual function in parent class LogisticModel
    void fit(const std::vector<std::vector<double>>& features, const std::vector<double>& labels) override {

        int classes {detect_classes(labels)};
        
        // Instantiate either Binary or Multiclass object depending on number of classes
        if (classes == 2) class_ptr = std::make_unique<Binary>(lr, iterations, reg_strength); // syntax: std::make_unique is function,
        else class_ptr = std::make_unique<Multiclass>(classes, lr, iterations, reg_strength); // pass arguments to constructor via ()
        
        // Call train() in Binary/Multi
        class_ptr->fitting(features, labels);
    }

    // PREDICT FUNCTION
    // Utilises dynamic binding to access child class predict() function
    // Find class with highest probability match
    std::vector<double> predict(const std::vector<std::vector<double>>& sample) const override {

        if (class_ptr==nullptr) { // Check for LogisticModel object (instantiated in fit() section)
            throw std::runtime_error("Model must be fitted before predicting!");
        }

        std::vector<std::vector<double>> probability {class_ptr->predicting(sample)}; // predicting() from Binary/Multiclass called

        int num_samples {static_cast<int>(probability.size())};
        int num_classes {static_cast<int>(probability[0].size())};
        std::vector<double> predicted_class(num_samples, 0.0); // Container for final class predictions 
        
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

}; // namespace linear_models


#endif 