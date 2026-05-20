// LogisticRegression:"traffic operator", holds pointer type LogisticModel to Binary and Multiclass
// LogisticModel(abstract): virtual fit() and predict(), and learning parameters 

// Alternative 1: Binary and Multiclass inherti directly from LogisticRegression
// Problem: Slicing because user instantiates LogisticRegression object in main
// Alternative 2: MEGA LogisticRegression fit() and predict() incorporting logic from both Binary and Multiclass
// Problem: Lots of if-else logic, waste of computational power when conducting Binary computations 
//          can become code spaghetti and hard to debug for amateurs like us
// Solution: LogisticRegression is access point class for user, uses dynamic binding to offload computation 
//           to children of LogisticModel, Binary and Multiclass. 


#ifndef SKLEARN_LITE_LOSGISTICREGRESSION_HPP
#define SKLEARN_LITE_LOSGISTICREGRESSION_HPP

#include <iostream>
#include <vector>   // for std::vector
#include <unordered_set> // to detect number of unique classes in label vector y
#include <memory>        // for std::unique_ptr and std::make_unique
#include "BaseModel.hpp"

namespace sklearn_lite::linear {

class LogisticModel {

    protected:
        double lr {}; // learning rate
        int iterations {}; // max number of iterations/ learning cycles
        double reg_strength {}; // Regularisation strength lamda 
        

    public:
        // Constructor passes learning rate and iterations to the base class
        // Also stores the regularisation strength for logistic regression
        LogisticModel(double lr, int iterations, double reg_strength)
            : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {}

        virtual void train(const std::vector<std::vector<double>>& X,const std::vector<double>& y) = 0;  
        virtual std::vector<std::vector<double>> premonish(const std::vector<std::vector<double>>& X) const = 0;
        
};
 
 // BINARY Logistic Regression
 // ECG classification (sick = 0, healthy = 1)
class Binary : public LogisticModel {
    private: 
    std::vector<double> weights{}; // Weights per feature for class 1 is stored

    // Sigmoid function to wrap raw probability 
    // Contains only class 1 probability as class 0 probability is complimentary to class 1
    void sigmoid(std::vector<double>& logits){
        for (int i{0}; i < logits.size(); i++){
            logits[i] = 1.0/(1.0 + std::exp(-logits[i]));
        }
    }
    public:
        Binary(double lr, int iterations, double reg_strength)
            : LogisticModel{lr, iterations, reg_strength} {}


        void train(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y) override{

            int num_samples {static_cast<int>(X.size())};
            int num_features {static_cast<int>(X[0].size())};

            weights.resize(num_features, 0.0);

            // Threshold value for change in weights between update
            double converge_threshold = 1e-12;
            // Vector to store weights from previous iteration to track how much it changed
            std::vector<double> old_weights(num_features,0.0);

            for (int e{0}; e < iterations; e++) {

                // Store previous iteration's weights after every interation
                old_weights = weights;

                // Wrapped/ relative probability vector used to calculate error in prediction
                std::vector<double> probabilities(num_samples, 0.0);

                // Loop through every sample to calculate individual probabilities(logits)
                for (int i {0}; i < num_samples; i++) {
                    // Reset logit to 0.0 for every samples
                    double logit {0.0}; 

                    // Calculate raw probability
                    for (int j {0}; j < num_features; j++) {
                        logit += weights[j] * X[i][j];
                    }
                    // Sigmoid function to convert to relative probabilities
                    probabilities[i] = 1.0 / (1.0 + std::exp(-logit));
                }

                // Store gradient (rate of change necessary) for each weight during this iteration
                // Reset to 0 at the start of every pass through the data
                std::vector<double> gradients(num_features, 0.0);

                // Calculate error in each sample
                for (int i {0}; i < num_samples; i++) {

                    double error {probabilities[i] - static_cast<double>(y[i])};
                    for (int j {0}; j < num_features; j++) {
                        gradients[j] += error * X[i][j]; // Error attributed to each feature
                    }
                }

                // Update Rule: subtract gradients from current weights, scaled using regularisation term
                for (int j {0}; j < num_features; j++) {
                    // Average gradient across all samples
                    double raw_gradient { gradients[j] / num_samples };
                    
                    // Apply regularisation term
                    double reg_gradient { raw_gradient + (2.0 * reg_strength * weights[j]) };
                    
                    // Apply update to current weights
                    weights[j] -= lr * reg_gradient; 
                }

                // Measure how much the weights changed overall in this iteration
                // Sum the squared change for each weight then take square root
                double overall_change {0.0};
                for (int j{0}; j < num_features; j++) {
                    double weight_change {old_weights[j] - weights[j]};
                    overall_change += weight_change * weight_change; 
                }
                overall_change = std::sqrt(overall_change); // Magnitude of total change

                // If weights have stabilised, stop training early
                if (overall_change < converge_threshold) {
                    std::cout << "Converged at iteration: " << e << "\n";
                    break;
                }

            }

            std::cout << "Training complete\n";
        }

        std::vector<std::vector<double>> premonish(const std::vector<std::vector<double>>& X) const override{

            if (weights.empty()) {
                std::cerr << "Model not trained. Call fit() first.\n";
                return {};
            }
            int num_samples { static_cast<int>(X.size()) };
            int num_features { static_cast<int>(X[0].size()) };

            // Vector to hold probability of class 1, complimentary to probability of class 0
            std::vector<std::vector<double>> probability(num_samples, std::vector<double>(2, 0.0));            

            // Calculate probability per sample 
            for (int i{0}; i < num_samples; i++) {
                double logit {0.0};
                for (int j{0}; j < num_features; j++) {
                    logit += weights[j] * X[i][j]; 
                }
                
                double class1_prob { 1.0 / (1.0 + std::exp(-logit))};
                probability[i][0] = 1.0 - class1_prob;  // Class 0 
                probability[i][1] = class1_prob;       // Class 1 

            }
            return probability;
        }
    
            
};


        

// MULTI-CLASS Logistic Regression
// Used for MNIST classification (digits 0-9)
// Calls softmax with k = 10 
class Multiclass : public LogisticModel {

private:

    std::vector<std::vector<double>> weights {};
    int num_classes{};
    //SOFTMAX FUNCTION wraps raw probabilities between 0 and 1, per sample per class
    // Inaccessible from main
    std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& logit) const {
        
        int num_samples = logit.size(); 
        int num_classes = logit[0].size(); 

        // Final wrapped probability matrix
        std::vector<std::vector<double>> probabilities(num_samples, std::vector<double>(num_classes,0.0));

        // Modification of usual softmax function:
        // e^z --> e^{z - max}: subtraction of max term is included in sigmoid function for computational stabilitiy
        // since handle high value exponents can crash programme
        
        // Find highest probability value in every sample 
        for (int i = 0; i < num_samples; i++){
            double max_logit = logit[i][0];
            for (int j = 0; j < num_classes; j++) {
                if (logit[i][j] > max_logit) {
                    max_logit = logit[i][j];
                }
            }
            
            // Track sum of probabilities, mathematically = 1
            double sum_probability = 0.0; 

            for (int j = 0; j < num_classes; j++) {
                probabilities[i][j] = std::exp(logit[i][j] - max_logit); //e^{z - max}
                sum_probability += probabilities[i][j];
            }
            // Relative Probability: divide individual probs by total prob sum 
            for (int j = 0; j < num_classes; j++) {
            probabilities[i][j] /= sum_probability;
            }
        }
        return probabilities;
    }
        
public:
    Multiclass(int num_classes, double lr, int iterations, double reg_strength)
    : num_classes {num_classes}, LogisticModel {lr, iterations, reg_strength} {}

    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y) override {
        int num_samples { static_cast<int>(X.size()) };
        int num_features { static_cast<int>(X[0].size()) };

        // Scale pixcel values to prevent explosion of values: resulted in prediction vector of only 0s
        std::vector<std::vector<double>> X_scaled = X;
        for (int i = 0; i < num_samples; ++i) {
            for (int k = 0; k < num_features; ++k) {
                X_scaled[i][k] /= 255.0;
            }
        }

        // Weights allocated to every feature for every class
        // num_classes x num_features
        weights.resize(num_classes, std::vector<double>(num_features, 0.0));

        // Threshold value for change in weights between update
        double converge_threshold = 1e-12;
        // Vector to store weights from previous iteration to track how much it changed
        std::vector<std::vector<double>> old_weights(num_classes, std::vector<double>(num_features, 0.0));
        
        for (int e {0}; e < iterations; e++) {
            
            old_weights = weights;
            // Store raw probability per sample per class
            std::vector<std::vector<double>> logits(num_samples, std::vector<double>(num_classes, 0.0));

            // Calculate raw probability per sample per class
            for (int i {0}; i < num_samples; i++) {
                for (int j {0}; j < num_classes; j++) {
                    for (int k {0}; k < num_features; k++) {
                        logits[i][j] += weights[j][k] * X[i][k];
                    }
                }
            }

            // Softmax convert raw peobability into relative probability between 0 and 1
            std::vector<std::vector<double>> probabilities {softmax(logits)};

            // 2D matrix to track necessary change in weights
            std::vector<std::vector<double>> gradients(num_classes, std::vector<double>(num_features, 0.0));

            // Loop through each sample
            for (int i {0}; i < num_samples; i++) {
                // Current class label of focus e.g. class 3
                int target_class = static_cast<int>(y[i]); // True category integer ID (e.g., 5)
                
                // Loop through sample's relatively probability for every class 
                for (int j{0}; j < num_classes; j++) {

                    // if current class is the target class: 1.0 (true) label. Else: 0.0 (false) for all other classes
                    double class_label { (j == target_class) ? 1.0 : 0.0 };

                    // Error attributed to each feature in current sample
                    double error { probabilities[i][j] - class_label };

                    // Accumulate sample error contribution to this specific class feature
                    for (int k{0}; k < num_features; k++) {

                        // Update gradients for each feature in current class
                        gradients[j][k] += error * X[i][k];
                    }
                }
            }

            // Update Rule
            for (int i {0}; i < num_classes; i++) {
                for (int j {0}; j < num_features; j++) {

                    // Average gradient across all samples
                    double regless_grad { gradients[i][j] / num_samples };

                    // Gradient with appled regularisation
                    double reg_gradient { regless_grad + (2.0 * reg_strength * weights[i][j]) };
                    
                    weights[i][j] -= lr * reg_gradient;
                }
            }

            // Measure how much the weights changed overall in this iteration
                // Sum the squared change for each weight then take square root
                double overall_change {0.0};

                for (int i{0}; i < num_features; i++) {
                    for (int j{0}; j < num_classes; j++) { 
                        // [num_classes][num_features]
                        double weight_change {old_weights[j][i] - weights[j][i]};
                        overall_change += weight_change * weight_change; 
                    } 
                }
                overall_change = std::sqrt(overall_change); 

                // If weights have stabilised, stop training early
                if (overall_change < converge_threshold) {
                    std::cout << "Converged at iteration: " << e << "\n";
                    break;
                }

        }
        std::cout << "Multiclass Engine training complete.\n";
        
    }


    // predict()
    std::vector<std::vector<double>> premonish(const std::vector<std::vector<double>>& X) const override{

            if (weights.empty()) {
                std::cerr << "Model not trained. Call fit() first.\n";
                return {};
            }

            int num_samples { static_cast<int>(X.size()) };
            int num_features { static_cast<int>(X[0].size()) };

            // Raw probability scores per sample per class
            std::vector<std::vector<double>> logits(num_samples, std::vector<double>(num_classes, 0.0));

            for (int i {0}; i < num_samples; i++) {
                for (int j{0}; j < num_classes; j++) {
                    for (int k{0}; k < num_features; k++) {
                        logits[i][j] += weights[j][k] * X[i][k];
                    }
                }
            }

            std::vector<std::vector<double>> probabilities{softmax(logits)};
            
            return probabilities;
        }

        
};

// User-facing LogisticRegression Class
// Utilises dynamic binding to offload Binary/Multiclass implementation
// Does not implement mathematical computation
class LogisticRegression : public BaseModel {
protected:

    double lr {}; // learning rate
    int iterations {}; // max number of iterations/ learning cycles
    double reg_strength {}; // Regularisation strength lamda 
    // LogisticRegression owns pointer to the abstract math class LogisticModel
    std::unique_ptr<LogisticModel> classptr; 

    // Count number of unique items in y label vector
    // std::unordered_set chosen for fast access > sorted order
    int detect_classes(const std::vector<double>& y) const {
        std::unordered_set<int> unique_labels(y.begin(), y.end()); 
        return static_cast<int>(unique_labels.size());
    }

public:

    LogisticRegression(double lr, int iterations, double reg_strength)
        : lr{lr}, iterations{iterations}, reg_strength{reg_strength} {} 
    // FIT FUNCTION
    // Allocates pointer to address of Binary/ Multiclass object
    // Allows child class function to override virtual function in parent class
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) override {

        int classes = detect_classes(y);
        
        // Instantiate either Binary or Multiclass object depending on number of classes
        // std::make_unique is function, pass arguments to constructor via ()
        if (classes == 2) classptr = std::make_unique<Binary>(lr, iterations, reg_strength);
        else classptr = std::make_unique<Multiclass>(classes, lr, iterations, reg_strength);
        
        // Call train() in Binary/Multi
        classptr->train(X, y);
    }

    // PREDICT
    // Utilises dynamic binding to access child class predict() function
    // Find class with highest probability match
    std::vector<double> predict(const std::vector<std::vector<double>>& sample) const override {
        // Check if there is existing LogisticModel object
        if (classptr==nullptr) {
            throw std::runtime_error("Model must be fitted before predicting!");
        }

        // predict() from Binary/Multiclass called
        std::vector<std::vector<double>> probability = classptr->premonish(sample);

        int num_samples {static_cast<int>(probability.size())};
        int num_classes {static_cast<int>(probability[0].size())};

        std::vector<double> predicted_class(num_samples, 0.0);
        
        // Loop through each sample to find the class with highest probability value
        for (int i{0}; i < num_samples; i++) {

            // Highest probability value across classes per sample, whose index corresponds to its class label
            double highest_prob{probability[i][0]};
            // Final class prediction

            for (int j{0}; j < num_classes; j++){
                if (probability[i][j] > highest_prob) {
                    highest_prob = probability[i][j];
                    predicted_class[i] = static_cast<double>(j); // row value 
                }
            }
        }

        return predicted_class;
    }
};

};


#endif 