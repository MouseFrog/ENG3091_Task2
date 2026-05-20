// Base class for all regression models in the sklearn-lite library
// It serves as an interface, only stating the compulsory fit() and predict() functions virtually
// LinearRegression and LogisticRegression inherit from BaseModel
// Binary and Multiclass logistic classes inherit from Logistic model 

#ifndef SKLEARN_LITE_BASEMODEL_HPP
#define SKLEARN_LITE_BASEMODEL_HPP

#include <iostream>
#include <vector>   
#include <fstream>  // File operations
#include <cmath>
#include <algorithm>  // 


// BASE CLASS for all models
namespace sklearn_lite::linear {
class BaseModel {

    public:
    // Explicit default constructor
    BaseModel() = default; 

    // fit and predict functions to be implemented in derived classes
    virtual void fit(const std::vector<std::vector<double>>& X,const std::vector<double>& y) = 0;  
    virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) const = 0;

    // Virtual destructor
    ~BaseModel() = default;
};

}; // namespace linear


#endif
        



