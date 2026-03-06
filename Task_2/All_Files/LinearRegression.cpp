#include "LinearRegression.hpp"

// Train the model using the dataset of bedrooms and prices
// The goal is to compute the line that best fits the data
void SimpleLinearRegression::train(const std::vector<double>& bedroom_values,
                                   const std::vector<double>& price_values)
{
    int num_points = bedroom_values.size();

    // Variables used to accumulate the sums needed for the regression calculation
    // They are updated during the loop so that we end up with the totals across the dataset
    double sum_bedrooms = 0.0;
    double sum_prices = 0.0;
    double sum_bedrooms_times_prices = 0.0;
    double sum_bedrooms_squared = 0.0;

    // Loop through the dataset and update the running sums for each value
    for (int i = 0; i < num_points; i++) {

        double bedrooms = bedroom_values[i];
        double price = price_values[i];

        sum_bedrooms += bedrooms;
        sum_prices += price;

        // Tracks how bedrooms and prices vary together
        sum_bedrooms_times_prices += bedrooms * price;

        // Tracks how spread out the bedroom values are
        sum_bedrooms_squared += bedrooms * bedrooms;
    }

    // Use the totals computed in the loop to determine the slope of the regression line
    w = (num_points * sum_bedrooms_times_prices - sum_bedrooms * sum_prices)
        /
        (num_points * sum_bedrooms_squared - sum_bedrooms * sum_bedrooms);

    b = (sum_prices - w * sum_bedrooms) / num_points;
}


// Predicts price from the regression model
double SimpleLinearRegression::predict(double bedrooms) {
    return w * bedrooms + b;
}