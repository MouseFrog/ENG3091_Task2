#include <iostream>
#include <cmath>
#include <limits>
using namespace std;


// Defining the function f(x) that Newton’s method will solve
double f(double x) {
    return exp(-0.1 * x) * sin(2.0 * x);
}

// Derivative of f(x), required for the Newton update formula
double df(double x) {
    return exp(-0.1 * x) * (2.0 * cos(2.0 * x) - 0.1 * sin(2.0 * x));
}

// oldGuess is the starting value that gets plugged into the Newton formula and updated each loop
// int &steps keeps track of how many iterations were used so main() can access that value
double newton_raphson(double(*func)(double), double(*dfunc)(double), double oldGuess, int &steps, int num_iters = 100, double tol = 1e-6)
{
// calcGuess stores the updated value calculated using the Newton formula
    double calcGuess;

// The loop evaluates f(x) and f'(x) before applying the Newton formula.
// If the derivative is too small, the division would be invalid, so the method stops and returns divergence.
for (int current_iter = 0; current_iter < num_iters; current_iter++) {

    double f_val  = func(oldGuess);
    double df_val = dfunc(oldGuess);

    if (std::fabs(df_val) < 1e-12) {
        steps = 0;
        cout << "Warning: divergence" << endl;
        return std::numeric_limits<double>::quiet_NaN();
    }


// Uses the Newton formula to update the current guess and move closer to the root
    calcGuess = oldGuess - (f_val / df_val);

// Compares the new and old guess to see if the value has stabilised.
// If the difference is below the tolerance, the loop stops and returns the result.    
    if (std::fabs(calcGuess - oldGuess) < tol) {
        steps = current_iter + 1;
        return calcGuess;
    }

    oldGuess = calcGuess;
}

// Runs if the loop finishes (it reached num_iters) without meeting the tolerance condition.
// It resets the step counter, prints a warning, and returns NaN to indicate that no valid root was found.
    steps = 0;
    cout << "Warning: divergence" << endl;
    return std::numeric_limits<double>::quiet_NaN();
}



int main() {

// iterations_taken stores how many steps were required for convergence
    int iterations_taken = 0;

    // Loops through the range [0, 10] in steps of 1.5
    for (double x0 = 0.0; x0 <= 10.0; x0 += 1.5) {

        // Call Newton’s method with the current starting value
        double root = newton_raphson(f, df, x0, iterations_taken);

        // If iterations_taken is not zero, convergence occurred
        if (iterations_taken != 0) {
            cout << "Root: " << root << " found in " << iterations_taken << " steps." << endl;
        }
    }

    return 0;
}
