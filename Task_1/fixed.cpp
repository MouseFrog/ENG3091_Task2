# include <iostream>
# include <cmath>
# include <limits>
using namespace std;

// Mathematical iteration fuction g(x)
double g(double x){

    return 0.5 * (x + 5.0 / x); 
}

/* oldGuess is the starting value for the algorithm
num_iters is the maximum number of times the loop will run to try to find the root 
tol is how small the change must be to stop for convergence*/
double fixed_point(double(*g)(double), double oldGuess, int num_iters = 1000, double tol = 1e-6)  
{

// calcGuess is the updated guess after applying the iteration formula to oldGuess
double calcGuess;

for (int current_iter = 0; current_iter < num_iters; current_iter ++) {

    calcGuess = g(oldGuess);

    // Prints each step of the iteration process until the value converges
    cout << "Iteration " 
         << current_iter + 1 
         << " : " 
         << calcGuess 
         << endl;

/* Compares the new and old guess to see how much the value is still changing.
If the difference is smaller than the tolerance, it means the updates are now extremely
small and the method has stabilised, so we stop and return the result */
    if (std::fabs(calcGuess - oldGuess) < tol) {
        cout << "Solution converges at iteration step " << current_iter + 1 << endl;
    return calcGuess; 
    }

    // Sets the current guess as the new starting point for the next loop
    oldGuess = calcGuess;
}

/* This only runs if the loop finishes (it reached num_iters) without converging
It prints a warning and returns NaN to indicate that no valid result was found */
cout << "Warning: Method did not converge within the maximum number of iterations" << endl;
return std::numeric_limits<double>::quiet_NaN();

}

int main() {
    // Runs the fixed point method using g and an initial guess of 2.0
    // num_iters and tol use their default values
    double my_sqrt = fixed_point(g, 2.0); 

// Printing the result and comparing with the math function
    cout << "Result from our fixed point method:   " << my_sqrt << endl;
    cout << "Result from the built-in sqrt function:   " << sqrt(5.0) << endl;

    return 0;
}
