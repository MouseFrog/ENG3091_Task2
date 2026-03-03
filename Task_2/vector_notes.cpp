#include <bits/stdc++.h>
using namespace std;

int main() {
    
    // INITIALISING VECTOR OF VECTORS

    // Method 1
    // 2D vector with initial size and value
    // This is the method we're using
    vector<vector<int>> v1(3, vector<int>(3, 7)); 
  
    // Method 2
    // A 2D vector initialized with initializer list
     vector<vector<int>> v = {{1, 2, 3},
                             {4, 5, 6}};

    // USING THE VECTOR OF VECTORS
    // Insert a new row at the end
    v.push_back({7, 8, 9});
    
    // Insert value in 2nd 
    // row at 2nd position
    v[1].insert(v[1].begin() + 1, 4);
    
    // Access first element 
    // of the first row
    cout << v[0][0] << endl;
    
    // Update first element 
    // of the second row
    v[1][0] = 5;
    
    // Access first element
    // of the second row using at()
    cout << v.at(1).at(0);
    
    return 0;

    
    return 0;
}