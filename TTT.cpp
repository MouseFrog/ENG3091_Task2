#include <iostream>
#include <random>

// Get user input 
int manual_move(std::string input_prompt) {
    while (true) { 
        std::cout << input_prompt << std::endl;
        // Validate input
        int input_value;
        if (std::cin >> input_value){ // If input is integer, it will pass from std::cin buffer stored into input_value variable
            if (input_value < 10 && input_value > 0) { // Check if input is within `range 1~9
            std::cin.ignore(99999, '\n'); // Clear buffer for next input
            return input_value;
            
            }
        }
        else {
        // Invalid input
        std::cout << "Invalid Input" << std::endl;
        std::cin.clear();
        std::cin.ignore(99999, '\n');
        }
    }
}   

// Computer move
int comp_move(){
    std::random_device rand; // Random number generated from hardware 
    std::mt19937 mt_num(rand()); // mt19937: Mersenne Twister pseudo-random generator 
    std::uniform_int_distribution<> distr(1, 9);  // Value range 1 to 9
    return distr(mt_num);
}

// Print board     
void printboard(char board[3][3]) {
    std::cout << "____________" << std::endl;
    std::cout << " " <<board[0][0] << " | " << board[0][1] << "  |" << board[0][2] << std::endl;
    std::cout << " " <<board[1][0] << " | " << board[1][1] << "  |" << board[1][2] << std::endl;
    std::cout << " " <<board[2][0] << " | " << board[2][1] << "  |" << board[2][2] << std::endl;
    std::cout << "____________" << std::endl;
}
    
// Check Win
bool isWinner(char board[3][3],char Player){    
    // Horizontal and Vertical Wins
    for (int i=0; i<3; i++) {
        if (board[i][0]==Player && board[i][1]==Player && board[i][2]==Player) {
            std::cout <<"Player"<<Player<<"wins!"<<std::endl;
            return true;
        }
        if (board[0][i]==Player && board[1][i]==Player && board[2][i]==Player) {
            std::cout <<"Player"<<Player<<" wins!"<<std::endl;
            return true;
        }
    }
    // Diagonal Wins
     if (board[0][0]==Player && board[1][1]==Player && board[2][2]==Player){
            std::cout <<"Player"<<Player<<" wins!"<<std::endl;
            return true;
        }
    if (board[0][2]==Player && board[1][1]==Player && board[2][0]==Player){
        std::cout <<"Player"<<Player<<" wins!"<<std::endl;
        return true;
    }
    return false;
}

// Check full board 
bool isFullboard(char board[3][3]) {
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            if (board[i][j] == '-') {
                return false; // board has space
            
            }
        }
    }
    return true; // full board
}


int main() {

    // Begin with player X, user input
    const char playerX = 'X';
    const char playerO = 'O';
    char current_player = playerX;

    // Initialise board
    char myboard[3][3]={
        {'-','-','-'},
        {'-','-','-'},
        {'-','-','-'}
        };
    
    printboard(myboard);
    

    while (true) {
        
        int input_val;
        
        // Computer's move
        if (current_player == playerO){
            input_val = comp_move();
        }

        // User's move
        else {
            input_val = manual_move("Enter move (1-9):");
        }
        
        // Unpack number for board assignment
        int row, col;
        row = 2 - (input_val - 1) / 3;
        col = (input_val - 1) % 3;


        // Check for cell occupancy
        if (myboard[row][col] != '-') {
            if (current_player == playerX){
                std::cout << "Cell  occupied, please select another cell." << std::endl;
            }
            continue;
        }
        
        // Update board with user input
        myboard[row][col] = current_player;
        printboard(myboard);

        // Check for win 
        if (isWinner(myboard,current_player)){
        break;
       }
            
        // Check full board
        if (isFullboard(myboard)){
            std::cout << "Game over!" <<std::endl;
            break;
        }

        // Switch player at end of turn
        if (current_player == playerX) {
            current_player = playerO;
            }
        else {
            current_player = playerX;
        }
            
    }
    
    return 0;
}

