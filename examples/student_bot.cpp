/**
 * STUDENT TEMPLATE BOT
 * 
 * Name: [Your Name]
 * Strategy: [Describe your strategy]
 * 
 * To compile:
 *   g++ -std=c++17 -fPIC -shared -I../DragonchessAI/src \
 *       ../DragonchessAI/src/simple_ai.cpp student_bot.cpp \
 *       -o student_bot.so
 * 
 * To test:
 *   cd ../DragonchessAI
 *   ./build/dragonchess --headless --mode tournament --games 100 \
 *       --gold-ai-plugin ../student_bot.so --scarlet-ai random \
 *       --output-csv results.csv --verbose
 */

#include "simple_ai.h"
#include <random>
#include <algorithm>

namespace dragonchess {

class StudentBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;  // Inherit constructor
    
    std::optional<Move> choose_move() override {
        // Get all legal moves
        auto moves = get_legal_moves();
        
        // Handle no legal moves
        if (moves.empty()) {
            return std::nullopt;
        }
        
        // ============================================================
        // YOUR CODE HERE: Implement your AI strategy!
        // ============================================================
        
        // Example: Simple greedy strategy
        // Find the best move by evaluating each one
        Move best_move = moves[0];
        int best_score = evaluate_my_move(best_move);
        
        for (const auto& move : moves) {
            int score = evaluate_my_move(move);
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
        
        // ============================================================
        // Other strategies you could try:
        // ============================================================
        
        // 1. Random (baseline)
        // static std::random_device rd;
        // static std::mt19937 gen(rd());
        // std::uniform_int_distribution<> dis(0, moves.size() - 1);
        // return moves[dis(gen)];
        
        // 2. First capture found
        // for (const auto& move : moves) {
        //     if (is_capture(move)) return move;
        // }
        // return moves[0];
        
        // 3. Highest value capture
        // Move best_capture = moves[0];
        // int best_value = 0;
        // for (const auto& move : moves) {
        //     if (is_capture(move)) {
        //         int value = get_piece_value(game.board[std::get<1>(move)]);
        //         if (value > best_value) {
        //             best_value = value;
        //             best_capture = move;
        //         }
        //     }
        // }
        // return best_value > 0 ? best_capture : moves[0];
    }
    
private:
    // Helper function: Evaluate a move
    int evaluate_my_move(const Move& move) {
        auto [from, to, flag] = move;
        int score = 0;
        
        // Reward capturing pieces
        if (is_capture(move)) {
            int captured_value = get_piece_value(game.board[to]);
            score += captured_value * 10;  // Captures are valuable!
        }
        
        // Small penalty for moving valuable pieces (they might be in danger)
        int my_piece_value = get_piece_value(game.board[from]);
        if (my_piece_value > 5) {
            score -= 2;  // Be careful with valuable pieces
        }
        
        // Add your own evaluation criteria here!
        // Ideas:
        // - Bonus for controlling center squares
        // - Bonus for developing pieces
        // - Penalty for leaving king exposed
        // - Bonus for threatening enemy pieces
        
        return score;
    }
    
    // Optional: Add more helper methods
    // bool is_center_square(int index) { ... }
    // bool is_king_safe() { ... }
    // int count_threats_to(int index) { ... }
};

// Required: Factory function for plugin system
extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new StudentBot(game, color);
}

} // namespace dragonchess
