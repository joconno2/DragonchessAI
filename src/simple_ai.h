#pragma once

#include "game.h"
#include <vector>
#include <optional>

namespace dragonchess {

/**
 * Simple AI Interface for Students
 * 
 * To create your own AI, inherit from this class and implement choose_move().
 * 
 * Example:
 *   class MyAI : public SimpleAI {
 *   public:
 *       using SimpleAI::SimpleAI;  // Inherit constructor
 *       
 *       std::optional<Move> choose_move() override {
 *           // Your AI logic here
 *           auto moves = get_legal_moves();
 *           if (moves.empty()) return std::nullopt;
 *           return moves[0];  // Pick first move
 *       }
 *   };
 */
class SimpleAI {
public:
    SimpleAI(Game& g, Color c) : game(g), color(c) {}
    virtual ~SimpleAI() = default;
    
    // Override this method with your AI logic
    virtual std::optional<Move> choose_move() = 0;
    
protected:
    // Helper methods for students
    
    // Get all legal moves for current player
    std::vector<Move> get_legal_moves() const;
    
    // Get all pieces of a specific color
    std::vector<int> get_my_pieces() const;
    std::vector<int> get_enemy_pieces() const;
    
    // Check if a square is occupied
    bool is_occupied(int index) const { return game.board[index] != EMPTY; }
    
    // Get piece at position
    int16_t get_piece(int index) const { return game.board[index]; }
    
    // Get piece value (for evaluation)
    int get_piece_value(int16_t piece) const;
    
    // Check if move is capture
    bool is_capture(const Move& move) const {
        auto [from, to, flag] = move;
        return game.board[to] != EMPTY;
    }
    
    // Get move value (higher is better)
    int evaluate_move(const Move& move) const;
    
    // Get current game state evaluation (higher favors current player)
    int evaluate_position() const;
    
    // Access to game state
    Game& game;
    Color color;
};

} // namespace dragonchess
