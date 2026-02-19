/**
 * Example AI Bots for Students
 * 
 * These are progressively more complex examples to help students
 * learn how to create their own Dragonchess AI.
 * 
 * Compile with:
 *   g++ -std=c++17 -I../src -c example_bots.cpp -o example_bots.o
 */

#include "simple_ai.h"
#include <random>
#include <algorithm>

namespace dragonchess {

// ============================================================================
// LEVEL 1: RANDOM BOT (Starter Example)
// Just picks a random legal move
// ============================================================================
class RandomBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // Pick random move
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, moves.size() - 1);
        
        return moves[dis(gen)];
    }
};

// ============================================================================
// LEVEL 2: CAPTURE BOT (Beginner Example)
// Prefers capturing moves, otherwise random
// ============================================================================
class CaptureBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // Find capturing moves
        std::vector<Move> captures;
        for (const auto& move : moves) {
            if (is_capture(move)) {
                captures.push_back(move);
            }
        }
        
        // Prefer captures if available
        if (!captures.empty()) {
            return captures[0];
        }
        
        return moves[0];
    }
};

// ============================================================================
// LEVEL 3: GREEDY BOT (Intermediate Example)
// Picks the move that captures the most valuable piece
// ============================================================================
class GreedyBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // Find best move by evaluation
        Move best_move = moves[0];
        int best_score = evaluate_move(best_move);
        
        for (const auto& move : moves) {
            int score = evaluate_move(move);
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
};

// ============================================================================
// LEVEL 4: DEFENSIVE BOT (Advanced Example)
// Balances offense and defense
// ============================================================================
class DefensiveBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // Evaluate each move considering both attack and defense
        Move best_move = moves[0];
        int best_score = score_move(best_move);
        
        for (const auto& move : moves) {
            int score = score_move(move);
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int score_move(const Move& move) {
        auto [from, to, flag] = move;
        int score = 0;
        
        // Reward captures
        if (is_capture(move)) {
            score += get_piece_value(game.board[to]) * 10;
        }
        
        // Penalize moving valuable pieces into danger
        int piece_value = get_piece_value(game.board[from]);
        if (piece_value > 5) {
            score -= 2;  // Be cautious with valuable pieces
        }
        
        // Bonus for center control (middle board positions)
        auto [layer, row, col] = index_to_pos(to);
        if (layer == 1 && row >= 3 && row <= 4 && col >= 5 && col <= 6) {
            score += 5;  // Center control bonus
        }
        
        return score;
    }
};

// ============================================================================
// LEVEL 5: STUDENT TEMPLATE (For Students to Fill In)
// ============================================================================
class StudentBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        // TODO: Implement your AI strategy here!
        //
        // Helpful methods you can use:
        // - get_legal_moves()        : Get all legal moves
        // - get_my_pieces()          : Get positions of your pieces
        // - get_enemy_pieces()       : Get positions of enemy pieces
        // - is_capture(move)         : Check if move captures
        // - evaluate_move(move)      : Get basic move score
        // - evaluate_position()      : Get current board score
        // - get_piece_value(piece)   : Get piece value
        //
        // Example strategies to try:
        // 1. Prioritize capturing high-value pieces
        // 2. Protect your king
        // 3. Control the center
        // 4. Develop pieces from starting positions
        // 5. Look ahead 1-2 moves (simulate moves)
        
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // YOUR CODE HERE
        return moves[0];  // Replace with your logic
    }
};

// ============================================================================
// LEVEL 6: LOOKAHEAD BOT (Expert Example)
// Simulates moves to look ahead
// ============================================================================
class LookaheadBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        Move best_move = moves[0];
        int best_score = -999999;
        
        for (const auto& move : moves) {
            int score = evaluate_move_with_lookahead(move);
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int evaluate_move_with_lookahead(const Move& move) {
        // Save game state
        Game temp_game = game;
        
        // Try the move
        temp_game.make_move(move);
        temp_game.update();
        
        // Evaluate resulting position
        int score = evaluate_move(move);
        
        // If game not over, consider opponent's best response
        if (!temp_game.game_over) {
            // Switch perspective for opponent
            Color opp_color = (color == Color::GOLD) ? Color::SCARLET : Color::GOLD;
            LookaheadBot opponent(temp_game, opp_color);
            
            auto opp_moves = opponent.get_legal_moves();
            if (!opp_moves.empty()) {
                // Assume opponent plays well
                int worst_response = 999999;
                for (const auto& opp_move : opp_moves) {
                    int opp_score = opponent.evaluate_move(opp_move);
                    if (opp_score < worst_response) {
                        worst_response = opp_score;
                    }
                }
                score -= worst_response / 2;  // Consider opponent's threat
            }
        }
        
        return score;
    }
};

} // namespace dragonchess
