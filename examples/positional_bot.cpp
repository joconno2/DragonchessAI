#include "../src/simple_ai.h"
#include <algorithm>

using namespace dragonchess;

/**
 * PositionalBot - Focuses on positional play
 * 
 * This bot:
 * 1. Develops pieces efficiently
 * 2. Controls center and key squares
 * 3. Maintains piece coordination
 * 4. Protects the king
 * 5. Balances material and position
 */
class PositionalBot : public SimpleAI {
public:
    PositionalBot(Game& g, Color c) : SimpleAI(g, c) {}
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        Move best_move = moves[0];
        int best_score = -999999;
        
        for (const auto& move : moves) {
            int score = evaluate_position_after_move(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int evaluate_position_after_move(const Move& move) const {
        auto [from, to, flag] = move;
        int score = 0;
        
        int moving_piece = std::abs(get_piece(from));
        
        // Material evaluation
        if (!!is_occupied(to)) {
            int captured = std::abs(get_piece(to));
            if (captured == 1) return 1000000;  // Checkmate
            score += get_piece_value(captured) * 150;
        }
        
        // Piece-specific positional bonuses
        score += evaluate_piece_placement(moving_piece, to);
        
        // Center control (ground level)
        score += evaluate_center_control(to);
        
        // Piece mobility (more options = better)
        score += count_squares_controlled(to, moving_piece);
        
        // King safety - keep king protected early
        score += evaluate_king_safety(move);
        
        // Piece development - get pieces out
        score += evaluate_development(from, to);
        
        // Coordination - pieces supporting each other
        score += evaluate_coordination(to);
        
        return score;
    }
    
    int evaluate_piece_placement(int piece, int square) const {
        int level = square / 64;
        int pos = square % 64;
        int row = pos / 8;
        int col = pos % 8;
        
        // Knights/Griffins better in center
        if (piece == 4 || piece == 14) {  // Thief, Griffin-like
            if (row >= 2 && row <= 5 && col >= 2 && col <= 5) {
                return 15;
            }
        }
        
        // Flying units (Sylphs, Dragons) better on sky level
        if ((piece == 10 || piece == 12) && level == 0) {  // Sylph, Dragon
            return 10;
        }
        
        // Ground units (Warriors, Dwarves) better on ground
        if ((piece == 7 || piece == 9) && level == 1) {  // Dwarf, Warrior
            return 10;
        }
        
        return 0;
    }
    
    int evaluate_center_control(int square) const {
        if (square < 64 || square >= 128) return 0;  // Only ground level
        
        int pos = square % 64;
        int row = pos / 8;
        int col = pos % 8;
        
        // Center 4 squares
        if ((row == 3 || row == 4) && (col == 3 || col == 4)) {
            return 30;
        }
        // Extended center
        if (row >= 2 && row <= 5 && col >= 2 && col <= 5) {
            return 15;
        }
        
        return 0;
    }
    
    int count_squares_controlled(int square, int piece) const {
        // Simplified mobility estimate
        int level = square / 64;
        int pos = square % 64;
        int row = pos / 8;
        int col = pos % 8;
        
        // Central pieces control more squares
        int center_dist = std::abs(row - 3) + std::abs(col - 3);
        return (8 - center_dist) * 2;
    }
    
    int evaluate_king_safety(const Move& move) const {
        auto [from, to, flag] = move;
        int piece = std::abs(get_piece(from));
        
        // Don't move king early unless necessary
        if (piece == 1) {  // King
            int my_pieces = get_my_pieces().size();
            if (my_pieces > 30) {  // Early game
                return -25;
            }
        }
        
        return 0;
    }
    
    int evaluate_development(int from, int to) const {
        int from_level = from / 64;
        int to_level = to / 64;
        
        // Bonus for moving to different level (developing)
        if (from_level != to_level) {
            return 20;
        }
        
        // Bonus for moving pieces forward
        int from_pos = from % 64;
        int to_pos = to % 64;
        int from_row = from_pos / 8;
        int to_row = to_pos / 8;
        
        if (color == Color::GOLD) {
            if (to_row > from_row) return 5;  // Forward for gold
        } else {
            if (to_row < from_row) return 5;  // Forward for scarlet
        }
        
        return 0;
    }
    
    int evaluate_coordination(int square) const {
        // Bonus if near friendly pieces (simplified)
        int nearby_friends = 0;
        auto my_pieces = get_my_pieces();
        
        for (int friendly_pos : my_pieces) {
            int dist = std::abs(friendly_pos - square);
            if (dist > 0 && dist <= 16) {  // Nearby
                nearby_friends++;
            }
        }
        
        return std::min(nearby_friends * 2, 20);  // Cap bonus
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new PositionalBot(game, color);
}
