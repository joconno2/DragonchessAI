#include "../src/simple_ai.h"
#include <algorithm>
#include <limits>

using namespace dragonchess;

/**
 * AggressiveBot - Maximally aggressive playstyle
 * 
 * This bot:
 * 1. Always attacks when possible
 * 2. Sacrifices pieces for checkmate
 * 3. Prioritizes king attacks
 * 4. Takes calculated risks
 * 5. Ignores defense in favor of offense
 */
class AggressiveBot : public SimpleAI {
public:
    AggressiveBot(Game& g, Color c) : SimpleAI(g, c) {}
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        Move best_move = moves[0];
        int best_score = std::numeric_limits<int>::min();
        
        for (const auto& move : moves) {
            int score = evaluate_aggressive_move(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int evaluate_aggressive_move(const Move& move) const {
        auto [from, to, flag] = move;
        int score = 0;
        
        int moving_piece = std::abs(get_piece(from));
        
        // Massive bonus for any capture
        if (!!is_occupied(to)) {
            int target = std::abs(get_piece(to));
            
            // Checkmate is everything
            if (target == 1) {
                return 10000000;
            }
            
            // All captures are good, high value captures are great
            score += get_piece_value(target) * 200;
            
            // BONUS: Sacrifice higher value piece for lower (aggressive trade)
            if (get_piece_value(moving_piece) > get_piece_value(target)) {
                score += 50;  // Willing to trade aggressively
            }
        }
        
        // Bonus for moving closer to enemy king
        int distance_to_enemy_king = calculate_distance_to_enemy_king(to);
        score += (20 - distance_to_enemy_king) * 10;
        
        // Bonus for forward movement (attack!)
        score += evaluate_forward_push(from, to) * 15;
        
        // Bonus for threatening multiple enemy pieces
        score += count_threats_from_square(to, moving_piece) * 20;
        
        // Penalty for defensive moves (we don't do defense!)
        if (is_retreating(from, to)) {
            score -= 50;
        }
        
        // Bonus for using powerful pieces aggressively
        if (moving_piece == 2 || moving_piece == 3) {  // Paladin, Mage
            score += 25;
        }
        
        return score;
    }
    
    int calculate_distance_to_enemy_king(int square) const {
        auto enemy_pieces = get_enemy_pieces();
        
        // Find enemy king
        for (int pos : enemy_pieces) {
            int piece = std::abs(get_piece(pos));
            if (piece == 1) {  // Found enemy king
                // Manhattan distance (simplified)
                int dx = std::abs((square % 64) % 8 - (pos % 64) % 8);
                int dy = std::abs((square % 64) / 8 - (pos % 64) / 8);
                int dz = std::abs(square / 64 - pos / 64);
                return dx + dy + dz;
            }
        }
        
        return 20;  // King not found (shouldn't happen)
    }
    
    int evaluate_forward_push(int from, int to) const {
        int from_row = (from % 64) / 8;
        int to_row = (to % 64) / 8;
        
        if (color == Color::GOLD) {
            return to_row - from_row;  // Gold moves "up"
        } else {
            return from_row - to_row;  // Scarlet moves "down"
        }
    }
    
    int count_threats_from_square(int square, int piece) const {
        // Simplified: count nearby enemy pieces
        int threats = 0;
        auto enemy_pieces = get_enemy_pieces();
        
        for (int enemy_pos : enemy_pieces) {
            int dist = std::abs(enemy_pos - square);
            if (dist > 0 && dist <= 12) {  // Within striking distance
                threats++;
            }
        }
        
        return threats;
    }
    
    bool is_retreating(int from, int to) const {
        // Moving away from opponent's side
        int from_row = (from % 64) / 8;
        int to_row = (to % 64) / 8;
        
        if (color == Color::GOLD) {
            return to_row < from_row;  // Moving backwards
        } else {
            return to_row > from_row;
        }
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new AggressiveBot(game, color);
}
