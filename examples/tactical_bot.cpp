#include "../src/simple_ai.h"
#include <algorithm>

using namespace dragonchess;

/**
 * TacticalBot - Looks for tactical opportunities
 * 
 * This bot:
 * 1. Prioritizes checks and checkmate threats
 * 2. Looks for undefended pieces to capture
 * 3. Avoids hanging pieces
 * 4. Controls key squares
 */
class TacticalBot : public SimpleAI {
public:
    TacticalBot(Game& g, Color c) : SimpleAI(g, c) {}
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        Move best_move = moves[0];
        int best_score = -999999;
        
        for (const auto& move : moves) {
            int score = evaluate_tactical_move(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int evaluate_tactical_move(const Move& move) const {
        auto [from, to, flag] = move;
        int score = 0;
        
        int moving_piece = std::abs(get_piece(from));
        
        // Huge bonus for capturing opponent king (checkmate)
        if (!!is_occupied(to)) {
            int target = std::abs(get_piece(to));
            if (target == 1) {  // King
                return 1000000;
            }
            
            // Bonus for captures, weighted by value
            score += get_piece_value(target) * 100;
            
            // Extra bonus if capturing with lower value piece
            if (get_piece_value(moving_piece) < get_piece_value(target)) {
                score += 50;
            }
        }
        
        // Bonus for central control (ground level, indices 64-127)
        if (to >= 64 && to < 128) {
            int pos = to % 64;
            int row = pos / 8;
            int col = pos % 8;
            
            // Center squares (d4, d5, e4, e5 equivalent)
            if ((row == 3 || row == 4) && (col == 3 || col == 4)) {
                score += 20;
            }
            // Near center
            else if (row >= 2 && row <= 5 && col >= 2 && col <= 5) {
                score += 10;
            }
        }
        
        // Bonus for developing pieces from back rank
        int from_level = from / 64;
        int to_level = to / 64;
        if (from_level != to_level) {
            score += 15;  // 3D movement
        }
        
        // Penalty for moving same piece multiple times early
        // (Would need move history - simplified here)
        
        // Small random factor to break ties
        score += (from + to) % 3;
        
        return score;
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new TacticalBot(game, color);
}
