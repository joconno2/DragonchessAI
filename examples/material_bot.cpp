#include "../src/simple_ai.h"

using namespace dragonchess;

/**
 * MaterialBot - Maximizes material advantage
 * 
 * This bot evaluates moves based on:
 * 1. Capturing high-value pieces
 * 2. Not losing our own pieces
 * 3. Overall material balance
 */
class MaterialBot : public SimpleAI {
public:
    MaterialBot(Game& g, Color c) : SimpleAI(g, c) {}
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        Move best_move = moves[0];
        int best_score = -999999;
        
        for (const auto& move : moves) {
            int score = evaluate_material_gain(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int evaluate_material_gain(const Move& move) const {
        auto [from, to, flag] = move;
        
        int score = 0;
        
        // Bonus for capturing pieces
        if (!!is_occupied(to)) {
            int captured_piece = std::abs(get_piece(to));
            score += get_piece_value(captured_piece) * 10;  // Prioritize captures
        }
        
        // Small bonus for advancing pieces toward center
        int center_distance = calculate_center_distance(to);
        score += (8 - center_distance);  // Closer to center = better
        
        // Penalty for moving king unnecessarily (early game)
        int piece = std::abs(get_piece(from));
        if (piece == 1 && get_my_pieces().size() > 30) {  // King = 1
            score -= 5;
        }
        
        return score;
    }
    
    int calculate_center_distance(int index) const {
        int level = index / 64;
        int pos = index % 64;
        int row = pos / 8;
        int col = pos % 8;
        
        // Distance from center (3.5, 3.5)
        int row_dist = std::abs(row - 3);
        int col_dist = std::abs(col - 3);
        return row_dist + col_dist;
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new MaterialBot(game, color);
}
