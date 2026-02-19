#include "simple_ai.h"
#include <algorithm>

namespace dragonchess {

std::vector<Move> SimpleAI::get_legal_moves() const {
    std::vector<Move> moves;
    
    // Generate all legal moves for current player
    for (int from = 0; from < TOTAL_SQUARES; ++from) {
        if (game.board[from] == EMPTY) continue;
        
        // Check if piece belongs to current player
        bool is_mine = (color == Color::GOLD && game.board[from] > 0) ||
                      (color == Color::SCARLET && game.board[from] < 0);
        
        if (!is_mine) continue;
        
        // Generate moves for this piece
        std::vector<Move> from_moves = game.get_legal_moves_for(from);
        moves.insert(moves.end(), from_moves.begin(), from_moves.end());
    }
    
    return moves;
}

std::vector<int> SimpleAI::get_my_pieces() const {
    std::vector<int> pieces;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        if (game.board[i] == EMPTY) continue;
        
        bool is_mine = (color == Color::GOLD && game.board[i] > 0) ||
                      (color == Color::SCARLET && game.board[i] < 0);
        
        if (is_mine) {
            pieces.push_back(i);
        }
    }
    return pieces;
}

std::vector<int> SimpleAI::get_enemy_pieces() const {
    std::vector<int> pieces;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        if (game.board[i] == EMPTY) continue;
        
        bool is_enemy = (color == Color::GOLD && game.board[i] < 0) ||
                       (color == Color::SCARLET && game.board[i] > 0);
        
        if (is_enemy) {
            pieces.push_back(i);
        }
    }
    return pieces;
}

int SimpleAI::get_piece_value(int16_t piece) const {
    if (piece == EMPTY) return 0;
    
    // Absolute value for piece type
    int16_t abs_piece = piece > 0 ? piece : -piece;
    
    // Piece values based on strategic importance
    switch (abs_piece) {
        // Sky pieces
        case GOLD_SYLPH:
        case SCARLET_SYLPH:
            return 3;
        case GOLD_GRIFFIN:
        case SCARLET_GRIFFIN:
            return 4;
        case GOLD_DRAGON:
        case SCARLET_DRAGON:
            return 6;
            
        // Ground pieces
        case GOLD_OLIPHANT:
        case SCARLET_OLIPHANT:
            return 5;
        case GOLD_UNICORN:
        case SCARLET_UNICORN:
            return 4;
        case GOLD_HERO:
        case SCARLET_HERO:
            return 5;
        case GOLD_THIEF:
        case SCARLET_THIEF:
            return 3;
        case GOLD_CLERIC:
        case SCARLET_CLERIC:
            return 4;
        case GOLD_MAGE:
        case SCARLET_MAGE:
            return 4;
        case GOLD_KING:
        case SCARLET_KING:
            return 1000;  // King is invaluable
        case GOLD_PALADIN:
        case SCARLET_PALADIN:
            return 6;
        case GOLD_WARRIOR:
        case SCARLET_WARRIOR:
            return 2;
            
        // Underworld pieces
        case GOLD_BASILISK:
        case SCARLET_BASILISK:
            return 5;
        case GOLD_ELEMENTAL:
        case SCARLET_ELEMENTAL:
            return 5;
        case GOLD_DWARF:
        case SCARLET_DWARF:
            return 2;
            
        default:
            return 1;
    }
}

int SimpleAI::evaluate_move(const Move& move) const {
    auto [from, to, flag] = move;
    int score = 0;
    
    // Capturing is good
    if (game.board[to] != EMPTY) {
        score += get_piece_value(game.board[to]) * 10;
    }
    
    // Moving valuable pieces is risky (slight penalty)
    score -= get_piece_value(game.board[from]) / 10;
    
    return score;
}

int SimpleAI::evaluate_position() const {
    int score = 0;
    
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        if (game.board[i] == EMPTY) continue;
        
        int value = get_piece_value(game.board[i]);
        
        if ((color == Color::GOLD && game.board[i] > 0) ||
            (color == Color::SCARLET && game.board[i] < 0)) {
            score += value;  // My pieces are good
        } else {
            score -= value;  // Enemy pieces are bad
        }
    }
    
    return score;
}

} // namespace dragonchess
