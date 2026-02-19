#include "moves.h"
#include <cmath>
#include <algorithm>

namespace dragonchess {

// Helper macro for checking enemy pieces
#define IS_ENEMY(piece, color) \
    ((color == Color::GOLD && (piece) < 0) || (color == Color::SCARLET && (piece) > 0))

std::vector<Move> generate_sylph_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    int direction = (color == Color::GOLD) ? -1 : 1;
    
    if (layer == 0) {
        // Non-capturing: diagonal moves (destination must be empty)
        for (int dc : {-1, 1}) {
            int new_row = row + direction;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY) {
                    moves.push_back({from_idx, to_idx, QUIET});
                }
            }
        }
        
        // Capturing: straight forward
        int new_row = row + direction;
        int new_col = col;
        if (in_bounds(layer, new_row, new_col)) {
            int to_idx = pos_to_index(layer, new_row, new_col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
        
        // Capturing: move down to middle board
        int new_layer = 1;
        if (in_bounds(new_layer, row, col)) {
            int to_idx = pos_to_index(new_layer, row, col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
    } else if (layer == 1) {
        // On middle board, allow quiet move back to top board if empty
        int new_layer = 0;
        if (in_bounds(new_layer, row, col) && board[pos_to_index(new_layer, row, col)] == EMPTY) {
            moves.push_back({from_idx, pos_to_index(new_layer, row, col), QUIET});
        }
        
        // Allow moves to designated home cells
        int target_layer = 0;
        if (color == Color::GOLD) {
            int home_row = BOARD_ROWS - 1; // row 7
            for (int c = 0; c < BOARD_COLS; c += 2) {
                if (in_bounds(target_layer, home_row, c) && 
                    board[pos_to_index(target_layer, home_row, c)] == EMPTY) {
                    moves.push_back({from_idx, pos_to_index(target_layer, home_row, c), QUIET});
                }
            }
        } else {
            int home_row = 0;
            for (int c = 1; c < BOARD_COLS; c += 2) {
                if (in_bounds(target_layer, home_row, c) && 
                    board[pos_to_index(target_layer, home_row, c)] == EMPTY) {
                    moves.push_back({from_idx, pos_to_index(target_layer, home_row, c), QUIET});
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_griffin_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 0) {
        // Extended knight moves
        std::array<std::pair<int,int>, 8> offsets = {{
            {3,2}, {3,-2}, {-3,2}, {-3,-2},
            {2,3}, {2,-3}, {-2,3}, {-2,-3}
        }};
        
        for (auto [dr, dc] : offsets) {
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
        
        // Diagonal moves to middle board
        int new_layer = 1;
        for (int dr : {-1, 1}) {
            for (int dc : {-1, 1}) {
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(new_layer, new_row, new_col)) {
                    int to_idx = pos_to_index(new_layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
    } else if (layer == 1) {
        // Diagonal moves on middle board
        for (int dr : {-1, 1}) {
            for (int dc : {-1, 1}) {
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(layer, new_row, new_col)) {
                    int to_idx = pos_to_index(layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
        
        // Diagonal moves to top board
        int new_layer = 0;
        for (int dr : {-1, 1}) {
            for (int dc : {-1, 1}) {
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(new_layer, new_row, new_col)) {
                    int to_idx = pos_to_index(new_layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_dragon_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 0) {
        return moves;
    }
    
    // King-like moves
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    }
    
    // Bishop-like sliding diagonal moves
    std::array<std::pair<int,int>, 4> diagonals = {{{-1,-1}, {-1,1}, {1,-1}, {1,1}}};
    for (auto [dr, dc] : diagonals) {
        int r = row;
        int c = col;
        while (true) {
            r += dr;
            c += dc;
            if (!in_bounds(layer, r, c)) break;
            int to_idx = pos_to_index(layer, r, c);
            if (board[to_idx] == EMPTY) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            } else {
                if (IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
                break;
            }
        }
    }
    
    // "Capture from afar" moves - Dragons can shoot DOWN to other layers
    // From Sky (layer 0) to Ground (layer 1)
    int target_layer = 1;
    if (in_bounds(target_layer, row, col)) {
        int to_idx = pos_to_index(target_layer, row, col);
        if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
            moves.push_back({from_idx, to_idx, AFAR});
        }
    }
    
    // Adjacent squares on Ground board
    std::array<std::pair<int,int>, 4> orthogonals = {{{0,1}, {0,-1}, {1,0}, {-1,0}}};
    for (auto [dr, dc] : orthogonals) {
        int new_row = row + dr;
        int new_col = col + dc;
        if (in_bounds(target_layer, new_row, new_col)) {
            int to_idx = pos_to_index(target_layer, new_row, new_col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AFAR});
            }
        }
    }
    
    // From Sky (layer 0) to Underworld (layer 2) - same position
    target_layer = 2;
    if (in_bounds(target_layer, row, col)) {
        int to_idx = pos_to_index(target_layer, row, col);
        if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
            moves.push_back({from_idx, to_idx, AFAR});
        }
    }
    
    // Adjacent squares on Underworld board
    for (auto [dr, dc] : orthogonals) {
        int new_row = row + dr;
        int new_col = col + dc;
        if (in_bounds(target_layer, new_row, new_col)) {
            int to_idx = pos_to_index(target_layer, new_row, new_col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AFAR});
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_oliphant_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 1) {
        return moves;
    }
    
    // Rook-like sliding moves
    std::array<std::pair<int,int>, 4> directions = {{{1,0}, {-1,0}, {0,1}, {0,-1}}};
    for (auto [dr, dc] : directions) {
        int r = row;
        int c = col;
        while (true) {
            r += dr;
            c += dc;
            if (!in_bounds(layer, r, c)) break;
            int to_idx = pos_to_index(layer, r, c);
            if (board[to_idx] == EMPTY) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            } else {
                if (IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
                break;
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_unicorn_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 1) {
        return moves;
    }
    
    // Knight moves
    std::array<std::pair<int,int>, 8> offsets = {{
        {2,1}, {2,-1}, {-2,1}, {-2,-1},
        {1,2}, {1,-2}, {-1,2}, {-1,-2}
    }};
    
    for (auto [dr, dc] : offsets) {
        int new_row = row + dr;
        int new_col = col + dc;
        if (in_bounds(layer, new_row, new_col)) {
            int to_idx = pos_to_index(layer, new_row, new_col);
            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_hero_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 1) {
        // Move 1 or 2 cells diagonally
        for (int dr : {-2, -1, 1, 2}) {
            for (int dc : {-2, -1, 1, 2}) {
                if (std::abs(dr) == std::abs(dc)) {
                    int new_row = row + dr;
                    int new_col = col + dc;
                    if (in_bounds(layer, new_row, new_col)) {
                        int to_idx = pos_to_index(layer, new_row, new_col);
                        if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                            moves.push_back({from_idx, to_idx, AMBIGUOUS});
                        }
                    }
                }
            }
        }
        
        // Move to top or bottom board via diagonal
        for (int target_layer : {0, 2}) {
            for (int dr : {-1, 1}) {
                for (int dc : {-1, 1}) {
                    int new_row = row + dr;
                    int new_col = col + dc;
                    if (in_bounds(target_layer, new_row, new_col)) {
                        int to_idx = pos_to_index(target_layer, new_row, new_col);
                        if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                            moves.push_back({from_idx, to_idx, AMBIGUOUS});
                        }
                    }
                }
            }
        }
    } else {
        // On top or bottom board, move to middle board via diagonal
        int target_layer = 1;
        for (int dr : {-1, 1}) {
            for (int dc : {-1, 1}) {
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(target_layer, new_row, new_col)) {
                    int to_idx = pos_to_index(target_layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_thief_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 1) {
        return moves;
    }
    
    // Bishop-like sliding diagonal moves
    std::array<std::pair<int,int>, 4> diagonals = {{{-1,-1}, {-1,1}, {1,-1}, {1,1}}};
    for (auto [dr, dc] : diagonals) {
        int r = row;
        int c = col;
        while (true) {
            r += dr;
            c += dc;
            if (!in_bounds(layer, r, c)) break;
            int to_idx = pos_to_index(layer, r, c);
            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            }
            if (board[to_idx] != EMPTY) break;
        }
    }
    
    return moves;
}

std::vector<Move> generate_cleric_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    // King-like moves on current layer
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    }
    
    // Inter-layer moves
    if (layer == 0) {
        int new_layer = 1;
        if (in_bounds(new_layer, row, col)) {
            int to_idx = pos_to_index(new_layer, row, col);
            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            }
        }
    } else if (layer == 1) {
        for (int target_layer : {0, 2}) {
            if (in_bounds(target_layer, row, col)) {
                int to_idx = pos_to_index(target_layer, row, col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    } else if (layer == 2) {
        int new_layer = 1;
        if (in_bounds(new_layer, row, col)) {
            int to_idx = pos_to_index(new_layer, row, col);
            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_mage_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 1) {
        // Queen-like sliding moves on middle board
        std::array<std::pair<int,int>, 8> directions = {{
            {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
        }};
        
        for (auto [dr, dc] : directions) {
            int r = row;
            int c = col;
            while (true) {
                r += dr;
                c += dc;
                if (!in_bounds(layer, r, c)) break;
                int to_idx = pos_to_index(layer, r, c);
                if (board[to_idx] == EMPTY) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                } else {
                    if (IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                    break;
                }
            }
        }
        
        // Vertical inter-layer moves
        for (int d_layer : {-1, 1}) {
            int new_layer = layer + d_layer;
            if (in_bounds(new_layer, row, col)) {
                int to_idx = pos_to_index(new_layer, row, col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    } else {
        // On top or bottom board, limited moves
        std::array<std::pair<int,int>, 4> orthogonals = {{{0,1}, {0,-1}, {1,0}, {-1,0}}};
        for (auto [dr, dc] : orthogonals) {
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
        
        // Can move 1 or 2 cells vertically
        for (int d : {-2, -1, 1, 2}) {
            int new_row = row + d;
            if (in_bounds(layer, new_row, col)) {
                int to_idx = pos_to_index(layer, new_row, col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_king_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 1) {
        // King moves on middle board
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(layer, new_row, new_col)) {
                    int to_idx = pos_to_index(layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
        
        // Vertical moves to top or bottom board
        for (int d_layer : {-1, 1}) {
            int new_layer = layer + d_layer;
            if (in_bounds(new_layer, row, col)) {
                int to_idx = pos_to_index(new_layer, row, col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    } else {
        // On top or bottom board, can only move to middle
        int new_layer = 1;
        if (in_bounds(new_layer, row, col)) {
            int to_idx = pos_to_index(new_layer, row, col);
            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, AMBIGUOUS});
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_paladin_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 1) {
        // King-like moves
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(layer, new_row, new_col)) {
                    int to_idx = pos_to_index(layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
        
        // Knight moves
        std::array<std::pair<int,int>, 8> knight_offsets = {{
            {2,1}, {2,-1}, {-2,1}, {-2,-1},
            {1,2}, {1,-2}, {-1,2}, {-1,-2}
        }};
        
        for (auto [dr, dc] : knight_offsets) {
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                    moves.push_back({from_idx, to_idx, AMBIGUOUS});
                }
            }
        }
    } else {
        // On top or bottom board, king-like moves
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int new_row = row + dr;
                int new_col = col + dc;
                if (in_bounds(layer, new_row, new_col)) {
                    int to_idx = pos_to_index(layer, new_row, new_col);
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    }
                }
            }
        }
        
        // 3D knight moves (unblockable)
        for (int d_layer = -2; d_layer <= 2; ++d_layer) {
            if (d_layer == 0) continue;
            for (int d_row = -2; d_row <= 2; ++d_row) {
                for (int d_col = -2; d_col <= 2; ++d_col) {
                    std::array<int, 3> diffs = {std::abs(d_layer), std::abs(d_row), std::abs(d_col)};
                    std::sort(diffs.begin(), diffs.end());
                    if (diffs[0] == 0 && diffs[1] == 1 && diffs[2] == 2) {
                        int new_layer = layer + d_layer;
                        int new_row = row + d_row;
                        int new_col = col + d_col;
                        if (in_bounds(new_layer, new_row, new_col)) {
                            int to_idx = pos_to_index(new_layer, new_row, new_col);
                            if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                                moves.push_back({from_idx, to_idx, THREED});
                            }
                        }
                    }
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_warrior_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 1) {
        return moves;
    }
    
    int direction = (color == Color::GOLD) ? -1 : 1;
    int new_row = row + direction;
    
    // Forward quiet move
    if (in_bounds(layer, new_row, col) && board[pos_to_index(layer, new_row, col)] == EMPTY) {
        moves.push_back({from_idx, pos_to_index(layer, new_row, col), QUIET});
    }
    
    // Diagonal captures
    for (int dc : {-1, 1}) {
        int new_col = col + dc;
        if (in_bounds(layer, new_row, new_col)) {
            int to_idx = pos_to_index(layer, new_row, new_col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_basilisk_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 2) {
        return moves;
    }
    
    int direction = (color == Color::GOLD) ? -1 : 1;
    
    // Forward vertical move
    int new_row = row + direction;
    if (in_bounds(layer, new_row, col)) {
        int to_idx = pos_to_index(layer, new_row, col);
        if (board[to_idx] == EMPTY) {
            moves.push_back({from_idx, to_idx, QUIET});
        } else if (IS_ENEMY(board[to_idx], color)) {
            moves.push_back({from_idx, to_idx, CAPTURE});
        }
    }
    
    // Diagonal forward moves
    for (int dc : {-1, 1}) {
        int new_col = col + dc;
        if (in_bounds(layer, new_row, new_col)) {
            int to_idx = pos_to_index(layer, new_row, new_col);
            if (board[to_idx] == EMPTY) {
                moves.push_back({from_idx, to_idx, QUIET});
            } else if (IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
    }
    
    // Backward move (non-capturing only)
    new_row = row - direction;
    if (in_bounds(layer, new_row, col) && board[pos_to_index(layer, new_row, col)] == EMPTY) {
        moves.push_back({from_idx, pos_to_index(layer, new_row, col), QUIET});
    }
    
    return moves;
}

std::vector<Move> generate_elemental_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer == 2) {
        // Orthogonal moves 1 or 2 cells
        std::array<std::pair<int,int>, 4> orthogonals = {{{1,0}, {-1,0}, {0,1}, {0,-1}}};
        for (auto [dr, dc] : orthogonals) {
            for (int dist = 1; dist <= 2; ++dist) {
                int new_row = row + dr * dist;
                int new_col = col + dc * dist;
                if (!in_bounds(layer, new_row, new_col)) break;
                int to_idx = pos_to_index(layer, new_row, new_col);
                
                if (dist == 1) {
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    } else {
                        break;
                    }
                } else {
                    int inter_idx = pos_to_index(layer, row + dr, col + dc);
                    if (board[inter_idx] != EMPTY) break;
                    if (board[to_idx] == EMPTY || IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, AMBIGUOUS});
                    } else {
                        break;
                    }
                }
            }
        }
        
        // Diagonal quiet moves
        std::array<std::pair<int,int>, 4> diagonals = {{{-1,-1}, {-1,1}, {1,-1}, {1,1}}};
        for (auto [dr, dc] : diagonals) {
            int new_row = row + dr;
            int new_col = col + dc;
            if (in_bounds(layer, new_row, new_col)) {
                int to_idx = pos_to_index(layer, new_row, new_col);
                if (board[to_idx] == EMPTY) {
                    moves.push_back({from_idx, to_idx, QUIET});
                }
            }
        }
        
        // Teleport capture to middle board
        for (auto [dr, dc] : orthogonals) {
            int inter_row = row + dr;
            int inter_col = col + dc;
            int target_layer = 1;
            if (in_bounds(layer, inter_row, inter_col) && 
                board[pos_to_index(layer, inter_row, inter_col)] == EMPTY) {
                int to_idx = pos_to_index(target_layer, row + dr, col + dc);
                if (in_bounds(target_layer, row + dr, col + dc)) {
                    moves.push_back({from_idx, to_idx, CAPTURE});
                }
            }
        }
    } else if (layer == 1) {
        // Teleport from middle to bottom board
        std::array<std::pair<int,int>, 4> orthogonals = {{{1,0}, {-1,0}, {0,1}, {0,-1}}};
        for (auto [dr, dc] : orthogonals) {
            int inter_row = row + dr;
            int inter_col = col + dc;
            int target_layer = 2;
            if (in_bounds(layer, inter_row, inter_col) && 
                board[pos_to_index(layer, inter_row, inter_col)] == EMPTY) {
                if (in_bounds(target_layer, row + dr, col + dc)) {
                    int to_idx = pos_to_index(target_layer, row + dr, col + dc);
                    if (board[to_idx] == EMPTY) {
                        moves.push_back({from_idx, to_idx, QUIET});
                    } else if (IS_ENEMY(board[to_idx], color)) {
                        moves.push_back({from_idx, to_idx, CAPTURE});
                    }
                }
            }
        }
    }
    
    return moves;
}

std::vector<Move> generate_dwarf_moves(const Position& pos, const Board& board, Color color) {
    std::vector<Move> moves;
    auto [layer, row, col] = pos;
    int from_idx = pos_to_index(layer, row, col);
    
    if (layer != 1 && layer != 2) {
        return moves;
    }
    
    int direction = (color == Color::GOLD) ? -1 : 1;
    int new_row = row + direction;
    
    // Forward quiet move
    if (in_bounds(layer, new_row, col) && board[pos_to_index(layer, new_row, col)] == EMPTY) {
        moves.push_back({from_idx, pos_to_index(layer, new_row, col), QUIET});
    }
    
    // Sideways quiet moves
    for (int dc : {-1, 1}) {
        if (in_bounds(layer, row, col + dc) && board[pos_to_index(layer, row, col + dc)] == EMPTY) {
            moves.push_back({from_idx, pos_to_index(layer, row, col + dc), QUIET});
        }
    }
    
    // Diagonal captures
    for (int dc : {-1, 1}) {
        if (in_bounds(layer, new_row, col + dc)) {
            int to_idx = pos_to_index(layer, new_row, col + dc);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
    }
    
    // Inter-layer moves
    if (layer == 2) {
        int target_layer = 1;
        if (in_bounds(target_layer, row, col)) {
            int to_idx = pos_to_index(target_layer, row, col);
            if (board[to_idx] != EMPTY && IS_ENEMY(board[to_idx], color)) {
                moves.push_back({from_idx, to_idx, CAPTURE});
            }
        }
    } else if (layer == 1) {
        int target_layer = 2;
        if (in_bounds(target_layer, row, col) && 
            board[pos_to_index(target_layer, row, col)] == EMPTY) {
            moves.push_back({from_idx, pos_to_index(target_layer, row, col), QUIET});
        }
    }
    
    // Reverse move if at edge
    if (!in_bounds(layer, row + direction, col)) {
        int reverse = -direction;
        int reverse_row = row + reverse;
        if (in_bounds(layer, reverse_row, col) && 
            board[pos_to_index(layer, reverse_row, col)] == EMPTY) {
            moves.push_back({from_idx, pos_to_index(layer, reverse_row, col), QUIET});
        }
    }
    
    return moves;
}

#undef IS_ENEMY

} // namespace dragonchess
