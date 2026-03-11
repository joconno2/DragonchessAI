#include "game.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

namespace dragonchess {

// Direct array indexed by absolute piece type [0..15]; 0 = EMPTY (nullptr).
// Eliminates hash-table overhead in get_all_moves (called at every search node).
using MoveGenerator = std::vector<Move>(*)(const Position&, const Board&, Color);

static const MoveGenerator move_gen_table[16] = {
    nullptr,                  // 0 = EMPTY
    generate_sylph_moves,     // 1
    generate_griffin_moves,   // 2
    generate_dragon_moves,    // 3
    generate_oliphant_moves,  // 4
    generate_unicorn_moves,   // 5
    generate_hero_moves,      // 6
    generate_thief_moves,     // 7
    generate_cleric_moves,    // 8
    generate_mage_moves,      // 9
    generate_king_moves,      // 10
    generate_paladin_moves,   // 11
    generate_warrior_moves,   // 12
    generate_basilisk_moves,  // 13
    generate_elemental_moves, // 14
    generate_dwarf_moves,     // 15
};

Game::Game() 
    : board(create_initial_board())
    , current_turn(Color::GOLD)
    , no_capture_count(0)
    , game_over(false)
    , winner("None")
{
    frozen.fill(false);
}

std::vector<Move> Game::get_all_moves() const {
    std::vector<Move> moves_list;
    
    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        int16_t piece = board[idx];
        if (piece == EMPTY) continue;
        
        // Skip frozen pieces
        if (frozen[idx]) continue;
        
        // Check if piece belongs to current player
        bool is_current_player = (current_turn == Color::GOLD && piece > 0) ||
                                 (current_turn == Color::SCARLET && piece < 0);
        
        if (!is_current_player) continue;
        
        // Get piece type
        int abs_code = std::abs(piece);
        if (abs_code < 1 || abs_code > 15) continue;
        MoveGenerator gen = move_gen_table[abs_code];
        if (!gen) continue;

        // Generate moves for this piece
        Position pos = index_to_pos(idx);
        std::vector<Move> candidate_moves = gen(pos, board, current_turn);
        
        // Validate moves
        for (const auto& move : candidate_moves) {
            auto [from_idx, to_idx, flag] = move;
            
            // Validate QUIET moves (destination must be empty)
            if (flag == QUIET && board[to_idx] != EMPTY) {
                continue;
            }
            
            // Validate CAPTURE/AFAR moves (destination must have enemy)
            if (flag == CAPTURE || flag == AFAR) {
                if (board[to_idx] == EMPTY) continue;
                
                bool is_enemy = (current_turn == Color::GOLD && board[to_idx] < 0) ||
                               (current_turn == Color::SCARLET && board[to_idx] > 0);
                
                if (!is_enemy) continue;
            }
            
            moves_list.push_back(move);
        }
    }
    
    return moves_list;
}

std::vector<Move> Game::get_legal_moves_for(int from_index) const {
    std::vector<Move> all_moves = get_all_moves();
    std::vector<Move> result;
    
    for (const auto& move : all_moves) {
        if (std::get<0>(move) == from_index) {
            result.push_back(move);
        }
    }
    
    return result;
}

std::string Game::move_to_algebraic(const Move& move, int16_t moving_piece) const {
    auto [from_idx, to_idx, flag] = move;
    char piece_char = piece_letter(moving_piece);
    const char* separator = (flag == CAPTURE || flag == AFAR) ? "x" : "-";
    
    return std::string(1, piece_char) + index_to_algebraic(from_idx) + 
           separator + index_to_algebraic(to_idx);
}

void Game::make_move(const Move& move) {
    auto [from_idx, to_idx, flag] = move;
    int16_t moving_piece = board[from_idx];
    
    // Save current state for undo
    GameState current_state;
    current_state.board = board;
    current_state.current_turn = current_turn;
    current_state.no_capture_count = no_capture_count;
    current_state.frozen = frozen;
    undo_stack.push_back(current_state);
    
    // Record move in algebraic notation
    std::string move_alg = move_to_algebraic(move, moving_piece);
    move_notations.push_back(move_alg);
    game_log.push_back(move);
    
    bool capture_occurred = false;
    
    // Handle captures
    if (flag == CAPTURE || flag == AFAR) {
        if (board[to_idx] != EMPTY) {
            capture_occurred = true;
        }
        board[to_idx] = EMPTY;
    }
    
    // Special case: AFAR moves with Dragon don't move the piece
    if (flag == AFAR && std::abs(moving_piece) == 3) {
        // Dragon stays in place
    } else {
        board[to_idx] = moving_piece;
        board[from_idx] = EMPTY;
    }
    
    // Update capture counter
    if (capture_occurred) {
        no_capture_count = 0;
    } else {
        no_capture_count++;
    }
    
    // Record state history
    state_history.push_back(board_state_hash());
    
    // Track last move for highlighting
    last_move = move;
    
    // Clear redo stack when a new move is made
    redo_stack.clear();
    
    // Switch turns
    current_turn = (current_turn == Color::GOLD) ? Color::SCARLET : Color::GOLD;
}

bool Game::undo_move() {
    // Can't undo if no moves or if game is over
    if (game_log.empty() || undo_stack.empty()) {
        return false;
    }
    
    // Save current state to redo stack
    Move last_move_made = game_log.back();
    redo_stack.push_back(last_move_made);
    
    // Restore previous state
    GameState prev_state = undo_stack.back();
    undo_stack.pop_back();
    
    board = prev_state.board;
    current_turn = prev_state.current_turn;
    no_capture_count = prev_state.no_capture_count;
    frozen = prev_state.frozen;
    
    // Remove from logs
    game_log.pop_back();
    move_notations.pop_back();
    if (!state_history.empty()) {
        state_history.pop_back();
    }
    
    // Update last move
    if (!game_log.empty()) {
        last_move = game_log.back();
    } else {
        last_move.reset();
    }
    
    // Reset game over state
    game_over = false;
    winner = "";
    
    return true;
}

bool Game::redo_move() {
    // Can't redo if redo stack is empty
    if (redo_stack.empty()) {
        return false;
    }
    
    // Get move to redo
    Move move = redo_stack.back();
    redo_stack.pop_back();
    
    // Make the move (this will clear redo stack, so we need to handle it differently)
    // Save the redo stack temporarily
    auto saved_redo = redo_stack;
    
    make_move(move);
    
    // Restore redo stack (minus the move we just redid)
    redo_stack = saved_redo;
    
    return true;
}

void Game::update() {
    // Reset frozen pieces
    frozen.fill(false);
    
    // Check for Basilisk freezing effect
    for (int row = 0; row < BOARD_ROWS; ++row) {
        for (int col = 0; col < BOARD_COLS; ++col) {
            int idx_bottom = pos_to_index(2, row, col);
            int16_t piece = board[idx_bottom];
            
            // Check if it's a Basilisk
            if (piece == GOLD_BASILISK || piece == SCARLET_BASILISK) {
                int idx_middle = pos_to_index(1, row, col);
                int16_t target = board[idx_middle];
                
                // Freeze enemy piece on middle board
                if (target != EMPTY && (target * piece < 0)) {
                    frozen[idx_middle] = true;
                }
            }
        }
    }
    
    // Check for draw by 250 moves without capture
    if (no_capture_count >= 250) {
        game_over = true;
        winner = "Draw";
    }
    
    // Check for threefold repetition (or more)
    if (state_history.size() >= 3) {
        uint64_t current_state = state_history.back();
        int repetition_count = 0;
        for (uint64_t h : state_history) {
            if (h == current_state) repetition_count++;
        }
        if (repetition_count >= 3) {
            game_over = true;
            winner = "Draw";
        }
    }
    
    // Check for king capture
    bool gold_king_exists = false;
    bool scarlet_king_exists = false;
    
    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        if (board[idx] == GOLD_KING) {
            gold_king_exists = true;
        } else if (board[idx] == SCARLET_KING) {
            scarlet_king_exists = true;
        }
    }
    
    if (!gold_king_exists) {
        game_over = true;
        winner = "Scarlet";
    } else if (!scarlet_king_exists) {
        game_over = true;
        winner = "Gold";
    }
}

char Game::piece_letter(int16_t piece) const {
    static const std::unordered_map<int, char> mapping = {
        {1, 'S'}, {2, 'G'}, {3, 'R'}, {4, 'O'}, {5, 'U'},
        {6, 'H'}, {7, 'T'}, {8, 'C'}, {9, 'M'}, {10, 'K'},
        {11, 'P'}, {12, 'W'}, {13, 'B'}, {14, 'E'}, {15, 'D'}
    };
    
    int abs_piece = std::abs(piece);
    auto it = mapping.find(abs_piece);
    if (it == mapping.end()) return '?';
    
    char letter = it->second;
    return (piece > 0) ? letter : static_cast<char>(std::tolower(letter));
}

std::string Game::index_to_algebraic(int idx) const {
    auto [layer, row, col] = index_to_pos(idx);
    int board_num = layer + 1;
    char file_letter = 'a' + col;
    int rank = BOARD_ROWS - row;
    
    std::ostringstream oss;
    oss << board_num << file_letter << rank;
    return oss.str();
}

uint64_t Game::board_state_hash() const {
    // Fast 64-bit hash of board + turn for repetition detection.
    // Uses the same Zobrist-style accumulation as the AlphaBeta TT hash.
    uint64_t hash = 0;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        int16_t piece = board[i];
        if (piece != EMPTY) {
            // Interleave square index and piece value for collision resistance
            uint64_t v = (static_cast<uint64_t>(static_cast<uint16_t>(piece)) << 10) | i;
            hash ^= v * 0x9e3779b97f4a7c15ULL;  // Fibonacci hashing
            hash = (hash << 13) | (hash >> 51);  // Rotation mix
        }
    }
    hash ^= (current_turn == Color::GOLD) ? 0xDEADBEEFCAFEBABEULL : 0;
    return hash;
}

} // namespace dragonchess
