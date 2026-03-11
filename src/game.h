#pragma once

#include "bitboard.h"
#include "moves.h"
#include <vector>
#include <string>
#include <array>
#include <unordered_set>
#include <optional>
#include <cstdint>

namespace dragonchess {

class Game {
public:
    Game();
    
    // Get all legal moves for current player
    std::vector<Move> get_all_moves() const;
    
    // Get legal moves for a specific piece
    std::vector<Move> get_legal_moves_for(int from_index) const;
    
    // Make a move on the board
    void make_move(const Move& move);
    
    // Undo last move (returns false if no moves to undo)
    bool undo_move();
    
    // Redo previously undone move (returns false if no moves to redo)
    bool redo_move();
    
    // Update game state (check for frozen pieces, win conditions, etc.)
    void update();
    
    // Convert move to algebraic notation
    std::string move_to_algebraic(const Move& move, int16_t moving_piece) const;
    
    // Convert index to algebraic notation
    std::string index_to_algebraic(int index) const;
    
    // Get piece letter for notation
    char piece_letter(int16_t piece) const;
    
    // Compute board state hash for repetition detection
    uint64_t board_state_hash() const;
    
    // Public members for game state
    Board board;
    Color current_turn;
    std::vector<uint64_t> state_history;
    std::vector<Move> game_log;
    std::vector<std::string> move_notations;
    std::optional<Move> last_move;  // Track last move for highlighting
    int no_capture_count;
    bool game_over;
    std::string winner;
    std::array<bool, TOTAL_SQUARES> frozen;
    
    // Undo/Redo support
    struct GameState {
        Board board;
        Color current_turn;
        int no_capture_count;
        std::array<bool, TOTAL_SQUARES> frozen;
    };
    std::vector<GameState> undo_stack;
    std::vector<Move> redo_stack;
    
    // Timer support
    struct Timer {
        int gold_time_ms = 600000;      // 10 minutes default
        int scarlet_time_ms = 600000;   // 10 minutes default
        int increment_ms = 0;           // No increment by default
        bool enabled = false;
        bool gold_time_up = false;
        bool scarlet_time_up = false;
    };
    Timer timer;
};

} // namespace dragonchess
