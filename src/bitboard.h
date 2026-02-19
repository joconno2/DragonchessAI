#pragma once

#include <cstdint>
#include <array>
#include <tuple>

namespace dragonchess {

// Board dimensions
constexpr int NUM_BOARDS = 3;
constexpr int BOARD_ROWS = 8;
constexpr int BOARD_COLS = 12;
constexpr int TOTAL_SQUARES = NUM_BOARDS * BOARD_ROWS * BOARD_COLS;

// Piece constants (positive for Gold; negative for Scarlet)
enum Piece : int16_t {
    EMPTY = 0,
    GOLD_SYLPH = 1,
    GOLD_GRIFFIN = 2,
    GOLD_DRAGON = 3,
    GOLD_OLIPHANT = 4,
    GOLD_UNICORN = 5,
    GOLD_HERO = 6,
    GOLD_THIEF = 7,
    GOLD_CLERIC = 8,
    GOLD_MAGE = 9,
    GOLD_KING = 10,
    GOLD_PALADIN = 11,
    GOLD_WARRIOR = 12,
    GOLD_BASILISK = 13,
    GOLD_ELEMENTAL = 14,
    GOLD_DWARF = 15,
    
    SCARLET_SYLPH = -1,
    SCARLET_GRIFFIN = -2,
    SCARLET_DRAGON = -3,
    SCARLET_OLIPHANT = -4,
    SCARLET_UNICORN = -5,
    SCARLET_HERO = -6,
    SCARLET_THIEF = -7,
    SCARLET_CLERIC = -8,
    SCARLET_MAGE = -9,
    SCARLET_KING = -10,
    SCARLET_PALADIN = -11,
    SCARLET_WARRIOR = -12,
    SCARLET_BASILISK = -13,
    SCARLET_ELEMENTAL = -14,
    SCARLET_DWARF = -15
};

// Color enum
enum class Color {
    GOLD,
    SCARLET
};

// Position type (layer, row, col)
using Position = std::tuple<int, int, int>;

// Board type - flat array indexed by layer, row, col
using Board = std::array<int16_t, TOTAL_SQUARES>;

// Convert (layer, row, col) to flat index
inline int pos_to_index(int layer, int row, int col) {
    return layer * (BOARD_ROWS * BOARD_COLS) + row * BOARD_COLS + col;
}

// Convert flat index to (layer, row, col)
inline Position index_to_pos(int index) {
    int layer = index / (BOARD_ROWS * BOARD_COLS);
    int rem = index % (BOARD_ROWS * BOARD_COLS);
    int row = rem / BOARD_COLS;
    int col = rem % BOARD_COLS;
    return {layer, row, col};
}

// Check if position is in bounds
inline bool in_bounds(int layer, int row, int col) {
    return (0 <= layer && layer < NUM_BOARDS) &&
           (0 <= row && row < BOARD_ROWS) &&
           (0 <= col && col < BOARD_COLS);
}

// Create initial board setup
Board create_initial_board();

} // namespace dragonchess
