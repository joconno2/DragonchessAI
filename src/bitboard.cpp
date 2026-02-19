#include "bitboard.h"

namespace dragonchess {

Board create_initial_board() {
    Board board{};
    board.fill(EMPTY);
    
    // TOP Board (layer 0)
    board[pos_to_index(0, 0, 2)] = SCARLET_GRIFFIN;
    board[pos_to_index(0, 0, 6)] = SCARLET_DRAGON;
    board[pos_to_index(0, 0, 10)] = SCARLET_GRIFFIN;
    
    for (int col = 0; col < BOARD_COLS; col += 2) {
        board[pos_to_index(0, 1, col)] = SCARLET_SYLPH;
    }
    
    for (int col = 0; col < BOARD_COLS; col += 2) {
        board[pos_to_index(0, 6, col)] = GOLD_SYLPH;
    }
    
    board[pos_to_index(0, 7, 2)] = GOLD_GRIFFIN;
    board[pos_to_index(0, 7, 6)] = GOLD_DRAGON;
    board[pos_to_index(0, 7, 10)] = GOLD_GRIFFIN;
    
    // MIDDLE Board (layer 1)
    std::array<int16_t, 12> scarlet_middle = {
        SCARLET_OLIPHANT, SCARLET_UNICORN, SCARLET_HERO, SCARLET_THIEF,
        SCARLET_CLERIC, SCARLET_MAGE, SCARLET_KING, SCARLET_PALADIN,
        SCARLET_THIEF, SCARLET_HERO, SCARLET_UNICORN, SCARLET_OLIPHANT
    };
    
    for (int col = 0; col < BOARD_COLS; ++col) {
        board[pos_to_index(1, 0, col)] = scarlet_middle[col];
    }
    
    for (int col = 0; col < BOARD_COLS; ++col) {
        board[pos_to_index(1, 1, col)] = SCARLET_WARRIOR;
    }
    
    for (int col = 0; col < BOARD_COLS; ++col) {
        board[pos_to_index(1, 6, col)] = GOLD_WARRIOR;
    }
    
    std::array<int16_t, 12> gold_middle = {
        GOLD_OLIPHANT, GOLD_UNICORN, GOLD_HERO, GOLD_THIEF,
        GOLD_CLERIC, GOLD_MAGE, GOLD_KING, GOLD_PALADIN,
        GOLD_THIEF, GOLD_HERO, GOLD_UNICORN, GOLD_OLIPHANT
    };
    
    for (int col = 0; col < BOARD_COLS; ++col) {
        board[pos_to_index(1, 7, col)] = gold_middle[col];
    }
    
    // BOTTOM Board (layer 2)
    board[pos_to_index(2, 0, 2)] = SCARLET_BASILISK;
    board[pos_to_index(2, 0, 6)] = SCARLET_ELEMENTAL;
    board[pos_to_index(2, 0, 10)] = SCARLET_BASILISK;
    
    for (int col = 1; col < BOARD_COLS; col += 2) {
        board[pos_to_index(2, 1, col)] = SCARLET_DWARF;
    }
    
    board[pos_to_index(2, 7, 2)] = GOLD_BASILISK;
    board[pos_to_index(2, 7, 6)] = GOLD_ELEMENTAL;
    board[pos_to_index(2, 7, 10)] = GOLD_BASILISK;
    
    for (int col = 1; col < BOARD_COLS; col += 2) {
        board[pos_to_index(2, 6, col)] = GOLD_DWARF;
    }
    
    return board;
}

} // namespace dragonchess
