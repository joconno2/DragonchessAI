#pragma once

#include "bitboard.h"
#include <vector>
#include <tuple>

namespace dragonchess {

// Move flag constants
enum MoveFlag : int {
    QUIET = 0,
    CAPTURE = 1,
    AFAR = 2,
    AMBIGUOUS = 3,
    THREED = 4
};

// Move type: (from_index, to_index, flag)
using Move = std::tuple<int, int, MoveFlag>;

// Move generation functions for each piece type
std::vector<Move> generate_sylph_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_griffin_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_dragon_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_oliphant_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_unicorn_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_hero_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_thief_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_cleric_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_mage_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_king_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_paladin_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_warrior_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_basilisk_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_elemental_moves(const Position& pos, const Board& board, Color color);
std::vector<Move> generate_dwarf_moves(const Position& pos, const Board& board, Color color);

} // namespace dragonchess
