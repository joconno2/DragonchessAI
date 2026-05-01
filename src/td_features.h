#pragma once

#include "game.h"
#include <vector>

namespace dragonchess {

// ---------------------------------------------------------------------------
// NNUE-style feature vector: piece-square table + strategic features.
//
// Piece-square block [0 .. 4031]:
//   14 piece types (King excluded) x 288 squares = 4032 binary features.
//   Index = piece_type_idx * 288 + square_idx
//   Gold piece on square: +1.0, Scarlet piece: -1.0
//
//   piece_type_idx mapping (from abs(piece) enum value):
//     Sylph(1)=0, Griffin(2)=1, Dragon(3)=2, Oliphant(4)=3, Unicorn(5)=4,
//     Hero(6)=5, Thief(7)=6, Cleric(8)=7, Mage(9)=8, Paladin(11)=9,
//     Warrior(12)=10, Basilisk(13)=11, Elemental(14)=12, Dwarf(15)=13
//
//   square_idx = layer * 96 + row * 12 + col  (0..287)
//
// Strategic features [4032 .. 4059]:
//   [4032] Gold frozen piece count
//   [4033] Scarlet frozen piece count
//   [4034] Gold frozen material value
//   [4035] Scarlet frozen material value
//   [4036] Gold pieces on Ground directly above enemy Basilisk
//   [4037] Scarlet pieces on Ground directly above enemy Basilisk
//   [4038] Friendly pieces within Chebyshev dist 1 of Gold king
//   [4039] Enemy pieces within Chebyshev dist 1 of Gold king
//   [4040] Enemy pieces within Chebyshev dist 2 of Gold king (excl dist 1)
//   [4041] Friendly pieces within Chebyshev dist 1 of Scarlet king
//   [4042] Enemy pieces within Chebyshev dist 1 of Scarlet king
//   [4043] Enemy pieces within Chebyshev dist 2 of Scarlet king (excl dist 1)
//   [4044] Gold king on home rank (row 7: 1.0/0.0)
//   [4045] Scarlet king on home rank (row 0: 1.0/0.0)
//   [4046] Total material on board / initial total
//   [4047] Total pieces on board / 52
//   [4048] Move count / 200
//   [4049] No-capture counter / 250
//   [4050] Gold isolated Warriors
//   [4051] Gold doubled Warriors
//   [4052] Gold connected Warriors
//   [4053] Scarlet isolated Warriors
//   [4054] Scarlet doubled Warriors
//   [4055] Scarlet connected Warriors
//   [4056] Gold cross-level capable piece count
//   [4057] Scarlet cross-level capable piece count
//   [4058] Gold material advantage (gold_total - scarlet_total, normalized)
//   [4059] Repetition count of current position / 3
// ---------------------------------------------------------------------------

constexpr int NUM_PIECE_TYPES = 14;       // King excluded
constexpr int NUM_KING_BUCKETS = 8;       // 4 col groups x 2 row halves on ground board
constexpr int NUM_PIECE_SQUARE = NUM_KING_BUCKETS * NUM_PIECE_TYPES * TOTAL_SQUARES;  // 8 * 14 * 288 = 32256
constexpr int NUM_STRATEGIC = 28;
constexpr int NUM_TD_FEATURES = NUM_PIECE_SQUARE + NUM_STRATEGIC;  // 32284

// Map a ground-board king position to a bucket (0..7).
// 4 column groups (0-2, 3-5, 6-8, 9-11) x 2 row halves (0-3, 4-7).
// If king is not on ground board, returns 0 (fallback).
inline int king_to_bucket(int king_sq) {
    if (king_sq < 0) return 0;
    constexpr int layer_size = BOARD_ROWS * BOARD_COLS;  // 96
    int layer = king_sq / layer_size;
    if (layer != 1) return 0;  // king should be on ground (layer 1)
    int rem = king_sq % layer_size;
    int row = rem / BOARD_COLS;
    int col = rem % BOARD_COLS;
    int col_group = col / 3;            // 0,1,2,3
    int row_half = (row >= 4) ? 1 : 0;  // 0 or 1
    return row_half * 4 + col_group;     // 0..7
}

// Map abs(piece) enum value (1..15, skip 10) to piece-type index (0..13).
// Returns -1 for EMPTY (0) and KING (10).
inline int piece_type_to_idx(int abs_piece) {
    static constexpr int TABLE[16] = {
        -1,  // 0: EMPTY
         0,  // 1: SYLPH
         1,  // 2: GRIFFIN
         2,  // 3: DRAGON
         3,  // 4: OLIPHANT
         4,  // 5: UNICORN
         5,  // 6: HERO
         6,  // 7: THIEF
         7,  // 8: CLERIC
         8,  // 9: MAGE
        -1,  // 10: KING
         9,  // 11: PALADIN
        10,  // 12: WARRIOR
        11,  // 13: BASILISK
        12,  // 14: ELEMENTAL
        13,  // 15: DWARF
    };
    return (abs_piece >= 0 && abs_piece <= 15) ? TABLE[abs_piece] : -1;
}

// Piece values for strategic features (indexed by abs piece type 0..15).
inline constexpr float PIECE_VAL[16] = {
    0.0f, 1.0f, 5.0f, 8.0f, 5.0f, 2.5f, 4.5f, 4.0f,
    9.0f, 11.0f, 0.0f, 10.0f, 1.0f, 3.0f, 4.0f, 2.0f,
};

// Initial total material value (both sides combined, no kings).
inline constexpr float INITIAL_TOTAL_MATERIAL = 2.0f * (
    6*1.0f + 2*5.0f + 1*8.0f + 2*5.0f + 2*2.5f + 2*4.5f + 2*4.0f +
    2*9.0f + 1*11.0f + 1*10.0f + 12*1.0f + 2*3.0f + 2*4.0f + 6*2.0f
);

// Sparse feature representation for efficient serialization.
struct SparseFeature {
    int index;
    float value;
};

// Extract sparse feature vector from the current game state.
// Returns only non-zero features. Much more efficient for 4060-dim
// vectors with ~80 non-zero entries.
std::vector<SparseFeature> extract_td_features_sparse(const Game& game);

// Reconstruct dense vector from sparse (for evaluation).
std::vector<float> extract_td_features(const Game& game);

} // namespace dragonchess
