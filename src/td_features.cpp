#include "td_features.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace dragonchess {

// Proxy piece values indexed by piece type (1-15) for level material features.
// Matches BaseAI::piece_values in ai.h (indexed by abs(piece)).
static const float FEAT_PIECE_VAL[16] = {
    0.0f,     // 0: EMPTY
    1.0f,     // 1: SYLPH
    5.0f,     // 2: GRIFFIN
    8.0f,     // 3: DRAGON
    5.0f,     // 4: OLIPHANT
    2.5f,     // 5: UNICORN
    4.5f,     // 6: HERO
    4.0f,     // 7: THIEF
    9.0f,     // 8: CLERIC
    11.0f,    // 9: MAGE
    0.0f,     // 10: KING (excluded from material)
    10.0f,    // 11: PALADIN
    1.0f,     // 12: WARRIOR
    3.0f,     // 13: BASILISK
    4.0f,     // 14: ELEMENTAL
    2.0f,     // 15: DWARF
};

// Piece types included in the 14 material count features (King excluded).
static const int FEAT_PIECE_TYPES[14] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15};

// Pieces that can move across levels (used for cross-level count feature).
static const bool IS_CROSS_LEVEL[16] = {
    false, // 0: EMPTY
    true,  // 1: SYLPH     (Sky native, can descend)
    true,  // 2: GRIFFIN   (Sky+Ground)
    true,  // 3: DRAGON    (long-range across levels)
    false, // 4: OLIPHANT
    false, // 5: UNICORN
    true,  // 6: HERO      (can move across levels)
    false, // 7: THIEF
    false, // 8: CLERIC
    true,  // 9: MAGE      (3D movement)
    false, // 10: KING
    true,  // 11: PALADIN  (cross-level movement)
    false, // 12: WARRIOR
    false, // 13: BASILISK
    false, // 14: ELEMENTAL
    false, // 15: DWARF
};

std::vector<float> extract_td_features(const Game& game) {
    std::vector<float> f(NUM_TD_FEATURES, 0.0f);

    // Per-piece-type counts
    int gold_counts[16] = {};
    int scarlet_counts[16] = {};

    // Per-level accumulators
    float gold_mat[3] = {};
    float scarlet_mat[3] = {};
    int   gold_lev[3] = {};
    int   scarlet_lev[3] = {};

    // Advancement accumulators (Ground = layer 1, Cavern = layer 2)
    float gold_ground_row_sum = 0.0f, gold_ground_count = 0.0f;
    float scarlet_ground_row_sum = 0.0f, scarlet_ground_count = 0.0f;
    float gold_cavern_row_sum = 0.0f, gold_cavern_count = 0.0f;
    float scarlet_cavern_row_sum = 0.0f, scarlet_cavern_count = 0.0f;

    int gold_total = 0, scarlet_total = 0;
    int gold_center = 0, scarlet_center = 0;  // Ground cols 3-8
    int gold_cross = 0, scarlet_cross = 0;
    int gold_king_sq = -1, scarlet_king_sq = -1;

    const int layer_size = BOARD_ROWS * BOARD_COLS;  // 96

    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        int16_t piece = game.board[idx];
        if (piece == EMPTY) continue;

        bool is_gold = (piece > 0);
        int ptype = static_cast<int>(std::abs(piece));
        int layer = idx / layer_size;
        int rem   = idx % layer_size;
        int row   = rem / BOARD_COLS;
        int col   = rem % BOARD_COLS;
        float pval = (ptype < 16) ? FEAT_PIECE_VAL[ptype] : 0.0f;

        if (is_gold) {
            gold_counts[ptype]++;
            gold_total++;
            gold_mat[layer] += pval;
            gold_lev[layer]++;
            if (ptype == 10) gold_king_sq = idx;
            if (ptype < 16 && IS_CROSS_LEVEL[ptype]) gold_cross++;
            if (layer == 1) {
                gold_ground_row_sum += static_cast<float>(row);
                gold_ground_count++;
                if (col >= 3 && col <= 8) gold_center++;
            } else if (layer == 2) {
                gold_cavern_row_sum += static_cast<float>(row);
                gold_cavern_count++;
            }
        } else {
            scarlet_counts[ptype]++;
            scarlet_total++;
            scarlet_mat[layer] += pval;
            scarlet_lev[layer]++;
            if (ptype == 10) scarlet_king_sq = idx;
            if (ptype < 16 && IS_CROSS_LEVEL[ptype]) scarlet_cross++;
            if (layer == 1) {
                scarlet_ground_row_sum += static_cast<float>(row);
                scarlet_ground_count++;
                if (col >= 3 && col <= 8) scarlet_center++;
            } else if (layer == 2) {
                scarlet_cavern_row_sum += static_cast<float>(row);
                scarlet_cavern_count++;
            }
        }

        // Also accumulate frozen counts in-loop (check frozen array)
        // (frozen handled below separately to keep logic clean)
    }

    // --- Fill features ---

    // [0-13] Material count diffs per piece type
    for (int i = 0; i < 14; ++i) {
        int pt = FEAT_PIECE_TYPES[i];
        f[i] = static_cast<float>(gold_counts[pt] - scarlet_counts[pt]);
    }

    // [14-16] Level piece count diffs
    for (int l = 0; l < 3; ++l)
        f[14 + l] = static_cast<float>(gold_lev[l] - scarlet_lev[l]);

    // [17-19] Level material value diffs
    for (int l = 0; l < 3; ++l)
        f[17 + l] = gold_mat[l] - scarlet_mat[l];

    // [20] Gold mean row on Ground (default 3.5 if no pieces)
    f[20] = (gold_ground_count > 0.0f)
            ? (gold_ground_row_sum / gold_ground_count)
            : 3.5f;

    // [21] Scarlet mean advancement on Ground (7 - mean_row)
    f[21] = (scarlet_ground_count > 0.0f)
            ? (7.0f - scarlet_ground_row_sum / scarlet_ground_count)
            : 3.5f;

    // [22] Gold mean row in Cavern
    f[22] = (gold_cavern_count > 0.0f)
            ? (gold_cavern_row_sum / gold_cavern_count)
            : 3.5f;

    // [23] Scarlet mean advancement in Cavern
    f[23] = (scarlet_cavern_count > 0.0f)
            ? (7.0f - scarlet_cavern_row_sum / scarlet_cavern_count)
            : 3.5f;

    // [24-25] King zone threats (Chebyshev distance <= 2, same level)
    auto king_zone_threats = [&](int king_sq, bool count_gold_pieces) -> float {
        if (king_sq < 0) return 0.0f;
        int kl = king_sq / layer_size;
        int kr = (king_sq % layer_size) / BOARD_COLS;
        int kc = king_sq % BOARD_COLS;
        float threats = 0.0f;
        int base = kl * layer_size;
        for (int i = base; i < base + layer_size; ++i) {
            int16_t p = game.board[i];
            if (p == EMPTY) continue;
            bool p_is_gold = (p > 0);
            if (p_is_gold != count_gold_pieces) continue;
            int r = (i % layer_size) / BOARD_COLS;
            int c = i % BOARD_COLS;
            int dist = std::max(std::abs(r - kr), std::abs(c - kc));
            if (dist > 0 && dist <= 2) threats += 1.0f;
        }
        return threats;
    };
    f[24] = king_zone_threats(gold_king_sq, false);    // scarlet near gold king
    f[25] = king_zone_threats(scarlet_king_sq, true);  // gold near scarlet king

    // [26-27] Frozen piece counts
    int gold_frozen = 0, scarlet_frozen = 0;
    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        if (game.frozen[idx]) {
            int16_t p = game.board[idx];
            if (p > 0) gold_frozen++;
            else if (p < 0) scarlet_frozen++;
        }
    }
    f[26] = static_cast<float>(gold_frozen);
    f[27] = static_cast<float>(scarlet_frozen);

    // [28] Cross-level piece count diff
    f[28] = static_cast<float>(gold_cross - scarlet_cross);

    // [29] Ground center control diff (cols 3-8)
    f[29] = static_cast<float>(gold_center - scarlet_center);

    // [30] Total piece count diff, normalized
    f[30] = static_cast<float>(gold_total - scarlet_total) / 26.0f;

    // [31] Gold king row on Ground (0 if king not on Ground)
    if (gold_king_sq >= 0) {
        int kl = gold_king_sq / layer_size;
        int kr = (gold_king_sq % layer_size) / BOARD_COLS;
        f[31] = (kl == 1) ? static_cast<float>(kr) : 0.0f;
    }

    // [32] Scarlet king advancement on Ground: 7-row (0 if king not on Ground)
    if (scarlet_king_sq >= 0) {
        int kl = scarlet_king_sq / layer_size;
        int kr = (scarlet_king_sq % layer_size) / BOARD_COLS;
        f[32] = (kl == 1) ? static_cast<float>(7 - kr) : 0.0f;
    }

    // [33-34] Absolute Sky material
    f[33] = gold_mat[0];
    f[34] = scarlet_mat[0];

    // [35-36] Absolute Cavern material
    f[35] = gold_mat[2];
    f[36] = scarlet_mat[2];

    // [37] Game progress (normalized)
    f[37] = std::min(static_cast<float>(game.game_log.size()) / 200.0f, 1.0f);

    // [38-39] Dragon presence flags
    f[38] = (gold_counts[3] > 0) ? 1.0f : 0.0f;
    f[39] = (scarlet_counts[3] > 0) ? 1.0f : 0.0f;

    return f;
}

} // namespace dragonchess
