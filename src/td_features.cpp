#include "td_features.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>

namespace dragonchess {

// Cross-level capable pieces.
static constexpr bool IS_CROSS_LEVEL[16] = {
    false, true, true, true, false, false, true, false,
    false, true, false, true, false, false, false, false,
    //     Syl   Gri   Dra                Hero
    //           Mage        Pal
};

std::vector<SparseFeature> extract_td_features_sparse(const Game& game) {
    std::vector<SparseFeature> out;
    out.reserve(80);  // ~52 pieces + strategic features

    // ---------- Accumulators for strategic features ----------

    // Per-side piece counts by type
    int gold_counts[16] = {};
    int scarlet_counts[16] = {};

    float gold_total_mat = 0.0f, scarlet_total_mat = 0.0f;
    int gold_total_pieces = 0, scarlet_total_pieces = 0;
    int gold_cross = 0, scarlet_cross = 0;
    int gold_frozen_count = 0, scarlet_frozen_count = 0;
    float gold_frozen_mat = 0.0f, scarlet_frozen_mat = 0.0f;
    int gold_king_sq = -1, scarlet_king_sq = -1;

    // Basilisk positions for freeze-threat feature
    int gold_basilisk_sq[4], scarlet_basilisk_sq[4];
    int n_gold_bas = 0, n_scarlet_bas = 0;

    // Warrior file tracking for pawn structure (Ground = layer 1, cols 0-11)
    bool gold_warrior_on_file[12] = {};
    bool scarlet_warrior_on_file[12] = {};
    int gold_warrior_file_count[12] = {};
    int scarlet_warrior_file_count[12] = {};

    struct WarriorPos { int row; int col; };
    WarriorPos gold_warriors[12];
    WarriorPos scarlet_warriors[12];
    int n_gold_warriors = 0, n_scarlet_warriors = 0;

    const int layer_size = BOARD_ROWS * BOARD_COLS;

    // ---------- Pre-scan: find king positions for king-bucket features ----------
    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        int16_t piece = game.board[idx];
        if (piece == 10) gold_king_sq = idx;
        else if (piece == -10) scarlet_king_sq = idx;
    }
    int gold_king_bucket = king_to_bucket(gold_king_sq);
    int scarlet_king_bucket = king_to_bucket(scarlet_king_sq);

    // ---------- Main board scan ----------

    for (int idx = 0; idx < TOTAL_SQUARES; ++idx) {
        int16_t piece = game.board[idx];
        if (piece == EMPTY) continue;

        bool is_gold = (piece > 0);
        int ptype = static_cast<int>(std::abs(piece));
        int layer = idx / layer_size;
        int rem = idx % layer_size;
        int row = rem / BOARD_COLS;
        int col = rem % BOARD_COLS;
        float pval = (ptype < 16) ? PIECE_VAL[ptype] : 0.0f;

        // --- Piece-square feature (king-bucket indexed) ---
        int pt_idx = piece_type_to_idx(ptype);
        if (pt_idx >= 0) {
            int bucket = is_gold ? gold_king_bucket : scarlet_king_bucket;
            int feat_idx = bucket * (NUM_PIECE_TYPES * TOTAL_SQUARES)
                         + pt_idx * TOTAL_SQUARES + idx;
            out.push_back({feat_idx, is_gold ? 1.0f : -1.0f});
        }

        // --- Accumulate for strategic features ---
        if (is_gold) {
            gold_counts[ptype]++;
            gold_total_pieces++;
            gold_total_mat += pval;
            if (ptype == 10) gold_king_sq = idx;
            if (ptype < 16 && IS_CROSS_LEVEL[ptype]) gold_cross++;
            if (game.frozen[idx]) { gold_frozen_count++; gold_frozen_mat += pval; }
            if (ptype == 13 && n_gold_bas < 4) gold_basilisk_sq[n_gold_bas++] = idx;
            if (ptype == 12 && layer == 1) {
                gold_warrior_on_file[col] = true;
                gold_warrior_file_count[col]++;
                if (n_gold_warriors < 12)
                    gold_warriors[n_gold_warriors++] = {row, col};
            }
        } else {
            scarlet_counts[ptype]++;
            scarlet_total_pieces++;
            scarlet_total_mat += pval;
            if (ptype == 10) scarlet_king_sq = idx;
            if (ptype < 16 && IS_CROSS_LEVEL[ptype]) scarlet_cross++;
            if (game.frozen[idx]) { scarlet_frozen_count++; scarlet_frozen_mat += pval; }
            if (ptype == 13 && n_scarlet_bas < 4) scarlet_basilisk_sq[n_scarlet_bas++] = idx;
            if (ptype == 12 && layer == 1) {
                scarlet_warrior_on_file[col] = true;
                scarlet_warrior_file_count[col]++;
                if (n_scarlet_warriors < 12)
                    scarlet_warriors[n_scarlet_warriors++] = {row, col};
            }
        }
    }

    // ---------- Strategic features (starting at NUM_PIECE_SQUARE = 4032) ----------
    constexpr int S = NUM_PIECE_SQUARE;

    // [S+0..S+1] Frozen counts
    if (gold_frozen_count > 0)
        out.push_back({S + 0, static_cast<float>(gold_frozen_count)});
    if (scarlet_frozen_count > 0)
        out.push_back({S + 1, static_cast<float>(scarlet_frozen_count)});

    // [S+2..S+3] Frozen material value
    if (gold_frozen_mat > 0.0f)
        out.push_back({S + 2, gold_frozen_mat});
    if (scarlet_frozen_mat > 0.0f)
        out.push_back({S + 3, scarlet_frozen_mat});

    // [S+4..S+5] Pieces on Ground directly above enemy Basilisk (freeze-threatened)
    // Basilisk is on Cavern (layer 2); "above" = same row/col on Ground (layer 1)
    {
        int gold_threatened = 0;
        for (int i = 0; i < n_scarlet_bas; ++i) {
            int bas_rem = scarlet_basilisk_sq[i] % layer_size;
            int ground_idx = layer_size + bas_rem;  // layer 1
            int16_t p = game.board[ground_idx];
            if (p > 0) gold_threatened++;
        }
        int scarlet_threatened = 0;
        for (int i = 0; i < n_gold_bas; ++i) {
            int bas_rem = gold_basilisk_sq[i] % layer_size;
            int ground_idx = layer_size + bas_rem;
            int16_t p = game.board[ground_idx];
            if (p < 0) scarlet_threatened++;
        }
        if (gold_threatened > 0)
            out.push_back({S + 4, static_cast<float>(gold_threatened)});
        if (scarlet_threatened > 0)
            out.push_back({S + 5, static_cast<float>(scarlet_threatened)});
    }

    // [S+6..S+11] King safety
    auto king_safety = [&](int king_sq, bool count_friendly) {
        // Returns {dist1_count, dist2_count} for friendly or enemy pieces
        int d1 = 0, d2 = 0;
        if (king_sq < 0) return std::make_pair(0, 0);
        int kl = king_sq / layer_size;
        int kr = (king_sq % layer_size) / BOARD_COLS;
        int kc = king_sq % BOARD_COLS;
        bool king_is_gold = (game.board[king_sq] > 0);
        int base = kl * layer_size;
        for (int i = base; i < base + layer_size; ++i) {
            int16_t p = game.board[i];
            if (p == EMPTY) continue;
            bool p_is_gold = (p > 0);
            bool is_friendly = (p_is_gold == king_is_gold);
            if (is_friendly != count_friendly) continue;
            int r = (i % layer_size) / BOARD_COLS;
            int c = i % BOARD_COLS;
            int dist = std::max(std::abs(r - kr), std::abs(c - kc));
            if (dist == 1) d1++;
            else if (dist == 2) d2++;
        }
        return std::make_pair(d1, d2);
    };

    auto [gk_friend_d1, gk_friend_d2] = king_safety(gold_king_sq, true);
    auto [gk_enemy_d1, gk_enemy_d2] = king_safety(gold_king_sq, false);
    auto [sk_friend_d1, sk_friend_d2] = king_safety(scarlet_king_sq, true);
    auto [sk_enemy_d1, sk_enemy_d2] = king_safety(scarlet_king_sq, false);

    if (gk_friend_d1 > 0) out.push_back({S + 6, static_cast<float>(gk_friend_d1)});
    if (gk_enemy_d1 > 0) out.push_back({S + 7, static_cast<float>(gk_enemy_d1)});
    if (gk_enemy_d2 > 0) out.push_back({S + 8, static_cast<float>(gk_enemy_d2)});
    if (sk_friend_d1 > 0) out.push_back({S + 9, static_cast<float>(sk_friend_d1)});
    if (sk_enemy_d1 > 0) out.push_back({S + 10, static_cast<float>(sk_enemy_d1)});
    if (sk_enemy_d2 > 0) out.push_back({S + 11, static_cast<float>(sk_enemy_d2)});

    // [S+12..S+13] King on home rank
    if (gold_king_sq >= 0) {
        int kr = (gold_king_sq % layer_size) / BOARD_COLS;
        int kl = gold_king_sq / layer_size;
        if (kl == 1 && kr == 7) out.push_back({S + 12, 1.0f});
    }
    if (scarlet_king_sq >= 0) {
        int kr = (scarlet_king_sq % layer_size) / BOARD_COLS;
        int kl = scarlet_king_sq / layer_size;
        if (kl == 1 && kr == 0) out.push_back({S + 13, 1.0f});
    }

    // [S+14] Total material / initial
    float total_mat = gold_total_mat + scarlet_total_mat;
    if (total_mat > 0.0f)
        out.push_back({S + 14, total_mat / INITIAL_TOTAL_MATERIAL});

    // [S+15] Total pieces / 52
    int total_pieces = gold_total_pieces + scarlet_total_pieces;
    if (total_pieces > 0)
        out.push_back({S + 15, static_cast<float>(total_pieces) / 52.0f});

    // [S+16] Move count / 200
    float move_progress = std::min(static_cast<float>(game.game_log.size()) / 200.0f, 1.0f);
    if (move_progress > 0.0f)
        out.push_back({S + 16, move_progress});

    // [S+17] No-capture counter / 250
    if (game.no_capture_count > 0)
        out.push_back({S + 17, static_cast<float>(game.no_capture_count) / 250.0f});

    // [S+18..S+23] Warrior pawn structure
    {
        int gold_isolated = 0, gold_doubled = 0, gold_connected = 0;
        int scarlet_isolated = 0, scarlet_doubled = 0, scarlet_connected = 0;

        for (int i = 0; i < n_gold_warriors; ++i) {
            int c = gold_warriors[i].col;
            int r = gold_warriors[i].row;
            bool has_neighbor = false;
            if (c > 0 && gold_warrior_on_file[c - 1]) has_neighbor = true;
            if (c < 11 && gold_warrior_on_file[c + 1]) has_neighbor = true;
            if (!has_neighbor) gold_isolated++;
            // Connected: friendly warrior on adjacent file, same or adjacent row
            for (int j = 0; j < n_gold_warriors; ++j) {
                if (i == j) continue;
                int dc = std::abs(gold_warriors[j].col - c);
                int dr = std::abs(gold_warriors[j].row - r);
                if (dc == 1 && dr <= 1) { gold_connected++; break; }
            }
        }
        for (int c = 0; c < 12; ++c) {
            if (gold_warrior_file_count[c] >= 2) gold_doubled += gold_warrior_file_count[c] - 1;
        }

        for (int i = 0; i < n_scarlet_warriors; ++i) {
            int c = scarlet_warriors[i].col;
            int r = scarlet_warriors[i].row;
            bool has_neighbor = false;
            if (c > 0 && scarlet_warrior_on_file[c - 1]) has_neighbor = true;
            if (c < 11 && scarlet_warrior_on_file[c + 1]) has_neighbor = true;
            if (!has_neighbor) scarlet_isolated++;
            for (int j = 0; j < n_scarlet_warriors; ++j) {
                if (i == j) continue;
                int dc = std::abs(scarlet_warriors[j].col - c);
                int dr = std::abs(scarlet_warriors[j].row - r);
                if (dc == 1 && dr <= 1) { scarlet_connected++; break; }
            }
        }
        for (int c = 0; c < 12; ++c) {
            if (scarlet_warrior_file_count[c] >= 2) scarlet_doubled += scarlet_warrior_file_count[c] - 1;
        }

        if (gold_isolated > 0) out.push_back({S + 18, static_cast<float>(gold_isolated)});
        if (gold_doubled > 0)  out.push_back({S + 19, static_cast<float>(gold_doubled)});
        if (gold_connected > 0) out.push_back({S + 20, static_cast<float>(gold_connected)});
        if (scarlet_isolated > 0) out.push_back({S + 21, static_cast<float>(scarlet_isolated)});
        if (scarlet_doubled > 0)  out.push_back({S + 22, static_cast<float>(scarlet_doubled)});
        if (scarlet_connected > 0) out.push_back({S + 23, static_cast<float>(scarlet_connected)});
    }

    // [S+24..S+25] Cross-level capable piece counts
    if (gold_cross > 0) out.push_back({S + 24, static_cast<float>(gold_cross)});
    if (scarlet_cross > 0) out.push_back({S + 25, static_cast<float>(scarlet_cross)});

    // [S+26] Material advantage (Gold - Scarlet, normalized)
    float mat_adv = (gold_total_mat - scarlet_total_mat) / INITIAL_TOTAL_MATERIAL;
    if (std::fabs(mat_adv) > 1e-6f)
        out.push_back({S + 26, mat_adv});

    // [S+27] Repetition count
    if (!game.state_history.empty()) {
        uint64_t current_hash = game.state_history.back();
        int reps = 0;
        for (size_t i = 0; i + 1 < game.state_history.size(); ++i) {
            if (game.state_history[i] == current_hash) reps++;
        }
        if (reps > 0)
            out.push_back({S + 27, static_cast<float>(reps) / 3.0f});
    }

    return out;
}

std::vector<float> extract_td_features(const Game& game) {
    std::vector<float> dense(NUM_TD_FEATURES, 0.0f);
    auto sparse = extract_td_features_sparse(game);
    for (const auto& sf : sparse)
        dense[sf.index] = sf.value;
    return dense;
}

} // namespace dragonchess
