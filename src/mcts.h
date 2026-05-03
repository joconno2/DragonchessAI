#pragma once

#include "game.h"
#include "moves.h"
#include "td_features.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <unordered_map>
#include <cassert>

namespace dragonchess {

// Action encoding: from_sq * TOTAL_SQUARES + to_sq.
// The flag (QUIET/CAPTURE/etc.) is determined by the game state, not encoded
// in the action. Two moves with same (from, to) but different flags don't
// occur in Dragonchess (a move is either a capture or not based on the target).
constexpr int ACTION_SPACE = TOTAL_SQUARES * TOTAL_SQUARES;  // 82944

inline int move_to_action(const Move& m) {
    return std::get<0>(m) * TOTAL_SQUARES + std::get<1>(m);
}

inline Move action_to_move_with_flag(int action, const Game& game) {
    int from = action / TOTAL_SQUARES;
    int to   = action % TOTAL_SQUARES;
    // Determine flag from board state
    MoveFlag flag = (game.board[to] != EMPTY) ? CAPTURE : QUIET;
    return std::make_tuple(from, to, flag);
}

// ---------------------------------------------------------------------------
// Dual-head neural network: policy + value from sparse features.
//
// Architecture: sparse_input(N) -> h1(H1, ReLU) -> h2(H2, ReLU) -> {
//     policy: h2 -> logits(ACTION_SPACE)   [only computed for legal moves]
//     value:  h2 -> scalar -> tanh
// }
// ---------------------------------------------------------------------------

struct DualHeadWeights {
    static constexpr int N_INPUT = NUM_TD_FEATURES;  // 32284
    static constexpr int N_H1 = 256;
    static constexpr int N_H2 = 128;
    static constexpr int N_ACTIONS = ACTION_SPACE;    // 83808

    // Shared trunk
    std::vector<float> w1;   // H1 x N_INPUT
    std::vector<float> b1;   // H1
    std::vector<float> w2;   // H2 x H1
    std::vector<float> b2;   // H2

    // Policy head
    std::vector<float> wp;   // N_ACTIONS x H2
    std::vector<float> bp;   // N_ACTIONS

    // Value head
    std::vector<float> wv;   // H2
    float bv;

    DualHeadWeights() : bv(0.0f) {
        w1.resize(N_H1 * N_INPUT, 0.0f);
        b1.resize(N_H1, 0.0f);
        w2.resize(N_H2 * N_H1, 0.0f);
        b2.resize(N_H2, 0.0f);
        wp.resize(N_ACTIONS * N_H2, 0.0f);
        bp.resize(N_ACTIONS, 0.0f);
        wv.resize(N_H2, 0.0f);
    }

    // Forward pass: compute h2 from sparse input.
    void forward_trunk(const std::vector<SparseFeature>& input,
                       float* h2_out) const {
        // Layer 1: sparse matmul + bias + ReLU
        float h1[N_H1];
        for (int j = 0; j < N_H1; ++j) h1[j] = b1[j];
        for (const auto& sf : input)
            for (int j = 0; j < N_H1; ++j)
                h1[j] += w1[j * N_INPUT + sf.index] * sf.value;
        for (int j = 0; j < N_H1; ++j)
            h1[j] = std::max(h1[j], 0.0f);

        // Layer 2
        for (int k = 0; k < N_H2; ++k) {
            float sum = b2[k];
            for (int j = 0; j < N_H1; ++j)
                sum += w2[k * N_H1 + j] * h1[j];
            h2_out[k] = std::max(sum, 0.0f);
        }
    }

    // Policy logit for a single action (used during MCTS expansion).
    float policy_logit(const float* h2, int action) const {
        float sum = bp[action];
        for (int k = 0; k < N_H2; ++k)
            sum += wp[action * N_H2 + k] * h2[k];
        return sum;
    }

    // Value from h2.
    float value(const float* h2) const {
        float sum = bv;
        for (int k = 0; k < N_H2; ++k)
            sum += wv[k] * h2[k];
        return std::tanh(sum);
    }

    // Evaluate a position: returns (policy_priors for legal moves, value).
    struct EvalResult {
        std::vector<std::pair<int, float>> move_priors;  // (action, probability)
        float value;
    };

    EvalResult evaluate(const Game& game) const {
        auto sparse = extract_td_features_sparse(game);
        float h2[N_H2];
        forward_trunk(sparse, h2);

        // Get legal moves and compute policy logits
        auto legal_moves = game.get_all_moves();
        std::vector<std::pair<int, float>> logits;
        logits.reserve(legal_moves.size());

        float max_logit = -1e30f;
        for (const auto& m : legal_moves) {
            int a = move_to_action(m);
            float l = policy_logit(h2, a);
            logits.push_back({a, l});
            max_logit = std::max(max_logit, l);
        }

        // Softmax over legal moves
        float sum_exp = 0.0f;
        for (auto& [a, l] : logits) {
            l = std::exp(l - max_logit);
            sum_exp += l;
        }
        for (auto& [a, l] : logits)
            l /= sum_exp;

        return {logits, value(h2)};
    }

    static constexpr int total_params() {
        return N_H1 * N_INPUT + N_H1          // trunk layer 1
             + N_H2 * N_H1 + N_H2             // trunk layer 2
             + N_ACTIONS * N_H2 + N_ACTIONS    // policy head
             + N_H2 + 1;                       // value head
    }

    bool from_flat(const std::vector<float>& flat) {
        if (static_cast<int>(flat.size()) != total_params())
            return false;
        int o = 0;
        auto copy = [&](std::vector<float>& dst, int n) {
            dst.assign(flat.begin() + o, flat.begin() + o + n);
            o += n;
        };
        copy(w1, N_H1 * N_INPUT);
        copy(b1, N_H1);
        copy(w2, N_H2 * N_H1);
        copy(b2, N_H2);
        copy(wp, N_ACTIONS * N_H2);
        copy(bp, N_ACTIONS);
        copy(wv, N_H2);
        bv = flat[o];
        return true;
    }
};

// ---------------------------------------------------------------------------
// MCTS Node
// ---------------------------------------------------------------------------

struct MCTSNode {
    int action = -1;            // action that led here (-1 for root)
    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    std::vector<std::unique_ptr<MCTSNode>> children;
    bool expanded = false;

    float q_value() const {
        return visit_count > 0 ? value_sum / visit_count : 0.0f;
    }
};

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

class MCTS {
public:
    MCTS(const DualHeadWeights& nn, float c_puct = 1.5f,
         float dirichlet_alpha = 0.3f, float dirichlet_frac = 0.25f)
        : nn_(nn), c_puct_(c_puct),
          dirichlet_alpha_(dirichlet_alpha), dirichlet_frac_(dirichlet_frac),
          rng_(std::random_device{}()) {}

    // Run N simulations from the given game state. Returns root node.
    MCTSNode* search(Game& game, int num_simulations);

    // Get MCTS policy (visit count distribution) from root.
    // Returns vector of (action, probability) for all visited children.
    std::vector<std::pair<int, float>> get_policy(const MCTSNode* root,
                                                   float temperature = 1.0f) const;

    // Select a move from the policy.
    int select_action(const std::vector<std::pair<int, float>>& policy);

private:
    // One simulation: select -> expand -> evaluate -> backprop.
    float simulate(MCTSNode* node, Game& game, bool is_root);

    // PUCT selection among children.
    MCTSNode* select_child(MCTSNode* node) const;

    // Expand a leaf node.
    void expand(MCTSNode* node, const Game& game,
                const std::vector<std::pair<int, float>>& priors);

    // Add Dirichlet noise to root priors.
    void add_dirichlet_noise(MCTSNode* node);

    const DualHeadWeights& nn_;
    float c_puct_;
    float dirichlet_alpha_;
    float dirichlet_frac_;
    std::mt19937 rng_;
    std::unique_ptr<MCTSNode> root_;

    // Map action -> Move for the current position's legal moves.
    std::unordered_map<int, Move> action_to_move_map_;
};

// ---------------------------------------------------------------------------
// Self-play data generation
// ---------------------------------------------------------------------------

struct SelfPlayPosition {
    std::vector<SparseFeature> features;
    std::vector<std::pair<int, float>> mcts_policy;  // (action, probability)
    float outcome;  // +1 gold win, -1 scarlet win, 0 draw (filled after game)
    Color turn;
};

struct SelfPlayGame {
    std::vector<SelfPlayPosition> positions;
    float outcome;  // +1 gold win, -1 scarlet, 0 draw
    int num_moves;
};

// Run one self-play game with MCTS.
SelfPlayGame run_mcts_selfplay(const DualHeadWeights& nn,
                                int num_simulations = 400,
                                float temperature = 1.0f,
                                int temp_drop_move = 30,
                                int max_moves = 500);

} // namespace dragonchess
