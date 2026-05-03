#pragma once

#include "game.h"
#include "td_features.h"
#include "nn_eval.h"
#include <optional>
#include <random>
#include <unordered_map>
#include <limits>

namespace dragonchess {

class BaseAI {
public:
    BaseAI(Game& game, Color color);
    virtual ~BaseAI() = default;
    
    virtual std::optional<Move> choose_move() = 0;
    
protected:
    Game& game;
    Color color;
    
    // Piece values for evaluation
    static constexpr float piece_values[16] = {
        0.0f,    // EMPTY
        1.0f,    // WARRIOR
        5.0f,    // OLIPHANT
        8.0f,    // UNICORN
        5.0f,    // HERO
        2.5f,    // THIEF
        4.5f,    // CLERIC
        4.0f,    // MAGE
        9.0f,    // PALADIN
        11.0f,   // DRAGON
        10000.0f, // KING
        10.0f,   // SYLPH
        1.0f,    // DWARF
        3.0f,    // BASILISK
        4.0f,    // ELEMENTAL
        2.0f     // GRIFFIN
    };
    
    virtual float evaluate_material(const Game& g) const;
};

class RandomAI : public BaseAI {
public:
    RandomAI(Game& game, Color color);
    
    std::optional<Move> choose_move() override;
    
private:
    std::mt19937 rng;
};

// Greedy bot that prefers captures
class GreedyAI : public BaseAI {
public:
    GreedyAI(Game& game, Color color);
    
    std::optional<Move> choose_move() override;
    
private:
    std::mt19937 rng;
};

// Greedy bot with piece value evaluation
class GreedyValueAI : public BaseAI {
public:
    GreedyValueAI(Game& game, Color color);
    
    std::optional<Move> choose_move() override;
    
private:
    std::mt19937 rng;
};

// Minimax with configurable depth
class MinimaxAI : public BaseAI {
public:
    MinimaxAI(Game& game, Color color, int depth = 2);
    
    std::optional<Move> choose_move() override;
    
private:
    int max_depth;
    int nodes_searched;  // For statistics
    std::mt19937 rng;    // For tie-breaking
    
    struct EvalResult {
        float score;
        std::optional<Move> move;
    };
    
    float evaluate_position() const;
    float evaluate_incremental(const Game& game_state, const Move& move, float current_eval) const;
    EvalResult minimax(Game& game_copy, int depth, bool maximizing);
};

// Alpha-beta with transposition table
class AlphaBetaAI : public BaseAI {
public:
    AlphaBetaAI(Game& game, Color color, int depth = 3);

    std::optional<Move> choose_move() override;

    // Iterative deepening with time limit (milliseconds). Searches d=1,2,3,...
    // until time runs out. Returns best move from deepest completed search.
    std::optional<Move> choose_move_timed(float time_limit_ms);

    void set_max_depth(int d) { max_depth = d; }
    void clear_tt() { transposition_table.clear(); }
    int get_last_depth() const { return last_search_depth; }

    // Run AB search on a game copy and return the score.
    float search_score(Game& game_copy, int depth) {
        nodes_searched = 0;
        transposition_table.clear();
        auto r = alphabeta(game_copy, depth,
            -std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::infinity(), true);
        return r.score;
    }

protected:
    int max_depth;
    int nodes_searched;  // For statistics
    int last_search_depth = 0;
    std::mt19937 rng;    // For tie-breaking
    std::unordered_map<uint64_t, std::pair<float, int>> transposition_table;

    struct EvalResult {
        float score;
        std::optional<Move> move;
        std::vector<Move> pv;  // Principal variation (root to leaf)
    };

    float evaluate_position() const;
    uint64_t hash_position(const Game& g) const;
    EvalResult alphabeta(Game& game_copy, int depth, float alpha, float beta, bool maximizing);
};

// AlphaBeta AI with externally supplied piece-value weights.
// Weights vector has 14 entries (piece types 1-15, excluding King at index 10).
// Piece type order: Sylph(1), Griffin(2), Dragon(3), Oliphant(4), Unicorn(5),
//   Hero(6), Thief(7), Cleric(8), Mage(9), Paladin(11), Warrior(12),
//   Basilisk(13), Elemental(14), Dwarf(15).
// King value is fixed at 10000 internally.
class EvolvableAI : public AlphaBetaAI {
public:
    EvolvableAI(Game& game, Color color, const std::vector<float>& weights, int depth = 2);

    float evaluate_material(const Game& g) const override;

private:
    std::vector<float> weights;  // 14 evolved piece values
};

// AlphaBeta AI with a TD-learned feature-weight evaluation function.
// Uses a NUM_TD_FEATURES-dimensional weight vector; evaluation is the dot
// product of extract_td_features(game) with the weight vector.
// Features are Gold-positive; the sign is flipped internally for Scarlet.
class TDEvalAI : public AlphaBetaAI {
public:
    TDEvalAI(Game& game, Color color, const std::vector<float>& weights, int depth = 1);

    float evaluate_material(const Game& g) const override;

    // TDLeaf: choose move and return PV leaf features for training.
    // Returns {best_move, leaf_features, minimax_value}.
    struct TDLeafResult {
        std::optional<Move> move;
        std::vector<SparseFeature> leaf_features;
        float value;
    };
    TDLeafResult choose_move_tdleaf();

private:
    std::vector<float> weights;  // NUM_TD_FEATURES learned weights
};

// NN-based evaluation (NNUE-style: sparse PST input → hidden → hidden → scalar).
// Uses incremental accumulator for the first hidden layer during AB search.
class NNEvalAI : public AlphaBetaAI {
public:
    NNEvalAI(Game& game, Color color, const NNWeights& nn, int depth = 1);

    float evaluate_material(const Game& g) const override;
    std::optional<Move> choose_move() override;

    struct TDLeafResult {
        std::optional<Move> move;
        std::vector<SparseFeature> leaf_features;
        float value;
    };
    TDLeafResult choose_move_tdleaf();

private:
    NNWeights nn;

    // Accumulator stack for incremental first-layer updates during search.
    mutable std::vector<NNAccumulator> acc_stack;
    mutable bool acc_valid = false;

    // Compute full accumulator from a game state.
    void acc_init(const Game& g) const;

    // Push current accumulator, apply move delta.
    void acc_push_move(const Game& g, const Move& m) const;

    // Pop to previous accumulator.
    void acc_pop() const;

    // Override AB search to manage accumulator around make/undo.
    EvalResult alphabeta_nn(Game& game_copy, int depth,
                            float alpha, float beta, bool maximizing);
};

} // namespace dragonchess
