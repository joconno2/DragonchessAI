#pragma once

#include "game.h"
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
    
    float evaluate_material() const;
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
    
private:
    int max_depth;
    int nodes_searched;  // For statistics
    std::mt19937 rng;    // For tie-breaking
    std::unordered_map<uint64_t, std::pair<float, int>> transposition_table;
    
    struct EvalResult {
        float score;
        std::optional<Move> move;
    };
    
    float evaluate_position() const;
    uint64_t hash_position() const;
    EvalResult alphabeta(Game& game_copy, int depth, float alpha, float beta, bool maximizing);
};

} // namespace dragonchess
