#include "ai.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

// Zobrist hash table for AlphaBeta transposition table.
// Initialized once at program startup with a fixed seed (reproducible).
// Layout: [square][piece_type][color]  — color 0=gold(+), 1=scarlet(-)
namespace {
    struct ZobristTable {
        uint64_t data[288][16][2];
        ZobristTable() {
            std::mt19937_64 rng(0xD4A6C0FFEE1234ULL);
            for (int sq = 0; sq < 288; ++sq)
                for (int pt = 1; pt < 16; ++pt)
                    for (int c = 0; c < 2; ++c)
                        data[sq][pt][c] = rng();
        }
    } const kZobrist;
} // anonymous namespace

namespace dragonchess {

BaseAI::BaseAI(Game& game, Color color)
    : game(game)
    , color(color)
{
}

float BaseAI::evaluate_material(const Game& g) const {
    float score = 0.0f;

    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        int16_t piece = g.board[i];
        if (piece != EMPTY) {
            int abs_piece = std::abs(piece);
            float value = piece_values[abs_piece];

            // Positive for our pieces, negative for opponent
            if ((piece > 0 && color == Color::GOLD) ||
                (piece < 0 && color == Color::SCARLET)) {
                score += value;
            } else {
                score -= value;
            }
        }
    }

    // Note: repetition draws enforced by Game::update() via threefold-repetition check.
    // No O(N) scan needed here at every leaf node.
    return score;
}

RandomAI::RandomAI(Game& game, Color color)
    : BaseAI(game, color)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
}

std::optional<Move> RandomAI::choose_move() {
    std::vector<Move> moves = game.get_all_moves();
    
    if (moves.empty()) {
        return std::nullopt;
    }
    
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng)];
}

// ===== GreedyAI (Novice) =====
GreedyAI::GreedyAI(Game& game, Color color)
    : BaseAI(game, color)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
}

std::optional<Move> GreedyAI::choose_move() {
    std::vector<Move> moves = game.get_all_moves();
    
    if (moves.empty()) {
        return std::nullopt;
    }
    
    // Separate captures from quiet moves
    std::vector<Move> captures;
    std::vector<Move> quiet_moves;
    
    for (const auto& move : moves) {
        int to_idx = std::get<1>(move);
        int16_t target = game.board[to_idx];
        if (target != EMPTY) {
            // Check if it's an enemy piece
            if ((target > 0 && color == Color::SCARLET) || 
                (target < 0 && color == Color::GOLD)) {
                captures.push_back(move);
            }
        } else {
            quiet_moves.push_back(move);
        }
    }
    
    // Prefer captures
    if (!captures.empty()) {
        std::uniform_int_distribution<size_t> dist(0, captures.size() - 1);
        return captures[dist(rng)];
    }
    
    std::uniform_int_distribution<size_t> dist(0, quiet_moves.size() - 1);
    return quiet_moves[dist(rng)];
}

// ===== GreedyValueAI (Apprentice) =====
GreedyValueAI::GreedyValueAI(Game& game, Color color)
    : BaseAI(game, color)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
}

std::optional<Move> GreedyValueAI::choose_move() {
    std::vector<Move> moves = game.get_all_moves();
    
    if (moves.empty()) {
        return std::nullopt;
    }
    
    // Score each move by captured piece value
    std::vector<std::pair<Move, float>> scored_moves;
    
    for (const auto& move : moves) {
        int to_idx = std::get<1>(move);
        int16_t target = game.board[to_idx];
        float score = 0.0f;
        
        if (target != EMPTY) {
            // Check if it's an enemy piece
            if ((target > 0 && color == Color::SCARLET) || 
                (target < 0 && color == Color::GOLD)) {
                score = piece_values[std::abs(target)];
            }
        }
        
        scored_moves.push_back({move, score});
    }
    
    // Sort by score descending
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take best moves (those with highest score)
    float best_score = scored_moves[0].second;
    std::vector<Move> best_moves;
    
    for (const auto& [move, score] : scored_moves) {
        if (score >= best_score - 0.01f) {  // Allow small epsilon
            best_moves.push_back(move);
        } else {
            break;
        }
    }
    
    std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
    return best_moves[dist(rng)];
}

// ===== MinimaxAI (Veteran/SimpleMinimaxAI) =====
MinimaxAI::MinimaxAI(Game& game, Color color, int depth)
    : BaseAI(game, color)
    , max_depth(depth)
    , nodes_searched(0)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
}

float MinimaxAI::evaluate_position() const {
    return evaluate_material(game);
}

MinimaxAI::EvalResult MinimaxAI::minimax(Game& game_copy, int depth, bool maximizing) {
    nodes_searched++;
    std::vector<Move> moves = game_copy.get_all_moves();
    
    // Terminal condition
    if (depth == 0 || moves.empty() || game_copy.game_over) {
        return {evaluate_material(game_copy), std::nullopt};
    }

    // Limit branching factor at deeper levels for speed
    size_t max_moves = (depth <= 1) ? moves.size() : std::min(moves.size(), size_t(25));
    
    // Move ordering: prioritize captures for better cutoffs
    std::sort(moves.begin(), moves.end(), [&game_copy](const Move& a, const Move& b) {
        int a_to = std::get<1>(a);
        int b_to = std::get<1>(b);
        int16_t a_target = game_copy.board[a_to];
        int16_t b_target = game_copy.board[b_to];
        
        bool a_capture = a_target != EMPTY;
        bool b_capture = b_target != EMPTY;
        
        if (a_capture != b_capture) return a_capture > b_capture;
        
        // Sort captures by value
        if (a_capture && b_capture) {
            return std::abs(a_target) > std::abs(b_target);
        }
        
        return false;
    });
    
    if (maximizing) {
        float best_score = -std::numeric_limits<float>::infinity();
        std::optional<Move> best_move;
        std::vector<Move> best_moves;  // Track all moves with best score
        
        for (size_t i = 0; i < max_moves; ++i) {
            const auto& move = moves[i];
            Game child = game_copy;
            child.make_move(move);
            child.update();
            
            auto result = minimax(child, depth - 1, false);
            
            if (result.score > best_score) {
                best_score = result.score;
                best_moves.clear();
                best_moves.push_back(move);
            } else if (std::abs(result.score - best_score) < 0.01f) {
                // Same score - add to candidates for random selection
                best_moves.push_back(move);
            }
        }
        
        // Randomly select from best moves to break ties
        if (!best_moves.empty()) {
            std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
            best_move = best_moves[dist(rng)];
        }
        
        return {best_score, best_move};
    } else {
        float best_score = std::numeric_limits<float>::infinity();
        std::optional<Move> best_move;
        std::vector<Move> best_moves;  // Track all moves with best score
        
        for (size_t i = 0; i < max_moves; ++i) {
            const auto& move = moves[i];
            Game child = game_copy;
            child.make_move(move);
            child.update();
            
            auto result = minimax(child, depth - 1, true);
            
            if (result.score < best_score) {
                best_score = result.score;
                best_moves.clear();
                best_moves.push_back(move);
            } else if (std::abs(result.score - best_score) < 0.01f) {
                // Same score - add to candidates for random selection
                best_moves.push_back(move);
            }
        }
        
        // Randomly select from best moves to break ties
        if (!best_moves.empty()) {
            std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
            best_move = best_moves[dist(rng)];
        }
        
        return {best_score, best_move};
    }
}

std::optional<Move> MinimaxAI::choose_move() {
    std::vector<Move> moves = game.get_all_moves();
    
    if (moves.empty()) {
        return std::nullopt;
    }
    
    nodes_searched = 0;
    Game game_copy = game;
    auto result = minimax(game_copy, max_depth, true);
    
    // Debug: show nodes searched
    // std::cout << "Minimax searched " << nodes_searched << " nodes" << std::endl;
    
    return result.move.has_value() ? result.move : moves[0];
}

// ===== AlphaBetaAI (Champion/FastMinimaxAI) =====
AlphaBetaAI::AlphaBetaAI(Game& game, Color color, int depth)
    : BaseAI(game, color)
    , max_depth(depth)
    , nodes_searched(0)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
}

float AlphaBetaAI::evaluate_position() const {
    return evaluate_material(game);
}

uint64_t AlphaBetaAI::hash_position(const Game& g) const {
    uint64_t hash = 0;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        int16_t piece = g.board[i];
        if (piece != EMPTY) {
            hash ^= kZobrist.data[i][std::abs(piece)][piece < 0 ? 1 : 0];
        }
    }
    return hash;
}

AlphaBetaAI::EvalResult AlphaBetaAI::alphabeta(Game& game_copy, int depth, 
                                                float alpha, float beta, bool maximizing) {
    nodes_searched++;
    
    // Check transposition table
    uint64_t hash = hash_position(game_copy);
    auto it = transposition_table.find(hash);
    if (it != transposition_table.end() && it->second.second >= depth) {
        return {it->second.first, std::nullopt};
    }

    std::vector<Move> moves = game_copy.get_all_moves();

    // Terminal condition
    if (depth == 0 || moves.empty() || game_copy.game_over) {
        float score = evaluate_material(game_copy);
        transposition_table[hash] = {score, depth};
        return {score, std::nullopt};
    }
    
    // Limit branching factor for speed
    size_t max_moves = (depth <= 1) ? moves.size() : std::min(moves.size(), size_t(30));
    
    // Move ordering: prioritize captures by value
    std::sort(moves.begin(), moves.end(), [&game_copy](const Move& a, const Move& b) {
        int a_to = std::get<1>(a);
        int b_to = std::get<1>(b);
        int16_t a_target = game_copy.board[a_to];
        int16_t b_target = game_copy.board[b_to];
        
        bool a_capture = a_target != EMPTY;
        bool b_capture = b_target != EMPTY;
        
        if (a_capture != b_capture) return a_capture > b_capture;
        
        // Sort captures by victim value - attacker value (MVV-LVA)
        if (a_capture && b_capture) {
            int a_from = std::get<0>(a);
            int b_from = std::get<0>(b);
            float a_score = std::abs(a_target) * 10 - std::abs(game_copy.board[a_from]);
            float b_score = std::abs(b_target) * 10 - std::abs(game_copy.board[b_from]);
            return a_score > b_score;
        }
        
        return false;
    });
    
    if (maximizing) {
        float best_score = -std::numeric_limits<float>::infinity();
        std::optional<Move> best_move;
        std::vector<Move> best_moves;

        for (size_t i = 0; i < max_moves; ++i) {
            const auto& move = moves[i];
            game_copy.make_move(move);   // make/unmake: no Game copy needed
            game_copy.update();

            auto result = alphabeta(game_copy, depth - 1, alpha, beta, false);

            game_copy.undo_move();       // restore board state

            if (result.score > best_score) {
                best_score = result.score;
                best_moves.clear();
                best_moves.push_back(move);
            } else if (std::abs(result.score - best_score) < 0.01f) {
                best_moves.push_back(move);
            }

            alpha = std::max(alpha, best_score);
            if (beta <= alpha) break;  // Beta cutoff
        }

        if (!best_moves.empty()) {
            std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
            best_move = best_moves[dist(rng)];
        }

        transposition_table[hash] = {best_score, depth};
        return {best_score, best_move};
    } else {
        float best_score = std::numeric_limits<float>::infinity();
        std::optional<Move> best_move;
        std::vector<Move> best_moves;

        for (size_t i = 0; i < max_moves; ++i) {
            const auto& move = moves[i];
            game_copy.make_move(move);   // make/unmake: no Game copy needed
            game_copy.update();

            auto result = alphabeta(game_copy, depth - 1, alpha, beta, true);

            game_copy.undo_move();       // restore board state

            if (result.score < best_score) {
                best_score = result.score;
                best_moves.clear();
                best_moves.push_back(move);
            } else if (std::abs(result.score - best_score) < 0.01f) {
                best_moves.push_back(move);
            }

            beta = std::min(beta, best_score);
            if (beta <= alpha) break;  // Alpha cutoff
        }

        if (!best_moves.empty()) {
            std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
            best_move = best_moves[dist(rng)];
        }

        transposition_table[hash] = {best_score, depth};
        return {best_score, best_move};
    }
}

std::optional<Move> AlphaBetaAI::choose_move() {
    std::vector<Move> moves = game.get_all_moves();
    
    if (moves.empty()) {
        return std::nullopt;
    }
    
    nodes_searched = 0;
    transposition_table.clear();  // Clear cache each turn
    
    Game game_copy = game;
    auto result = alphabeta(game_copy, max_depth, 
                           -std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity(),
                           true);
    
    // Debug: show nodes and cache hits
    // std::cout << "AlphaBeta searched " << nodes_searched << " nodes, cache size: " << transposition_table.size() << std::endl;
    
    return result.move.has_value() ? result.move : moves[0];
}

// ===== EvolvableAI =====
// Piece type -> weight index (King excluded, fixed at 10000):
//   1=Sylph→0, 2=Griffin→1, 3=Dragon→2, 4=Oliphant→3, 5=Unicorn→4,
//   6=Hero→5, 7=Thief→6, 8=Cleric→7, 9=Mage→8, 10=King(fixed),
//   11=Paladin→9, 12=Warrior→10, 13=Basilisk→11, 14=Elemental→12, 15=Dwarf→13
static const int PIECE_TO_WEIGHT_IDX[16] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 9, 10, 11, 12, 13
};

EvolvableAI::EvolvableAI(Game& game, Color color, const std::vector<float>& weights, int depth)
    : AlphaBetaAI(game, color, depth)
    , weights(weights)
{
}

float EvolvableAI::evaluate_material(const Game& g) const {
    float score = 0.0f;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        int16_t piece = g.board[i];
        if (piece == EMPTY) continue;
        int abs_piece = std::abs(piece);
        float value;
        if (abs_piece == 10) {
            value = 10000.0f;  // King always fixed
        } else {
            int idx = PIECE_TO_WEIGHT_IDX[abs_piece];
            value = (idx >= 0 && idx < (int)weights.size()) ? weights[idx] : 0.0f;
        }
        if ((piece > 0 && color == Color::GOLD) || (piece < 0 && color == Color::SCARLET))
            score += value;
        else
            score -= value;
    }
    return score;
}

// ===== TDEvalAI =====
TDEvalAI::TDEvalAI(Game& game, Color color, const std::vector<float>& weights, int depth)
    : AlphaBetaAI(game, color, depth)
    , weights(weights)
{
}

float TDEvalAI::evaluate_material(const Game& g) const {
    const std::vector<float> features = extract_td_features(g);
    float score = 0.0f;
    int n = static_cast<int>(std::min(features.size(), weights.size()));
    for (int i = 0; i < n; ++i)
        score += weights[i] * features[i];
    // Features are Gold-positive; flip for Scarlet so the maximizer always
    // maximizes from its own perspective (matching BaseAI convention).
    return (color == Color::GOLD) ? score : -score;
}

} // namespace dragonchess
