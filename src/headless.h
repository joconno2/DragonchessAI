#pragma once

#include "game.h"
#include "ai.h"
#include <string>
#include <vector>
#include <memory>

namespace dragonchess {

struct AIConfig {
    std::string type;  // "random", "greedy", "greedyvalue", "minimax", "alphabeta", "plugin", "evolvable"
    int depth = 2;     // For minimax/alphabeta/evolvable
    float time_per_move_ms = 0.0f;  // If > 0, use iterative deepening with time limit
    std::string name;  // Optional custom name
    std::string plugin_path;  // Path to .so file for plugin type
    std::vector<float> weights;  // For evolvable: 14 piece-value weights (no King)
};

struct MatchResult {
    std::string gold_ai;
    std::string scarlet_ai;
    std::string winner;
    int total_moves;
    int gold_pieces_remaining;
    int scarlet_pieces_remaining;
    bool is_checkmate;
    double duration_ms;
};

struct TournamentResults {
    std::vector<MatchResult> matches;
    int total_games;
    int gold_wins;
    int scarlet_wins;
    int draws;
    double total_time_ms;
    double avg_game_length;
};

// Create AI from config
std::unique_ptr<BaseAI> create_ai(const AIConfig& config, Game& game, Color color);

// Run a single headless match (no rendering, maximum speed)
MatchResult run_headless_match(const AIConfig& gold_config, const AIConfig& scarlet_config, 
                               int max_moves = 1000, bool verbose = false);

// Run a headless tournament (parallel execution)
TournamentResults run_headless_tournament(const AIConfig& gold_config, const AIConfig& scarlet_config,
                                         int num_games = 100, int num_threads = 0, bool verbose = false);

// Output results
void print_match_result(const MatchResult& result);
void print_tournament_results(const TournamentResults& results);
void export_results_csv(const TournamentResults& results, const std::string& filename);
void export_results_json(const TournamentResults& results, const std::string& filename);

// ---------------------------------------------------------------------------
// Self-play recording (TD training data generation)
// ---------------------------------------------------------------------------

// One position sampled during a self-play game.
// Sparse features, Gold-positive (see td_features.h).
struct PositionRecord {
    std::vector<SparseFeature> features;
};

// Full record for a single self-play game.
struct GameRecord {
    float outcome;                       // +1 Gold wins, -1 Scarlet wins, 0 draw
    std::vector<PositionRecord> positions; // feature snapshot before each half-move
};

// Play one game and record position features at every half-move.
GameRecord run_selfplay_game(const AIConfig& gold_config, const AIConfig& scarlet_config,
                             int max_moves = 500);

// Play num_games games in parallel, write NDJSON game records to `out`.
// Each line: {"o": <outcome>, "p": [{"i":[idx,...],"v":[val,...]}, ...]}
void run_selfplay_batch(const AIConfig& gold_config, const AIConfig& scarlet_config,
                        int num_games, int num_threads, std::ostream& out);

// ---------------------------------------------------------------------------
// Search-supervised label generation (Stockfish NNUE approach)
// ---------------------------------------------------------------------------

// One labeled position: sparse features + AB search score at given depth.
struct LabeledPosition {
    std::vector<SparseFeature> features;
    float search_score;  // AB(depth=N) score from Gold's perspective
};

// Play num_games random/shallow games. At each position, run AB(label_depth)
// with the handcrafted eval and record (features, score). Write NDJSON to out.
// Each line: {"i":[idx,...],"v":[val,...],"s":<score>}
// random_plies: first N plies use random moves for opening diversity.
void run_genlabels_batch(int num_games, int label_depth, int random_plies,
                         int num_threads, std::ostream& out);

} // namespace dragonchess
