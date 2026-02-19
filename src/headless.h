#pragma once

#include "game.h"
#include "ai.h"
#include <string>
#include <vector>
#include <memory>

namespace dragonchess {

struct AIConfig {
    std::string type;  // "random", "greedy", "greedyvalue", "minimax", "alphabeta", "plugin"
    int depth = 2;     // For minimax/alphabeta
    std::string name;  // Optional custom name
    std::string plugin_path;  // Path to .so file for plugin type
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

} // namespace dragonchess
