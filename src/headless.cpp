#include "headless.h"
#include "ai_plugin.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace dragonchess {

std::unique_ptr<BaseAI> create_ai(const AIConfig& config, Game& game, Color color) {
    std::string type_lower = config.type;
    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);
    
    if (type_lower == "plugin") {
        // Load plugin AI
        auto plugin = std::make_unique<AIPlugin>();
        if (plugin->load(config.plugin_path, game, color)) {
            // Wrap SimpleAI in BaseAI adapter
            class PluginAdapter : public BaseAI {
            public:
                PluginAdapter(std::unique_ptr<AIPlugin> p, Game& g, Color c) 
                    : BaseAI(g, c), plugin(std::move(p)) {
                    plugin_ai = plugin->get_ai();
                }
                std::optional<Move> choose_move() override {
                    return plugin_ai->choose_move();
                }
                ~PluginAdapter() override = default;
            private:
                std::unique_ptr<AIPlugin> plugin;  // Keep plugin loaded
                SimpleAI* plugin_ai;
            };
            return std::make_unique<PluginAdapter>(std::move(plugin), game, color);
        } else {
            std::cerr << "Failed to load plugin, falling back to RandomAI" << std::endl;
            return std::make_unique<RandomAI>(game, color);
        }
    } else if (type_lower == "random") {
        return std::make_unique<RandomAI>(game, color);
    } else if (type_lower == "greedy") {
        return std::make_unique<GreedyAI>(game, color);
    } else if (type_lower == "greedyvalue") {
        return std::make_unique<GreedyValueAI>(game, color);
    } else if (type_lower == "minimax") {
        return std::make_unique<MinimaxAI>(game, color, config.depth);
    } else if (type_lower == "alphabeta") {
        return std::make_unique<AlphaBetaAI>(game, color, config.depth);
    } else if (type_lower == "evolvable") {
        return std::make_unique<EvolvableAI>(game, color, config.weights, config.depth);
    } else {
        std::cerr << "Unknown AI type: " << config.type << ", defaulting to RandomAI" << std::endl;
        return std::make_unique<RandomAI>(game, color);
    }
}

MatchResult run_headless_match(const AIConfig& gold_config, const AIConfig& scarlet_config,
                               int max_moves, bool verbose) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create game
    Game game;
    
    // Create AIs
    auto gold_ai = create_ai(gold_config, game, Color::GOLD);
    auto scarlet_ai = create_ai(scarlet_config, game, Color::SCARLET);
    
    int move_count = 0;
    int moves_without_progress = 0;
    const int MAX_NO_PROGRESS = 100;
    
    // Play game at maximum speed (no delays, no rendering)
    while (!game.game_over && move_count < max_moves) {
        auto move = (game.current_turn == Color::GOLD) ? 
                    gold_ai->choose_move() : scarlet_ai->choose_move();
        
        if (move.has_value()) {
            int prev_no_capture = game.no_capture_count;
            game.make_move(move.value());
            game.update();
            move_count++;
            
            // Track progress
            if (game.no_capture_count == 0) {
                moves_without_progress = 0;
            } else {
                moves_without_progress++;
            }
            
            // Force draw if no progress
            if (moves_without_progress >= MAX_NO_PROGRESS) {
                game.game_over = true;
                game.winner = "Draw";
                break;
            }
        } else {
            // No legal moves - stalemate
            game.game_over = true;
            game.winner = "Draw";
            break;
        }
    }
    
    // Handle move limit
    if (move_count >= max_moves && !game.game_over) {
        game.game_over = true;
        game.winner = "Draw";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Count remaining pieces
    int gold_pieces = 0;
    int scarlet_pieces = 0;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        if (game.board[i] != EMPTY) {
            if (game.board[i] > 0) {  // Gold pieces are positive
                gold_pieces++;
            } else {  // Scarlet pieces are negative
                scarlet_pieces++;
            }
        }
    }
    
    MatchResult result;
    result.gold_ai = gold_config.name.empty() ? gold_config.type : gold_config.name;
    result.scarlet_ai = scarlet_config.name.empty() ? scarlet_config.type : scarlet_config.name;
    result.winner = game.winner;
    result.total_moves = move_count;
    result.gold_pieces_remaining = gold_pieces;
    result.scarlet_pieces_remaining = scarlet_pieces;
    result.is_checkmate = game.winner != "Draw" && game.game_over;
    result.duration_ms = duration_ms;
    
    if (verbose) {
        std::cout << "Match completed in " << duration_ms << "ms, " 
                  << move_count << " moves, Winner: " << game.winner << std::endl;
    }
    
    return result;
}

TournamentResults run_headless_tournament(const AIConfig& gold_config, const AIConfig& scarlet_config,
                                         int num_games, int num_threads, bool verbose) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    
    if (verbose) {
        std::cout << "Running " << num_games << " games with " << num_threads << " threads..." << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<MatchResult> all_results(num_games);
    std::mutex results_mutex;
    std::atomic<int> games_completed{0};
    
    // Worker function
    auto worker = [&](int start_idx, int end_idx) {
        for (int i = start_idx; i < end_idx; ++i) {
            MatchResult result = run_headless_match(gold_config, scarlet_config, 1000, false);
            
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                all_results[i] = result;
            }
            
            int completed = ++games_completed;
            if (verbose && completed % 10 == 0) {
                std::cout << "Completed " << completed << "/" << num_games << " games" << std::endl;
            }
        }
    };
    
    // Launch threads
    std::vector<std::thread> threads;
    int games_per_thread = num_games / num_threads;
    int remaining_games = num_games % num_threads;
    
    int current_idx = 0;
    for (int i = 0; i < num_threads; ++i) {
        int start = current_idx;
        int count = games_per_thread + (i < remaining_games ? 1 : 0);
        int end = start + count;
        
        threads.emplace_back(worker, start, end);
        current_idx = end;
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Compile results
    TournamentResults tournament_results;
    tournament_results.matches = all_results;
    tournament_results.total_games = num_games;
    tournament_results.gold_wins = 0;
    tournament_results.scarlet_wins = 0;
    tournament_results.draws = 0;
    tournament_results.total_time_ms = total_time_ms;
    
    int total_moves = 0;
    for (const auto& result : all_results) {
        total_moves += result.total_moves;
        
        if (result.winner == "Gold") {
            tournament_results.gold_wins++;
        } else if (result.winner == "Scarlet") {
            tournament_results.scarlet_wins++;
        } else {
            tournament_results.draws++;
        }
    }
    
    tournament_results.avg_game_length = static_cast<double>(total_moves) / num_games;
    
    return tournament_results;
}

void print_match_result(const MatchResult& result) {
    std::cout << "\n=== Match Result ===" << std::endl;
    std::cout << "Gold AI:    " << result.gold_ai << std::endl;
    std::cout << "Scarlet AI: " << result.scarlet_ai << std::endl;
    std::cout << "Winner:     " << result.winner << std::endl;
    std::cout << "Moves:      " << result.total_moves << std::endl;
    std::cout << "Duration:   " << std::fixed << std::setprecision(2) << result.duration_ms << "ms" << std::endl;
    std::cout << "Gold pieces remaining:    " << result.gold_pieces_remaining << std::endl;
    std::cout << "Scarlet pieces remaining: " << result.scarlet_pieces_remaining << std::endl;
    std::cout << "Checkmate:  " << (result.is_checkmate ? "Yes" : "No") << std::endl;
}

void print_tournament_results(const TournamentResults& results) {
    std::cout << "\n=== Tournament Results ===" << std::endl;
    std::cout << "Total games:     " << results.total_games << std::endl;
    std::cout << "Gold wins:       " << results.gold_wins 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * results.gold_wins / results.total_games) << "%)" << std::endl;
    std::cout << "Scarlet wins:    " << results.scarlet_wins 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * results.scarlet_wins / results.total_games) << "%)" << std::endl;
    std::cout << "Draws:           " << results.draws 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * results.draws / results.total_games) << "%)" << std::endl;
    std::cout << "Avg game length: " << std::fixed << std::setprecision(1) << results.avg_game_length << " moves" << std::endl;
    std::cout << "Total time:      " << std::fixed << std::setprecision(2) << results.total_time_ms << "ms" << std::endl;
    std::cout << "Time per game:   " << std::fixed << std::setprecision(2) 
              << (results.total_time_ms / results.total_games) << "ms" << std::endl;
    std::cout << "Games per second:" << std::fixed << std::setprecision(2) 
              << (1000.0 * results.total_games / results.total_time_ms) << std::endl;
}

void export_results_csv(const TournamentResults& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    
    // Header
    file << "game_id,gold_ai,scarlet_ai,winner,moves,duration_ms,gold_pieces,scarlet_pieces,checkmate\n";
    
    // Data
    for (size_t i = 0; i < results.matches.size(); ++i) {
        const auto& m = results.matches[i];
        file << i << ","
             << m.gold_ai << ","
             << m.scarlet_ai << ","
             << m.winner << ","
             << m.total_moves << ","
             << std::fixed << std::setprecision(3) << m.duration_ms << ","
             << m.gold_pieces_remaining << ","
             << m.scarlet_pieces_remaining << ","
             << (m.is_checkmate ? "yes" : "no") << "\n";
    }
    
    file.close();
    std::cout << "Results exported to " << filename << std::endl;
}

static void write_results_json(std::ostream& out, const TournamentResults& results) {
    out << "{\n";
    out << "  \"summary\": {\n";
    out << "    \"total_games\": " << results.total_games << ",\n";
    out << "    \"gold_wins\": " << results.gold_wins << ",\n";
    out << "    \"scarlet_wins\": " << results.scarlet_wins << ",\n";
    out << "    \"draws\": " << results.draws << ",\n";
    out << "    \"avg_game_length\": " << std::fixed << std::setprecision(2) << results.avg_game_length << ",\n";
    out << "    \"total_time_ms\": " << std::fixed << std::setprecision(2) << results.total_time_ms << ",\n";
    out << "    \"avg_time_per_game_ms\": " << std::fixed << std::setprecision(2)
        << (results.total_time_ms / results.total_games) << "\n";
    out << "  },\n";
    out << "  \"matches\": [\n";

    for (size_t i = 0; i < results.matches.size(); ++i) {
        const auto& m = results.matches[i];
        out << "    {\n";
        out << "      \"game_id\": " << i << ",\n";
        out << "      \"gold_ai\": \"" << m.gold_ai << "\",\n";
        out << "      \"scarlet_ai\": \"" << m.scarlet_ai << "\",\n";
        out << "      \"winner\": \"" << m.winner << "\",\n";
        out << "      \"moves\": " << m.total_moves << ",\n";
        out << "      \"duration_ms\": " << std::fixed << std::setprecision(3) << m.duration_ms << ",\n";
        out << "      \"gold_pieces_remaining\": " << m.gold_pieces_remaining << ",\n";
        out << "      \"scarlet_pieces_remaining\": " << m.scarlet_pieces_remaining << ",\n";
        out << "      \"checkmate\": " << (m.is_checkmate ? "true" : "false") << "\n";
        out << "    }" << (i < results.matches.size() - 1 ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

void export_results_json(const TournamentResults& results, const std::string& filename) {
    // Special value "-" writes to stdout (used by training scripts to avoid temp files)
    if (filename == "-") {
        write_results_json(std::cout, results);
        return;
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    write_results_json(file, results);
    file.close();
    std::cout << "Results exported to " << filename << std::endl;
}

} // namespace dragonchess
