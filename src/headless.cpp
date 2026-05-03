#include "headless.h"
#include "ai_plugin.h"
#include "td_features.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <sstream>
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
    } else if (type_lower == "tdeval") {
        return std::make_unique<TDEvalAI>(game, color, config.weights, config.depth);
    } else if (type_lower == "nneval") {
        NNWeights nn;
        if (!nn.from_flat(config.weights)) {
            std::cerr << "nneval: expected " << NNWeights::total_params()
                      << " weights, got " << config.weights.size()
                      << ". Falling back to RandomAI." << std::endl;
            return std::make_unique<RandomAI>(game, color);
        }
        return std::make_unique<NNEvalAI>(game, color, nn, config.depth);
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
        std::optional<Move> move;
        if (game.current_turn == Color::GOLD) {
            auto* ab = dynamic_cast<AlphaBetaAI*>(gold_ai.get());
            if (ab && gold_config.time_per_move_ms > 0)
                move = ab->choose_move_timed(gold_config.time_per_move_ms);
            else
                move = gold_ai->choose_move();
        } else {
            auto* ab = dynamic_cast<AlphaBetaAI*>(scarlet_ai.get());
            if (ab && scarlet_config.time_per_move_ms > 0)
                move = ab->choose_move_timed(scarlet_config.time_per_move_ms);
            else
                move = scarlet_ai->choose_move();
        }
        
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

// ---------------------------------------------------------------------------
// Self-play recording
// ---------------------------------------------------------------------------

GameRecord run_selfplay_game(const AIConfig& gold_config, const AIConfig& scarlet_config,
                             int max_moves) {
    Game game;
    auto gold_ai    = create_ai(gold_config,    game, Color::GOLD);
    auto scarlet_ai = create_ai(scarlet_config, game, Color::SCARLET);

    // Try to use TDLeaf if AIs support it (TDEvalAI or NNEvalAI)
    auto* gold_td = dynamic_cast<TDEvalAI*>(gold_ai.get());
    auto* scarlet_td = dynamic_cast<TDEvalAI*>(scarlet_ai.get());
    auto* gold_nn = dynamic_cast<NNEvalAI*>(gold_ai.get());
    auto* scarlet_nn = dynamic_cast<NNEvalAI*>(scarlet_ai.get());

    GameRecord record;
    int moves_without_progress = 0;
    const int MAX_NO_PROGRESS = 100;

    while (!game.game_over && static_cast<int>(game.game_log.size()) < max_moves) {
        std::optional<Move> move;

        if (game.current_turn == Color::GOLD) {
            if (gold_nn) {
                auto result = gold_nn->choose_move_tdleaf();
                record.positions.push_back({std::move(result.leaf_features)});
                move = result.move;
            } else if (gold_td) {
                auto result = gold_td->choose_move_tdleaf();
                record.positions.push_back({std::move(result.leaf_features)});
                move = result.move;
            } else {
                record.positions.push_back({extract_td_features_sparse(game)});
                move = gold_ai->choose_move();
            }
        } else {
            if (scarlet_nn) {
                auto result = scarlet_nn->choose_move_tdleaf();
                record.positions.push_back({std::move(result.leaf_features)});
                move = result.move;
            } else if (scarlet_td) {
                auto result = scarlet_td->choose_move_tdleaf();
                record.positions.push_back({std::move(result.leaf_features)});
                move = result.move;
            } else {
                record.positions.push_back({extract_td_features_sparse(game)});
                move = scarlet_ai->choose_move();
            }
        }

        if (!move.has_value()) {
            game.game_over = true;
            game.winner = "Draw";
            break;
        }

        game.make_move(move.value());
        game.update();

        if (game.no_capture_count == 0)
            moves_without_progress = 0;
        else
            ++moves_without_progress;

        if (moves_without_progress >= MAX_NO_PROGRESS) {
            game.game_over = true;
            game.winner = "Draw";
            break;
        }
    }

    if (!game.game_over) {
        game.game_over = true;
        game.winner = "Draw";
    }

    if (game.winner == "Gold")
        record.outcome = 1.0f;
    else if (game.winner == "Scarlet")
        record.outcome = -1.0f;
    else
        record.outcome = 0.0f;

    return record;
}

// Serialize one GameRecord as a compact JSON line (sparse format).
// Format: {"o":<outcome>,"p":[{"i":[idx,...],"v":[val,...]}, ...]}
static std::string game_record_to_json(const GameRecord& rec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5);
    ss << "{\"o\":" << rec.outcome << ",\"p\":[";
    for (size_t pi = 0; pi < rec.positions.size(); ++pi) {
        if (pi > 0) ss << ',';
        const auto& feat = rec.positions[pi].features;
        ss << "{\"i\":[";
        for (size_t fi = 0; fi < feat.size(); ++fi) {
            if (fi > 0) ss << ',';
            ss << feat[fi].index;
        }
        ss << "],\"v\":[";
        for (size_t fi = 0; fi < feat.size(); ++fi) {
            if (fi > 0) ss << ',';
            ss << feat[fi].value;
        }
        ss << "]}";
    }
    ss << "]}";
    return ss.str();
}

void run_selfplay_batch(const AIConfig& gold_config, const AIConfig& scarlet_config,
                        int num_games, int num_threads, std::ostream& out) {
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }

    std::vector<GameRecord> records(static_cast<size_t>(num_games));
    std::vector<std::thread> threads;
    std::atomic<int> next_game{0};

    auto worker = [&]() {
        for (;;) {
            int idx = next_game.fetch_add(1);
            if (idx >= num_games) break;
            records[static_cast<size_t>(idx)] =
                run_selfplay_game(gold_config, scarlet_config);
        }
    };

    int nw = std::min(num_threads, num_games);
    threads.reserve(static_cast<size_t>(nw));
    for (int i = 0; i < nw; ++i)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    // Write NDJSON — one game per line, serial to preserve ordering
    for (const auto& rec : records)
        out << game_record_to_json(rec) << '\n';
}

// ---------------------------------------------------------------------------
// Search-supervised label generation
// ---------------------------------------------------------------------------

// Generate labeled positions from one game.
// First random_plies moves are random for opening diversity,
// then AB(d=1) plays out the rest. At every position, AB(label_depth) with
// the handcrafted eval scores the position from Gold's perspective.
static std::vector<LabeledPosition> generate_labeled_game(
    int label_depth, int random_plies, int max_moves, std::mt19937& rng)
{
    Game game;
    // Labeler uses handcrafted eval at deep search for ground truth
    AlphaBetaAI labeler(game, Color::GOLD);

    AlphaBetaAI playout_gold(game, Color::GOLD);
    playout_gold.set_max_depth(1);
    AlphaBetaAI playout_scarlet(game, Color::SCARLET);
    playout_scarlet.set_max_depth(1);

    std::vector<LabeledPosition> positions;
    int no_progress = 0;

    for (int ply = 0; ply < max_moves && !game.game_over; ++ply) {
        // Extract features and label with deep search
        auto features = extract_td_features_sparse(game);
        Game label_copy = game;
        float score = labeler.search_score(label_copy, label_depth);
        positions.push_back({std::move(features), score});

        // Choose move: random for opening diversity, then AB(d=1)
        std::optional<Move> move;
        auto moves = game.get_all_moves();
        if (moves.empty()) break;

        if (ply < random_plies) {
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            move = moves[dist(rng)];
        } else if (game.current_turn == Color::GOLD) {
            move = playout_gold.choose_move();
        } else {
            move = playout_scarlet.choose_move();
        }

        if (!move.has_value()) break;
        game.make_move(move.value());
        game.update();

        if (game.no_capture_count == 0) no_progress = 0;
        else ++no_progress;
        if (no_progress >= 100) break;
    }
    return positions;
}

// Serialize one LabeledPosition as JSON.
static std::string labeled_pos_to_json(const LabeledPosition& lp) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(5);
    ss << "{\"i\":[";
    for (size_t i = 0; i < lp.features.size(); ++i) {
        if (i > 0) ss << ',';
        ss << lp.features[i].index;
    }
    ss << "],\"v\":[";
    for (size_t i = 0; i < lp.features.size(); ++i) {
        if (i > 0) ss << ',';
        ss << lp.features[i].value;
    }
    ss << "],\"s\":" << lp.search_score << "}";
    return ss.str();
}

void run_genlabels_batch(int num_games, int label_depth, int random_plies,
                         int num_threads, std::ostream& out) {
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }

    std::vector<std::vector<LabeledPosition>> all_positions(
        static_cast<size_t>(num_games));
    std::atomic<int> next_game{0};

    auto worker = [&](unsigned seed) {
        std::mt19937 rng(seed);
        for (;;) {
            int idx = next_game.fetch_add(1);
            if (idx >= num_games) break;
            all_positions[static_cast<size_t>(idx)] =
                generate_labeled_game(label_depth, random_plies, 300, rng);
        }
    };

    int nw = std::min(num_threads, num_games);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nw));
    std::mt19937 seed_rng(std::random_device{}());
    for (int i = 0; i < nw; ++i)
        threads.emplace_back(worker, seed_rng());
    for (auto& t : threads)
        t.join();

    // Write one position per line
    for (const auto& game_positions : all_positions)
        for (const auto& lp : game_positions)
            out << labeled_pos_to_json(lp) << '\n';
}

} // namespace dragonchess
