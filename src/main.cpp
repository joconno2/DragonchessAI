#include "game.h"
#include "ai.h"
#include "renderer.h"
#include "input.h"
#include "input_helpers.h"
#include "menu.h"
#include "tournament.h"
#include "campaign.h"
#include "headless.h"
#include <iostream>
#include <unordered_set>
#include <optional>
#include <thread>
#include <chrono>
#include <memory>
#include <cstring>

using namespace dragonchess;

void run_two_player_game(Renderer& renderer, InputHandler& input);
void run_ai_vs_player_game(Renderer& renderer, InputHandler& input, const AISettings& settings);
void run_ai_vs_ai_game(Renderer& renderer, InputHandler& input, const AISettings& settings);
void run_tournament_mode(Renderer& renderer, InputHandler& input);
void run_campaign_mode(Renderer& renderer, InputHandler& input);
void print_usage(const char* program_name);
int run_headless_mode(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    // Check for headless mode
    if (argc > 1 && (strcmp(argv[1], "--headless") == 0 || strcmp(argv[1], "-h") == 0)) {
        return run_headless_mode(argc, argv);
    }
    
    // Check for help
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-?") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Initialize renderer with smaller, more manageable window size
    // Default to 1280x720 which will scale the 1600x900 virtual resolution
    Renderer renderer;
    if (!renderer.init("Dragonchess", 1280, 720)) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return 1;
    }
    
    if (!renderer.load_assets()) {
        std::cerr << "Failed to load assets" << std::endl;
        return 1;
    }
    
    // Initialize menu
    Menu menu;
    if (!menu.init(renderer.get_renderer())) {
        std::cerr << "Failed to initialize menu" << std::endl;
        return 1;
    }
    
    // Create input handler
    InputHandler input;
    
    // Main menu loop
    bool running = true;
    while (running) {
        GameMode mode = menu.show_main_menu(renderer.get_renderer());
        
        switch (mode) {
            case GameMode::TWO_PLAYER:
                run_two_player_game(renderer, input);
                break;
                
            case GameMode::AI_VS_PLAYER: {
                AISettings settings = menu.show_ai_vs_player_menu(renderer.get_renderer());
                if (!settings.player_side.empty()) {
                    run_ai_vs_player_game(renderer, input, settings);
                }
                break;
            }
                
            case GameMode::AI_VS_AI: {
                AISettings settings = menu.show_ai_vs_ai_menu(renderer.get_renderer());
                run_ai_vs_ai_game(renderer, input, settings);
                break;
            }
                
            case GameMode::TOURNAMENT:
                run_tournament_mode(renderer, input);
                break;
                
            case GameMode::CAMPAIGN:
                run_campaign_mode(renderer, input);
                break;
                
            case GameMode::QUIT:
                running = false;
                break;
        }
    }
    
    return 0;
}

void run_two_player_game(Renderer& renderer, InputHandler& input) {
    // Create game
    Game game;
    
    // Game state
    std::optional<int> selected;
    std::unordered_set<int> legal_dest;
    
    // Main game loop
    bool running = true;
    while (running && !game.game_over) {
        int mouse_x = 0, mouse_y = 0;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        switch (event) {
            case InputEvent::QUIT:
                running = false;
                break;
                
            case InputEvent::MOUSE_CLICK: {
                // Convert screen to board coordinates
                auto pos = renderer.screen_to_board(mouse_x, mouse_y);
                if (pos.has_value()) {
                    auto [layer, row, col] = pos.value();
                    int idx = pos_to_index(layer, row, col);
                    int16_t piece = game.board[idx];
                    
                    // Check if this is a valid piece for current player
                    bool valid_piece = (game.current_turn == Color::GOLD && piece > 0) ||
                                      (game.current_turn == Color::SCARLET && piece < 0);
                    
                    if (!selected.has_value()) {
                        // Select piece
                        if (valid_piece) {
                            selected = idx;
                            auto moves = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        }
                    } else {
                        // Try to make a move
                        if (legal_dest.count(idx) > 0) {
                            // Find the matching move
                            auto moves = game.get_legal_moves_for(selected.value());
                            for (const auto& move : moves) {
                                if (std::get<1>(move) == idx) {
                                    game.make_move(move);
                                    game.update();
                                    selected.reset();
                                    legal_dest.clear();
                                    break;
                                }
                            }
                        } else if (valid_piece) {
                            // Select different piece
                            selected = idx;
                            auto moves = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        } else {
                            // Deselect
                            selected.reset();
                            legal_dest.clear();
                        }
                    }
                }
                break;
            }
                
            case InputEvent::NONE:
                break;
        }
        
        // Render
        renderer.render_board(game, selected, legal_dest);
        renderer.present();
        
        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
    
    // Show post-game screen if game ended normally
    if (game.game_over && running) {
        bool showing = true;
        while (showing) {
            int mouse_x = 0, mouse_y = 0;
            InputEvent event = input.poll(mouse_x, mouse_y);
            
            if (event == InputEvent::QUIT) {
                showing = false;
            } else if (event == InputEvent::MOUSE_CLICK) {
                // Check if clicked on "Back to Menu" button
                auto [win_w, win_h] = renderer.get_window_size();
                SDL_Rect button = {win_w / 2 - 100, win_h * 3 / 4, 200, 50};
                if (mouse_x >= button.x && mouse_x < button.x + button.w &&
                    mouse_y >= button.y && mouse_y < button.y + button.h) {
                    showing = false;
                }
            }
            
            renderer.render_post_game(game);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    
    std::cout << "Game Over! Winner: " << game.winner << std::endl;
    std::cout << "Total moves: " << game.move_notations.size() << std::endl;
}

void run_ai_vs_player_game(Renderer& renderer, InputHandler& input, const AISettings& settings) {
    // Create game
    Game game;
    
    // Game state
    std::optional<int> selected;
    std::unordered_set<int> legal_dest;
    
    // Determine AI color
    Color ai_color = (settings.player_side == "Gold") ? Color::SCARLET : Color::GOLD;
    RandomAI ai(game, ai_color);
    
    std::cout << "Starting AI vs Player game. Player is " << settings.player_side << std::endl;
    
    // Main game loop
    bool running = true;
    while (running && !game.game_over) {
        int mouse_x = 0, mouse_y = 0;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        switch (event) {
            case InputEvent::QUIT:
                running = false;
                break;
                
            case InputEvent::MOUSE_CLICK: {
                if (game.current_turn == ai_color) {
                    // Don't process clicks during AI turn
                    break;
                }
                
                // Convert screen to board coordinates
                auto pos = renderer.screen_to_board(mouse_x, mouse_y);
                if (pos.has_value()) {
                    auto [layer, row, col] = pos.value();
                    int idx = pos_to_index(layer, row, col);
                    int16_t piece = game.board[idx];
                    
                    // Check if this is a valid piece for current player
                    bool valid_piece = (game.current_turn == Color::GOLD && piece > 0) ||
                                      (game.current_turn == Color::SCARLET && piece < 0);
                    
                    if (!selected.has_value()) {
                        // Select piece
                        if (valid_piece) {
                            selected = idx;
                            auto moves = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        }
                    } else {
                        // Try to make a move
                        if (legal_dest.count(idx) > 0) {
                            // Find the matching move
                            auto moves = game.get_legal_moves_for(selected.value());
                            for (const auto& move : moves) {
                                if (std::get<1>(move) == idx) {
                                    game.make_move(move);
                                    game.update();
                                    selected.reset();
                                    legal_dest.clear();
                                    break;
                                }
                            }
                        } else if (valid_piece) {
                            // Select different piece
                            selected = idx;
                            auto moves = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        } else {
                            // Deselect
                            selected.reset();
                            legal_dest.clear();
                        }
                    }
                }
                break;
            }
                
            case InputEvent::NONE:
                break;
        }
        
        // AI turn
        if (game.current_turn == ai_color && !game.game_over) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            auto move = ai.choose_move();
            if (move.has_value()) {
                game.make_move(move.value());
                game.update();
                selected.reset();
                legal_dest.clear();
            }
        }
        
        // Render
        renderer.render_board(game, selected, legal_dest);
        renderer.present();
        
        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
    
    // Show post-game screen if game ended normally
    if (game.game_over && running) {
        bool showing = true;
        while (showing) {
            int mouse_x = 0, mouse_y = 0;
            InputEvent event = input.poll(mouse_x, mouse_y);
            
            if (event == InputEvent::QUIT) {
                showing = false;
            } else if (event == InputEvent::MOUSE_CLICK) {
                // Check if clicked on "Back to Menu" button
                auto [win_w, win_h] = renderer.get_window_size();
                SDL_Rect button = {win_w / 2 - 100, win_h * 3 / 4, 200, 50};
                if (mouse_x >= button.x && mouse_x < button.x + button.w &&
                    mouse_y >= button.y && mouse_y < button.y + button.h) {
                    showing = false;
                }
            }
            
            renderer.render_post_game(game);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    
    std::cout << "Game Over! Winner: " << game.winner << std::endl;
    std::cout << "Total moves: " << game.move_notations.size() << std::endl;
}

void run_ai_vs_ai_game(Renderer& renderer, InputHandler& input, const AISettings& settings) {
    // Check if cancelled
    if (settings.gold_ai_path == "CANCELLED") {
        return;
    }
    
    // Create game
    Game game;
    
    // Helper to create AI based on type string
    auto create_ai = [&game](const std::string& ai_type, Color color) -> std::unique_ptr<BaseAI> {
        if (ai_type.find("GreedyValueAI") != std::string::npos) {
            return std::make_unique<GreedyValueAI>(game, color);
        } else if (ai_type.find("GreedyAI") != std::string::npos) {
            return std::make_unique<GreedyAI>(game, color);
        } else if (ai_type.find("MinimaxAI") != std::string::npos) {
            return std::make_unique<MinimaxAI>(game, color, 2);  // depth 2
        } else if (ai_type.find("AlphaBetaAI") != std::string::npos) {
            return std::make_unique<AlphaBetaAI>(game, color, 3);  // depth 3
        } else {
            return std::make_unique<RandomAI>(game, color);
        }
    };
    
    // Create both AIs based on user selection
    auto gold_ai = create_ai(settings.gold_ai_path, Color::GOLD);
    auto scarlet_ai = create_ai(settings.scarlet_ai_path, Color::SCARLET);
    
    std::cout << "Starting AI vs AI game: " << settings.gold_ai_path 
              << " (Gold) vs " << settings.scarlet_ai_path << " (Scarlet)" << std::endl;
    
    // Game state
    std::optional<int> selected;
    std::unordered_set<int> legal_dest;
    
    // Main game loop
    bool running = true;
    auto last_move_time = std::chrono::steady_clock::now();
    
    while (running && !game.game_over) {
        int mouse_x = 0, mouse_y = 0;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        if (event == InputEvent::QUIT || event == InputEvent::ESC_PRESSED) {
            running = false;
            break;
        }
        
        // AI makes move every 500ms
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move_time);
        
        if (elapsed.count() >= 500 && !game.game_over) {
            auto move = (game.current_turn == Color::GOLD) ? gold_ai->choose_move() : scarlet_ai->choose_move();
            if (move.has_value()) {
                game.make_move(move.value());
                game.update();
                last_move_time = now;
            }
        }
        
        // Render
        renderer.render_board(game, selected, legal_dest);
        renderer.present();
        
        // Small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    // Show post-game screen
    if (game.game_over && running) {
        bool showing = true;
        while (showing) {
            int mouse_x = 0, mouse_y = 0;
            InputEvent event = input.poll(mouse_x, mouse_y);
            
            if (event == InputEvent::QUIT) {
                showing = false;
            } else if (event == InputEvent::MOUSE_CLICK) {
                auto [win_w, win_h] = renderer.get_window_size();
                SDL_Rect button = {win_w / 2 - 100, win_h * 3 / 4, 200, 50};
                if (mouse_x >= button.x && mouse_x < button.x + button.w &&
                    mouse_y >= button.y && mouse_y < button.y + button.h) {
                    showing = false;
                }
            }
            
            renderer.render_post_game(game);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    
    std::cout << "Game Over! Winner: " << game.winner << std::endl;
    std::cout << "Total moves: " << game.move_notations.size() << std::endl;
}

void run_tournament_mode(Renderer& renderer, InputHandler& input) {
    Tournament tournament;
    
    // Let user select bots
    if (!tournament.select_bots(renderer, input)) {
        std::cout << "Tournament cancelled." << std::endl;
        return;
    }
    
    // Run 5 rounds
    tournament.run(renderer, input, 5);
}

void run_campaign_mode(Renderer& renderer, InputHandler& input) {
    Campaign campaign;
    campaign.run(renderer, input);
}

void print_usage(const char* program_name) {
    std::cout << "Dragonchess AI - 3D Chess Research Platform\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << program_name << "                    # Launch GUI mode\n";
    std::cout << "  " << program_name << " --headless [OPTIONS] # Run headless for research\n\n";
    
    std::cout << "HEADLESS MODE OPTIONS:\n";
    std::cout << "  --mode <match|tournament>  Mode: single match or tournament (default: match)\n";
    std::cout << "  --games <N>                Number of games for tournament (default: 100)\n";
    std::cout << "  --threads <N>              Number of threads (default: auto-detect)\n";
    std::cout << "  --max-moves <N>            Maximum moves per game (default: 1000)\n\n";
    
    std::cout << "AI CONFIGURATION:\n";
    std::cout << "  --gold-ai <type>           Gold AI type (required)\n";
    std::cout << "  --scarlet-ai <type>        Scarlet AI type (required)\n";
    std::cout << "  --gold-ai-plugin <file>    Load gold AI from .so plugin file\n";
    std::cout << "  --scarlet-ai-plugin <file> Load scarlet AI from .so plugin file\n";
    std::cout << "  --gold-depth <N>           Search depth for minimax/alphabeta (default: 2)\n";
    std::cout << "  --scarlet-depth <N>        Search depth for minimax/alphabeta (default: 2)\n";
    std::cout << "  --gold-name <name>         Custom name for gold AI\n";
    std::cout << "  --scarlet-name <name>      Custom name for scarlet AI\n\n";
    
    std::cout << "AI TYPES:\n";
    std::cout << "  random                     Random legal move selection\n";
    std::cout << "  greedy                     Greedy capture-focused\n";
    std::cout << "  greedyvalue                Greedy with piece value evaluation\n";
    std::cout << "  minimax                    Minimax search (specify depth)\n";
    std::cout << "  alphabeta                  Alpha-beta pruning (specify depth)\n\n";
    
    std::cout << "OUTPUT OPTIONS:\n";
    std::cout << "  --output-csv <file>        Export results to CSV\n";
    std::cout << "  --output-json <file>       Export results to JSON\n";
    std::cout << "  --verbose                  Print detailed progress\n";
    std::cout << "  --quiet                    Minimal output\n\n";
    
    std::cout << "EXAMPLES:\n";
    std::cout << "  # Single match: Greedy vs Random\n";
    std::cout << "  " << program_name << " --headless --gold-ai greedy --scarlet-ai random\n\n";
    
    std::cout << "  # Tournament: 100 games, minimax depth 3 vs alphabeta depth 3\n";
    std::cout << "  " << program_name << " --headless --mode tournament --games 100 \\\n";
    std::cout << "    --gold-ai minimax --gold-depth 3 \\\n";
    std::cout << "    --scarlet-ai alphabeta --scarlet-depth 3 \\\n";
    std::cout << "    --output-csv results.csv --verbose\n\n";
    
    std::cout << "  # Fast tournament: Random vs Random, 1000 games, 8 threads\n";
    std::cout << "  " << program_name << " --headless --mode tournament --games 1000 --threads 8 \\\n";
    std::cout << "    --gold-ai random --scarlet-ai random --output-json results.json\n\n";
}

int run_headless_mode(int argc, char* argv[]) {
    // Default config
    std::string mode = "match";
    int num_games = 100;
    int num_threads = 0;  // auto-detect
    int max_moves = 1000;
    bool verbose = false;
    bool quiet = false;
    std::string output_csv;
    std::string output_json;
    
    AIConfig gold_config;
    AIConfig scarlet_config;
    gold_config.depth = 2;
    scarlet_config.depth = 2;
    
    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--max-moves" && i + 1 < argc) {
            max_moves = std::atoi(argv[++i]);
        } else if (arg == "--gold-ai" && i + 1 < argc) {
            gold_config.type = argv[++i];
        } else if (arg == "--scarlet-ai" && i + 1 < argc) {
            scarlet_config.type = argv[++i];
        } else if (arg == "--gold-depth" && i + 1 < argc) {
            gold_config.depth = std::atoi(argv[++i]);
        } else if (arg == "--scarlet-depth" && i + 1 < argc) {
            scarlet_config.depth = std::atoi(argv[++i]);
        } else if (arg == "--gold-name" && i + 1 < argc) {
            gold_config.name = argv[++i];
        } else if (arg == "--scarlet-name" && i + 1 < argc) {
            scarlet_config.name = argv[++i];
        } else if (arg == "--gold-ai-plugin" && i + 1 < argc) {
            gold_config.type = "plugin";
            gold_config.plugin_path = argv[++i];
        } else if (arg == "--scarlet-ai-plugin" && i + 1 < argc) {
            scarlet_config.type = "plugin";
            scarlet_config.plugin_path = argv[++i];
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--output-json" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--help" || arg == "-?") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required parameters
    if (gold_config.type.empty() || scarlet_config.type.empty()) {
        std::cerr << "Error: Both --gold-ai and --scarlet-ai are required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (!quiet) {
        std::cout << "=== Dragonchess Headless Mode ===" << std::endl;
        std::cout << "Mode: " << mode << std::endl;
        std::cout << "Gold AI: " << gold_config.type;
        if (!gold_config.name.empty()) std::cout << " (" << gold_config.name << ")";
        std::cout << " [depth: " << gold_config.depth << "]" << std::endl;
        std::cout << "Scarlet AI: " << scarlet_config.type;
        if (!scarlet_config.name.empty()) std::cout << " (" << scarlet_config.name << ")";
        std::cout << " [depth: " << scarlet_config.depth << "]" << std::endl;
    }
    
    if (mode == "match") {
        // Run single match
        if (!quiet) std::cout << "\nRunning single match..." << std::endl;
        MatchResult result = run_headless_match(gold_config, scarlet_config, max_moves, verbose);
        
        if (!quiet) {
            print_match_result(result);
        }
        
    } else if (mode == "tournament") {
        // Run tournament
        if (!quiet) {
            std::cout << "Games: " << num_games << std::endl;
            std::cout << "Threads: " << (num_threads > 0 ? std::to_string(num_threads) : "auto") << std::endl;
            std::cout << std::endl;
        }
        
        TournamentResults results = run_headless_tournament(
            gold_config, scarlet_config, num_games, num_threads, verbose
        );
        
        if (!quiet) {
            print_tournament_results(results);
        }
        
        // Export results
        if (!output_csv.empty()) {
            export_results_csv(results, output_csv);
        }
        if (!output_json.empty()) {
            export_results_json(results, output_json);
        }
        
    } else {
        std::cerr << "Unknown mode: " << mode << " (use 'match' or 'tournament')" << std::endl;
        return 1;
    }
    
    return 0;
}
