#include "headless.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using namespace dragonchess;

static void print_usage(const char* program_name) {
    std::cout << "Dragonchess AI - Headless Research Platform\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << program_name << " --headless [OPTIONS]\n\n";

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
    std::cout << "  alphabeta                  Alpha-beta pruning (specify depth)\n";
    std::cout << "  evolvable                  Weight-driven evaluation AI\n\n";

    std::cout << "OUTPUT OPTIONS:\n";
    std::cout << "  --output-csv <file>        Export results to CSV\n";
    std::cout << "  --output-json <file>       Export results to JSON\n";
    std::cout << "  --verbose                  Print detailed progress\n";
    std::cout << "  --quiet                    Minimal output\n\n";
}

static int run_headless_mode(int argc, char* argv[]) {
    std::string mode = "match";
    int num_games = 100;
    int num_threads = 0;
    int max_moves = 1000;
    bool verbose = false;
    bool quiet = false;
    std::string output_csv;
    std::string output_json;

    AIConfig gold_config;
    AIConfig scarlet_config;
    gold_config.depth = 2;
    scarlet_config.depth = 2;

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
        } else if (arg == "--gold-weights" && i + 1 < argc) {
            gold_config.type = "evolvable";
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                gold_config.weights.push_back(std::stof(token));
            }
        } else if (arg == "--scarlet-weights" && i + 1 < argc) {
            scarlet_config.type = "evolvable";
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                scarlet_config.weights.push_back(std::stof(token));
            }
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

    if (gold_config.type.empty() || scarlet_config.type.empty()) {
        std::cerr << "Error: Both --gold-ai and --scarlet-ai are required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (!quiet) {
        std::cout << "=== Dragonchess Headless Mode ===" << std::endl;
        std::cout << "Mode: " << mode << std::endl;
        std::cout << "Gold AI: " << gold_config.type;
        if (!gold_config.name.empty()) {
            std::cout << " (" << gold_config.name << ")";
        }
        std::cout << " [depth: " << gold_config.depth << "]" << std::endl;
        std::cout << "Scarlet AI: " << scarlet_config.type;
        if (!scarlet_config.name.empty()) {
            std::cout << " (" << scarlet_config.name << ")";
        }
        std::cout << " [depth: " << scarlet_config.depth << "]" << std::endl;
    }

    if (mode == "match") {
        if (!quiet) {
            std::cout << "\nRunning single match..." << std::endl;
        }
        MatchResult result = run_headless_match(gold_config, scarlet_config, max_moves, verbose);
        if (!quiet) {
            print_match_result(result);
        }
    } else if (mode == "tournament") {
        if (!quiet) {
            std::cout << "Games: " << num_games << std::endl;
            std::cout << "Threads: " << (num_threads > 0 ? std::to_string(num_threads) : "auto") << std::endl;
            std::cout << std::endl;
        }

        TournamentResults results = run_headless_tournament(
            gold_config,
            scarlet_config,
            num_games,
            num_threads,
            verbose
        );

        if (!quiet) {
            print_tournament_results(results);
        }
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

int main(int argc, char* argv[]) {
    if (argc > 1 && (std::strcmp(argv[1], "--headless") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        return run_headless_mode(argc, argv);
    }

    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-?") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    std::cerr << "This binary was built with HEADLESS_ONLY=ON.\n";
    std::cerr << "Run it with --headless, or build the full GUI target instead.\n";
    print_usage(argv[0]);
    return 1;
}
