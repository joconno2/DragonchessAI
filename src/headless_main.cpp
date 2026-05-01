#include "headless.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace dragonchess;

static void print_usage(const char* program_name) {
    std::cout << "Dragonchess AI - Headless Research Platform\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << program_name << " --headless [OPTIONS]\n\n";

    std::cout << "HEADLESS MODE OPTIONS:\n";
    std::cout << "  --mode <match|tournament|selfplay|genlabels>  Mode (default: match)\n";
    std::cout << "     selfplay:   generate TD training data; outputs NDJSON to stdout\n";
    std::cout << "     genlabels:  generate search-supervised labels (NNUE-style)\n";
    std::cout << "  --label-depth <N>          AB search depth for labeling (default: 6)\n";
    std::cout << "  --random-plies <N>         Random opening moves for diversity (default: 8)\n";
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
    std::cout << "  evolvable                  Piece-value evolved AI (14 weights)\n";
    std::cout << "  tdeval                     TD-learned feature evaluation (40 weights)\n\n";
    std::cout << "TD SELF-PLAY OPTIONS (--mode selfplay):\n";
    std::cout << "  --td-weights <w0,w1,...>   Set both gold and scarlet to tdeval with weights\n";
    std::cout << "  --gold-td-weights <csv>    Set gold AI to tdeval with given weights\n";
    std::cout << "  --scarlet-td-weights <csv> Set scarlet AI to tdeval with given weights\n";
    std::cout << "  --td-depth <N>             Search depth for tdeval (default: 1)\n\n";

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
    bool td_depth_set = false;
    int label_depth = 6;
    int random_plies = 8;

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
        } else if (arg == "--td-weights" && i + 1 < argc) {
            // Set both players to TDEvalAI with the same weights
            gold_config.type = "tdeval";
            scarlet_config.type = "tdeval";
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                float v = std::stof(token);
                gold_config.weights.push_back(v);
                scarlet_config.weights.push_back(v);
            }
        } else if (arg == "--gold-td-weights" && i + 1 < argc) {
            gold_config.type = "tdeval";
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                gold_config.weights.push_back(std::stof(token));
            }
        } else if (arg == "--scarlet-td-weights" && i + 1 < argc) {
            scarlet_config.type = "tdeval";
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                scarlet_config.weights.push_back(std::stof(token));
            }
        } else if (arg == "--nn-weights" && i + 1 < argc) {
            // Load NN weights from a binary file (flat float32 array)
            std::string path = argv[++i];
            std::ifstream wf(path, std::ios::binary);
            if (!wf) {
                std::cerr << "Cannot open nn-weights file: " << path << std::endl;
                return 1;
            }
            wf.seekg(0, std::ios::end);
            size_t bytes = wf.tellg();
            wf.seekg(0);
            size_t count = bytes / sizeof(float);
            std::vector<float> w(count);
            wf.read(reinterpret_cast<char*>(w.data()), bytes);
            gold_config.type = "nneval";
            scarlet_config.type = "nneval";
            gold_config.weights = w;
            scarlet_config.weights = w;
        } else if (arg == "--gold-nn-weights" && i + 1 < argc) {
            std::string path = argv[++i];
            std::ifstream wf(path, std::ios::binary);
            if (!wf) { std::cerr << "Cannot open: " << path << std::endl; return 1; }
            wf.seekg(0, std::ios::end);
            size_t bytes = wf.tellg();
            wf.seekg(0);
            gold_config.weights.resize(bytes / sizeof(float));
            wf.read(reinterpret_cast<char*>(gold_config.weights.data()), bytes);
            gold_config.type = "nneval";
        } else if (arg == "--scarlet-nn-weights" && i + 1 < argc) {
            std::string path = argv[++i];
            std::ifstream wf(path, std::ios::binary);
            if (!wf) { std::cerr << "Cannot open: " << path << std::endl; return 1; }
            wf.seekg(0, std::ios::end);
            size_t bytes = wf.tellg();
            wf.seekg(0);
            scarlet_config.weights.resize(bytes / sizeof(float));
            wf.read(reinterpret_cast<char*>(scarlet_config.weights.data()), bytes);
            scarlet_config.type = "nneval";
        } else if (arg == "--td-depth" && i + 1 < argc) {
            int td_depth = std::atoi(argv[++i]);
            gold_config.depth = td_depth;
            scarlet_config.depth = td_depth;
            td_depth_set = true;
        } else if (arg == "--label-depth" && i + 1 < argc) {
            label_depth = std::atoi(argv[++i]);
        } else if (arg == "--random-plies" && i + 1 < argc) {
            random_plies = std::atoi(argv[++i]);
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

    // selfplay and genlabels don't need explicit AI configs
    if (mode != "selfplay" && mode != "genlabels" &&
        (gold_config.type.empty() || scarlet_config.type.empty())) {
        std::cerr << "Error: Both --gold-ai and --scarlet-ai are required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Modes that write pure NDJSON to stdout need quiet output.
    if (mode == "selfplay" || mode == "genlabels") quiet = true;

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

    if (mode == "selfplay") {
        // Selfplay mode: generate TD training data, output NDJSON to stdout.
        // Requires --td-weights (or --gold-td-weights / --scarlet-td-weights).
        if (gold_config.type.empty())   gold_config.type   = "tdeval";
        if (scarlet_config.type.empty()) scarlet_config.type = "tdeval";
        // Default depth 1 for selfplay only if --td-depth was NOT explicitly set
        if (!td_depth_set) {
            gold_config.depth   = 1;
            scarlet_config.depth = 1;
        }
        run_selfplay_batch(gold_config, scarlet_config, num_games, num_threads, std::cout);
        return 0;
    } else if (mode == "match") {
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
    } else if (mode == "genlabels") {
        // Generate search-supervised training labels (NNUE approach).
        // Plays games with random openings, labels each position with AB(label_depth).
        run_genlabels_batch(num_games, label_depth, random_plies, num_threads, std::cout);
        return 0;
    } else {
        std::cerr << "Unknown mode: " << mode << " (use 'match', 'tournament', 'selfplay', or 'genlabels')" << std::endl;
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
