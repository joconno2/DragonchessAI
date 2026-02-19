#pragma once

#include "game.h"
#include "ai.h"
#include "renderer.h"
#include "input.h"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

namespace dragonchess {

struct BotInfo {
    std::string name;
    std::string path;  // Path to bot file (or empty for built-in)
    std::string type;  // "RandomAI", "Custom", etc.
};

struct TournamentParticipant {
    std::string name;
    std::string bot_path;
    float score;
    int wins;
    int losses;
    int draws;
    float elo;
    
    // Statistics for debugging/training
    int total_moves;
    int avg_moves_per_game;
    int shortest_game;
    int longest_game;
    int captures_made;
    int pieces_lost;
    int checkmates_delivered;
    int total_pieces_remaining;
};

struct MatchState {
    int match_id;
    int player1_idx;
    int player2_idx;
    std::unique_ptr<Game> game;
    std::atomic<int> current_move;
    std::atomic<bool> finished;
    std::string winner;
    int total_moves;
    
    // For visualization
    std::mutex game_mutex;
};

class Tournament {
public:
    Tournament();
    ~Tournament();
    
    // Bot selection
    std::vector<BotInfo> scan_available_bots();
    bool select_bots(Renderer& renderer, InputHandler& input);
    
    void add_participant(const std::string& name, const std::string& bot_path = "");
    void run(Renderer& renderer, InputHandler& input, int rounds = 5);
    
private:
    std::vector<TournamentParticipant> participants;
    std::vector<std::shared_ptr<MatchState>> active_matches;
    std::atomic<bool> should_quit{false};  // Signal for early termination
    
    // Swiss pairing
    std::vector<std::pair<int, int>> generate_pairings();
    
    // ELO calculations
    float expected_score(float rating_a, float rating_b);
    void update_elo(int p1_idx, int p2_idx, float p1_score);
    
    // Multi-threaded match execution
    void run_match_async(std::shared_ptr<MatchState> match);
    void run_round_parallel(Renderer& renderer, InputHandler& input, 
                           std::vector<std::pair<int, int>>& pairings, int round_num);
    
    // Rendering
    void render_bot_selection(Renderer& renderer, const std::vector<BotInfo>& bots,
                             const std::vector<int>& selection_count, int hover_idx);
    void render_parallel_matches(Renderer& renderer, int round_num, int total_rounds);
    void render_match_board(Renderer& renderer, const MatchState& match, 
                           int x, int y, int width, int height);
    void render_standings(Renderer& renderer, int current_round, int total_rounds);
    void render_detailed_stats(Renderer& renderer);
};

} // namespace dragonchess
