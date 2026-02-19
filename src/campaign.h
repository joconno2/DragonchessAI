#pragma once

#include "game.h"
#include "ai.h"
#include "renderer.h"
#include "input.h"
#include <string>
#include <vector>

namespace dragonchess {

struct CampaignLevel {
    int level;
    std::string name;
    std::string description;
    bool completed;
    int best_moves;  // Fewest moves to complete (0 = not completed)
};

struct PlayerProfile {
    std::string name;
    int current_level;
    int total_wins;
    int total_losses;
    std::vector<CampaignLevel> levels;
};

class Campaign {
public:
    Campaign();
    
    void init_new_profile(const std::string& player_name);
    void run(Renderer& renderer, InputHandler& input);
    
private:
    PlayerProfile profile;
    
    void init_levels();
    bool play_level(Renderer& renderer, InputHandler& input, int level_idx);
    void render_level_select(Renderer& renderer);
    void render_level_intro(Renderer& renderer, const CampaignLevel& level);
};

} // namespace dragonchess
