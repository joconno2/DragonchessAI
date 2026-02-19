#include "campaign.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <SDL2/SDL_ttf.h>

namespace dragonchess {

Campaign::Campaign() {
    init_levels();
}

void Campaign::init_levels() {
    // Create 5 campaign levels with increasing difficulty
    std::vector<CampaignLevel> levels = {
        {1, "Training Grounds", "Learn the basics against a simple opponent", false, 0},
        {2, "First Challenge", "Face a slightly more aggressive AI", false, 0},
        {3, "Rising Difficulty", "The AI is getting smarter", false, 0},
        {4, "Master Challenge", "Only skilled players succeed here", false, 0},
        {5, "Final Battle", "The ultimate test of your skills", false, 0}
    };
    
    profile.levels = levels;
}

void Campaign::init_new_profile(const std::string& player_name) {
    profile.name = player_name;
    profile.current_level = 1;
    profile.total_wins = 0;
    profile.total_losses = 0;
    init_levels();
}

void Campaign::render_level_select(Renderer& renderer) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    
    // Clear screen
    SDL_SetRenderDrawColor(sdl_renderer, 20, 20, 40, 255);
    SDL_RenderClear(sdl_renderer);
    
    auto [win_w, win_h] = renderer.get_window_size();
    
    TTF_Font* font_large = TTF_OpenFont("assets/font.ttf", 48);  // Decorative title
    TTF_Font* font = TTF_OpenFont("assets/pixel.ttf", 20);  // Pixel font for menu items
    
    if (font_large) {
        // Title
        std::string title = "Campaign: " + profile.name;
        SDL_Color gold = {255, 215, 0, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font_large, title.c_str(), gold);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(sdl_renderer, surface);
            SDL_Rect dest = {win_w / 2 - surface->w / 2, 30, surface->w, surface->h};
            SDL_RenderCopy(sdl_renderer, texture, nullptr, &dest);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
        TTF_CloseFont(font_large);
    }
    
    if (font) {
        // Stats
        std::string stats = "Wins: " + std::to_string(profile.total_wins) + 
                          "  Losses: " + std::to_string(profile.total_losses);
        SDL_Color white = {255, 255, 255, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, stats.c_str(), white);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(sdl_renderer, surface);
            SDL_Rect dest = {win_w / 2 - surface->w / 2, 100, surface->w, surface->h};
            SDL_RenderCopy(sdl_renderer, texture, nullptr, &dest);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
        
        // Level list
        int y = 180;
        for (size_t i = 0; i < profile.levels.size(); ++i) {
            const auto& level = profile.levels[i];
            
            // Level button background
            SDL_Rect button = {win_w / 2 - 300, y, 600, 60};
            
            bool unlocked = (i == 0 || profile.levels[i - 1].completed);
            if (unlocked) {
                if (level.completed) {
                    SDL_SetRenderDrawColor(sdl_renderer, 50, 100, 50, 255);  // Green for completed
                } else if (static_cast<int>(i) == profile.current_level - 1) {
                    SDL_SetRenderDrawColor(sdl_renderer, 100, 100, 150, 255);  // Blue for current
                } else {
                    SDL_SetRenderDrawColor(sdl_renderer, 80, 80, 80, 255);  // Gray for available
                }
            } else {
                SDL_SetRenderDrawColor(sdl_renderer, 40, 40, 40, 255);  // Dark for locked
            }
            
            SDL_RenderFillRect(sdl_renderer, &button);
            SDL_SetRenderDrawColor(sdl_renderer, 200, 200, 200, 255);
            SDL_RenderDrawRect(sdl_renderer, &button);
            
            // Level text
            std::string level_text = "Level " + std::to_string(level.level) + ": " + level.name;
            if (!unlocked) {
                level_text += " [LOCKED]";
            } else if (level.completed) {
                level_text += " [COMPLETED]";
            }
            
            SDL_Color text_color = unlocked ? white : SDL_Color{100, 100, 100, 255};
            SDL_Surface* surf = TTF_RenderText_Blended(font, level_text.c_str(), text_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {button.x + 20, button.y + 10, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                
                // Description on second line
                SDL_Surface* desc_surf = TTF_RenderText_Blended(font, level.description.c_str(), text_color);
                if (desc_surf) {
                    SDL_Texture* desc_tex = SDL_CreateTextureFromSurface(sdl_renderer, desc_surf);
                    SDL_Rect desc_dest = {button.x + 40, button.y + 30, desc_surf->w, desc_surf->h};
                    SDL_RenderCopy(sdl_renderer, desc_tex, nullptr, &desc_dest);
                    SDL_DestroyTexture(desc_tex);
                    SDL_FreeSurface(desc_surf);
                }
                
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
            
            y += 80;
        }
        
        // Instructions
        std::string instr = "Click a level to play. ESC or click outside to exit.";
        SDL_Surface* instr_surf = TTF_RenderText_Blended(font, instr.c_str(), white);
        if (instr_surf) {
            SDL_Texture* instr_tex = SDL_CreateTextureFromSurface(sdl_renderer, instr_surf);
            SDL_Rect dest = {win_w / 2 - instr_surf->w / 2, win_h - 60, instr_surf->w, instr_surf->h};
            SDL_RenderCopy(sdl_renderer, instr_tex, nullptr, &dest);
            SDL_DestroyTexture(instr_tex);
            SDL_FreeSurface(instr_surf);
        }
        
        TTF_CloseFont(font);
    }
    
    SDL_RenderPresent(sdl_renderer);
}

bool Campaign::play_level(Renderer& renderer, InputHandler& input, int level_idx) {
    Game game;
    RandomAI ai(game, Color::SCARLET);
    
    std::optional<int> selected;
    std::unordered_set<int> legal_dest;
    
    int moves = 0;
    bool running = true;
    
    while (running && !game.game_over) {
        int mouse_x = 0, mouse_y = 0;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        switch (event) {
            case InputEvent::QUIT:
                return false;
                
            case InputEvent::MOUSE_CLICK: {
                if (game.current_turn == Color::SCARLET) {
                    break;  // Don't process clicks during AI turn
                }
                
                auto pos = renderer.screen_to_board(mouse_x, mouse_y);
                if (pos.has_value()) {
                    auto [layer, row, col] = pos.value();
                    int idx = pos_to_index(layer, row, col);
                    int16_t piece = game.board[idx];
                    
                    bool valid_piece = (game.current_turn == Color::GOLD && piece > 0);
                    
                    if (!selected.has_value()) {
                        if (valid_piece) {
                            selected = idx;
                            auto moves_list = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves_list) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        }
                    } else {
                        if (legal_dest.count(idx) > 0) {
                            auto moves_list = game.get_legal_moves_for(selected.value());
                            for (const auto& move : moves_list) {
                                if (std::get<1>(move) == idx) {
                                    game.make_move(move);
                                    game.update();
                                    moves++;
                                    selected.reset();
                                    legal_dest.clear();
                                    break;
                                }
                            }
                        } else if (valid_piece) {
                            selected = idx;
                            auto moves_list = game.get_legal_moves_for(idx);
                            legal_dest.clear();
                            for (const auto& move : moves_list) {
                                legal_dest.insert(std::get<1>(move));
                            }
                        } else {
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
        if (game.current_turn == Color::SCARLET && !game.game_over) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            auto move = ai.choose_move();
            if (move.has_value()) {
                game.make_move(move.value());
                game.update();
                selected.reset();
                legal_dest.clear();
            }
        }
        
        renderer.render_board(game, selected, legal_dest);
        renderer.present();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    // Game finished
    bool player_won = (game.winner == "Gold");
    
    if (player_won) {
        profile.total_wins++;
        profile.levels[level_idx].completed = true;
        if (profile.levels[level_idx].best_moves == 0 || moves < profile.levels[level_idx].best_moves) {
            profile.levels[level_idx].best_moves = moves;
        }
        if (profile.current_level == level_idx + 1 && static_cast<size_t>(level_idx + 1) < profile.levels.size()) {
            profile.current_level++;
        }
    } else {
        profile.total_losses++;
    }
    
    // Show result screen briefly
    renderer.render_post_game(game);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    return true;  // Continue campaign
}

void Campaign::run(Renderer& renderer, InputHandler& input) {
    // For now, create a default profile
    init_new_profile("Player");
    
    std::cout << "Starting campaign mode for " << profile.name << std::endl;
    
    bool running = true;
    while (running) {
        render_level_select(renderer);
        
        // Wait for level selection
        bool selecting = true;
        while (selecting) {
            int mouse_x, mouse_y;
            InputEvent event = input.poll(mouse_x, mouse_y);
            
            if (event == InputEvent::QUIT) {
                return;
            }
            
            if (event == InputEvent::MOUSE_CLICK) {
                auto [win_w, win_h] = renderer.get_window_size();
                
                // Check which level was clicked
                int y = 180;
                for (size_t i = 0; i < profile.levels.size(); ++i) {
                    SDL_Rect button = {win_w / 2 - 300, y, 600, 60};
                    
                    if (mouse_x >= button.x && mouse_x < button.x + button.w &&
                        mouse_y >= button.y && mouse_y < button.y + button.h) {
                        
                        // Check if level is unlocked
                        bool unlocked = (i == 0 || profile.levels[i - 1].completed);
                        if (unlocked) {
                            if (!play_level(renderer, input, i)) {
                                return;  // User quit
                            }
                            selecting = false;
                            break;
                        }
                    }
                    
                    y += 80;
                }
                
                // Click outside levels = exit
                if (selecting && (mouse_y < 180 || mouse_y > 180 + 80 * static_cast<int>(profile.levels.size()))) {
                    return;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
}

} // namespace dragonchess
