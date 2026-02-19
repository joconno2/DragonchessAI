#include "tournament.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <SDL2/SDL_ttf.h>

namespace dragonchess {

Tournament::Tournament() {
}

Tournament::~Tournament() {
    // Wait for any active matches to finish
    active_matches.clear();
}

std::vector<BotInfo> Tournament::scan_available_bots() {
    std::vector<BotInfo> bots;
    
    // Built-in bots
    bots.push_back({"RandomAI", "", "Built-in"});
    bots.push_back({"GreedyAI", "", "Built-in"});
    bots.push_back({"GreedyValueAI", "", "Built-in"});
    bots.push_back({"MinimaxAI-D2", "", "Built-in"});
    bots.push_back({"MinimaxAI-D3", "", "Built-in"});
    bots.push_back({"AlphaBetaAI-D2", "", "Built-in"});
    bots.push_back({"AlphaBetaAI-D3", "", "Built-in"});
    bots.push_back({"AlphaBetaAI-D4", "", "Built-in"});
    
    // Scan bots directory for custom bots (placeholders for now)
    if (std::filesystem::exists("bots")) {
        for (const auto& entry : std::filesystem::directory_iterator("bots")) {
            if (entry.path().extension() == ".cpp") {
                BotInfo info;
                info.name = entry.path().stem().string();
                info.path = entry.path().string();
                info.type = "Placeholder";
                // Don't add placeholders for now, they're mapped to built-ins
            }
        }
    }
    
    return bots;
}

void Tournament::render_bot_selection(Renderer& renderer, const std::vector<BotInfo>& bots,
                                      const std::vector<int>& selection_count, int hover_idx) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    
    // Modern dark gradient background
    SDL_SetRenderDrawColor(sdl_renderer, 25, 28, 35, 255);
    SDL_RenderClear(sdl_renderer);
    
    auto [win_w, win_h] = renderer.get_window_size();
    
    TTF_Font* font_title = TTF_OpenFont("assets/font.ttf", 48);
    TTF_Font* pixel_font = TTF_OpenFont("assets/pixel.ttf", 20);  // Increased for readability
    TTF_Font* pixel_font_small = TTF_OpenFont("assets/pixel.ttf", 16);  // Increased for readability
    
    // Draw title with shadow effect (decorative font)
    if (font_title) {
        SDL_Color shadow = {0, 0, 0, 180};
        SDL_Color gold = {255, 215, 0, 255};
        
        const char* title_text = "Tournament Setup";
        
        // Shadow
        SDL_Surface* surf = TTF_RenderText_Blended(font_title, title_text, shadow);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2 + 2, 42, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        // Main title
        surf = TTF_RenderText_Blended(font_title, title_text, gold);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2, 40, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        TTF_CloseFont(font_title);
    }
    
    if (pixel_font && pixel_font_small) {
        // Instructions with pixel font
        SDL_Color light_gray = {180, 185, 195, 255};
        std::string instr = "Select up to 8 bots (same bot can be selected multiple times)";
        SDL_Surface* surf = TTF_RenderText_Blended(pixel_font_small, instr.c_str(), light_gray);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2, 105, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        // Selected count with progress bar (pixel font)
        int y = 145;
        int total_selected = 0;
        for (int count : selection_count) {
            total_selected += count;
        }
        
        std::string count_text = "Selected: " + std::to_string(total_selected) + " / 8";
        SDL_Color white = {220, 220, 230, 255};
        surf = TTF_RenderText_Blended(pixel_font, count_text.c_str(), white);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - 400, y, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        // Progress bar
        SDL_Rect progress_bg = {win_w / 2 + 100, y + 5, 200, 20};
        SDL_SetRenderDrawColor(sdl_renderer, 40, 45, 55, 255);
        SDL_RenderFillRect(sdl_renderer, &progress_bg);
        SDL_SetRenderDrawColor(sdl_renderer, 80, 85, 95, 255);
        SDL_RenderDrawRect(sdl_renderer, &progress_bg);
        
        if (total_selected > 0) {
            SDL_Rect progress_fill = {progress_bg.x + 2, progress_bg.y + 2, 
                                     (progress_bg.w - 4) * total_selected / 8, progress_bg.h - 4};
            SDL_SetRenderDrawColor(sdl_renderer, 100, 200, 100, 255);
            SDL_RenderFillRect(sdl_renderer, &progress_fill);
        }
        
        y += 55;
        
        // Bot list with card-style design
        for (size_t i = 0; i < bots.size(); ++i) {
            SDL_Rect box = {win_w / 2 - 400, y, 800, 50};
            
            // Card background with gradient effect
            if (static_cast<int>(i) == hover_idx) {
                SDL_SetRenderDrawColor(sdl_renderer, 65, 75, 95, 255);
            } else if (selection_count[i] > 0) {
                SDL_SetRenderDrawColor(sdl_renderer, 45, 85, 60, 255);
            } else {
                SDL_SetRenderDrawColor(sdl_renderer, 40, 45, 55, 255);
            }
            SDL_RenderFillRect(sdl_renderer, &box);
            
            // Border
            if (static_cast<int>(i) == hover_idx) {
                SDL_SetRenderDrawColor(sdl_renderer, 120, 140, 180, 255);
            } else if (selection_count[i] > 0) {
                SDL_SetRenderDrawColor(sdl_renderer, 100, 200, 120, 255);
            } else {
                SDL_SetRenderDrawColor(sdl_renderer, 70, 75, 85, 255);
            }
            SDL_RenderDrawRect(sdl_renderer, &box);
            
            // Count badge (circular style)
            SDL_Rect badge = {box.x + 15, box.y + 12, 40, 26};
            if (selection_count[i] > 0) {
                SDL_SetRenderDrawColor(sdl_renderer, 100, 200, 100, 255);
            } else {
                SDL_SetRenderDrawColor(sdl_renderer, 60, 65, 75, 255);
            }
            SDL_RenderFillRect(sdl_renderer, &badge);
            SDL_SetRenderDrawColor(sdl_renderer, 120, 125, 135, 255);
            SDL_RenderDrawRect(sdl_renderer, &badge);
            
            if (selection_count[i] > 0) {
                std::string count_str = std::to_string(selection_count[i]);
                SDL_Color badge_color = {255, 255, 255, 255};
                surf = TTF_RenderText_Blended(pixel_font, count_str.c_str(), badge_color);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                    SDL_Rect dest = {badge.x + (badge.w - surf->w) / 2, 
                                    badge.y + (badge.h - surf->h) / 2, 
                                    surf->w, surf->h};
                    SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                    SDL_DestroyTexture(tex);
                    SDL_FreeSurface(surf);
                }
            } else {
                // Draw "0" in grey
                SDL_Color grey = {100, 105, 115, 255};
                surf = TTF_RenderText_Blended(pixel_font, "0", grey);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                    SDL_Rect dest = {badge.x + (badge.w - surf->w) / 2, 
                                    badge.y + (badge.h - surf->h) / 2, 
                                    surf->w, surf->h};
                    SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                    SDL_DestroyTexture(tex);
                    SDL_FreeSurface(surf);
                }
            }
            
            // Bot name with pixel font
            SDL_Color name_color = {220, 220, 230, 255};
            surf = TTF_RenderText_Blended(pixel_font, bots[i].name.c_str(), name_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {box.x + 70, box.y + 8, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
            
            // Bot type in smaller pixel font
            std::string type_text = "(" + bots[i].type + ")";
            SDL_Color type_color = {140, 145, 155, 255};
            surf = TTF_RenderText_Blended(pixel_font_small, type_text.c_str(), type_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {box.x + 70, box.y + 28, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
            
            y += 60;
        }
        
        // Bottom instruction
        // Bottom instruction with pixel font
        y = win_h - 60;
        std::string hint = total_selected >= 2 ? 
            "Press SPACE to start tournament" : 
            "Select at least 2 bots to continue";
        SDL_Color hint_color = total_selected >= 2 ? 
            SDL_Color{100, 255, 100, 255} : SDL_Color{180, 180, 190, 255};
        surf = TTF_RenderText_Blended(pixel_font, hint.c_str(), hint_color);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2, y, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        TTF_CloseFont(pixel_font);
        TTF_CloseFont(pixel_font_small);
    }
    
    SDL_RenderPresent(sdl_renderer);
}

bool Tournament::select_bots(Renderer& renderer, InputHandler& input) {
    auto bots = scan_available_bots();
    if (bots.empty()) {
        std::cerr << "No bots available!" << std::endl;
        return false;
    }
    
    // Track number of times each bot is selected (0-8)
    std::vector<int> selection_count(bots.size(), 0);
    int hover_idx = -1;
    
    bool running = true;
    while (running) {
        int mouse_x, mouse_y;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        auto [win_w, win_h] = renderer.get_window_size();
        
        // Update hover - must match rendering exactly!
        hover_idx = -1;
        int y_start = 200;  // Matches rendering: 145 + 55 = 200
        for (size_t i = 0; i < bots.size(); ++i) {
            SDL_Rect box = {win_w / 2 - 400, y_start + static_cast<int>(i) * 60, 800, 50};
            if (mouse_x >= box.x && mouse_x < box.x + box.w &&
                mouse_y >= box.y && mouse_y < box.y + box.h) {
                hover_idx = static_cast<int>(i);
            }
        }
        
        if (event == InputEvent::QUIT) {
            return false;
        } else if (event == InputEvent::MOUSE_CLICK && hover_idx >= 0) {
            int total_selected = 0;
            for (int count : selection_count) total_selected += count;
            
            // Left click to add, right click (or shift+click) to remove
            // For now, just cycle: click to add up to 8 total, then wrap to 0
            if (total_selected < 8) {
                selection_count[hover_idx]++;
            } else if (selection_count[hover_idx] > 0) {
                // Already at max, clicking again removes this bot
                selection_count[hover_idx] = 0;
            }
        } else if (event == InputEvent::SPACE_PRESSED) {
            // Space bar to continue
            int total_selected = 0;
            for (int count : selection_count) total_selected += count;
            
            if (total_selected >= 2) {
                // Add selected bots to tournament (with duplicates)
                participants.clear();
                for (size_t i = 0; i < bots.size(); ++i) {
                    for (int j = 0; j < selection_count[i]; ++j) {
                        std::string name = bots[i].name;
                        if (selection_count[i] > 1) {
                            name += " #" + std::to_string(j + 1);
                        }
                        add_participant(name, bots[i].path);
                    }
                }
                return true;
            }
        }
        
        render_bot_selection(renderer, bots, selection_count, hover_idx);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    return false;
}

void Tournament::add_participant(const std::string& name, const std::string& bot_path) {
    TournamentParticipant p;
    p.name = name;
    p.bot_path = bot_path;
    p.score = 0.0f;
    p.wins = 0;
    p.losses = 0;
    p.draws = 0;
    p.elo = 1500.0f;
    p.total_moves = 0;
    p.avg_moves_per_game = 0;
    p.shortest_game = 999999;
    p.longest_game = 0;
    p.captures_made = 0;
    p.pieces_lost = 0;
    p.checkmates_delivered = 0;
    p.total_pieces_remaining = 0;
    participants.push_back(p);
}

float Tournament::expected_score(float rating_a, float rating_b) {
    return 1.0f / (1.0f + std::pow(10.0f, (rating_b - rating_a) / 400.0f));
}

void Tournament::update_elo(int p1_idx, int p2_idx, float p1_score) {
    const float K = 32.0f;
    
    float p1_rating = participants[p1_idx].elo;
    float p2_rating = participants[p2_idx].elo;
    
    float p1_expected = expected_score(p1_rating, p2_rating);
    float p2_expected = expected_score(p2_rating, p1_rating);
    
    participants[p1_idx].elo += K * (p1_score - p1_expected);
    participants[p2_idx].elo += K * ((1.0f - p1_score) - p2_expected);
}

std::vector<std::pair<int, int>> Tournament::generate_pairings() {
    // Sort by score, then ELO (Swiss system)
    std::vector<int> indices;
    for (size_t i = 0; i < participants.size(); ++i) {
        indices.push_back(i);
    }
    
    std::sort(indices.begin(), indices.end(), [this](int a, int b) {
        if (participants[a].score != participants[b].score) {
            return participants[a].score > participants[b].score;
        }
        return participants[a].elo > participants[b].elo;
    });
    
    // Simple pairing: pair adjacent players
    std::vector<std::pair<int, int>> pairings;
    for (size_t i = 0; i + 1 < indices.size(); i += 2) {
        pairings.push_back({indices[i], indices[i + 1]});
    }
    
    // If odd number, give bye to last player
    if (indices.size() % 2 == 1) {
        int bye_idx = indices.back();
        participants[bye_idx].score += 1.0f;
        participants[bye_idx].wins += 1;
        std::cout << participants[bye_idx].name << " gets a bye." << std::endl;
    }
    
    return pairings;
}

void Tournament::run_match_async(std::shared_ptr<MatchState> match) {
    // Helper to create AI based on bot name
    auto create_ai = [](const std::string& bot_name, Game& game, Color color) -> std::unique_ptr<BaseAI> {
        if (bot_name.find("GreedyValueAI") != std::string::npos || 
            bot_name.find("apprentice") != std::string::npos) {
            return std::make_unique<GreedyValueAI>(game, color);
        } else if (bot_name.find("GreedyAI") != std::string::npos || 
                   bot_name.find("novice") != std::string::npos) {
            return std::make_unique<GreedyAI>(game, color);
        } else if (bot_name.find("AlphaBetaAI-D4") != std::string::npos || 
                   bot_name.find("champion") != std::string::npos) {
            return std::make_unique<AlphaBetaAI>(game, color, 4);
        } else if (bot_name.find("AlphaBetaAI-D3") != std::string::npos || 
                   bot_name.find("fast_minimax") != std::string::npos) {
            return std::make_unique<AlphaBetaAI>(game, color, 3);
        } else if (bot_name.find("AlphaBetaAI-D2") != std::string::npos) {
            return std::make_unique<AlphaBetaAI>(game, color, 2);
        } else if (bot_name.find("MinimaxAI-D3") != std::string::npos || 
                   bot_name.find("simple_minimax") != std::string::npos) {
            return std::make_unique<MinimaxAI>(game, color, 3);
        } else if (bot_name.find("MinimaxAI-D2") != std::string::npos || 
                   bot_name.find("veteran") != std::string::npos) {
            return std::make_unique<MinimaxAI>(game, color, 2);
        } else {
            // Default to RandomAI
            return std::make_unique<RandomAI>(game, color);
        }
    };
    
    // Create AIs for both players based on their bot names
    auto gold_ai = create_ai(participants[match->player1_idx].name, *match->game, Color::GOLD);
    auto scarlet_ai = create_ai(participants[match->player2_idx].name, *match->game, Color::SCARLET);
    
    int moves_without_progress = 0;
    const int MAX_MOVES = 300;  // Reduced from 500
    const int MAX_NO_PROGRESS = 50;  // Stop if 50 moves with no change
    
    while (!match->game->game_over && match->current_move < MAX_MOVES) {
        std::lock_guard<std::mutex> lock(match->game_mutex);
        
        auto move = (match->game->current_turn == Color::GOLD) ? 
                    gold_ai->choose_move() : scarlet_ai->choose_move();
        
        if (move.has_value()) {
            int prev_no_capture = match->game->no_capture_count;
            match->game->make_move(move.value());
            match->game->update();
            match->current_move++;
            
            // Track progress (captures reset the counter)
            if (match->game->no_capture_count == 0) {
                moves_without_progress = 0;
            } else {
                moves_without_progress++;
            }
            
            // Force draw if no progress for too long
            if (moves_without_progress >= MAX_NO_PROGRESS) {
                match->game->game_over = true;
                match->game->winner = "Draw";
                break;
            }
        } else {
            // No legal moves - stalemate
            match->game->game_over = true;
            match->game->winner = "Draw";
            break;
        }
        
        // No delay - let games run at full speed!
        // Rendering loop handles visualization at appropriate FPS
    }
    
    // If we hit the move limit, call it a draw
    if (match->current_move >= MAX_MOVES && !match->game->game_over) {
        match->game->game_over = true;
        match->game->winner = "Draw";
    }
    
    match->winner = match->game->winner;
    match->total_moves = match->current_move;
    match->finished = true;
}

void Tournament::render_match_board(Renderer& renderer, const MatchState& match,
                                   int x, int y, int width, int height) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    
    // Note: We avoid locking here for rendering performance.
    // Reading game state for visualization without modification is safe enough
    // as the worst case is just a visual glitch for one frame.
    
    // Draw background
    SDL_Rect bg = {x, y, width, height};
    SDL_SetRenderDrawColor(sdl_renderer, 25, 28, 35, 255);
    SDL_RenderFillRect(sdl_renderer, &bg);
    
    // Draw border
    SDL_SetRenderDrawColor(sdl_renderer, 80, 85, 95, 255);
    SDL_RenderDrawRect(sdl_renderer, &bg);
    
    // Calculate cell dimensions for 3 boards side by side
    const int padding = 10;
    const int board_spacing = 8;  // Increased from 3 to 8 for better visual separation
    const int available_width = width - 2 * padding - 2 * board_spacing;
    const int available_height = height - 2 * padding;
    
    const int cell_w = available_width / 36;  // 3 boards * 12 cols
    const int cell_h = available_height / 8;   // 8 rows
    const int cell_size = std::min(cell_w, cell_h);
    
    // Center the boards
    const int total_width = cell_size * 36 + 2 * board_spacing;
    const int total_height = cell_size * 8;
    const int start_x = x + (width - total_width) / 2;
    const int start_y = y + (height - total_height) / 2;
    
    // Draw the 3 boards
    for (int layer = 0; layer < NUM_BOARDS; ++layer) {
        int board_x = start_x + layer * (cell_size * 12 + board_spacing);
        
        // Draw checkerboard pattern
        for (int row = 0; row < BOARD_ROWS; ++row) {
            for (int col = 0; col < BOARD_COLS; ++col) {
                int px = board_x + col * cell_size;
                int py = start_y + row * cell_size;
                
                // Checkerboard colors
                bool is_light = (row + col) % 2 == 0;
                if (is_light) {
                    SDL_SetRenderDrawColor(sdl_renderer, 200, 180, 150, 255);
                } else {
                    SDL_SetRenderDrawColor(sdl_renderer, 140, 110, 80, 255);
                }
                
                SDL_Rect cell = {px, py, cell_size, cell_size};
                SDL_RenderFillRect(sdl_renderer, &cell);
            }
        }
        
        // Draw pieces as scaled sprites
        for (int row = 0; row < BOARD_ROWS; ++row) {
            for (int col = 0; col < BOARD_COLS; ++col) {
                int idx = pos_to_index(layer, row, col);
                int16_t piece = match.game->board[idx];
                
                if (piece != EMPTY) {
                    SDL_Texture* piece_texture = renderer.get_piece_texture(piece);
                    if (piece_texture) {
                        // Render the actual piece sprite scaled down
                        int px = board_x + col * cell_size;
                        int py = start_y + row * cell_size;
                        
                        // Add small padding so piece doesn't fill entire cell
                        int padding = cell_size / 8;
                        SDL_Rect dest = {px + padding, py + padding, 
                                       cell_size - 2 * padding, cell_size - 2 * padding};
                        SDL_RenderCopy(sdl_renderer, piece_texture, nullptr, &dest);
                    }
                }
            }
        }
        
        // Draw vertical dividing line between boards
        if (layer < NUM_BOARDS - 1) {
            int line_x = board_x + cell_size * 12 + board_spacing / 2;
            SDL_SetRenderDrawColor(sdl_renderer, 150, 150, 170, 255);
            SDL_RenderDrawLine(sdl_renderer, line_x, start_y, line_x, start_y + total_height);
        }
    }
}

void Tournament::render_parallel_matches(Renderer& renderer, int round_num, int total_rounds) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    
    // Modern dark background
    SDL_SetRenderDrawColor(sdl_renderer, 25, 28, 35, 255);
    SDL_RenderClear(sdl_renderer);
    
    auto [win_w, win_h] = renderer.get_window_size();
    
    TTF_Font* font_title = TTF_OpenFont("assets/font.ttf", 40);
    TTF_Font* pixel_font = TTF_OpenFont("assets/pixel.ttf", 20);  // Increased for readability
    
    if (font_title) {
        std::string title = "Round " + std::to_string(round_num) + " / " + std::to_string(total_rounds);
        
        // Shadow
        SDL_Color shadow = {0, 0, 0, 180};
        SDL_Surface* surf = TTF_RenderText_Blended(font_title, title.c_str(), shadow);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2 + 2, 12, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        // Main title
        SDL_Color gold = {255, 215, 0, 255};
        surf = TTF_RenderText_Blended(font_title, title.c_str(), gold);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2, 10, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        TTF_CloseFont(font_title);
    }
    
    if (pixel_font) {
        // Draw quit button
        SDL_Rect quit_button = {win_w - 150, win_h - 50, 130, 35};
        SDL_SetRenderDrawColor(sdl_renderer, 150, 50, 50, 255);
        SDL_RenderFillRect(sdl_renderer, &quit_button);
        SDL_SetRenderDrawColor(sdl_renderer, 200, 100, 100, 255);
        SDL_RenderDrawRect(sdl_renderer, &quit_button);
        
        SDL_Color white = {255, 255, 255, 255};
        SDL_Surface* surf = TTF_RenderText_Blended(pixel_font, "Return to Menu", white);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {quit_button.x + (quit_button.w - surf->w) / 2, 
                           quit_button.y + (quit_button.h - surf->h) / 2, 
                           surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
        
        // Draw all active matches
        int matches_per_row = 2;
        int match_width = (win_w - 60) / matches_per_row - 20;
        int match_height = 180;
        
        for (size_t i = 0; i < active_matches.size(); ++i) {
            auto& match = active_matches[i];
            
            int row = i / matches_per_row;
            int col = i % matches_per_row;
            int mx = 30 + col * (match_width + 20);
            int my = 70 + row * (match_height + 100);
            
            // Match names with pixel font
            std::string p1_name = participants[match->player1_idx].name;
            std::string p2_name = participants[match->player2_idx].name;
            std::string match_text = p1_name + " vs " + p2_name;
            
            SDL_Color white = {220, 220, 230, 255};
            SDL_Surface* surf = TTF_RenderText_Blended(pixel_font, match_text.c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {mx, my, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
            
            // Draw mini board with checkerboard and pieces
            render_match_board(renderer, *match, mx, my + 22, match_width, match_height);
            
            // Status with pixel font
            std::string status;
            SDL_Color status_color = white;
            if (match->finished) {
                status = "Finished: " + match->winner + " (" + std::to_string(match->total_moves) + " moves)";
                status_color = {100, 255, 100, 255};  // Green when finished
            } else {
                status = "Playing... Move: " + std::to_string(match->current_move.load());
                status_color = {100, 200, 255, 255};  // Blue while playing
            }
            
            surf = TTF_RenderText_Blended(pixel_font, status.c_str(), status_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {mx, my + match_height + 28, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        TTF_CloseFont(pixel_font);
    }
    
    SDL_RenderPresent(sdl_renderer);
}

void Tournament::run_round_parallel(Renderer& renderer, InputHandler& input,
                                   std::vector<std::pair<int, int>>& pairings, int round_num) {
    // Create match states
    active_matches.clear();
    std::vector<std::thread> threads;
    
    for (size_t i = 0; i < pairings.size(); ++i) {
        auto match = std::make_shared<MatchState>();
        match->match_id = i;
        match->player1_idx = pairings[i].first;
        match->player2_idx = pairings[i].second;
        match->game = std::make_unique<Game>();
        match->current_move = 0;
        match->finished = false;
        
        active_matches.push_back(match);
        
        // Start match thread
        threads.emplace_back(&Tournament::run_match_async, this, match);
    }
    
    // Visualize while matches run
    bool all_finished = false;
    auto [win_w, win_h] = renderer.get_window_size();
    
    while (!all_finished && !should_quit) {
        // Check for quit or button click
        int mouse_x, mouse_y;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        if (event == InputEvent::QUIT) {
            should_quit = true;
        } else if (event == InputEvent::MOUSE_CLICK) {
            // Check if quit button was clicked
            SDL_Rect quit_button = {win_w - 150, win_h - 50, 130, 35};
            if (mouse_x >= quit_button.x && mouse_x < quit_button.x + quit_button.w &&
                mouse_y >= quit_button.y && mouse_y < quit_button.y + quit_button.h) {
                should_quit = true;
            }
        }
        
        if (should_quit) {
            // Signal all matches to finish
            for (auto& match : active_matches) {
                match->finished = true;
            }
            break;
        }
        
        // Check if all matches finished
        all_finished = true;
        for (const auto& match : active_matches) {
            if (!match->finished) {
                all_finished = false;
                break;
            }
        }
        
        render_parallel_matches(renderer, round_num, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS rendering
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }
    
    // Update statistics
    for (const auto& match : active_matches) {
        int p1_idx = match->player1_idx;
        int p2_idx = match->player2_idx;
        
        participants[p1_idx].total_moves += match->total_moves;
        participants[p2_idx].total_moves += match->total_moves;
        
        // Track shortest and longest games
        participants[p1_idx].shortest_game = std::min(participants[p1_idx].shortest_game, match->total_moves);
        participants[p1_idx].longest_game = std::max(participants[p1_idx].longest_game, match->total_moves);
        participants[p2_idx].shortest_game = std::min(participants[p2_idx].shortest_game, match->total_moves);
        participants[p2_idx].longest_game = std::max(participants[p2_idx].longest_game, match->total_moves);
        
        // Count remaining pieces
        int p1_pieces = 0, p2_pieces = 0;
        for (int i = 0; i < TOTAL_SQUARES; ++i) {
            int16_t piece = match->game->board[i];
            if (piece > 0) p1_pieces++;
            else if (piece < 0) p2_pieces++;
        }
        participants[p1_idx].total_pieces_remaining += p1_pieces;
        participants[p2_idx].total_pieces_remaining += p2_pieces;
        
        if (match->winner == "Gold") {
            participants[p1_idx].score += 1.0f;
            participants[p1_idx].wins += 1;
            participants[p1_idx].checkmates_delivered += 1;
            participants[p2_idx].losses += 1;
            update_elo(p1_idx, p2_idx, 1.0f);
        } else if (match->winner == "Scarlet") {
            participants[p2_idx].score += 1.0f;
            participants[p2_idx].wins += 1;
            participants[p2_idx].checkmates_delivered += 1;
            participants[p1_idx].losses += 1;
            update_elo(p1_idx, p2_idx, 0.0f);
        } else {
            participants[p1_idx].score += 0.5f;
            participants[p2_idx].score += 0.5f;
            participants[p1_idx].draws += 1;
            participants[p2_idx].draws += 1;
            update_elo(p1_idx, p2_idx, 0.5f);
        }
        
        // Update averages
        int p1_games = participants[p1_idx].wins + participants[p1_idx].losses + participants[p1_idx].draws;
        int p2_games = participants[p2_idx].wins + participants[p2_idx].losses + participants[p2_idx].draws;
        if (p1_games > 0) participants[p1_idx].avg_moves_per_game = participants[p1_idx].total_moves / p1_games;
        if (p2_games > 0) participants[p2_idx].avg_moves_per_game = participants[p2_idx].total_moves / p2_games;
    }
}

void Tournament::render_standings(Renderer& renderer, int current_round, int total_rounds) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    
    // Modern dark background matching the selection screen
    SDL_SetRenderDrawColor(sdl_renderer, 25, 28, 35, 255);
    SDL_RenderClear(sdl_renderer);
    
    auto [win_w, win_h] = renderer.get_window_size();
    
    TTF_Font* font_title = TTF_OpenFont("assets/font.ttf", 40);
    TTF_Font* font = TTF_OpenFont("assets/font.ttf", 24);
    
    if (font_title) {
        std::string title = "Round " + std::to_string(current_round) + " / " + std::to_string(total_rounds) + " Complete";
        
        // Shadow
        SDL_Color shadow = {0, 0, 0, 180};
        SDL_Surface* surface = TTF_RenderText_Blended(font_title, title.c_str(), shadow);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(sdl_renderer, surface);
            SDL_Rect dest = {win_w / 2 - surface->w / 2 + 2, 22, surface->w, surface->h};
            SDL_RenderCopy(sdl_renderer, texture, nullptr, &dest);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
        
        // Main title
        SDL_Color gold = {255, 215, 0, 255};
        surface = TTF_RenderText_Blended(font_title, title.c_str(), gold);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(sdl_renderer, surface);
            SDL_Rect dest = {win_w / 2 - surface->w / 2, 20, surface->w, surface->h};
            SDL_RenderCopy(sdl_renderer, texture, nullptr, &dest);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
        
        TTF_CloseFont(font_title);
    }
    
    render_detailed_stats(renderer);
    
    // Bottom hint with pixel font
    TTF_Font* pixel_font = TTF_OpenFont("assets/pixel.ttf", 20);  // Increased for readability
    if (pixel_font) {
        std::string hint = current_round < total_rounds ? 
            "Press SPACE to continue to next round" : 
            "Tournament Complete! Press SPACE to continue";
        SDL_Color hint_color = {100, 200, 255, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(pixel_font, hint.c_str(), hint_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(sdl_renderer, surface);
            SDL_Rect dest = {win_w / 2 - surface->w / 2, win_h - 50, surface->w, surface->h};
            SDL_RenderCopy(sdl_renderer, texture, nullptr, &dest);
            SDL_DestroyTexture(texture);
            SDL_FreeSurface(surface);
        }
        TTF_CloseFont(pixel_font);
    }
    
    SDL_RenderPresent(sdl_renderer);
}

void Tournament::render_detailed_stats(Renderer& renderer) {
    SDL_Renderer* sdl_renderer = renderer.get_renderer();
    auto [win_w, win_h] = renderer.get_window_size();
    
    TTF_Font* pixel_font = TTF_OpenFont("assets/pixel.ttf", 20);  // Increased for readability
    TTF_Font* pixel_font_small = TTF_OpenFont("assets/pixel.ttf", 16);  // Increased for readability
    if (!pixel_font) return;
    
    // Sort participants
    auto sorted = participants;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.elo > b.elo;
    });
    
    // Draw header background
    int table_start_y = 80;
    SDL_Rect header_bg = {20, table_start_y, win_w - 40, 50};
    SDL_SetRenderDrawColor(sdl_renderer, 40, 45, 55, 255);
    SDL_RenderFillRect(sdl_renderer, &header_bg);
    SDL_SetRenderDrawColor(sdl_renderer, 80, 85, 95, 255);
    SDL_RenderDrawRect(sdl_renderer, &header_bg);
    
    // Table headers
    int y = table_start_y + 8;
    SDL_Color gold = {255, 215, 0, 255};
    SDL_Color white = {220, 220, 230, 255};
    SDL_Color gray = {140, 145, 155, 255};
    
    std::vector<std::string> headers = {"#", "Name", "Score", "W-L-D", "ELO", "Avg", "Min", "Max", "Mates"};
    std::vector<int> x_pos = {35, 70, 290, 370, 470, 540, 610, 680, 750};
    
    for (size_t i = 0; i < headers.size(); ++i) {
        SDL_Surface* surf = TTF_RenderText_Blended(pixel_font, headers[i].c_str(), gold);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {x_pos[i], y, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
    }
    
    // Column labels (second row)
    if (pixel_font_small) {
        y += 22;
        std::vector<std::string> sublabels = {"", "", "", "", "", "Moves", "Game", "Game", "Given"};
        for (size_t i = 5; i < sublabels.size(); ++i) {
            if (!sublabels[i].empty()) {
                SDL_Surface* surf = TTF_RenderText_Blended(pixel_font_small, sublabels[i].c_str(), gray);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                    SDL_Rect dest = {x_pos[i], y, surf->w, surf->h};
                    SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                    SDL_DestroyTexture(tex);
                    SDL_FreeSurface(surf);
                }
            }
        }
    }
    
    y = table_start_y + 60;
    
    // Participant rows with card-style design
    for (size_t i = 0; i < sorted.size(); ++i) {
        const auto& p = sorted[i];
        
        int games_played = p.wins + p.losses + p.draws;
        int avg_pieces = games_played > 0 ? p.total_pieces_remaining / games_played : 0;
        
        // Row background
        SDL_Rect row_bg = {20, y - 3, win_w - 40, 35};
        if (i % 2 == 0) {
            SDL_SetRenderDrawColor(sdl_renderer, 40, 45, 55, 255);
        } else {
            SDL_SetRenderDrawColor(sdl_renderer, 35, 40, 50, 255);
        }
        SDL_RenderFillRect(sdl_renderer, &row_bg);
        
        // Highlight border for top 3
        if (i < 3) {
            SDL_Color border_color;
            if (i == 0) border_color = {255, 215, 0, 255};        // Gold
            else if (i == 1) border_color = {192, 192, 192, 255}; // Silver
            else border_color = {205, 127, 50, 255};              // Bronze
            
            SDL_SetRenderDrawColor(sdl_renderer, border_color.r, border_color.g, border_color.b, border_color.a);
            SDL_RenderDrawRect(sdl_renderer, &row_bg);
        }
        
        std::vector<std::string> row = {
            std::to_string(i + 1),
            p.name.length() > 18 ? p.name.substr(0, 15) + "..." : p.name,
            std::to_string(static_cast<int>(p.score * 10) / 10.0f),
            std::to_string(p.wins) + "-" + std::to_string(p.losses) + "-" + std::to_string(p.draws),
            std::to_string(static_cast<int>(p.elo)),
            std::to_string(p.avg_moves_per_game),
            p.shortest_game < 999999 ? std::to_string(p.shortest_game) : "---",
            p.longest_game > 0 ? std::to_string(p.longest_game) : "---",
            std::to_string(p.checkmates_delivered)
        };
        
        // Color code by rank
        SDL_Color row_color;
        if (i == 0) row_color = {255, 215, 0, 255};        // Gold
        else if (i == 1) row_color = {192, 192, 192, 255}; // Silver
        else if (i == 2) row_color = {205, 127, 50, 255};  // Bronze
        else row_color = white;
        
        for (size_t j = 0; j < row.size(); ++j) {
            SDL_Surface* surf = TTF_RenderText_Blended(pixel_font_small ? pixel_font_small : pixel_font, row[j].c_str(), row_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
                SDL_Rect dest = {x_pos[j], y, surf->w, surf->h};
                SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        y += 28;
    }
    
    // Add summary statistics at the bottom
    y += 20;
    if (pixel_font_small && !sorted.empty()) {
        SDL_Color cyan = {100, 200, 255, 255};
        
        // Calculate tournament stats
        int total_games = 0;
        int total_moves = 0;
        int total_checkmates = 0;
        for (const auto& p : sorted) {
            total_games += p.wins + p.losses + p.draws;
            total_moves += p.total_moves;
            total_checkmates += p.checkmates_delivered;
        }
        total_games /= 2;  // Each game counted twice
        
        std::string summary = "Tournament: " + std::to_string(total_games) + " games, " +
                            std::to_string(total_moves) + " total moves, " +
                            std::to_string(total_checkmates) + " checkmates";
        
        SDL_Surface* surf = TTF_RenderText_Blended(pixel_font_small, summary.c_str(), cyan);
        if (surf) {
            SDL_Texture* tex = SDL_CreateTextureFromSurface(sdl_renderer, surf);
            SDL_Rect dest = {win_w / 2 - surf->w / 2, y, surf->w, surf->h};
            SDL_RenderCopy(sdl_renderer, tex, nullptr, &dest);
            SDL_DestroyTexture(tex);
            SDL_FreeSurface(surf);
        }
    }
    
    if (pixel_font_small) TTF_CloseFont(pixel_font_small);
    TTF_CloseFont(pixel_font);
}

void Tournament::run(Renderer& renderer, InputHandler& input, int rounds) {
    std::cout << "Starting tournament with " << participants.size() << " participants for " << rounds << " rounds" << std::endl;
    
    should_quit = false;  // Reset quit flag
    
    for (int round = 1; round <= rounds && !should_quit; ++round) {
        std::cout << "\n=== Round " << round << " ===" << std::endl;
        
        auto pairings = generate_pairings();
        
        // Run round with parallel visualization
        run_round_parallel(renderer, input, pairings, round);
        
        if (should_quit) break;
        
        // Show standings
        render_standings(renderer, round, rounds);
        
        // Wait for user to continue or quit
        bool waiting = true;
        auto start_time = std::chrono::steady_clock::now();
        while (waiting && !should_quit) {
            int mouse_x, mouse_y;
            InputEvent event = input.poll(mouse_x, mouse_y);
            
            if (event == InputEvent::QUIT) {
                should_quit = true;
            } else if (event == InputEvent::MOUSE_CLICK || event == InputEvent::SPACE_PRESSED) {
                waiting = false;
            }
            
            // Auto-continue after 3 seconds
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed > std::chrono::seconds(3)) {
                waiting = false;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    
    if (should_quit) {
        std::cout << "\nTournament cancelled by user." << std::endl;
        return;
    }
    
    // Final standings
    std::cout << "\n=== Final Standings ===" << std::endl;
    auto sorted = participants;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.elo > b.elo;
    });
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        const auto& p = sorted[i];
        std::cout << (i + 1) << ". " << p.name 
                  << " - Score: " << p.score 
                  << " (W:" << p.wins << " L:" << p.losses << " D:" << p.draws << ")"
                  << " ELO: " << static_cast<int>(p.elo)
                  << " Avg Moves: " << p.avg_moves_per_game << std::endl;
    }
    
    // Wait for user
    render_detailed_stats(renderer);
    SDL_RenderPresent(renderer.get_renderer());
    
    bool running = true;
    while (running) {
        int mouse_x, mouse_y;
        InputEvent event = input.poll(mouse_x, mouse_y);
        
        if (event == InputEvent::QUIT || event == InputEvent::MOUSE_CLICK || event == InputEvent::SPACE_PRESSED) {
            running = false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

} // namespace dragonchess
