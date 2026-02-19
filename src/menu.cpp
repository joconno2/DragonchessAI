#include "menu.h"
#include <SDL2/SDL_image.h>
#include <iostream>
#include <algorithm>

namespace dragonchess {

Menu::Menu() 
    : title_font(nullptr)
    , button_font(nullptr)
    , background(nullptr)
{
}

Menu::~Menu() {
    cleanup();
}

bool Menu::init(SDL_Renderer* renderer) {
    // Load fonts
    title_font = TTF_OpenFont("assets/font.ttf", 48);
    if (!title_font) {
        std::cerr << "Failed to load title font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    button_font = TTF_OpenFont("assets/font.ttf", 32);
    if (!button_font) {
        std::cerr << "Failed to load button font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    // Try to load menu background
    SDL_Surface* bg_surface = IMG_Load("assets/main_menu_1.png");
    if (bg_surface) {
        background = SDL_CreateTextureFromSurface(renderer, bg_surface);
        SDL_FreeSurface(bg_surface);
    }
    
    return true;
}

void Menu::cleanup() {
    if (title_font) {
        TTF_CloseFont(title_font);
        title_font = nullptr;
    }
    
    if (button_font) {
        TTF_CloseFont(button_font);
        button_font = nullptr;
    }
    
    if (background) {
        SDL_DestroyTexture(background);
        background = nullptr;
    }
}

void Menu::render_text(SDL_Renderer* renderer, const std::string& text, int x, int y,
                       SDL_Color color, TTF_Font* font) {
    SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), color);
    if (!surface) return;
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return;
    }
    
    SDL_Rect dest = {x, y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, nullptr, &dest);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

void Menu::render_button(SDL_Renderer* renderer, const MenuButton& button) {
    // Button background
    if (button.hovered) {
        SDL_SetRenderDrawColor(renderer, 100, 100, 200, 255); // Lighter when hovered
    } else {
        SDL_SetRenderDrawColor(renderer, 60, 60, 120, 255);
    }
    SDL_RenderFillRect(renderer, &button.rect);
    
    // Button border
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    SDL_RenderDrawRect(renderer, &button.rect);
    
    // Button text (centered)
    SDL_Color text_color = {255, 255, 255, 255};
    SDL_Surface* text_surface = TTF_RenderText_Blended(button_font, button.text.c_str(), text_color);
    if (text_surface) {
        int text_x = button.rect.x + (button.rect.w - text_surface->w) / 2;
        int text_y = button.rect.y + (button.rect.h - text_surface->h) / 2;
        render_text(renderer, button.text, text_x, text_y, text_color, button_font);
        SDL_FreeSurface(text_surface);
    }
}

void Menu::update_hover_state(int mouse_x, int mouse_y) {
    for (auto& button : buttons) {
        button.hovered = (mouse_x >= button.rect.x && 
                         mouse_x < button.rect.x + button.rect.w &&
                         mouse_y >= button.rect.y && 
                         mouse_y < button.rect.y + button.rect.h);
    }
}

int Menu::get_button_at_position(int x, int y) {
    for (size_t i = 0; i < buttons.size(); ++i) {
        const auto& button = buttons[i];
        if (x >= button.rect.x && x < button.rect.x + button.rect.w &&
            y >= button.rect.y && y < button.rect.y + button.rect.h) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void Menu::render_menu(SDL_Renderer* renderer, const std::string& title) {
    // Clear screen
    SDL_SetRenderDrawColor(renderer, 20, 20, 40, 255);
    SDL_RenderClear(renderer);
    
    // Get logical size for positioning
    int window_w, window_h;
    SDL_RenderGetLogicalSize(renderer, &window_w, &window_h);
    
    // Draw background if available (centered, maintaining aspect ratio)
    if (background) {
        int bg_w, bg_h;
        SDL_QueryTexture(background, nullptr, nullptr, &bg_w, &bg_h);
        
        // Scale background to fit while maintaining aspect ratio
        float scale_x = static_cast<float>(window_w) / bg_w;
        float scale_y = static_cast<float>(window_h) / bg_h;
        float scale = std::min(scale_x, scale_y);
        
        int scaled_w = static_cast<int>(bg_w * scale);
        int scaled_h = static_cast<int>(bg_h * scale);
        int bg_x = (window_w - scaled_w) / 2;
        int bg_y = (window_h - scaled_h) / 2;
        
        SDL_Rect bg_rect = {bg_x, bg_y, scaled_w, scaled_h};
        SDL_RenderCopy(renderer, background, nullptr, &bg_rect);
    }
    
    // Draw title
    SDL_Color title_color = {255, 215, 0, 255}; // Gold color
    SDL_Surface* title_surface = TTF_RenderText_Blended(title_font, title.c_str(), title_color);
    if (title_surface) {
        int title_x = (window_w - title_surface->w) / 2;
        render_text(renderer, title, title_x, 80, title_color, title_font);
        SDL_FreeSurface(title_surface);
    }
    
    // Draw buttons
    for (const auto& button : buttons) {
        render_button(renderer, button);
    }
    
    SDL_RenderPresent(renderer);
}

GameMode Menu::show_main_menu(SDL_Renderer* renderer) {
    // Setup buttons
    buttons.clear();
    
    int window_w, window_h;
    SDL_RenderGetLogicalSize(renderer, &window_w, &window_h);
    
    int button_width = 250;
    int button_height = 60;
    int button_x = (window_w - button_width) / 2;
    int button_y_start = 250;
    int button_gap = 80;
    
    buttons.push_back({
        {button_x, button_y_start + 0 * button_gap, button_width, button_height},
        "2 Player", GameMode::TWO_PLAYER, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 1 * button_gap, button_width, button_height},
        "AI vs Player", GameMode::AI_VS_PLAYER, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 2 * button_gap, button_width, button_height},
        "AI vs AI", GameMode::AI_VS_AI, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 3 * button_gap, button_width, button_height},
        "Tournament", GameMode::TOURNAMENT, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 4 * button_gap, button_width, button_height},
        "Campaign", GameMode::CAMPAIGN, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 5 * button_gap, button_width, button_height},
        "Quit", GameMode::QUIT, false
    });
    
    // Main menu loop
    bool running = true;
    GameMode selected_mode = GameMode::QUIT;
    
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                return GameMode::QUIT;
            }
            
            if (event.type == SDL_MOUSEMOTION) {
                update_hover_state(event.motion.x, event.motion.y);
            }
            
            if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                int button_index = get_button_at_position(event.button.x, event.button.y);
                if (button_index >= 0) {
                    selected_mode = buttons[button_index].mode;
                    running = false;
                }
            }
        }
        
        render_menu(renderer, "Dragonchess");
        SDL_Delay(16); // ~60 FPS
    }
    
    return selected_mode;
}

AISettings Menu::show_ai_vs_player_menu(SDL_Renderer* renderer) {
    AISettings settings;
    settings.player_side = "Gold"; // Default to player as Gold
    settings.scarlet_ai_path = ""; // Use RandomAI by default
    
    // Simple menu for now - just select side
    buttons.clear();
    
    int window_w, window_h;
    SDL_RenderGetLogicalSize(renderer, &window_w, &window_h);
    
    int button_width = 250;
    int button_height = 60;
    int button_x = (window_w - button_width) / 2;
    int button_y_start = 300;
    int button_gap = 80;
    
    buttons.push_back({
        {button_x, button_y_start, button_width, button_height},
        "Play as Gold", GameMode::TWO_PLAYER, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + button_gap, button_width, button_height},
        "Play as Scarlet", GameMode::AI_VS_PLAYER, false
    });
    
    buttons.push_back({
        {button_x, button_y_start + 2 * button_gap, button_width, button_height},
        "Back", GameMode::QUIT, false
    });
    
    bool running = true;
    SDL_Event event;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                settings.player_side = ""; // Signal to quit
                return settings;
            }
            
            if (event.type == SDL_MOUSEMOTION) {
                update_hover_state(event.motion.x, event.motion.y);
            }
            
            if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                int button_index = get_button_at_position(event.button.x, event.button.y);
                if (button_index == 0) {
                    settings.player_side = "Gold";
                    running = false;
                } else if (button_index == 1) {
                    settings.player_side = "Scarlet";
                    running = false;
                } else if (button_index == 2) {
                    settings.player_side = ""; // Back to main menu
                    return settings;
                }
            }
        }
        
        render_menu(renderer, "AI vs Player");
        SDL_Delay(16);
    }
    
    return settings;
}

AISettings Menu::show_ai_vs_ai_menu(SDL_Renderer* renderer) {
    AISettings settings;
    
    // AI types available
    std::vector<std::string> ai_types = {
        "RandomAI",
        "GreedyAI", 
        "GreedyValueAI",
        "MinimaxAI (depth 2)",
        "AlphaBetaAI (depth 3)"
    };
    
    int gold_ai_index = 0;  // Default to RandomAI
    int scarlet_ai_index = 0;
    bool selecting_gold = true;  // true = selecting gold, false = selecting scarlet
    
    bool done = false;
    bool cancelled = false;
    
    TTF_Font* title_font = TTF_OpenFont("assets/font.ttf", 48);
    TTF_Font* font = TTF_OpenFont("assets/font.ttf", 32);
    TTF_Font* small_font = TTF_OpenFont("assets/pixel.ttf", 20);
    
    // Layout constants for proper centering
    const int screen_width = 1920;
    const int column_width = 450;
    const int column_spacing = 100;
    const int total_width = column_width * 2 + column_spacing;
    const int left_margin = (screen_width - total_width) / 2;
    const int gold_x = left_margin;
    const int scarlet_x = left_margin + column_width + column_spacing;
    
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                cancelled = true;
                done = true;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    cancelled = true;
                    done = true;
                } else if (event.key.keysym.sym == SDLK_RETURN || event.key.keysym.sym == SDLK_SPACE) {
                    if (selecting_gold) {
                        selecting_gold = false;  // Move to scarlet selection
                    } else {
                        done = true;  // Both selected, start game
                    }
                } else if (event.key.keysym.sym == SDLK_UP) {
                    if (selecting_gold) {
                        gold_ai_index = (gold_ai_index - 1 + ai_types.size()) % ai_types.size();
                    } else {
                        scarlet_ai_index = (scarlet_ai_index - 1 + ai_types.size()) % ai_types.size();
                    }
                } else if (event.key.keysym.sym == SDLK_DOWN) {
                    if (selecting_gold) {
                        gold_ai_index = (gold_ai_index + 1) % ai_types.size();
                    } else {
                        scarlet_ai_index = (scarlet_ai_index + 1) % ai_types.size();
                    }
                }
            } else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                int mouse_x = event.button.x;
                int mouse_y = event.button.y;
                
                // Check AI type clicks
                int start_y = 200;
                for (size_t i = 0; i < ai_types.size(); ++i) {
                    int y = start_y + i * 60;
                    SDL_Rect gold_box = {gold_x, y, column_width, 50};
                    SDL_Rect scarlet_box = {scarlet_x, y, column_width, 50};
                    
                    if (mouse_x >= gold_box.x && mouse_x < gold_box.x + gold_box.w &&
                        mouse_y >= gold_box.y && mouse_y < gold_box.y + gold_box.h) {
                        gold_ai_index = i;
                        if (!selecting_gold) selecting_gold = true;
                    }
                    if (mouse_x >= scarlet_box.x && mouse_x < scarlet_box.x + scarlet_box.w &&
                        mouse_y >= scarlet_box.y && mouse_y < scarlet_box.y + scarlet_box.h) {
                        scarlet_ai_index = i;
                        if (selecting_gold) selecting_gold = false;
                    }
                }
                
                // Check confirm button
                SDL_Rect confirm_button = {750, 550, 250, 60};
                if (mouse_x >= confirm_button.x && mouse_x < confirm_button.x + confirm_button.w &&
                    mouse_y >= confirm_button.y && mouse_y < confirm_button.y + confirm_button.h) {
                    if (selecting_gold) {
                        selecting_gold = false;
                    } else {
                        done = true;
                    }
                }
            }
        }
        
        // Render
        SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
        SDL_RenderClear(renderer);
        
        // Title
        if (title_font) {
            SDL_Color white = {255, 255, 255, 255};
            SDL_Surface* surf = TTF_RenderText_Blended(title_font, "AI vs AI Setup", white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_Rect dest = {960 - surf->w / 2, 50, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        // Column headers
        if (font) {
            SDL_Color gold_color = {255, 215, 0, 255};
            SDL_Color scarlet_color = {220, 20, 60, 255};
            SDL_Color gray = {150, 150, 150, 255};
            
            // Gold header - centered in column
            SDL_Surface* surf = TTF_RenderText_Blended(font, "Gold AI", 
                selecting_gold ? gold_color : gray);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_Rect dest = {gold_x + column_width / 2 - surf->w / 2, 140, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
            
            // Scarlet header - centered in column
            surf = TTF_RenderText_Blended(font, "Scarlet AI",
                selecting_gold ? gray : scarlet_color);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_Rect dest = {scarlet_x + column_width / 2 - surf->w / 2, 140, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        // Draw AI options
        if (small_font) {
            int start_y = 200;
            for (size_t i = 0; i < ai_types.size(); ++i) {
                int y = start_y + i * 60;
                
                // Gold column
                SDL_Rect gold_box = {gold_x, y, column_width, 50};
                bool gold_selected = (selecting_gold && i == gold_ai_index);
                bool gold_chosen = (i == gold_ai_index);
                
                if (gold_selected) {
                    SDL_SetRenderDrawColor(renderer, 255, 215, 0, 100);
                } else if (gold_chosen && !selecting_gold) {
                    SDL_SetRenderDrawColor(renderer, 255, 215, 0, 50);
                } else {
                    SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
                }
                SDL_RenderFillRect(renderer, &gold_box);
                SDL_SetRenderDrawColor(renderer, gold_selected ? 255 : 100, 
                                      gold_selected ? 215 : 100, 
                                      gold_selected ? 0 : 100, 255);
                SDL_RenderDrawRect(renderer, &gold_box);
                
                SDL_Color text_color = gold_selected ? SDL_Color{255, 255, 255, 255} : 
                                      SDL_Color{200, 200, 200, 255};
                SDL_Surface* surf = TTF_RenderText_Blended(small_font, ai_types[i].c_str(), text_color);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                    SDL_Rect dest = {gold_box.x + 20, gold_box.y + 12, surf->w, surf->h};
                    SDL_RenderCopy(renderer, tex, nullptr, &dest);
                    SDL_DestroyTexture(tex);
                    SDL_FreeSurface(surf);
                }
                
                // Scarlet column
                SDL_Rect scarlet_box = {scarlet_x, y, column_width, 50};
                bool scarlet_selected = (!selecting_gold && i == scarlet_ai_index);
                bool scarlet_chosen = (i == scarlet_ai_index);
                
                if (scarlet_selected) {
                    SDL_SetRenderDrawColor(renderer, 220, 20, 60, 100);
                } else if (scarlet_chosen && selecting_gold) {
                    SDL_SetRenderDrawColor(renderer, 220, 20, 60, 50);
                } else {
                    SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
                }
                SDL_RenderFillRect(renderer, &scarlet_box);
                SDL_SetRenderDrawColor(renderer, scarlet_selected ? 220 : 100,
                                      scarlet_selected ? 20 : 100,
                                      scarlet_selected ? 60 : 100, 255);
                SDL_RenderDrawRect(renderer, &scarlet_box);
                
                text_color = scarlet_selected ? SDL_Color{255, 255, 255, 255} : 
                            SDL_Color{200, 200, 200, 255};
                surf = TTF_RenderText_Blended(small_font, ai_types[i].c_str(), text_color);
                if (surf) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                    SDL_Rect dest = {scarlet_box.x + 20, scarlet_box.y + 12, surf->w, surf->h};
                    SDL_RenderCopy(renderer, tex, nullptr, &dest);
                    SDL_DestroyTexture(tex);
                    SDL_FreeSurface(surf);
                }
            }
        }
        
        // Confirm button - centered
        if (font) {
            SDL_Rect confirm_button = {960 - 125, 550, 250, 60};
            SDL_SetRenderDrawColor(renderer, 100, 200, 100, 255);
            SDL_RenderFillRect(renderer, &confirm_button);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderDrawRect(renderer, &confirm_button);
            
            std::string button_text = selecting_gold ? "Next >" : "Start Game";
            SDL_Color white = {255, 255, 255, 255};
            SDL_Surface* surf = TTF_RenderText_Blended(font, button_text.c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_Rect dest = {875 - surf->w / 2, 568, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        // Instructions
        if (small_font) {
            SDL_Color gray = {150, 150, 150, 255};
            std::string inst = "Use arrow keys or click to select AI | ENTER/SPACE to confirm | ESC to cancel";
            SDL_Surface* surf = TTF_RenderText_Blended(small_font, inst.c_str(), gray);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_Rect dest = {960 - surf->w / 2, 650, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, nullptr, &dest);
                SDL_DestroyTexture(tex);
                SDL_FreeSurface(surf);
            }
        }
        
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    
    if (title_font) TTF_CloseFont(title_font);
    if (font) TTF_CloseFont(font);
    if (small_font) TTF_CloseFont(small_font);
    
    if (cancelled) {
        settings.gold_ai_path = "CANCELLED";
        settings.scarlet_ai_path = "CANCELLED";
        return settings;
    }
    
    // Map indices to AI type strings
    settings.gold_ai_path = ai_types[gold_ai_index];
    settings.scarlet_ai_path = ai_types[scarlet_ai_index];
    
    return settings;
}

} // namespace dragonchess
