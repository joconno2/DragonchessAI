#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <vector>
#include <functional>

namespace dragonchess {

enum class GameMode {
    TWO_PLAYER,
    AI_VS_PLAYER,
    AI_VS_AI,
    TOURNAMENT,
    CAMPAIGN,
    QUIT
};

struct AISettings {
    std::string gold_ai_path;
    std::string scarlet_ai_path;
    std::string player_side; // "Gold" or "Scarlet"
};

struct MenuButton {
    SDL_Rect rect;
    std::string text;
    GameMode mode;
    bool hovered;
};

class Menu {
public:
    Menu();
    ~Menu();
    
    bool init(SDL_Renderer* renderer);
    void cleanup();
    
    // Show main menu and return selected mode
    GameMode show_main_menu(SDL_Renderer* renderer);
    
    // Show AI vs Player setup
    AISettings show_ai_vs_player_menu(SDL_Renderer* renderer);
    
    // Show AI vs AI setup
    AISettings show_ai_vs_ai_menu(SDL_Renderer* renderer);
    
private:
    TTF_Font* title_font;
    TTF_Font* button_font;
    SDL_Texture* background;
    
    std::vector<MenuButton> buttons;
    
    void render_menu(SDL_Renderer* renderer, const std::string& title);
    void render_button(SDL_Renderer* renderer, const MenuButton& button);
    void render_text(SDL_Renderer* renderer, const std::string& text, int x, int y, 
                    SDL_Color color, TTF_Font* font);
    int get_button_at_position(int x, int y);
    void update_hover_state(int mouse_x, int mouse_y);
};

} // namespace dragonchess
