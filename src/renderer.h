#pragma once

#include "game.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <unordered_map>
#include <optional>

namespace dragonchess {

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    // Initialize SDL and create window
    bool init(const std::string& title, int width, int height);
    
    // Cleanup SDL resources
    void cleanup();
    
    // Load all game assets
    bool load_assets();
    
    // Render the game board
    void render_board(const Game& game, 
                     std::optional<int> selected = std::nullopt,
                     const std::unordered_set<int>& legal_dest = {},
                     bool show_move_history = true,
                     bool show_timer = true);
    
    // Render move history panel
    void render_move_history(const Game& game, int panel_x, int panel_y, int panel_w, int panel_h);
    
    // Render game clock
    void render_clock(const Game& game, int panel_x, int panel_y, int panel_w, int panel_h);
    
    // Render post-game screen
    void render_post_game(const Game& game);
    
    // Convert screen coordinates to board position
    std::optional<Position> screen_to_board(int x, int y) const;
    
    // Present the rendered frame
    void present();
    
    // Clear the screen
    void clear();
    
    // Get window size
    std::pair<int, int> get_window_size() const;
    
    // Get SDL renderer (for menu system)
    SDL_Renderer* get_renderer() { return renderer; }
    
    // Get piece texture for rendering
    SDL_Texture* get_piece_texture(int16_t piece) const {
        auto it = piece_textures.find(piece);
        return (it != piece_textures.end()) ? it->second : nullptr;
    }
    
    // Set fullscreen mode
    void set_fullscreen(bool fullscreen);
    
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;           // Decorative font (font.ttf) for titles
    TTF_Font* font_large;     // Large decorative font
    TTF_Font* pixel_font;     // Pixel font for UI/statistics
    TTF_Font* pixel_font_small; // Small pixel font
    
    // Textures for pieces
    std::unordered_map<int16_t, SDL_Texture*> piece_textures;
    
    // Background texture
    SDL_Texture* bg_texture;
    
    // Virtual resolution (logical coordinates)
    // Optimized layout for 1920x1080 with all UI elements visible
    static constexpr int VIRTUAL_WIDTH = 1920;
    static constexpr int VIRTUAL_HEIGHT = 1080;
    static constexpr int CELL_SIZE = 42;  // 12*42=504, 3 boards = 1512 + spacing
    
    // Layout sections
    static constexpr int SIDEBAR_WIDTH = 260;   // For move history
    static constexpr int TOP_PANEL_HEIGHT = 70; // For clock/info
    static constexpr int BOTTOM_PANEL_HEIGHT = 50; // For controls hint
    
    // Current window dimensions
    int window_width;
    int window_height;
    
    // Scaling factors
    float scale_x;
    float scale_y;
    float scale_uniform;  // Uniform scale to maintain aspect ratio
    int offset_x;
    int offset_y;
    
    // Helper functions
    void update_scale();
    SDL_Rect scale_rect(int x, int y, int w, int h) const;
    void scale_point(int& x, int& y) const;
    void unscale_point(int& x, int& y) const;
    
    // Helper functions
    SDL_Texture* load_texture(const std::string& path);
    void render_text(const std::string& text, int x, int y, SDL_Color color, TTF_Font* use_font = nullptr);
    void render_outlined_text(const std::string& text, int x, int y, TTF_Font* use_font);
    void draw_cell(int layer, int row, int col, const Game& game, 
                  bool is_selected, bool is_legal_dest, bool is_last_move = false);
};

} // namespace dragonchess
