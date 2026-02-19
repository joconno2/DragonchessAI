#include "renderer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace dragonchess {

Renderer::Renderer()
    : window(nullptr)
    , renderer(nullptr)
    , font(nullptr)
    , font_large(nullptr)
    , pixel_font(nullptr)
    , pixel_font_small(nullptr)
    , bg_texture(nullptr)
    , window_width(1280)
    , window_height(720)
    , scale_x(1.0f)
    , scale_y(1.0f)
    , scale_uniform(1.0f)
    , offset_x(0)
    , offset_y(0)
{
}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::init(const std::string& title, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL Init failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    if (IMG_Init(IMG_INIT_PNG) == 0) {
        std::cerr << "SDL_image Init failed: " << IMG_GetError() << std::endl;
        return false;
    }
    
    if (TTF_Init() < 0) {
        std::cerr << "SDL_ttf Init failed: " << TTF_GetError() << std::endl;
        return false;
    }
    
    window_width = width;
    window_height = height;
    
    window = SDL_CreateWindow(
        title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    // Enable logical size for automatic scaling
    SDL_RenderSetLogicalSize(renderer, VIRTUAL_WIDTH, VIRTUAL_HEIGHT);
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");  // Linear filtering
    
    update_scale();
    
    return true;
}

void Renderer::cleanup() {
    // Clean up textures
    for (auto& [piece, texture] : piece_textures) {
        if (texture) {
            SDL_DestroyTexture(texture);
        }
    }
    piece_textures.clear();
    
    if (bg_texture) {
        SDL_DestroyTexture(bg_texture);
        bg_texture = nullptr;
    }
    
    if (font) {
        TTF_CloseFont(font);
        font = nullptr;
    }
    
    if (font_large) {
        TTF_CloseFont(font_large);
        font_large = nullptr;
    }
    
    if (pixel_font) {
        TTF_CloseFont(pixel_font);
        pixel_font = nullptr;
    }
    
    if (pixel_font_small) {
        TTF_CloseFont(pixel_font_small);
        pixel_font_small = nullptr;
    }
    
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    
    if (window) {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    
    TTF_Quit();
    IMG_Quit();
    SDL_Quit();
}

SDL_Texture* Renderer::load_texture(const std::string& path) {
    SDL_Surface* surface = IMG_Load(path.c_str());
    if (!surface) {
        std::cerr << "Failed to load image " << path << ": " << IMG_GetError() << std::endl;
        return nullptr;
    }
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    
    if (!texture) {
        std::cerr << "Failed to create texture from " << path << ": " << SDL_GetError() << std::endl;
    }
    
    return texture;
}

bool Renderer::load_assets() {
    // Load decorative fonts (font.ttf for titles)
    font = TTF_OpenFont("assets/font.ttf", 32);
    if (!font) {
        std::cerr << "Failed to load font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    font_large = TTF_OpenFont("assets/font.ttf", 48);
    if (!font_large) {
        std::cerr << "Failed to load large font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    // Load pixel fonts (pixel.ttf for UI and statistics)
    pixel_font = TTF_OpenFont("assets/pixel.ttf", 20);  // Increased from 16 for better readability
    if (!pixel_font) {
        std::cerr << "Failed to load pixel font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    pixel_font_small = TTF_OpenFont("assets/pixel.ttf", 16);  // Increased from 12 for better readability
    if (!pixel_font_small) {
        std::cerr << "Failed to load small pixel font: " << TTF_GetError() << std::endl;
        return false;
    }
    
    // Load background (not used currently - using flat color)
    bg_texture = load_texture("assets/bg.png");
    if (!bg_texture) {
        return false;
    }
    
    // Load piece textures
    std::vector<std::pair<int16_t, std::string>> piece_files = {
        {GOLD_SYLPH, "assets/gold_sylph.png"},
        {GOLD_GRIFFIN, "assets/gold_griffin.png"},
        {GOLD_DRAGON, "assets/gold_dragon.png"},
        {GOLD_OLIPHANT, "assets/gold_oliphant.png"},
        {GOLD_UNICORN, "assets/gold_unicorn.png"},
        {GOLD_HERO, "assets/gold_hero.png"},
        {GOLD_THIEF, "assets/gold_thief.png"},
        {GOLD_CLERIC, "assets/gold_cleric.png"},
        {GOLD_MAGE, "assets/gold_mage.png"},
        {GOLD_KING, "assets/gold_king.png"},
        {GOLD_PALADIN, "assets/gold_paladin.png"},
        {GOLD_WARRIOR, "assets/gold_warrior.png"},
        {GOLD_BASILISK, "assets/gold_basilisk.png"},
        {GOLD_ELEMENTAL, "assets/gold_elemental.png"},
        {GOLD_DWARF, "assets/gold_dwarf.png"},
        {SCARLET_SYLPH, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-sylph.png"},
        {SCARLET_GRIFFIN, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-griffin.png"},
        {SCARLET_DRAGON, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-dragon.png"},
        {SCARLET_OLIPHANT, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-oliphant.png"},
        {SCARLET_UNICORN, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-unicorn.png"},
        {SCARLET_HERO, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-hero.png"},
        {SCARLET_THIEF, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-thief.png"},
        {SCARLET_CLERIC, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-cleric.png"},
        {SCARLET_MAGE, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-mage.png"},
        {SCARLET_KING, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-king.png"},
        {SCARLET_PALADIN, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-paladin.png"},
        {SCARLET_WARRIOR, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-warrior.png"},
        {SCARLET_BASILISK, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-basilisk.png"},
        {SCARLET_ELEMENTAL, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-elemental.png"},
        {SCARLET_DWARF, "assets/Joconno2-dragon-chess-pieces-design-Scarlet-dwarf.png"}
    };
    
    for (const auto& [piece, path] : piece_files) {
        SDL_Texture* tex = load_texture(path);
        if (tex) {
            piece_textures[piece] = tex;
        } else {
            std::cerr << "Warning: Failed to load piece texture: " << path << std::endl;
        }
    }
    
    return true;
}

void Renderer::clear() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
}

void Renderer::present() {
    SDL_RenderPresent(renderer);
}

std::pair<int, int> Renderer::get_window_size() const {
    // Return virtual size for consistent UI rendering
    return {VIRTUAL_WIDTH, VIRTUAL_HEIGHT};
}

void Renderer::update_scale() {
    SDL_GetWindowSize(window, &window_width, &window_height);
    
    scale_x = static_cast<float>(window_width) / VIRTUAL_WIDTH;
    scale_y = static_cast<float>(window_height) / VIRTUAL_HEIGHT;
    
    // Use uniform scaling to maintain aspect ratio
    scale_uniform = std::min(scale_x, scale_y);
    
    // Calculate letterbox offsets
    int scaled_width = static_cast<int>(VIRTUAL_WIDTH * scale_uniform);
    int scaled_height = static_cast<int>(VIRTUAL_HEIGHT * scale_uniform);
    offset_x = (window_width - scaled_width) / 2;
    offset_y = (window_height - scaled_height) / 2;
}

SDL_Rect Renderer::scale_rect(int x, int y, int w, int h) const {
    // SDL_RenderSetLogicalSize handles this automatically
    return {x, y, w, h};
}

void Renderer::scale_point(int& x, int& y) const {
    // SDL_RenderSetLogicalSize handles this automatically
    // No transformation needed
}

void Renderer::unscale_point(int& x, int& y) const {
    // SDL_RenderSetLogicalSize handles this automatically
    // No transformation needed
}

void Renderer::set_fullscreen(bool fullscreen) {
    if (fullscreen) {
        SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    } else {
        SDL_SetWindowFullscreen(window, 0);
    }
    update_scale();
}

void Renderer::render_text(const std::string& text, int x, int y, SDL_Color color, TTF_Font* use_font) {
    if (!use_font) use_font = font;
    if (!use_font) return;
    
    SDL_Surface* surface = TTF_RenderText_Blended(use_font, text.c_str(), color);
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

void Renderer::render_outlined_text(const std::string& text, int x, int y, TTF_Font* use_font) {
    if (!use_font) use_font = font;
    
    // Draw outline in black
    SDL_Color black = {0, 0, 0, 255};
    for (int ox = -2; ox <= 2; ox += 4) {
        for (int oy = -2; oy <= 2; oy += 4) {
            render_text(text, x + ox, y + oy, black, use_font);
        }
    }
    
    // Draw main text in white
    SDL_Color white = {255, 255, 255, 255};
    render_text(text, x, y, white, use_font);
}

void Renderer::draw_cell(int layer, int row, int col, const Game& game, 
                        bool is_selected, bool is_legal_dest, bool is_last_move) {
    // Responsive layout:
    // - Always use horizontal layout (3 boards side by side)
    // - Scale down cell size if needed to fit screen
    // - Right sidebar for move history
    // - Top panel for timer/info
    
    const int board_pixel_width = BOARD_COLS * CELL_SIZE;   // 12 * 45 = 540
    const int board_pixel_height = BOARD_ROWS * CELL_SIZE;  // 8 * 45 = 360
    
    // Available space for boards (excluding sidebar and panels)
    const int available_width = VIRTUAL_WIDTH - SIDEBAR_WIDTH - 40;  // 1920 - 280 - 40 = 1600
    const int available_height = VIRTUAL_HEIGHT - TOP_PANEL_HEIGHT - BOTTOM_PANEL_HEIGHT - 40;
    
    // Calculate positions for horizontal layout (3 boards side by side)
    const int board_spacing = 15;  // Space between boards
    const int total_boards_width = board_pixel_width * 3 + board_spacing * 2; // 540*3 + 30 = 1650
    
    // Center the boards in available space
    int boards_start_x = (available_width - total_boards_width) / 2 + 20;
    int boards_start_y = TOP_PANEL_HEIGHT + (available_height - board_pixel_height) / 2 + 20;
    
    int x = boards_start_x + layer * (board_pixel_width + board_spacing);
    int y = boards_start_y;
    
    x += col * CELL_SIZE;
    y += row * CELL_SIZE;
    
    // Draw cell background
    SDL_Rect cell_rect = {x, y, CELL_SIZE, CELL_SIZE};
    
    // Checkerboard pattern with highlighting
    bool is_light = (row + col) % 2 == 0;
    if (is_selected) {
        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow for selected
    } else if (is_last_move) {
        SDL_SetRenderDrawColor(renderer, 255, 255, 100, 180); // Light yellow for last move
    } else if (is_legal_dest) {
        SDL_SetRenderDrawColor(renderer, 100, 255, 100, 200); // Green for legal destinations
    } else if (is_light) {
        SDL_SetRenderDrawColor(renderer, 240, 217, 181, 255); // Light square
    } else {
        SDL_SetRenderDrawColor(renderer, 181, 136, 99, 255); // Dark square
    }
    
    SDL_RenderFillRect(renderer, &cell_rect);
    
    // Draw grid lines
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawRect(renderer, &cell_rect);
    
    // Draw piece if present
    int idx = pos_to_index(layer, row, col);
    int16_t piece = game.board[idx];
    
    if (piece != EMPTY) {
        auto it = piece_textures.find(piece);
        if (it != piece_textures.end() && it->second) {
            SDL_Rect piece_rect = {x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8};
            SDL_RenderCopy(renderer, it->second, nullptr, &piece_rect);
        }
        
        // Draw frozen indicator
        if (game.frozen[idx]) {
            SDL_SetRenderDrawColor(renderer, 100, 100, 255, 180);
            SDL_Rect freeze_rect = {x + CELL_SIZE - 15, y + 4, 11, 11};
            SDL_RenderFillRect(renderer, &freeze_rect);
        }
    }
}

void Renderer::render_board(const Game& game, 
                           std::optional<int> selected,
                           const std::unordered_set<int>& legal_dest,
                           bool show_move_history,
                           bool show_timer) {
    // Clear with flat background color
    SDL_SetRenderDrawColor(renderer, 35, 35, 45, 255);  // Dark blue-grey
    SDL_RenderClear(renderer);
    
    // Calculate layout
    const int board_pixel_width = BOARD_COLS * CELL_SIZE;
    const int board_pixel_height = BOARD_ROWS * CELL_SIZE;
    
    const int available_width = VIRTUAL_WIDTH - SIDEBAR_WIDTH - 40;
    const int available_height = VIRTUAL_HEIGHT - TOP_PANEL_HEIGHT - BOTTOM_PANEL_HEIGHT - 40;
    
    const int horizontal_total_width = board_pixel_width * 3 + 40;
    const int vertical_stacked_height = board_pixel_height * 3 + 80;
    
    bool use_vertical_layout = (horizontal_total_width > available_width) || 
                               (vertical_stacked_height < available_height);
    
    // Extract last move for highlighting
    std::optional<int> last_from, last_to;
    if (game.last_move.has_value()) {
        auto [from, to, flag] = game.last_move.value();
        last_from = from;
        last_to = to;
    }
    
    // Draw all three boards (no labels)
    for (int layer = 0; layer < NUM_BOARDS; ++layer) {
        // Draw board cells
        for (int row = 0; row < BOARD_ROWS; ++row) {
            for (int col = 0; col < BOARD_COLS; ++col) {
                int idx = pos_to_index(layer, row, col);
                bool is_selected = selected.has_value() && selected.value() == idx;
                bool is_legal = legal_dest.count(idx) > 0;
                bool is_last_move = (last_from.has_value() && last_from.value() == idx) ||
                                   (last_to.has_value() && last_to.value() == idx);
                
                draw_cell(layer, row, col, game, is_selected, is_legal, is_last_move);
            }
        }
    }
    
    // Draw top info panel
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 220);
    SDL_Rect top_panel = {0, 0, VIRTUAL_WIDTH, TOP_PANEL_HEIGHT};
    SDL_RenderFillRect(renderer, &top_panel);
    
    // Draw clock if enabled
    if (show_timer && game.timer.enabled) {
        render_clock(game, 20, 10, VIRTUAL_WIDTH - SIDEBAR_WIDTH - 40, 60);
    } else {
        // Show turn (decorative font) and move count (pixel font)
        std::string turn_text = "Turn: " + std::string(game.current_turn == Color::GOLD ? "Gold" : "Scarlet");
        SDL_Color turn_color = game.current_turn == Color::GOLD ? 
                              SDL_Color{255, 215, 0, 255} : SDL_Color{220, 20, 60, 255};
        render_text(turn_text, 20, 22, turn_color, font_large);
        
        std::string moves_text = "Moves: " + std::to_string(game.move_notations.size());
        render_text(moves_text, 300, 28, {200, 200, 200, 255}, pixel_font);
    }
    
    // Draw move history sidebar
    if (show_move_history) {
        int sidebar_x = VIRTUAL_WIDTH - SIDEBAR_WIDTH;
        int sidebar_y = 0;
        render_move_history(game, sidebar_x, sidebar_y, SIDEBAR_WIDTH, VIRTUAL_HEIGHT - BOTTOM_PANEL_HEIGHT);
    }
    
    // Draw bottom hint panel
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 220);
    SDL_Rect bottom_panel = {0, VIRTUAL_HEIGHT - BOTTOM_PANEL_HEIGHT, VIRTUAL_WIDTH, BOTTOM_PANEL_HEIGHT};
    SDL_RenderFillRect(renderer, &bottom_panel);
    
    std::string hint = "Controls: U-Undo | R-Redo | F11-Fullscreen | ESC-Quit";
    render_text(hint, 20, VIRTUAL_HEIGHT - BOTTOM_PANEL_HEIGHT + 15, {150, 150, 150, 255}, pixel_font);  // Increased from pixel_font_small
    
    // Draw game over message
    if (game.game_over) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
        SDL_Rect overlay = {0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT};
        SDL_RenderFillRect(renderer, &overlay);
        
        std::string winner_text = "Winner: " + game.winner;
        render_outlined_text(winner_text, VIRTUAL_WIDTH / 2 - 150, VIRTUAL_HEIGHT / 2 - 50, font_large);
        
        std::string hint_text = "Press ESC to return to menu";
        render_text(hint_text, VIRTUAL_WIDTH / 2 - 180, VIRTUAL_HEIGHT / 2 + 50, {200, 200, 200, 255}, pixel_font);  // Use pixel_font for better readability
    }
}

void Renderer::render_post_game(const Game& game) {
    clear();
    
    // Draw background with board
    render_board(game);
    
    // Draw semi-transparent overlay
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
    SDL_Rect overlay = {0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT};
    SDL_RenderFillRect(renderer, &overlay);
    
    // Draw winner text
    std::string winner_text = "Winner: " + game.winner;
    render_outlined_text(winner_text, VIRTUAL_WIDTH / 2 - 150, VIRTUAL_HEIGHT / 3, font_large);
    
    // Draw move count
    std::string moves_text = "Moves: " + std::to_string(game.move_notations.size());
    render_outlined_text(moves_text, VIRTUAL_WIDTH / 2 - 100, VIRTUAL_HEIGHT / 2, font);
    
    // Draw "Back to Menu" button
    SDL_Rect button = {VIRTUAL_WIDTH / 2 - 100, VIRTUAL_HEIGHT * 3 / 4, 200, 50};
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    SDL_RenderFillRect(renderer, &button);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &button);
    
    render_outlined_text("Back to Menu", VIRTUAL_WIDTH / 2 - 80, VIRTUAL_HEIGHT * 3 / 4 + 10, font);
    
    present();
}

std::optional<Position> Renderer::screen_to_board(int x, int y) const {
    // Horizontal layout only (matches draw_cell)
    const int board_pixel_width = BOARD_COLS * CELL_SIZE;  // 12 * 42 = 504
    const int board_pixel_height = BOARD_ROWS * CELL_SIZE; // 8 * 42 = 336
    
    const int available_width = VIRTUAL_WIDTH - SIDEBAR_WIDTH - 40;  // 1920 - 260 - 40 = 1620
    const int available_height = VIRTUAL_HEIGHT - TOP_PANEL_HEIGHT - BOTTOM_PANEL_HEIGHT - 40;
    
    const int board_spacing = 15;
    const int total_boards_width = board_pixel_width * 3 + board_spacing * 2;
    
    int boards_start_x = (available_width - total_boards_width) / 2 + 20;
    int boards_start_y = TOP_PANEL_HEIGHT + (available_height - board_pixel_height) / 2 + 20;
    
    // Check each board
    for (int layer = 0; layer < NUM_BOARDS; ++layer) {
        int board_x = boards_start_x + layer * (board_pixel_width + board_spacing);
        int board_y = boards_start_y;
        
        if (x >= board_x && x < board_x + board_pixel_width &&
            y >= board_y && y < board_y + board_pixel_height) {
            
            int col = (x - board_x) / CELL_SIZE;
            int row = (y - board_y) / CELL_SIZE;
            
            if (in_bounds(layer, row, col)) {
                return Position{layer, row, col};
            }
        }
    }
    
    return std::nullopt;
}

void Renderer::render_move_history(const Game& game, int panel_x, int panel_y, int panel_w, int panel_h) {
    // Draw panel background
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 40, 40, 40, 200);
    SDL_Rect panel = {panel_x, panel_y, panel_w, panel_h};
    SDL_RenderFillRect(renderer, &panel);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    SDL_RenderDrawRect(renderer, &panel);
    
    // Draw title with decorative font
    render_text("Move History", panel_x + 10, panel_y + 5, {255, 255, 255, 255}, font);
    
    // Draw moves with pixel font (show last 20 moves, scrollable area)
    int start_y = panel_y + 40;
    int line_height = 18;
    int max_visible = (panel_h - 50) / line_height;
    
    int total_moves = game.move_notations.size();
    int start_move = std::max(0, total_moves - max_visible);
    
    for (int i = start_move; i < total_moves; ++i) {
        int y = start_y + (i - start_move) * line_height;
        
        // Alternate colors for readability
        bool is_gold = (i % 2 == 0);
        SDL_Color color = is_gold ? SDL_Color{255, 215, 0, 255} : SDL_Color{220, 20, 60, 255};
        
        std::string move_text = std::to_string(i / 2 + 1) + ". " + game.move_notations[i];
        render_text(move_text, panel_x + 10, y, color, pixel_font);
    }
    
    // Undo/Redo hints at bottom with pixel font
    if (panel_h > 100) {
        SDL_Color hint_color = {150, 150, 150, 255};
        render_text("U: Undo  R: Redo", panel_x + 10, panel_y + panel_h - 25, hint_color, pixel_font);  // Increased from pixel_font_small
    }
}

void Renderer::render_clock(const Game& game, int panel_x, int panel_y, int panel_w, int panel_h) {
    // Draw clock background
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 20, 20, 20, 220);
    SDL_Rect panel = {panel_x, panel_y, panel_w, panel_h};
    SDL_RenderFillRect(renderer, &panel);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    SDL_RenderDrawRect(renderer, &panel);
    
    // Convert milliseconds to MM:SS format
    auto format_time = [](int ms) -> std::string {
        int total_seconds = ms / 1000;
        int minutes = total_seconds / 60;
        int seconds = total_seconds % 60;
        
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << minutes << ":"
            << std::setfill('0') << std::setw(2) << seconds;
        return oss.str();
    };
    
    // Gold time (left side) with pixel font
    std::string gold_time = "Gold: " + format_time(game.timer.gold_time_ms);
    SDL_Color gold_color = (game.current_turn == Color::GOLD) ? 
                          SDL_Color{255, 255, 0, 255} : SDL_Color{200, 200, 200, 255};
    if (game.timer.gold_time_up) gold_color = {255, 0, 0, 255};
    render_text(gold_time, panel_x + 20, panel_y + 15, gold_color, pixel_font);
    
    // Scarlet time (right side) with pixel font
    std::string scarlet_time = "Scarlet: " + format_time(game.timer.scarlet_time_ms);
    SDL_Color scarlet_color = (game.current_turn == Color::SCARLET) ? 
                              SDL_Color{255, 255, 0, 255} : SDL_Color{200, 200, 200, 255};
    if (game.timer.scarlet_time_up) scarlet_color = {255, 0, 0, 255};
    render_text(scarlet_time, panel_x + panel_w - 150, panel_y + 15, scarlet_color, pixel_font);
}

} // namespace dragonchess
