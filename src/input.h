#pragma once

#include <SDL2/SDL.h>
#include <functional>

namespace dragonchess {

enum class InputEvent {
    QUIT,
    MOUSE_CLICK,
    FULLSCREEN_TOGGLE,
    SPACE_PRESSED,
    UNDO_PRESSED,      // Ctrl+Z or U
    REDO_PRESSED,      // Ctrl+Y or R
    FLIP_BOARD,        // F key
    ESC_PRESSED,       // ESC key to exit/cancel
    NONE
};

class InputHandler {
public:
    InputHandler() = default;
    
    // Poll for input events
    InputEvent poll(int& mouse_x, int& mouse_y);
    
private:
    SDL_Event event;
};

} // namespace dragonchess
