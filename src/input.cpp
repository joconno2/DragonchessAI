#include "input.h"

namespace dragonchess {

InputEvent InputHandler::poll(int& mouse_x, int& mouse_y) {
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                return InputEvent::QUIT;
                
            case SDL_MOUSEBUTTONUP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouse_x = event.button.x;
                    mouse_y = event.button.y;
                    return InputEvent::MOUSE_CLICK;
                }
                break;
                
            case SDL_KEYDOWN:
                // F11 for fullscreen toggle
                if (event.key.keysym.sym == SDLK_F11) {
                    return InputEvent::FULLSCREEN_TOGGLE;
                }
                // Space bar
                if (event.key.keysym.sym == SDLK_SPACE) {
                    return InputEvent::SPACE_PRESSED;
                }
                // Undo: Ctrl+Z or U
                if (event.key.keysym.sym == SDLK_z && (event.key.keysym.mod & KMOD_CTRL)) {
                    return InputEvent::UNDO_PRESSED;
                }
                if (event.key.keysym.sym == SDLK_u) {
                    return InputEvent::UNDO_PRESSED;
                }
                // Redo: Ctrl+Y or R
                if (event.key.keysym.sym == SDLK_y && (event.key.keysym.mod & KMOD_CTRL)) {
                    return InputEvent::REDO_PRESSED;
                }
                if (event.key.keysym.sym == SDLK_r) {
                    return InputEvent::REDO_PRESSED;
                }
                // Flip board: F
                if (event.key.keysym.sym == SDLK_f) {
                    return InputEvent::FLIP_BOARD;
                }
                // ESC to exit/cancel
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    return InputEvent::ESC_PRESSED;
                }
                break;
        }
    }
    
    return InputEvent::NONE;
}

} // namespace dragonchess
