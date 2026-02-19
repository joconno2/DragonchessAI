#pragma once

#include "renderer.h"
#include "input.h"

namespace dragonchess {

// Global fullscreen state
inline bool g_is_fullscreen = false;

// Handle common input events (fullscreen toggle)
inline bool handle_common_input(InputEvent event, Renderer& renderer) {
    if (event == InputEvent::FULLSCREEN_TOGGLE) {
        g_is_fullscreen = !g_is_fullscreen;
        renderer.set_fullscreen(g_is_fullscreen);
        return true;
    }
    return false;
}

} // namespace dragonchess
