#include "ai_plugin.h"
#include <iostream>

namespace dragonchess {

bool AIPlugin::load(const std::string& library_path, Game& game, Color color) {
    // Clean up any existing plugin
    unload();
    
    // Load shared library
    handle = dlopen(library_path.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load plugin: " << dlerror() << std::endl;
        return false;
    }
    
    // Clear any existing errors
    dlerror();
    
    // Get factory function
    create_ai_func create_ai = (create_ai_func)dlsym(handle, "create_ai");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Failed to load symbol create_ai: " << dlsym_error << std::endl;
        dlclose(handle);
        handle = nullptr;
        return false;
    }
    
    // Create AI instance
    ai = create_ai(game, color);
    if (!ai) {
        std::cerr << "Factory function returned nullptr" << std::endl;
        dlclose(handle);
        handle = nullptr;
        return false;
    }
    
    return true;
}

void AIPlugin::unload() {
    if (ai) {
        delete ai;
        ai = nullptr;
    }
    
    if (handle) {
        dlclose(handle);
        handle = nullptr;
    }
}

} // namespace dragonchess
