#pragma once

#include "simple_ai.h"
#include <string>
#include <memory>
#include <dlfcn.h>

namespace dragonchess {

/**
 * Plugin Loader for External AI Bots
 * 
 * Allows loading AI implementations from shared libraries (.so files)
 * 
 * To create a plugin:
 * 1. Implement a class inheriting from SimpleAI
 * 2. Add extern "C" factory function:
 *    extern "C" SimpleAI* create_ai(Game& game, Color color) {
 *        return new YourBot(game, color);
 *    }
 * 3. Compile as shared library:
 *    g++ -std=c++17 -fPIC -shared -I../src yourbot.cpp -o yourbot.so
 */

class AIPlugin {
public:
    AIPlugin() : handle(nullptr), ai(nullptr) {}
    ~AIPlugin() { unload(); }
    
    // Load AI from shared library
    bool load(const std::string& library_path, Game& game, Color color);
    
    // Unload plugin
    void unload();
    
    // Get AI instance
    SimpleAI* get_ai() { return ai; }
    
    // Check if loaded
    bool is_loaded() const { return ai != nullptr; }
    
private:
    void* handle;
    SimpleAI* ai;
};

// Factory function type
typedef SimpleAI* (*create_ai_func)(Game&, Color);

} // namespace dragonchess
