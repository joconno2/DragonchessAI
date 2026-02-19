#include "../src/simple_ai.h"
#include <random>
#include <ctime>

using namespace dragonchess;

/**
 * RandomBot - Picks a random legal move
 * 
 * This is the simplest possible AI - it just randomly
 * selects from all available legal moves.
 */
class RandomBot : public SimpleAI {
private:
    std::mt19937 rng;

public:
    RandomBot(Game& g, Color c) : SimpleAI(g, c) {
        rng.seed(std::time(nullptr));
    }
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        
        if (moves.empty()) {
            return std::nullopt;
        }
        
        // Pick a random move
        std::uniform_int_distribution<int> dist(0, moves.size() - 1);
        return moves[dist(rng)];
    }
};

// Factory function - required for plugin loading
extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new RandomBot(game, color);
}
