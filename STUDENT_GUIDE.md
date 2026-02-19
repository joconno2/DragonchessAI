# Dragonchess AI - Student Guide

## Quick Start for AI Development

Welcome! This guide will help you create your own Dragonchess AI bot.

## Method 1: Simple Single-File Bot (Recommended for Beginners)

### Step 1: Create Your Bot

Create a file `my_bot.cpp`:

```cpp
#include "simple_ai.h"
#include <random>

namespace dragonchess {

class MyBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;  // Inherit constructor
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // TODO: Add your strategy here
        // For now, pick first move
        return moves[0];
    }
};

// Factory function for plugin system
extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new MyBot(game, color);
}

} // namespace dragonchess
```

### Step 2: Compile Your Bot

```bash
cd DragonchessAI
g++ -std=c++17 -fPIC -shared -Isrc \
    src/simple_ai.cpp \
    my_bot.cpp \
    -o my_bot.so
```

### Step 3: Test Your Bot

```bash
# In GUI mode (if implemented)
./build/dragonchess --gold-ai-plugin my_bot.so

# In headless mode
./build/dragonchess --headless \
    --gold-ai-plugin my_bot.so \
    --scarlet-ai random \
    --mode tournament --games 100 \
    --output-csv my_bot_results.csv
```

## Available Helper Methods

Your bot has access to these useful methods:

### Getting Information
- `get_legal_moves()` - Returns all legal moves you can make
- `get_my_pieces()` - Returns positions of your pieces
- `get_enemy_pieces()` - Returns positions of enemy pieces
- `is_occupied(index)` - Check if a square has a piece
- `get_piece(index)` - Get the piece at a position

### Evaluating Moves
- `is_capture(move)` - Check if move captures an enemy piece
- `evaluate_move(move)` - Get basic score for a move (higher = better)
- `get_piece_value(piece)` - Get strategic value of a piece
- `evaluate_position()` - Get overall board evaluation

### Game State
- `game` - Reference to current game state
- `color` - Your color (Color::GOLD or Color::SCARLET)

## Example Strategies

### Level 1: Random Bot
```cpp
std::optional<Move> choose_move() override {
    auto moves = get_legal_moves();
    if (moves.empty()) return std::nullopt;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, moves.size() - 1);
    
    return moves[dis(gen)];
}
```

### Level 2: Capture Bot
```cpp
std::optional<Move> choose_move() override {
    auto moves = get_legal_moves();
    if (moves.empty()) return std::nullopt;
    
    // Prefer capturing moves
    for (const auto& move : moves) {
        if (is_capture(move)) {
            return move;
        }
    }
    
    return moves[0];
}
```

### Level 3: Greedy Bot
```cpp
std::optional<Move> choose_move() override {
    auto moves = get_legal_moves();
    if (moves.empty()) return std::nullopt;
    
    Move best_move = moves[0];
    int best_score = evaluate_move(best_move);
    
    for (const auto& move : moves) {
        int score = evaluate_move(move);
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }
    
    return best_move;
}
```

### Level 4: Value-Based Bot
```cpp
std::optional<Move> choose_move() override {
    auto moves = get_legal_moves();
    if (moves.empty()) return std::nullopt;
    
    Move best_move = moves[0];
    int best_score = -999999;
    
    for (const auto& move : moves) {
        auto [from, to, flag] = move;
        int score = 0;
        
        // Capture high-value pieces
        if (is_capture(move)) {
            score += get_piece_value(game.board[to]) * 10;
        }
        
        // Protect valuable pieces
        if (get_piece_value(game.board[from]) > 5) {
            score -= 5;
        }
        
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }
    
    return best_move;
}
```

## Piece Values

| Piece | Value | Description |
|-------|-------|-------------|
| King | 1000 | Must protect! |
| Paladin | 6 | Powerful ground unit |
| Dragon | 6 | Powerful sky unit |
| Oliphant | 5 | Strong ground piece |
| Hero | 5 | Versatile ground piece |
| Basilisk | 5 | Freezing ability |
| Elemental | 5 | Underground power |
| Griffin | 4 | Flying attacker |
| Unicorn | 4 | Mobile ground piece |
| Cleric | 4 | Support piece |
| Mage | 4 | Magic abilities |
| Sylph | 3 | Light flyer |
| Thief | 3 | Sneaky ground piece |
| Warrior | 2 | Basic ground unit |
| Dwarf | 2 | Underground fighter |

## Board Structure

The game has 3 boards stacked vertically:
- **Layer 0** (Sky): Dragons, Griffins, Sylphs fly here
- **Layer 1** (Ground): Most pieces, main battlefield
- **Layer 2** (Underworld): Dwarves, Elementals, Basilisks

Each layer is 8 rows × 12 columns.

## Tips for Success

1. **Start Simple**: Begin with random or greedy strategies
2. **Test Frequently**: Use headless mode to run 100s of games quickly
3. **Analyze Results**: Use CSV output to see where your bot struggles
4. **Iterate**: Small improvements lead to big wins
5. **Protect King**: Losing your king = instant loss
6. **Capture Value**: High-value captures are usually good
7. **Think Ahead**: Can you simulate one move ahead?

## Assignment Ideas

### Assignment 1: Beat RandomBot
Create a bot that wins >60% against random play

### Assignment 2: Specialized Strategy
Implement one of these strategies:
- Aggressive: Maximum captures
- Defensive: Protect valuable pieces
- Positional: Control center squares
- Material: Maximize piece advantage

### Assignment 3: Adaptive Bot
Create a bot that changes strategy based on game state:
- Aggressive when ahead
- Defensive when behind
- Different strategies for different game phases

### Assignment 4: Learning Bot
Track which moves work well and prefer them in future games

### Assignment 5: Tournament Champion
Create the best bot in the class!

## Testing Your Bot

### Quick Test (10 games)
```bash
./build/dragonchess --headless --mode tournament --games 10 \
    --gold-ai-plugin my_bot.so --scarlet-ai random \
    --verbose
```

### Full Evaluation (100 games)
```bash
./build/dragonchess --headless --mode tournament --games 100 \
    --gold-ai-plugin my_bot.so --scarlet-ai greedy \
    --output-csv evaluation.csv --output-json evaluation.json
```

### Tournament vs All Built-in AIs
```bash
for ai in random greedy greedyvalue; do
    ./build/dragonchess --headless --mode tournament --games 50 \
        --gold-ai-plugin my_bot.so --scarlet-ai $ai \
        --output-csv "my_bot_vs_${ai}.csv"
done
```

## Debugging Tips

1. **Print statements**: Use `std::cout` to debug
2. **Count moves**: Track how many times choose_move() is called
3. **Visualize**: Run in GUI mode to see your bot play
4. **Check legality**: Make sure you return legal moves!
5. **Handle edge cases**: What if no moves available?

## Common Mistakes

❌ Returning an illegal move → Game will crash  
✅ Only return moves from `get_legal_moves()`

❌ Infinite loops in choose_move()  
✅ Keep it fast (< 1ms per move ideal)

❌ Not checking for empty move list  
✅ Always check `if (moves.empty()) return std::nullopt;`

❌ Forgetting to compile  
✅ Recompile after every change!

## Advanced: Look-Ahead

To think ahead, copy the game state and simulate:

```cpp
int evaluate_move_with_lookahead(const Move& move) {
    Game temp_game = game;  // Copy game
    temp_game.make_move(move);
    temp_game.update();
    
    // Now evaluate the resulting position
    // ... your evaluation code ...
    
    return score;
}
```

## Need Help?

1. Check `example_bots.cpp` for working examples
2. Read `simple_ai.h` for available methods
3. Run example bots to see them in action
4. Ask your instructor or TA

Good luck and have fun! 🎮
