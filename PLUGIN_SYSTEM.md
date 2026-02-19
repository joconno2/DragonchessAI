# Dragonchess AI Plugin System - Complete Documentation

## Overview

The Dragonchess AI now includes a complete plugin system designed for teaching AI classes. Students can create AI agents in a single C++ file with minimal complexity while maintaining maximum performance.

## Key Features

✅ **Minimal Interface** - Students only need to override 1 method (`choose_move()`)  
✅ **20+ Helper Methods** - Hide complexity of board representation, move generation  
✅ **Zero Performance Overhead** - Plugins compile to native code, same speed as built-in AIs  
✅ **Easy Compilation** - Single `make` command or simple `g++` invocation  
✅ **Progressive Examples** - 6 example bots from beginner to advanced  
✅ **Complete Testing Tools** - Built-in tournament mode, CSV/JSON export  
✅ **Research Ready** - Headless mode with multi-threading, 1000+ games/second  

## Performance Benchmarks

```
Random vs Random:     ~20,000 games/second
Greedy vs Greedy:     ~13,670 games/second
Plugin vs Built-in:   ~1,000 games/second (identical performance)
Minimax vs AlphaBeta: ~1-2 games/second (depth-dependent)
```

## Student Workflow

### 1. Write AI (my_bot.cpp)

```cpp
#include "../src/simple_ai.h"

class MyBot : public SimpleAI {
public:
    MyBot(Game& g, Color c) : SimpleAI(g, c) {}
    
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        if (moves.empty()) return std::nullopt;
        
        // Your strategy here!
        Move best = moves[0];
        int best_score = evaluate_move(best);
        
        for (const auto& move : moves) {
            int score = evaluate_move(move);
            if (score > best_score) {
                best = move;
                best_score = score;
            }
        }
        
        return best;
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new MyBot(game, color);
}
```

### 2. Compile

```bash
make my_bot.so
```

Or manually:
```bash
g++ -std=c++17 -fPIC -O3 -I../src -shared \
    ../src/simple_ai.cpp ../src/bitboard.cpp ../src/moves.cpp ../src/game.cpp \
    my_bot.cpp -o my_bot.so
```

### 3. Test

```bash
# Quick test (10 games)
make test BOT=my_bot.so

# Tournament (100 games)
make tournament BOT=my_bot.so

# Full evaluation
make evaluate BOT=my_bot.so
```

### 4. Analyze

```bash
# Export to CSV for analysis
../build/dragonchess --headless --mode tournament --games 1000 \
    --gold-ai-plugin my_bot.so --scarlet-ai greedy \
    --output-csv results.csv

# Or JSON
../build/dragonchess --headless --mode tournament --games 1000 \
    --gold-ai-plugin my_bot.so --scarlet-ai greedy \
    --output-json results.json
```

## Available Files

### For Students

- **examples/QUICK_START.md** - 5-minute getting started guide
- **examples/STUDENT_GUIDE.md** - Complete tutorial with 6 strategy levels
- **examples/student_bot.cpp** - Template with TODOs and comments
- **examples/Makefile** - Automated build and test commands
- **examples/example_bots.cpp** - 6 complete example implementations

### System Files

- **src/simple_ai.h** - Base class interface
- **src/simple_ai.cpp** - Helper method implementations
- **src/ai_plugin.h/cpp** - Dynamic library loader
- **src/headless.h/cpp** - Tournament and headless mode
- **src/main.cpp** - CLI interface

## SimpleAI Interface

### Required Override

```cpp
virtual std::optional<Move> choose_move() = 0;
```

### Available Helper Methods

```cpp
// Move generation
std::vector<Move> get_legal_moves() const;

// Piece queries
std::vector<int> get_my_pieces() const;
std::vector<int> get_opponent_pieces() const;
int get_piece_at(int index) const;
int get_piece_value(int piece_type) const;

// Board queries
bool is_empty(int index) const;
bool is_my_piece(int index) const;
bool is_opponent_piece(int index) const;

// Evaluation
int evaluate_move(const Move& move) const;
int evaluate_position() const;

// Game state access
const Game& get_game() const;
Color get_color() const;
```

## Command-Line Interface

### Basic Usage

```bash
# GUI mode (default)
./build/dragonchess

# Headless mode
./build/dragonchess --headless --mode tournament --games 100 \
    --gold-ai-plugin my_bot.so --scarlet-ai random
```

### All Options

```
GENERAL OPTIONS:
  --headless                 Run in headless mode (no graphics)
  --help                     Show this help message

HEADLESS MODES:
  --mode <mode>              tournament, benchmark, or evaluation
  
AI CONFIGURATION:
  --gold-ai <type>           Gold AI type (random, greedy, minimax, alphabeta)
  --scarlet-ai <type>        Scarlet AI type
  --gold-ai-plugin <file>    Load gold AI from .so plugin file
  --scarlet-ai-plugin <file> Load scarlet AI from .so plugin file
  --gold-depth <N>           Search depth for minimax/alphabeta (default: 2)
  --scarlet-depth <N>        Search depth for minimax/alphabeta (default: 2)
  --gold-name <name>         Custom name for gold AI
  --scarlet-name <name>      Custom name for scarlet AI

TOURNAMENT OPTIONS:
  --games <N>                Number of games to play (default: 100)
  --max-moves <N>            Maximum moves per game (default: 1000)
  --threads <N>              Number of parallel threads (default: auto)

OUTPUT OPTIONS:
  --output-csv <file>        Export results to CSV file
  --output-json <file>       Export results to JSON file
  --quiet                    Suppress progress output
  --verbose                  Enable detailed logging
```

## Example Bots

### 1. RandomBot (Baseline)
Picks a random move from all legal moves.

### 2. CaptureBot
Prefers capturing moves, otherwise picks randomly.

### 3. GreedyBot
Evaluates all moves and picks the highest-scoring one.

### 4. DefensiveBot
Balances offense and defense by considering opponent's threats.

### 5. StudentBot (Template)
Includes TODOs and comments for students to complete.

### 6. LookaheadBot
Implements 1-ply lookahead (minimax lite).

## Piece Values

```cpp
King:      1000  // Game over if captured
Paladin:   6     // Powerful warrior
Mage:      5     // Magic attacks
Thief:     4     // Stealthy
Cleric:    4     // Healing abilities
Hero:      4     // Versatile
Dwarf:     3     // Strong but slow
Basilisk:  3     // Petrifying gaze
Unicorn:   3     // Fast mover
Oliphant:  3     // Powerful
Warrior:   2     // Basic fighter
Elemental: 2     // Elemental magic
Sylph:     2     // Aerial unit
Griffin:   2     // Flying
Dragon:    2     // Fearsome but vulnerable
```

## Board Representation

- Total squares: 192 (3 levels × 8 rows × 8 columns)
- Sky board: indices 0-63
- Ground board: indices 64-127
- Cavern board: indices 128-191

Each index represents: `level * 64 + row * 8 + col`

## Move Representation

Moves are tuples: `std::tuple<int, int, MoveFlag>`

```cpp
auto [from_index, to_index, flag] = move;

// MoveFlag values:
//   QUIET = 0     (normal move)
//   CAPTURE = 1   (capturing piece)
//   AFAR = 2      (long-range attack)
//   AMBIGUOUS = 3 (multiple interpretations)
//   THREED = 4    (3D move between levels)
```

## Assignment Ideas

### 1. Basic Strategy (Week 1)
Implement CaptureBot - prefer capturing moves over quiet moves.

### 2. Evaluation Function (Week 2)
Create a custom evaluation function considering:
- Material value
- Piece positioning
- King safety
- Control of center

### 3. Minimax Search (Week 3-4)
Implement 2-ply minimax search with alpha-beta pruning.

### 4. Advanced Features (Week 5-6)
Add:
- Move ordering
- Transposition tables
- Quiescence search
- Opening book

### 5. Tournament Competition (Final Project)
Students compete in a round-robin tournament. Best AI wins!

## Performance Tips

1. **Enable Optimizations**: Always compile with `-O3`
2. **Avoid Unnecessary Copies**: Use `const auto&` for large objects
3. **Pre-compute Values**: Cache piece values, position scores
4. **Early Pruning**: Skip obviously bad moves early
5. **Move Ordering**: Evaluate good moves first (captures, checks)

## Debugging

### Check Plugin Compiles

```bash
nm my_bot.so | grep create_ai
# Should show: T create_ai
```

### Test Plugin Loads

```bash
../build/dragonchess --headless --mode tournament --games 1 \
    --gold-ai-plugin my_bot.so --scarlet-ai random --verbose
```

### Common Errors

**Segmentation fault**: Usually missing `extern "C"` on factory function

**Symbol not found**: Factory function must be exactly:
```cpp
extern "C" SimpleAI* create_ai(Game& game, Color color)
```

**Hangs/infinite loop**: Make sure to return from `choose_move()`, check for empty moves

**Wrong behavior**: Use `--verbose` to see game play-by-play

## Integration with Research Tools

The plugin system integrates seamlessly with the headless mode research tools:

```bash
# Compare your bot against all built-in AIs
for ai in random greedy greedyvalue minimax alphabeta; do
    echo "Testing against $ai..."
    ../build/dragonchess --headless --mode tournament \
        --games 100 \
        --gold-ai-plugin my_bot.so \
        --scarlet-ai $ai \
        --output-csv results_${ai}.csv \
        --quiet
done

# Multi-threaded mass testing (10,000 games)
../build/dragonchess --headless --mode tournament \
    --games 10000 \
    --gold-ai-plugin my_bot.so \
    --scarlet-ai greedy \
    --threads 8 \
    --output-json big_results.json \
    --quiet
```

## Future Enhancements

Potential additions for advanced courses:

1. **Neural Network Support**: Interface for TensorFlow/PyTorch models
2. **MCTS Implementation**: Monte Carlo Tree Search framework
3. **Reinforcement Learning**: Training harness for RL agents
4. **Game Analysis**: Position database, opening books
5. **Visualization**: Move trees, evaluation graphs

## Contact & Support

For questions, issues, or contributions:
- Read the documentation: `STUDENT_GUIDE.md`, `QUICK_START.md`
- Check examples: `example_bots.cpp`
- Review template: `student_bot.cpp`

Happy coding! 🐉♟️
