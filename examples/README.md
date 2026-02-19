# Dragonchess Example Bots

This directory contains 5 example AI bots demonstrating increasing levels of sophistication, plus all the tools needed to compile, test, and run tournaments.

## Quick Start

```bash
# Compile a bot
make material_bot.so

# Test it (10 games)
make test BOT=material_bot.so

# Run tournament (100 games)
make tournament BOT=material_bot.so

# Compare against all AIs
make evaluate BOT=material_bot.so
```

## Example Bots

### 1. random_bot.cpp (Baseline)
**Win Rate**: 4.2%  
**Strategy**: Picks random legal moves  
**Use Case**: Baseline for comparison

### 2. material_bot.cpp 🥈
**Win Rate**: 41.3%  
**Strategy**: Maximizes material advantage and center control  
**Highlights**:
- Captures high-value pieces
- Controls center squares
- 93-1 record vs Random AI
- 55-16 record vs Greedy AI

**Code Complexity**: ~70 lines  
**Best For**: Learning material evaluation

### 3. tactical_bot.cpp 🥉
**Win Rate**: 40.5%  
**Strategy**: Tactical opportunities and key square control  
**Highlights**:
- Prioritizes checks and threats
- Bonus for 3D movement
- 86-2 record vs Random AI
- **57-25 vs Greedy AI** (best performance!)

**Code Complexity**: ~100 lines  
**Best For**: Learning tactical evaluation

### 4. positional_bot.cpp
**Win Rate**: 37.8%  
**Strategy**: Balanced positional play with piece coordination  
**Highlights**:
- Multi-factor evaluation function
- Piece development principles
- King safety considerations
- 82-1 record vs Random AI
- 45-42 vs Greedy AI (nearly 50-50!)

**Code Complexity**: ~180 lines  
**Best For**: Learning complex evaluation

### 5. aggressive_bot.cpp
**Win Rate**: 36.7%  
**Strategy**: Maximum aggression, willing to sacrifice  
**Highlights**:
- **92-0 record vs Random AI (undefeated!)**
- Always attacks when possible
- Ignores defense for offense
- 28-41 vs Greedy AI (struggles vs defense)

**Code Complexity**: ~150 lines  
**Best For**: Learning aggressive play styles

## Files

### Source Code
- `random_bot.cpp` - Simple random baseline
- `material_bot.cpp` - Material-focused strategy
- `tactical_bot.cpp` - Tactical opportunities
- `positional_bot.cpp` - Positional evaluation
- `aggressive_bot.cpp` - Aggressive play
- `student_bot.cpp` - Template for students
- `example_bots.cpp` - All 6 examples in one file

### Documentation
- `QUICK_START.md` - Get started in 5 minutes
- `STUDENT_GUIDE.md` - Complete tutorial
- `TOURNAMENT_REPORT.md` - Full tournament analysis

### Tools
- `Makefile` - Compilation and testing
- `run_tournament.sh` - Round-robin tournament script

## Tournament Results

A complete round-robin tournament was run with all bots playing 100 games against each other (2,000 total games):

| Rank | Bot | Win % | Record |
|------|-----|-------|---------|
| 🥇 | Greedy (Built-in) | 44.6% | 223-185-92 |
| 🥈 | **Material Bot** | 41.3% | 248-17-335 |
| 🥉 | **Tactical Bot** | 40.5% | 243-27-330 |
| 4 | **Positional Bot** | 37.8% | 227-43-330 |
| 5 | **Aggressive Bot** | 36.7% | 220-41-339 |

Full results available in `tournament_results/` directory.

## Compilation

All bots compile with a single command:

```bash
g++ -std=c++17 -fPIC -O3 -I../src -shared \
    ../src/simple_ai.cpp ../src/bitboard.cpp ../src/moves.cpp ../src/game.cpp \
    your_bot.cpp -o your_bot.so
```

Or use the Makefile:
```bash
make your_bot.so
```

## Testing

### Quick Test (10 games)
```bash
../build/dragonchess --headless --mode tournament --games 10 \
    --gold-ai-plugin your_bot.so --scarlet-ai random
```

### Full Tournament (100 games)
```bash
../build/dragonchess --headless --mode tournament --games 100 \
    --gold-ai-plugin your_bot.so --scarlet-ai greedy \
    --output-csv results.csv
```

### Round-Robin Tournament
```bash
./run_tournament.sh
```

This runs all bots against each other (100 games per matchup) and generates:
- Individual CSV files for each matchup
- Overall standings
- Win/loss/draw statistics
- Performance analysis

## Learning Path

1. **Start Simple**: Study `random_bot.cpp` (20 lines)
2. **Add Strategy**: Study `material_bot.cpp` (captures + center)
3. **Go Deeper**: Study `tactical_bot.cpp` (tactical awareness)
4. **Get Advanced**: Study `positional_bot.cpp` (multi-factor evaluation)
5. **Be Creative**: Study `aggressive_bot.cpp` (aggressive sacrifices)

## Performance Metrics

- **Random vs Random**: ~20,000 games/second
- **Plugin vs Built-in**: ~1,000 games/second  
- **Tournament (2,000 games)**: ~3 minutes total

## Data Analysis

Each CSV export includes:
- Game ID
- AI player names
- Winner
- Move count
- Duration (ms)
- Final piece counts
- Checkmate status

Perfect for:
- Statistical analysis
- Strategy research
- Machine learning training
- Performance profiling

## Next Steps

To beat Greedy AI consistently (>75% win rate):

1. **Implement lookahead** - Add 2-ply minimax search
2. **Better evaluation** - Add king safety, mobility, threats
3. **Move ordering** - Try captures and checks first
4. **Endgame strategy** - Recognize won/lost positions
5. **Opening principles** - Use development heuristics

## Support

- Read `QUICK_START.md` for immediate help
- Check `STUDENT_GUIDE.md` for detailed tutorials
- Review `PLUGIN_SYSTEM.md` for technical details
- Examine working bots for examples

## License

Part of the Dragonchess AI project. Use these bots as references for learning and teaching AI concepts.

Happy coding! 🐉♟️
