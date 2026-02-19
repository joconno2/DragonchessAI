# Dragonchess Headless Mode - Research Guide

## Overview

The Dragonchess engine now supports **fully headless, multi-threaded execution** for research purposes. This allows you to run thousands of games at maximum CPU speed without any rendering overhead.

## Performance Benchmarks

- **Random vs Random**: ~20,000+ games/second
- **Greedy vs Greedy**: ~13,600 games/second  
- **Minimax vs AlphaBeta**: ~1-2 games/second (depth dependent)

## Quick Start

### Single Match
```bash
./build/dragonchess --headless --gold-ai greedy --scarlet-ai random
```

### Tournament (100 games)
```bash
./build/dragonchess --headless --mode tournament --games 100 \
  --gold-ai greedy --scarlet-ai greedy \
  --output-csv results.csv
```

### High-Performance Research Run
```bash
# 1000 games, 16 threads, CSV output
./build/dragonchess --headless --mode tournament --games 1000 --threads 16 \
  --gold-ai random --scarlet-ai random \
  --output-csv random_baseline.csv --verbose
```

### AI Comparison Study
```bash
# Compare minimax depths
./build/dragonchess --headless --mode tournament --games 50 \
  --gold-ai minimax --gold-depth 3 \
  --scarlet-ai minimax --scarlet-depth 2 \
  --output-json minimax_depth_study.json \
  --verbose
```

## Available AI Types

| AI Type | Description | Speed | Strength |
|---------|-------------|-------|----------|
| `random` | Random legal moves | Fastest | Weakest |
| `greedy` | Capture-focused | Fast | Weak |
| `greedyvalue` | Value-based captures | Fast | Medium |
| `minimax` | Minimax search | Slow | Strong |
| `alphabeta` | Alpha-beta pruning | Medium | Strongest |

## Command-Line Options

### Core Options
- `--headless` - Enable headless mode (required)
- `--mode <match|tournament>` - Single match or tournament
- `--games <N>` - Number of games (tournament mode)
- `--threads <N>` - Thread count (0 = auto-detect)
- `--max-moves <N>` - Maximum moves per game

### AI Configuration
- `--gold-ai <type>` - Gold player AI type
- `--scarlet-ai <type>` - Scarlet player AI type
- `--gold-depth <N>` - Search depth for minimax/alphabeta
- `--scarlet-depth <N>` - Search depth for minimax/alphabeta
- `--gold-name <name>` - Custom name for reports
- `--scarlet-name <name>` - Custom name for reports

### Output Options
- `--output-csv <file>` - Export results to CSV
- `--output-json <file>` - Export results to JSON
- `--verbose` - Detailed progress updates
- `--quiet` - Minimal output

## Output Formats

### CSV Format
```csv
game_id,gold_ai,scarlet_ai,winner,moves,duration_ms,gold_pieces,scarlet_pieces,checkmate
0,greedy,random,Gold,167,0.510,39,5,yes
```

Fields:
- `game_id`: Unique game identifier
- `gold_ai`, `scarlet_ai`: AI types
- `winner`: "Gold", "Scarlet", or "Draw"
- `moves`: Total moves in game
- `duration_ms`: Game duration in milliseconds
- `gold_pieces`, `scarlet_pieces`: Remaining pieces
- `checkmate`: "yes" or "no"

### JSON Format
```json
{
  "summary": {
    "total_games": 100,
    "gold_wins": 53,
    "scarlet_wins": 44,
    "draws": 3,
    "avg_game_length": 145.1,
    "total_time_ms": 7.31,
    "avg_time_per_game_ms": 0.07
  },
  "matches": [
    {
      "game_id": 0,
      "gold_ai": "greedy",
      "scarlet_ai": "greedy",
      "winner": "Gold",
      "moves": 53,
      "duration_ms": 0.299,
      "gold_pieces_remaining": 29,
      "scarlet_pieces_remaining": 21,
      "checkmate": true
    }
  ]
}
```

## Research Examples

### Baseline Performance Study
```bash
# Test all AI types against random
for ai in greedy greedyvalue minimax alphabeta; do
  ./build/dragonchess --headless --mode tournament --games 100 \
    --gold-ai $ai --scarlet-ai random \
    --output-csv "${ai}_vs_random.csv"
done
```

### Depth Analysis
```bash
# Compare different search depths
for depth in 1 2 3 4; do
  ./build/dragonchess --headless --mode tournament --games 20 \
    --gold-ai alphabeta --gold-depth $depth \
    --scarlet-ai alphabeta --scarlet-depth 2 \
    --output-json "alphabeta_depth${depth}_vs_depth2.json"
done
```

### Large-Scale Tournament
```bash
# 10,000 games for statistical significance
./build/dragonchess --headless --mode tournament --games 10000 --threads 16 \
  --gold-ai greedyvalue --scarlet-ai greedy \
  --output-csv large_tournament.csv \
  --output-json large_tournament.json \
  --verbose
```

## Performance Tips

1. **Use maximum threads**: `--threads $(nproc)` for parallel execution
2. **Simple AIs for speed**: Use `random` or `greedy` for fast iterations
3. **Batch processing**: Run multiple tournaments in parallel using shell scripts
4. **CSV for analysis**: Easier to import into R, Python pandas, Excel
5. **JSON for archiving**: Better for hierarchical data and metadata

## Integration with Analysis Tools

### Python (pandas)
```python
import pandas as pd

# Load results
df = pd.read_csv('results.csv')

# Calculate win rates
win_rate = df[df['winner'] == 'Gold'].shape[0] / len(df)
print(f"Gold win rate: {win_rate:.1%}")

# Average game length by winner
print(df.groupby('winner')['moves'].mean())
```

### R
```r
# Load results
data <- read.csv('results.csv')

# Statistical analysis
summary(data$moves)
table(data$winner)

# Visualization
hist(data$moves, main="Game Length Distribution")
```

## Notes

- Games run at **full CPU speed** with no artificial delays
- **Parallel execution** uses all available CPU cores
- **Deterministic results** may vary due to move ordering in random AIs
- **Memory efficient**: Each game is independent
- **Thread-safe**: Proper synchronization for parallel tournaments

## Example Output

```
=== Tournament Results ===
Total games:     100
Gold wins:       53 (53.0%)
Scarlet wins:    44 (44.0%)
Draws:           3 (3.0%)
Avg game length: 145.1 moves
Total time:      7.31ms
Time per game:   0.07ms
Games per second:13670.67
```

For complete options, run: `./build/dragonchess --help`
