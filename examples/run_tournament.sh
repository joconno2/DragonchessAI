#!/bin/bash

# Round-Robin Tournament Script
# Each bot plays 100 games against every other bot

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRAGONCHESS="$SCRIPT_DIR/../build/dragonchess"
GAMES=100
OUTPUT_DIR="$SCRIPT_DIR/tournament_results"

# List of bots (use absolute paths)
BOTS=(
    "$SCRIPT_DIR/random_bot.so:Random"
    "$SCRIPT_DIR/material_bot.so:Material"
    "$SCRIPT_DIR/tactical_bot.so:Tactical"
    "$SCRIPT_DIR/positional_bot.so:Positional"
    "$SCRIPT_DIR/aggressive_bot.so:Aggressive"
)

# Also include built-in AIs
BUILTIN_AIS=(
    "random:Random(Built-in)"
    "greedy:Greedy(Built-in)"
)

echo "=========================================="
echo "  Dragonchess Round-Robin Tournament"
echo "=========================================="
echo "Games per matchup: $GAMES"
echo "Bots competing: ${#BOTS[@]} plugins + ${#BUILTIN_AIS[@]} built-in"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Results summary file
SUMMARY="$OUTPUT_DIR/tournament_summary.txt"
echo "Round-Robin Tournament Results" > "$SUMMARY"
echo "===============================" >> "$SUMMARY"
echo "Games per matchup: $GAMES" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# Track overall statistics
declare -A wins
declare -A losses
declare -A draws
declare -A total_games

# Initialize stats
for bot_entry in "${BOTS[@]}"; do
    name="${bot_entry##*:}"
    wins[$name]=0
    losses[$name]=0
    draws[$name]=0
    total_games[$name]=0
done

for ai_entry in "${BUILTIN_AIS[@]}"; do
    name="${ai_entry##*:}"
    wins[$name]=0
    losses[$name]=0
    draws[$name]=0
    total_games[$name]=0
done

# Function to run a matchup
run_matchup() {
    local gold_type=$1
    local gold_name=$2
    local scarlet_type=$3
    local scarlet_name=$4
    local matchup_name="${gold_name}_vs_${scarlet_name}"
    local csv_file="$OUTPUT_DIR/${matchup_name}.csv"
    
    echo "Running: $gold_name (Gold) vs $scarlet_name (Scarlet)"
    
    # Build command based on whether it's a plugin or built-in
    local cmd="$DRAGONCHESS --headless --mode tournament --games $GAMES"
    
    if [[ $gold_type == *.so ]]; then
        cmd="$cmd --gold-ai-plugin $gold_type"
    else
        cmd="$cmd --gold-ai $gold_type"
    fi
    
    if [[ $scarlet_type == *.so ]]; then
        cmd="$cmd --scarlet-ai-plugin $scarlet_type"
    else
        cmd="$cmd --scarlet-ai $scarlet_type"
    fi
    
    cmd="$cmd --output-csv $csv_file --quiet"
    
    # Run the match
    $cmd
    
    # Parse results
    local gold_wins=$(tail -n +2 "$csv_file" | grep -c ",Gold,")
    local scarlet_wins=$(tail -n +2 "$csv_file" | grep -c ",Scarlet,")
    local match_draws=$(tail -n +2 "$csv_file" | grep -c ",Draw,")
    
    # Update statistics
    wins[$gold_name]=$((${wins[$gold_name]} + gold_wins))
    losses[$gold_name]=$((${losses[$gold_name]} + scarlet_wins))
    draws[$gold_name]=$((${draws[$gold_name]} + match_draws))
    total_games[$gold_name]=$((${total_games[$gold_name]} + GAMES))
    
    wins[$scarlet_name]=$((${wins[$scarlet_name]} + scarlet_wins))
    losses[$scarlet_name]=$((${losses[$scarlet_name]} + gold_wins))
    draws[$scarlet_name]=$((${draws[$scarlet_name]} + match_draws))
    total_games[$scarlet_name]=$((${total_games[$scarlet_name]} + GAMES))
    
    echo "  Results: Gold $gold_wins - Scarlet $scarlet_wins - Draws $match_draws"
    echo ""
    
    # Add to summary
    echo "$gold_name vs $scarlet_name:" >> "$SUMMARY"
    echo "  Gold wins: $gold_wins ($(awk "BEGIN {printf \"%.1f\", $gold_wins*100/$GAMES}")%)" >> "$SUMMARY"
    echo "  Scarlet wins: $scarlet_wins ($(awk "BEGIN {printf \"%.1f\", $scarlet_wins*100/$GAMES}")%)" >> "$SUMMARY"
    echo "  Draws: $match_draws ($(awk "BEGIN {printf \"%.1f\", $match_draws*100/$GAMES}")%)" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
}

# Run all matchups between plugins
echo "=== Plugin vs Plugin Matchups ==="
for i in "${!BOTS[@]}"; do
    bot1_entry="${BOTS[$i]}"
    bot1_file="${bot1_entry%%:*}"
    bot1_name="${bot1_entry##*:}"
    
    for j in "${!BOTS[@]}"; do
        if [ $i -lt $j ]; then
            bot2_entry="${BOTS[$j]}"
            bot2_file="${bot2_entry%%:*}"
            bot2_name="${bot2_entry##*:}"
            
            run_matchup "$bot1_file" "$bot1_name" "$bot2_file" "$bot2_name"
        fi
    done
done

# Run matchups between plugins and built-in AIs
echo "=== Plugin vs Built-in AI Matchups ==="
for bot_entry in "${BOTS[@]}"; do
    bot_file="${bot_entry%%:*}"
    bot_name="${bot_entry##*:}"
    
    for ai_entry in "${BUILTIN_AIS[@]}"; do
        ai_type="${ai_entry%%:*}"
        ai_name="${ai_entry##*:}"
        
        run_matchup "$bot_file" "$bot_name" "$ai_type" "$ai_name"
    done
done

# Generate final standings
echo "" >> "$SUMMARY"
echo "==================================" >> "$SUMMARY"
echo "        FINAL STANDINGS" >> "$SUMMARY"
echo "==================================" >> "$SUMMARY"
echo "" >> "$SUMMARY"

printf "%-25s %6s %6s %6s %6s %8s\n" "Bot" "Wins" "Losses" "Draws" "Games" "Win%" >> "$SUMMARY"
echo "---------------------------------------------------------------------" >> "$SUMMARY"

# Create sorted array by wins
declare -a sorted_bots=()
for bot_entry in "${BOTS[@]}"; do
    sorted_bots+=("${bot_entry##*:}")
done
for ai_entry in "${BUILTIN_AIS[@]}"; do
    sorted_bots+=("${ai_entry##*:}")
done

# Sort by wins (simple bubble sort)
for ((i = 0; i < ${#sorted_bots[@]}; i++)); do
    for ((j = i + 1; j < ${#sorted_bots[@]}; j++)); do
        if [ ${wins[${sorted_bots[$i]}]} -lt ${wins[${sorted_bots[$j]}]} ]; then
            temp="${sorted_bots[$i]}"
            sorted_bots[$i]="${sorted_bots[$j]}"
            sorted_bots[$j]="$temp"
        fi
    done
done

# Print standings
for name in "${sorted_bots[@]}"; do
    w=${wins[$name]}
    l=${losses[$name]}
    d=${draws[$name]}
    g=${total_games[$name]}
    win_pct=$(awk "BEGIN {printf \"%.1f\", $w*100/$g}")
    
    printf "%-25s %6d %6d %6d %6d %7s%%\n" "$name" "$w" "$l" "$d" "$g" "$win_pct" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "Detailed results available in: $OUTPUT_DIR/" >> "$SUMMARY"

# Display summary
cat "$SUMMARY"

echo ""
echo "=========================================="
echo "Tournament complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
