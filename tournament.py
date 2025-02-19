import copy
import math
import random
import csv
import importlib.util
import os
from game import Game
from ai import RandomAI

def load_custom_ai(filepath, game, color):
    """
    Load a custom AI from a given file path.
    If the file path is None or "None", use RandomAI.
    """
    if filepath is None or filepath == "None":
        return RandomAI(game, color)
    try:
        spec = importlib.util.spec_from_file_location("custom_ai", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CustomAI(game, color)
    except Exception as e:
        print(f"Error loading bot {filepath}: {e}. Using RandomAI instead.")
        return RandomAI(game, color)

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, score, expected, k=32):
    return rating + k * (score - expected)

def run_tournament(options):
    rounds = options.get("tournament_rounds", 5)
    bot_file_paths = options.get("bot_file_paths", [None]*8)
    output_csv = options.get("output_csv", "tournament_results.csv")
    
    # Create participant list (8 bots)
    participants = []
    for i, fp in enumerate(bot_file_paths):
        if fp is None or fp == "None":
            name = "RandomAI"
            fp_str = "None"
        else:
            name = os.path.basename(fp)
            fp_str = fp
        participants.append({
            "name": name,
            "file": fp_str,
            "score": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "elo": 1500.0
        })
    
    num_players = len(participants)
    
    # Run Swiss rounds
    for rnd in range(1, rounds + 1):
        print(f"Round {rnd}")
        # Sort participants by score and then Elo
        participants.sort(key=lambda p: (-p["score"], -p["elo"]))
        
        # Pair players in order (simple Swiss pairing)
        pairs = []
        unmatched = []
        used = [False] * num_players
        i = 0
        while i < num_players:
            if not used[i]:
                if i + 1 < num_players and not used[i + 1]:
                    pairs.append((participants[i], participants[i + 1]))
                    used[i] = True
                    used[i + 1] = True
                else:
                    unmatched.append(participants[i])
                    used[i] = True
            i += 1
        
        # Give unmatched player(s) a bye (counts as a win, no Elo change)
        for p in unmatched:
            p["score"] += 1
            p["wins"] += 1
            print(f"{p['name']} gets a bye.")
        
        # Play each pair
        for p1, p2 in pairs:
            game = Game()
            ai_gold = load_custom_ai(p1["file"], game, "Gold")
            ai_scarlet = load_custom_ai(p2["file"], game, "Scarlet")
            while not game.game_over:
                if game.current_turn == "Gold":
                    move = ai_gold.choose_move()
                else:
                    move = ai_scarlet.choose_move()
                if move:
                    game.make_move(move)
                game.update()
            winner = game.winner
            print(f"Match: {p1['name']} vs {p2['name']} | Winner: {winner}")
            if winner == "Gold":
                p1["score"] += 1
                p1["wins"] += 1
                p2["losses"] += 1
                outcome = 1  # from Gold's perspective
            elif winner == "Scarlet":
                p2["score"] += 1
                p2["wins"] += 1
                p1["losses"] += 1
                outcome = 0  # from Gold's perspective, loss
            else:
                p1["score"] += 0.5
                p2["score"] += 0.5
                p1["draws"] += 1
                p2["draws"] += 1
                outcome = 0.5

            # Elo update (using Gold perspective for pair; reciprocal for Scarlet)
            expected_p1 = expected_score(p1["elo"], p2["elo"])
            expected_p2 = expected_score(p2["elo"], p1["elo"])
            p1["elo"] = update_elo(p1["elo"], outcome, expected_p1)
            p2["elo"] = update_elo(p2["elo"], 1 - outcome, expected_p2)
    
    # Print final standings
    print("Tournament finished. Final standings:")
    for p in participants:
        print(p)
    
    # Write final standings to CSV.
    # Note: We include the "file" key in fieldnames to avoid ValueError.
    fieldnames = ["name", "file", "score", "wins", "losses", "draws", "elo"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in participants:
            writer.writerow(p)
