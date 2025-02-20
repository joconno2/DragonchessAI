# tournament.py
import copy
import math
import random
import csv
import importlib.util
import os
import time
import pygame
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from game import Game
from ai import RandomAI

# Estimated average moves per match
AVERAGE_MOVES = 150

def draw_tournament_window(screen, font, progress_dict, match_info, participants, current_round):
    """
    Draws the persistent tournament window.
    Top half: current round match progress.
    Bottom half: cumulative tournament stats.
    """
    screen.fill((30, 30, 30))
    window_width, window_height = screen.get_size()

    # Header for current round
    header_text = font.render(f"Tournament - Round {current_round}", True, (255, 255, 0))
    screen.blit(header_text, (window_width // 2 - header_text.get_width() // 2, 10))

    # Top half for match progress
    top_area_height = window_height // 2 - 20
    y_offset = 40
    for match_id, label in match_info.items():
        move_count = progress_dict.get(match_id, 0)
        progress = min(move_count / AVERAGE_MOVES, 1.0)
        # Draw match label
        text_surface = font.render(label, True, (255, 255, 255))
        screen.blit(text_surface, (20, y_offset))
        # Draw progress bar
        bar_x = 250
        bar_y = y_offset
        bar_width = 400
        bar_height = 20
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        fill_width = int(bar_width * progress)
        pygame.draw.rect(screen, (0, 200, 0), (bar_x, bar_y, fill_width, bar_height))
        # Display moves count
        moves_text = font.render(f"{move_count}/{AVERAGE_MOVES}", True, (255, 255, 255))
        screen.blit(moves_text, (bar_x + bar_width + 10, bar_y))
        y_offset += 40
        if y_offset > top_area_height:
            break  # Only show as many matches as can fit

    # Bottom half for cumulative stats
    stats_header = font.render("Cumulative Tournament Stats", True, (0, 255, 255))
    screen.blit(stats_header, (20, window_height // 2 + 10))
    # Draw table headers
    headers = ["Name", "Score", "Wins", "Losses", "Draws", "Elo"]
    col_widths = [150, 70, 70, 70, 70, 100]
    x_positions = [20]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w + 10)
    y_table = window_height // 2 + 40
    for i, header in enumerate(headers):
        header_text = font.render(header, True, (200, 200, 200))
        screen.blit(header_text, (x_positions[i], y_table))
    y_table += 30
    # Sort participants (highest score then Elo)
    sorted_participants = sorted(participants, key=lambda p: (-p["score"], -p["elo"]))
    for p in sorted_participants:
        row = [p["name"], f"{p['score']:.1f}", str(p["wins"]), str(p["losses"]), str(p["draws"]), f"{p['elo']:.1f}"]
        for i, cell in enumerate(row):
            cell_text = font.render(cell, True, (255, 255, 255))
            screen.blit(cell_text, (x_positions[i], y_table))
        y_table += 25

def simulate_tournament_match(p1_file, p2_file, p1_name, p2_name, match_id, progress_dict):
    """
    Simulate a single tournament match between two bots.
    Updates progress_dict[match_id] with the current move count every 10 moves.
    """
    from game import Game
    from ai import RandomAI
    label = f"{p1_name} vs {p2_name}"
    print(f"Starting match {label}")
    game = Game()
    ai_gold = load_custom_ai(p1_file, game, "Gold")
    ai_scarlet = load_custom_ai(p2_file, game, "Scarlet")
    move_count = 0
    while not game.game_over:
        if game.current_turn == "Gold":
            move = ai_gold.choose_move()
        else:
            move = ai_scarlet.choose_move()
        if move:
            game.make_move(move)
            move_count += 1
            if move_count % 10 == 0:
                progress_dict[match_id] = move_count
        game.update()
    progress_dict[match_id] = move_count
    print(f"Finished match {label} with winner {game.winner}")
    return {"winner": game.winner, "move_count": move_count}

def load_custom_ai(filepath, game, color):
    """
    Dynamically load a custom AI. If not provided, falls back to RandomAI.
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
    """
    Runs a Swiss tournament over several rounds.
    A persistent Pygame window is used to display live match progress and cumulative stats.
    """
    rounds = options.get("tournament_rounds", 5)
    bot_file_paths = options.get("bot_file_paths", [None] * 8)
    output_csv = options.get("output_csv", "logs/tournament_results.csv")
    
    # Build initial participant list (8 bots)
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
    
    # Initialize one persistent, larger Pygame window.
    pygame.init()
    window_width = 1000
    window_height = 600
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Tournament Progress and Stats")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    
    for rnd in range(1, rounds + 1):
        print(f"\nStarting Round {rnd}")
        # Sort participants by score and Elo.
        participants.sort(key=lambda p: (-p["score"], -p["elo"]))
        
        # Simple Swiss pairing.
        pairs = []
        unmatched = []
        used = [False] * num_players
        for i in range(num_players):
            if not used[i]:
                if i + 1 < num_players and not used[i + 1]:
                    pairs.append((i, i + 1))
                    used[i] = True
                    used[i + 1] = True
                else:
                    unmatched.append(i)
                    used[i] = True
        
        # Give unmatched players a bye.
        for idx in unmatched:
            participants[idx]["score"] += 1
            participants[idx]["wins"] += 1
            print(f"{participants[idx]['name']} gets a bye.")
        
        # Set up shared progress state for this round.
        manager = multiprocessing.Manager()
        progress_dict = manager.dict()
        match_info = {}  # Maps match_id to a label (e.g. "Bot1 vs Bot2")
        futures = {}
        match_id_counter = 1
        
        # Use increased concurrency (e.g. twice the number of CPU cores).
        max_workers = os.cpu_count() * 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, j in pairs:
                p1 = participants[i]
                p2 = participants[j]
                match_id = f"match_{rnd}_{match_id_counter}"
                match_info[match_id] = f"{p1['name']} vs {p2['name']}"
                progress_dict[match_id] = 0
                future = executor.submit(simulate_tournament_match,
                                         p1["file"], p2["file"],
                                         p1["name"], p2["name"],
                                         match_id, progress_dict)
                futures[future] = (i, j, match_id)
                match_id_counter += 1
            
            # While matches are running, update the persistent window.
            while not all(f.done() for f in futures):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                draw_tournament_window(screen, font, progress_dict, match_info, participants, current_round=rnd)
                pygame.display.flip()
                clock.tick(30)
            
            # One final update for this round before processing results.
            draw_tournament_window(screen, font, progress_dict, match_info, participants, current_round=rnd)
            pygame.display.flip()
            time.sleep(2)
            
            # Process match results as they complete.
            for future in as_completed(futures):
                i, j, match_id = futures[future]
                result = future.result()
                winner = result["winner"]
                if winner == "Gold":
                    outcome = 1
                elif winner == "Scarlet":
                    outcome = 0
                else:
                    outcome = 0.5
                p1 = participants[i]
                p2 = participants[j]
                if winner == "Gold":
                    p1["score"] += 1
                    p1["wins"] += 1
                    p2["losses"] += 1
                elif winner == "Scarlet":
                    p2["score"] += 1
                    p2["wins"] += 1
                    p1["losses"] += 1
                else:
                    p1["score"] += 0.5
                    p2["score"] += 0.5
                    p1["draws"] += 1
                    p2["draws"] += 1
                expected_p1 = expected_score(p1["elo"], p2["elo"])
                expected_p2 = expected_score(p2["elo"], p1["elo"])
                p1["elo"] = update_elo(p1["elo"], outcome, expected_p1)
                p2["elo"] = update_elo(p2["elo"], 1 - outcome, expected_p2)
                print(f"Round {rnd}: {p1['name']} vs {p2['name']} - Winner: {winner}, Moves: {result['move_count']}")
            
            manager.shutdown()
    
    # After all rounds, display final standings in the window until the user closes it.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((30, 30, 30))
        final_text = font.render("Tournament Finished - Final Standings", True, (255, 215, 0))
        screen.blit(final_text, (window_width // 2 - final_text.get_width() // 2, 20))
        y_offset = 60
        sorted_participants = sorted(participants, key=lambda p: (-p["score"], -p["elo"]))
        for p in sorted_participants:
            line = (f"{p['name']}: Score {p['score']:.1f}, Wins {p['wins']}, "
                    f"Losses {p['losses']}, Draws {p['draws']}, Elo {p['elo']:.1f}")
            text_line = font.render(line, True, (255, 255, 255))
            screen.blit(text_line, (20, y_offset))
            y_offset += 30
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    
    # Write final standings to CSV, omitting the 'file' column.
    fieldnames = ["name", "score", "wins", "losses", "draws", "elo"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in participants:
            # Create a copy of p without the 'file' key.
            row = {k: v for k, v in p.items() if k != "file"}
            writer.writerow(row)
