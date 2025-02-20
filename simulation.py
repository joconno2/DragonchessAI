# simulation.py
import os
import copy
import random
import sys
import pygame
from game import Game
from ai import RandomAI
import importlib.util

def print_progress_bar(label, current, total, bar_length=40):
    """
    Prints a progress bar on the same CLI line.
    """
    percent = min(current / total, 1.0)
    filled_length = int(round(bar_length * percent))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    # \r returns carriage to start of line; flush ensures immediate printing
    print(f'\r{label} Progress: |{bar}| {current}/{total} moves', end='', flush=True)

def load_custom_ai(filepath, game, color):
    """Dynamically load a custom AI from a given file path."""
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)

def simulate_ai_vs_ai_game(game_num, options):
    """
    Simulate one AI vs AI game (headless) and return:
    (game_num, move_notations, winner)
    """
    AVERAGE_MOVES = 150  # Estimated average moves per game
    pid = os.getpid()
    label = f"[Process {pid}] Game {game_num}"
    print(f"{label} Starting")
    
    # Create a new game instance.
    game = Game()
    
    # Create AIs.
    if options.get("gold_ai"):
        ai_gold = load_custom_ai(options["gold_ai"], game, "Gold")
    else:
        ai_gold = RandomAI(game, "Gold")
    if options.get("scarlet_ai"):
        ai_scarlet = load_custom_ai(options["scarlet_ai"], game, "Scarlet")
    else:
        ai_scarlet = RandomAI(game, "Scarlet")
    
    move_count = 0
    # Run the game simulation.
    while not game.game_over:
        if game.current_turn == "Gold":
            move = ai_gold.choose_move()
        else:
            move = ai_scarlet.choose_move()
        if move:
            game.make_move(move)
            move_count += 1
            # Update the progress bar every 10 moves.
            if move_count % 10 == 0:
                print_progress_bar(label, move_count, AVERAGE_MOVES)
        game.update()
    
    print()  # Ensure a newline after the progress bar is complete.
    print(f"{label} Finished with winner {game.winner}")
    return game_num, game.move_notations, game.winner
