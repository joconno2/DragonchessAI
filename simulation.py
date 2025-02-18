import os
import copy
import random
import sys
import pygame
from game import Game
from ai import RandomAI
import importlib.util

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
    pid = os.getpid()
    print(f"[Process {pid}] Starting game {game_num}")
    
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
    
    # Run the game simulation.
    while not game.game_over:
        if game.current_turn == "Gold":
            move = ai_gold.choose_move()
        else:
            move = ai_scarlet.choose_move()
        if move:
            game.make_move(move)
        game.update()
    
    print(f"[Process {pid}] Finished game {game_num} with winner {game.winner}")
    return game_num, game.move_notations, game.winner
