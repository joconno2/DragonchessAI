import sys
import os
import pygame
import importlib.util
import csv
from game import Game, BOARD_COLS, BOARD_GAP, BOARD_LEFT_MARGIN, BOARD_TOP_MARGIN, SIDE_PANEL_WIDTH, BOARD_ROWS, NUM_BOARDS
from ai import RandomAI 
from menu import run_menu, run_ai_vs_ai_menu

def load_custom_ai(filepath, game, color):
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)

def main():
    pygame.init()

    icon_path = os.path.join("assets", "scarlet_dragon.png")
    try:
        icon_image = pygame.image.load(icon_path)
        pygame.display.set_icon(icon_image)
    except Exception as e:
        print(f"Could not load icon from {icon_path}: {e}")

    mode, custom_ai_paths = run_menu()
    
    headless_override = False
    num_games = 1
    log_filename = "game_log.csv"
    if mode == "AI vs AI":
        options = run_ai_vs_ai_menu()
        num_games = options["num_games"]
        log_filename = options["log_filename"]
        headless_override = options["headless"]
        custom_ai_paths["scarlet"] = options["scarlet_ai"]
        custom_ai_paths["gold"] = options["gold_ai"]
    
    # Compute initial window dimensions.
    initial_cell_size = 40
    board_width = BOARD_COLS * initial_cell_size
    total_board_width = BOARD_LEFT_MARGIN * 2 + NUM_BOARDS * board_width + (NUM_BOARDS - 1) * BOARD_GAP
    win_width = total_board_width + SIDE_PANEL_WIDTH
    board_height = BOARD_ROWS * initial_cell_size
    win_height = BOARD_TOP_MARGIN + board_height + 20  # 20 for bottom margin
    
    if headless_override:
        screen = None
    else:
        screen = pygame.display.set_mode((win_width, win_height), pygame.RESIZABLE)
        pygame.display.set_caption("Dragonchess")
    
    game = Game(screen, headless=headless_override)
    gold_ai = None
    scarlet_ai = None

    if mode == "AI as Scarlet":
        if custom_ai_paths["scarlet"]:
            scarlet_ai = load_custom_ai(custom_ai_paths["scarlet"], game, "Scarlet")
        else:
            scarlet_ai = RandomAI(game, "Scarlet")
    elif mode == "AI as Gold":
        if custom_ai_paths["gold"]:
            gold_ai = load_custom_ai(custom_ai_paths["gold"], game, "Gold")
        else:
            gold_ai = RandomAI(game, "Gold")
    
    clock = pygame.time.Clock()
    
    if mode == "AI vs AI":
        with open(log_filename, "w", newline="") as csvfile:
            fieldnames = ["game_number", "move_number", "move", "winner"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for game_num in range(1, num_games + 1):
                game = Game(screen, headless=headless_override)
                if custom_ai_paths["gold"]:
                    gold_ai = load_custom_ai(custom_ai_paths["gold"], game, "Gold")
                else:
                    gold_ai = RandomAI(game, "Gold")
                if custom_ai_paths["scarlet"]:
                    scarlet_ai = load_custom_ai(custom_ai_paths["scarlet"], game, "Scarlet")
                else:
                    scarlet_ai = RandomAI(game, "Scarlet")
                move_counter = 0
                running_game = True
                while running_game:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.VIDEORESIZE:
                            new_width, new_height = event.size
                            # Recompute new cell size based on new width.
                            available_width = new_width - SIDE_PANEL_WIDTH - 2 * BOARD_LEFT_MARGIN - (NUM_BOARDS - 1) * BOARD_GAP
                            new_cell_size = available_width // (NUM_BOARDS * BOARD_COLS)
                            game.set_cell_size(new_cell_size)
                            board_width = BOARD_COLS * new_cell_size
                            total_board_width = BOARD_LEFT_MARGIN * 2 + NUM_BOARDS * board_width + (NUM_BOARDS - 1) * BOARD_GAP
                            win_width = total_board_width + SIDE_PANEL_WIDTH
                            pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    current_color = game.current_turn
                    if current_color == "Gold":
                        move = gold_ai.choose_move()
                        if move:
                            game.make_move(move)
                            move_counter += 1
                    elif current_color == "Scarlet":
                        move = scarlet_ai.choose_move()
                        if move:
                            game.make_move(move)
                            move_counter += 1
                    game.update()
                    if not headless_override:
                        game.draw()
                        pygame.display.flip()
                    pygame.time.delay(10)
                    if game.game_over:
                        running_game = False
                for i, alg_move in enumerate(game.game_log):
                    writer.writerow({
                        "game_number": game_num,
                        "move_number": i + 1,
                        "move": alg_move,
                        "winner": game.winner if i == len(game.game_log) - 1 else ""
                    })
                print(f"Game {game_num} finished. Winner: {game.winner}")
        pygame.quit()
        sys.exit()
    else:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    new_width, new_height = event.size
                    available_width = new_width - SIDE_PANEL_WIDTH - 2 * BOARD_LEFT_MARGIN - (NUM_BOARDS - 1) * BOARD_GAP
                    new_cell_size = available_width // (NUM_BOARDS * BOARD_COLS)
                    game.set_cell_size(new_cell_size)
                    pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                else:
                    game.handle_event(event)
            current_color = game.current_turn
            if current_color == "Gold" and gold_ai:
                move = gold_ai.choose_move()
                if move:
                    game.make_move(move)
            elif current_color == "Scarlet" and scarlet_ai:
                move = scarlet_ai.choose_move()
                if move:
                    game.make_move(move)
            game.update()
            if not headless_override:
                game.draw()
                pygame.display.flip()
            clock.tick(30)
        pygame.quit()

if __name__ == "__main__":
    main()
