import sys, os, pygame, importlib.util, csv, time
import numpy as np
from menu import run_menu, run_ai_vs_ai_menu, run_ai_vs_player_menu, run_tournament_menu
from game import Game, pos_to_index, index_to_pos, move_generators
from bitboard import BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES
from ai import RandomAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from common import (CELL_SIZE, BOARD_GAP, BOARD_LEFT_MARGIN, BOARD_TOP_MARGIN, SIDE_PANEL_WIDTH,
                    load_assets, screen_to_board, draw_board, window_to_design,
                    BOARD_ROWS, BOARD_COLS, NUM_BOARDS, LIGHT_SQUARE, DARK_SQUARE, LINE_COLOR)
from simulation import simulate_ai_vs_ai_game  # Make sure simulation.py is available

def warmup_moves():
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode((360, 640))
    font = pygame.font.Font("assets/pixel.ttf", 36)
    text = font.render("Loading...", True, (255, 255, 255))
    screen.fill((0, 0, 0))
    screen.blit(text, (screen.get_width()//2 - text.get_width()//2,
                        screen.get_height()//2 - text.get_height()//2))
    pygame.display.update()
    dummy_board = np.zeros(TOTAL_SQUARES, dtype=np.int16)
    dummy_pos = (0, 0, 0)
    for key, func in move_generators.items():
        try:
            func(dummy_pos, dummy_board, "Gold")
        except Exception as e:
            print(f"Error warming up function for piece {key}: {e}")
    time.sleep(1)

def load_custom_ai(filepath, game, color):
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)

def main():
    pygame.init()
    flags = pygame.RESIZABLE

    mode, custom_ai_menu = run_menu()
    warmup_moves()

    if mode == "Campaign":
        import campaign
        campaign.run_campaign_menu()
        return

    if mode == "AI vs AI":
        options = run_ai_vs_ai_menu()
    elif mode == "AI vs Player":
        options = run_ai_vs_player_menu()
    elif mode == "Tournament":
        options = run_tournament_menu()
    else:
        options = {}

    board_width = BOARD_COLS * CELL_SIZE
    total_board_width = BOARD_LEFT_MARGIN * 2 + NUM_BOARDS * board_width + (NUM_BOARDS - 1) * BOARD_GAP
    design_width = total_board_width + SIDE_PANEL_WIDTH
    design_height = BOARD_TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 20
    design_resolution = (design_width, design_height)
    print(f"Gameplay design resolution: {design_width}x{design_height}")

    # For Tournament mode, run tournament simulation.
    if mode == "Tournament":
        import tournament
        tournament.run_tournament(options)
        pygame.quit()
        return

    # Create a resizable game window (initially 1280x720, 16:9)
    screen = pygame.display.set_mode((1280, 720), flags)
    pygame.display.set_caption("Dragonchess")
    base_surface = pygame.Surface(design_resolution)
    assets = load_assets(CELL_SIZE)
    clock = pygame.time.Clock()
    bg = pygame.image.load(os.path.join("assets", "bg.png")).convert()
    bg = pygame.transform.scale(bg, (design_width - SIDE_PANEL_WIDTH, design_height))

    if mode in ["2 Player", "AI vs Player"]:
        game = Game()
        if mode == "AI vs Player":
            # Determine AI side; human controls the opposite.
            ai_side = options.get("ai_side", "Gold")
            human_side = "Scarlet" if ai_side == "Gold" else "Gold"
            ai = load_custom_ai(options.get("ai_file"), game, ai_side) if options.get("ai_file") else RandomAI(game, ai_side)
        else:
            human_side = None  # Two-player mode: allow moves for both.
        selected_index = None
        legal_destinations = None
        running = True

        while running:
            time_delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    # In AI vs Player, only process clicks when it's human's turn.
                    if mode == "AI vs Player" and game.current_turn != human_side:
                        continue
                    design_mouse = window_to_design(pygame.mouse.get_pos(), design_resolution)
                    if design_mouse:
                        board_pos = screen_to_board(design_mouse)
                        if board_pos:
                            idx = pos_to_index(*board_pos)
                            # In AI vs Player, ensure the piece belongs to the human.
                            if mode == "AI vs Player":
                                if human_side == "Gold":
                                    valid_piece = game.board[idx] > 0
                                else:
                                    valid_piece = game.board[idx] < 0
                            else:
                                valid_piece = (game.board[idx] != 0)
                            if selected_index is None:
                                if valid_piece:
                                    selected_index = idx
                                    moves_list = game.get_legal_moves_for(selected_index)
                                    legal_destinations = {move[1] for move in moves_list}
                            else:
                                if idx in legal_destinations:
                                    moves_list = game.get_legal_moves_for(selected_index)
                                    for move in moves_list:
                                        if move[1] == idx:
                                            game.make_move(move)
                                            break
                                    selected_index = None
                                    legal_destinations = None
                                else:
                                    if valid_piece:
                                        selected_index = idx
                                        moves_list = game.get_legal_moves_for(selected_index)
                                        legal_destinations = {move[1] for move in moves_list}
                                    else:
                                        selected_index = None
                                        legal_destinations = None
            if mode == "AI vs Player" and game.current_turn != human_side:
                move = ai.choose_move()
                if move:
                    game.make_move(move)
            game.update()
            # If game is over, exit the loop.
            if game.game_over:
                running = False
            draw_board(base_surface, game, assets, bg, selected_index, legal_destinations)
            window_width, window_height = pygame.display.get_surface().get_size()
            scale = min(window_width / design_width, window_height / design_height)
            scaled_width = int(design_width * scale)
            scaled_height = int(design_height * scale)
            scaled_surface = pygame.transform.scale(base_surface, (scaled_width, scaled_height))
            final_surface = pygame.Surface((window_width, window_height))
            final_surface.fill((0, 0, 0))
            final_surface.blit(scaled_surface, ((window_width - scaled_width) // 2, (window_height - scaled_height) // 2))
            screen.blit(final_surface, (0, 0))
            pygame.display.update()
        pygame.quit()

    elif mode == "AI vs AI":
        headless = options.get("headless", False)
        num_games = options.get("num_games", 1)
        log_filename = options.get("log_filename", "logs/ai_vs_ai_log.csv")
        if headless:
            results = []
            max_workers = min(num_games, 10)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(simulate_ai_vs_ai_game, game_num, options)
                           for game_num in range(1, num_games+1)]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            with open(log_filename, "w", newline="") as csvfile:
                fieldnames = ["game_number", "full_record", "winner"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\")
                writer.writeheader()
                for game_num, move_notations, winner in results:
                    writer.writerow({"game_number": game_num, "full_record": "|".join(move_notations), "winner": winner})
        else:
            # Non-headless AI vs AI: run a game loop with rendering.
            game = Game()
            ai_gold = RandomAI(game, "Gold")
            ai_scarlet = RandomAI(game, "Scarlet")
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if game.current_turn == "Gold":
                    move = ai_gold.choose_move()
                else:
                    move = ai_scarlet.choose_move()
                if move:
                    game.make_move(move)
                game.update()
                draw_board(base_surface, game, assets, bg)
                window_width, window_height = pygame.display.get_surface().get_size()
                scale = min(window_width / design_width, window_height / design_height)
                scaled_width = int(design_width * scale)
                scaled_height = int(design_height * scale)
                scaled_surface = pygame.transform.scale(base_surface, (scaled_width, scaled_height))
                final_surface = pygame.Surface((window_width, window_height))
                final_surface.fill((0,0,0))
                final_surface.blit(scaled_surface, ((window_width-scaled_width)//2, (window_height-scaled_height)//2))
                pygame.display.get_surface().blit(final_surface, (0, 0))
                pygame.display.update()
                if game.game_over:
                    running = False
        pygame.quit()
    else:
        pygame.quit()

if __name__ == "__main__":
    main()
