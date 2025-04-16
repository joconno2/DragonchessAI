import sys, os, pygame, importlib.util, csv, time
import numpy as np
from menu import run_menu, run_ai_vs_ai_menu, run_ai_vs_player_menu, run_tournament_menu
from game import Game, pos_to_index, index_to_pos, move_generators
from bitboard import BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES
from ai import RandomAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from common import BG_SCALE_X, BG_SCALE_Y, CELL_SIZE, LIGHT_SQUARE, DARK_SQUARE, LINE_COLOR, load_assets, draw_board, screen_to_board, DESIGN_WIDTH, DESIGN_HEIGHT
from simulation import simulate_ai_vs_ai_game  # Ensure simulation.py is available

def warmup_moves():
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode((DESIGN_WIDTH, DESIGN_HEIGHT))
    font = pygame.font.Font("assets/pixel.ttf", 36)
    text = font.render("Loading...", True, (255, 255, 255))
    screen.fill((0, 0, 0))
    screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2,
                        screen.get_height() // 2 - text.get_height() // 2))
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

    pygame.display.set_icon(pygame.image.load(os.path.join("assets", "small_icon.png")))
    pygame.display.set_caption("Dragonchess")

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

    # Use the horizontal design resolution: 1600x900 (16:9 aspect ratio)
    design_width, design_height = (DESIGN_WIDTH, DESIGN_HEIGHT)
    design_resolution = (design_width, design_height)
    print(f"Gameplay design resolution: {design_width}x{design_height}")

    if mode == "Tournament":
        import tournament
        tournament.run_tournament(options)
        pygame.quit()
        return

    # Create a resizable game window (initially 1600x900)
    screen = pygame.display.set_mode((design_width, design_height), flags)
    pygame.display.set_caption("Dragonchess")
    base_surface = pygame.Surface(design_resolution)
    assets = load_assets(CELL_SIZE)
    clock = pygame.time.Clock()
    # Load the horizontal background image.
    bg = pygame.image.load(os.path.join("assets", "bg.png")).convert()
    bg = pygame.transform.scale(bg, (int(bg.get_width() * BG_SCALE_X), int(bg.get_height() * BG_SCALE_Y)))

    if mode in ["2 Player", "AI vs Player"]:
        game = Game()
        if mode == "AI vs Player":
            ai_side = options.get("ai_side", "Gold")
            human_side = "Scarlet" if ai_side == "Gold" else "Gold"
            ai = load_custom_ai(options.get("ai_file"), game, ai_side) if options.get("ai_file") else RandomAI(game, ai_side)
        else:
            human_side = None  # Two-player mode: both sides controlled by players.
        selected_index = None
        legal_destinations = None
        running = True

        while running:
            time_delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    # In AI vs Player, process clicks only when it's the human's turn.
                    if mode == "AI vs Player" and game.current_turn != human_side:
                        continue
                    # Convert mouse coordinates (window space) back to design coordinates.
                    win_width, win_height = screen.get_size()
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    scale_x = design_width / win_width
                    scale_y = design_height / win_height
                    design_mouse = (mouse_x * scale_x, mouse_y * scale_y)
                    board_pos = screen_to_board(design_mouse)
                    if board_pos:
                        idx = pos_to_index(*board_pos)
                        if mode == "AI vs Player":
                            valid_piece = game.board[idx] > 0 if human_side == "Gold" else game.board[idx] < 0
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
            if game.game_over:
                running = False
            # Draw on the base surface (at 1600x900 design resolution).
            base_surface.fill((0, 0, 0))
            draw_board(base_surface, game, assets, bg, selected_index, legal_destinations)
            # Scale the base surface to the current window size.
            scaled_surface = pygame.transform.scale(base_surface, screen.get_size())
            screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
    elif mode == "AI vs AI":
        game = Game()
        running = True
        while running:
            time_delta = clock.tick(60) / 1000.0
            if game.current_turn == "Gold":
                move = RandomAI(game, "Gold").choose_move()
            else:
                move = RandomAI(game, "Scarlet").choose_move()
            if move:
                game.make_move(move)
            game.update()
            if game.game_over:
                running = False
            base_surface.fill((0, 0, 0))
            draw_board(base_surface, game, assets, bg)
            scaled_surface = pygame.transform.scale(base_surface, screen.get_size())
            screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
