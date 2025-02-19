import sys, os, pygame, importlib.util, csv
from menu import run_menu, run_ai_vs_ai_menu, run_ai_vs_player_menu, run_tournament_menu
from game import Game, pos_to_index, index_to_pos
from bitboard import BOARD_ROWS, BOARD_COLS, NUM_BOARDS
from ai import RandomAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# UI layout constants
CELL_SIZE = 40
BOARD_GAP = 20
BOARD_LEFT_MARGIN = 20
BOARD_TOP_MARGIN = 60
SIDE_PANEL_WIDTH = 200

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LINE_COLOR = (0, 0, 0)
BG_COLOR = (50, 50, 50)

def load_assets(cell_size):
    assets = {}
    asset_names = {
        1: "gold_sylph",
        2: "gold_griffin",
        3: "gold_dragon",
        4: "gold_oliphant",
        5: "gold_unicorn",
        6: "gold_hero",
        7: "gold_thief",
        8: "gold_cleric",
        9: "gold_mage",
        10: "gold_king",
        11: "gold_paladin",
        12: "gold_warrior",
        13: "gold_basilisk",
        14: "gold_elemental",
        15: "gold_dwarf",
        -1: "scarlet_sylph",
        -2: "scarlet_griffin",
        -3: "scarlet_dragon",
        -4: "scarlet_oliphant",
        -5: "scarlet_unicorn",
        -6: "scarlet_hero",
        -7: "scarlet_thief",
        -8: "scarlet_cleric",
        -9: "scarlet_mage",
        -10: "scarlet_king",
        -11: "scarlet_paladin",
        -12: "scarlet_warrior",
        -13: "scarlet_basilisk",
        -14: "scarlet_elemental",
        -15: "scarlet_dwarf"
    }
    for code, name in asset_names.items():
        path = os.path.join("assets", f"{name}.png")
        try:
            image = pygame.image.load(path)
            image = pygame.transform.scale(image, (cell_size, cell_size))
            assets[code] = image
        except Exception:
            assets[code] = None
    return assets

def load_custom_ai(filepath, game, color):
    """Dynamically load a custom AI from a given file path."""
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)

def draw_board(screen, game, assets, selected_index=None, legal_destinations=None):
    screen.fill(BG_COLOR)
    board_width = BOARD_COLS * CELL_SIZE
    board_height = BOARD_ROWS * CELL_SIZE
    font = pygame.font.Font("pixel.ttf", 20)
    frozen_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    frozen_overlay.fill((0,150,255,100))
    
    for layer in range(NUM_BOARDS):
        board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
        board_y_start = BOARD_TOP_MARGIN
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                rect = pygame.Rect(board_x_start + col * CELL_SIZE,
                                   board_y_start + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                square_color = LIGHT_SQUARE if (row+col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, square_color, rect)
                pygame.draw.rect(screen, LINE_COLOR, rect, 1)
                idx = pos_to_index(layer, row, col)
                if selected_index is not None and idx == selected_index:
                    pygame.draw.rect(screen, (0,255,0), rect, 3)
                if legal_destinations and idx in legal_destinations:
                    pygame.draw.rect(screen, (0,0,255), rect, 3)
                piece = game.board[idx]
                if piece != 0:
                    asset = assets.get(piece)
                    if asset:
                        screen.blit(asset, rect.topleft)
                    else:
                        text = font.render(game.piece_letter(piece), True, (0,0,0))
                        text_rect = text.get_rect(center=rect.center)
                        screen.blit(text, text_rect)
                if game.frozen[idx]:
                    screen.blit(frozen_overlay, rect.topleft)
        board_rect = pygame.Rect(board_x_start, board_y_start, board_width, board_height)
        pygame.draw.rect(screen, LINE_COLOR, board_rect, 3)
        titles = ["Sky", "Ground", "Underworld"]
        title_font = pygame.font.Font("pixel.ttf", 36)
        title_text = title_font.render(titles[layer], True, (255,255,255))
        title_rect = title_text.get_rect(center=(board_x_start + board_width//2, BOARD_TOP_MARGIN//2))
        screen.blit(title_text, title_rect)
    total_width = screen.get_width()
    total_height = screen.get_height()
    pane_rect = pygame.Rect(total_width - SIDE_PANEL_WIDTH, 0, SIDE_PANEL_WIDTH, total_height)
    pygame.draw.rect(screen, (30,30,30), pane_rect)
    pygame.draw.rect(screen, LINE_COLOR, pane_rect, 3)
    log_font = pygame.font.Font("pixel.ttf", 24)
    y_offset = 10
    for move_str in game.move_notations[-int((total_height - y_offset)/20):]:
        text = log_font.render(move_str, True, (200,200,200))
        screen.blit(text, (pane_rect.x + 5, y_offset))
        y_offset += 20

def screen_to_board(pos):
    x, y = pos
    board_width = BOARD_COLS * CELL_SIZE
    board_height = BOARD_ROWS * CELL_SIZE
    for layer in range(NUM_BOARDS):
        board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
        board_x_end = board_x_start + board_width
        board_y_start = BOARD_TOP_MARGIN
        board_y_end = board_y_start + board_height
        if board_x_start <= x < board_x_end and board_y_start <= y < board_y_end:
            col = (x - board_x_start) // CELL_SIZE
            row = (y - board_y_start) // CELL_SIZE
            return (layer, row, col)
    return None

def draw_game_over(screen, message, win_width, win_height):
    overlay = pygame.Surface((win_width, win_height), pygame.SRCALPHA)
    overlay.fill((0,0,0,150))
    screen.blit(overlay, (0,0))
    game_over_font = pygame.font.Font("pixel.ttf", 48)
    text = game_over_font.render(f"Game Over! Winner: {message}", True, (255,255,255))
    text_rect = text.get_rect(center=(win_width//2, win_height//2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.delay(3000)

def compress_move_log(move_notations):
    """Compress the full move record into a single string."""
    return "|".join(move_notations)

# Import simulation function for AI vs AI games.
from simulation import simulate_ai_vs_ai_game

def main():
    pygame.init()
    mode, custom_ai_menu = run_menu()
    
    if mode == "AI vs AI":
        options = run_ai_vs_ai_menu()
        headless = options["headless"]
        num_games = options["num_games"]
    elif mode == "AI vs Player":
        options = run_ai_vs_player_menu()
        headless = False
        num_games = 1
        ai_side = options["ai_side"]  # "Gold" or "Scarlet"
    elif mode == "Tournament":
        options = run_tournament_menu()
        from tournament import run_tournament
        run_tournament(options)
        pygame.quit()
        sys.exit()
    else:
        headless = False
        num_games = 1

    board_width = BOARD_COLS * CELL_SIZE
    total_board_width = BOARD_LEFT_MARGIN * 2 + NUM_BOARDS * board_width + (NUM_BOARDS - 1) * BOARD_GAP
    win_width = total_board_width + SIDE_PANEL_WIDTH
    win_height = BOARD_TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 20

    if headless:
        screen = None
    else:
        screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("Dragonchess")
    assets = load_assets(CELL_SIZE)
    clock = pygame.time.Clock()

    if mode == "AI vs AI":
        log_filename = options["log_filename"]
        with open(log_filename, "w", newline="") as csvfile:
            fieldnames = ["game_number", "full_record", "winner"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writeheader()
            if headless:
                results = []
                max_workers = min(num_games, 10)
                finished_count = 0
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(simulate_ai_vs_ai_game, game_num, options)
                               for game_num in range(1, num_games + 1)]
                    for future in as_completed(futures):
                        results.append(future.result())
                        finished_count += 1
                        print(f"Progress: {finished_count}/{num_games} games finished.")
                for game_num, move_notations, winner in sorted(results, key=lambda x: x[0]):
                    full_record = compress_move_log(move_notations)
                    writer.writerow({
                        "game_number": game_num,
                        "full_record": full_record,
                        "winner": winner
                    })
                    print(f"Game {game_num} finished. Winner: {winner}")
            else:
                for game_num in range(1, num_games + 1):
                    game = Game()
                    if options.get("gold_ai"):
                        ai_gold = load_custom_ai(options["gold_ai"], game, "Gold")
                    else:
                        ai_gold = RandomAI(game, "Gold")
                    if options.get("scarlet_ai"):
                        ai_scarlet = load_custom_ai(options["scarlet_ai"], game, "Scarlet")
                    else:
                        ai_scarlet = RandomAI(game, "Scarlet")
                    while not game.game_over:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                        if game.current_turn == "Gold":
                            move = ai_gold.choose_move()
                        else:
                            move = ai_scarlet.choose_move()
                        if move:
                            game.make_move(move)
                        game.update()
                        if screen:
                            draw_board(screen, game, assets)
                            pygame.display.flip()
                    full_record = compress_move_log(game.move_notations)
                    writer.writerow({
                        "game_number": game_num,
                        "full_record": full_record,
                        "winner": game.winner
                    })
                    print(f"Game {game_num} finished. Winner: {game.winner}")
    else:
        # Two-player or AI vs Player mode.
        game = Game()
        ai = None
        two_player = (mode == "2 Player")
        if mode == "AI vs Player":
            if options.get("ai_file"):
                ai = load_custom_ai(options["ai_file"], game, ai_side)
            else:
                ai = RandomAI(game, ai_side)
        selected_index = None
        running = True
        while running and not game.game_over:
            if two_player or (mode == "AI vs Player" and game.current_turn != ai_side):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        board_cell = screen_to_board(event.pos)
                        if board_cell is not None:
                            clicked_index = pos_to_index(*board_cell)
                            if selected_index is None:
                                if game.board[clicked_index] != 0:
                                    piece = game.board[clicked_index]
                                    if (game.current_turn == "Gold" and piece > 0) or (game.current_turn == "Scarlet" and piece < 0):
                                        selected_index = clicked_index
                            else:
                                legal_moves = game.get_legal_moves_for(selected_index)
                                chosen_move = None
                                for m in legal_moves:
                                    if m[1] == clicked_index:
                                        chosen_move = m
                                        break
                                if chosen_move:
                                    game.make_move(chosen_move)
                                    selected_index = None
                                else:
                                    if game.board[clicked_index] != 0:
                                        piece = game.board[clicked_index]
                                        if (game.current_turn == "Gold" and piece > 0) or (game.current_turn == "Scarlet" and piece < 0):
                                            selected_index = clicked_index
                                        else:
                                            selected_index = None
                                    else:
                                        selected_index = None
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            if mode == "AI vs Player" and game.current_turn == ai_side:
                move = ai.choose_move()
                if move:
                    game.make_move(move)
            game.update()
            legal_destinations = None
            if selected_index is not None:
                legal_destinations = { m[1] for m in game.get_legal_moves_for(selected_index) }
            if screen:
                draw_board(screen, game, assets, selected_index, legal_destinations)
                pygame.display.flip()
        if screen and (two_player or mode == "AI vs Player"):
            draw_game_over(screen, game.winner, win_width, win_height)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
