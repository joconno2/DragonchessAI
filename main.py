import sys, os, pygame, importlib.util, csv
from menu import run_menu, run_ai_vs_ai_menu, run_ai_vs_player_menu, run_tournament_menu
from game import Game, pos_to_index, index_to_pos
from bitboard import BOARD_ROWS, BOARD_COLS, NUM_BOARDS
from ai import RandomAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# UI layout constants (for gameplay design resolution)
CELL_SIZE = 40
BOARD_GAP = 50
BOARD_LEFT_MARGIN = 50
BOARD_TOP_MARGIN = 120
SIDE_PANEL_WIDTH = 200

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LINE_COLOR = (0, 0, 0)

# Helper: enforce design aspect ratio.
def enforce_aspect_ratio(w, h, design_ratio):
    if w / h > design_ratio:
        w = int(h * design_ratio)
    else:
        h = int(w / design_ratio)
    return w, h

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

def draw_board(surface, game, assets, bg, selected_index=None, legal_destinations=None):
    surface.blit(bg, (0, 0))
    
    board_width = BOARD_COLS * CELL_SIZE
    board_height = BOARD_ROWS * CELL_SIZE
    font = pygame.font.Font("assets/pixel.ttf", 20)
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
                square_color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(surface, square_color, rect)
                pygame.draw.rect(surface, LINE_COLOR, rect, 1)
                idx = pos_to_index(layer, row, col)
                
                piece = game.board[idx]
                if piece != 0:
                    asset = assets.get(piece)
                    if asset:
                        surface.blit(asset, rect.topleft)
                    else:
                        text = font.render(game.piece_letter(piece), True, (0,0,0))
                        text_rect = text.get_rect(center=rect.center)
                        surface.blit(text, text_rect)
                if game.frozen[idx]:
                    surface.blit(frozen_overlay, rect.topleft)
                
                if selected_index is not None and idx == selected_index:
                    pygame.draw.rect(surface, (0,255,0), rect, 3)
                if legal_destinations and idx in legal_destinations:
                    pygame.draw.rect(surface, (0,0,255), rect, 3)
                    
        board_rect = pygame.Rect(board_x_start, board_y_start, board_width, board_height)
        pygame.draw.rect(surface, LINE_COLOR, board_rect, 3)
        titles = ["Sky", "Ground", "Underworld"]
        title_font = pygame.font.Font("assets/pixel.ttf", 36)
        title_text = title_font.render(titles[layer], True, (255,255,255))
        title_rect = title_text.get_rect(center=(board_x_start + board_width // 2, BOARD_TOP_MARGIN // 2))
        surface.blit(title_text, title_rect)
    
    total_width = surface.get_width()
    total_height = surface.get_height()
    pane_rect = pygame.Rect(total_width - SIDE_PANEL_WIDTH, 0, SIDE_PANEL_WIDTH, total_height)
    pygame.draw.rect(surface, (30,30,30), pane_rect)
    pygame.draw.rect(surface, LINE_COLOR, pane_rect, 3)
    log_font = pygame.font.Font("assets/pixel.ttf", 24)
    y_offset = 10
    for move_str in game.move_notations[-int((total_height - y_offset) / 20):]:
        text = log_font.render(move_str, True, (200,200,200))
        surface.blit(text, (pane_rect.x + 5, y_offset))
        y_offset += 20

def screen_to_board(design_pos):
    x, y = design_pos
    board_width = BOARD_COLS * CELL_SIZE
    board_height = BOARD_ROWS * CELL_SIZE
    for layer in range(NUM_BOARDS):
        board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
        board_x_end = board_x_start + board_width
        board_y_start = BOARD_TOP_MARGIN
        board_y_end = board_y_start + board_height
        if board_x_start <= x < board_x_end and board_y_start <= y < board_y_end:
            col = int((x - board_x_start) // CELL_SIZE)
            row = int((y - board_y_start) // CELL_SIZE)
            return (layer, row, col)
    return None

def draw_game_over(surface, message, width, height):
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0,0,0,150))
    surface.blit(overlay, (0,0))
    game_over_font = pygame.font.Font("assets/pixel.ttf", 48)
    text = game_over_font.render(f"Game Over! Winner: {message}", True, (255,255,255))
    text_rect = text.get_rect(center=(width // 2, height // 2))
    surface.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.delay(3000)

def compress_move_log(move_notations):
    return "|".join(move_notations)

# Import simulation function for AI vs AI games.
from simulation import simulate_ai_vs_ai_game

def main():
    pygame.init()
    flags = pygame.RESIZABLE

    # --- Run Menu First (menu uses its own resolution) ---
    mode, custom_ai_menu = run_menu()
    if mode == "AI vs AI":
        options = run_ai_vs_ai_menu()
    elif mode == "AI vs Player":
        options = run_ai_vs_player_menu()
    elif mode == "Tournament":
        options = run_tournament_menu()
    else:
        options = {}

    # --- Now Compute Gameplay Design Resolution ---
    board_width = BOARD_COLS * CELL_SIZE
    total_board_width = BOARD_LEFT_MARGIN * 2 + NUM_BOARDS * board_width + (NUM_BOARDS - 1) * BOARD_GAP
    design_width = total_board_width + SIDE_PANEL_WIDTH
    design_height = BOARD_TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 20
    design_ratio = design_width / design_height
    print(f"Gameplay design resolution: {design_width}x{design_height}")
    
    # Reinitialize display for gameplay.
    screen = pygame.display.set_mode((design_width, design_height), flags)
    pygame.display.set_caption("Dragonchess")
    base_surface = pygame.Surface((design_width, design_height))
    current_size = (design_width, design_height)
    
    assets = load_assets(CELL_SIZE)
    clock = pygame.time.Clock()
    
    # Game area (design): entire area minus side panel.
    game_area_width = design_width - SIDE_PANEL_WIDTH
    game_area_height = design_height
    bg = pygame.image.load(os.path.join("assets", "bg.png")).convert()
    bg = pygame.transform.scale(bg, (game_area_width, game_area_height))
    
    # --- Gameplay Loop ---
    if mode == "AI vs AI":
        headless = options.get("headless", False)
        num_games = options.get("num_games", 1)
        log_filename = options.get("log_filename", "ai_vs_ai_log.csv")
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
                            if event.type == pygame.VIDEORESIZE:
                                new_w, new_h = enforce_aspect_ratio(event.w, event.h, design_ratio)
                                current_size = (new_w, new_h)
                                screen = pygame.display.set_mode(current_size, flags)
                        if game.current_turn == "Gold":
                            move = ai_gold.choose_move()
                        else:
                            move = ai_scarlet.choose_move()
                        if move:
                            game.make_move(move)
                        game.update()
                        draw_board(base_surface, game, assets, bg)
                        scaled = pygame.transform.smoothscale(base_surface, current_size)
                        screen.blit(scaled, (0, 0))
                        pygame.display.flip()
                        clock.tick(60)
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
        if mode == "AI vs Player":
            ai_side = options.get("ai_side", "Gold")
            if options.get("ai_file"):
                ai = load_custom_ai(options["ai_file"], game, ai_side)
            else:
                ai = RandomAI(game, ai_side)
        selected_index = None
        running = True
        while running and not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.VIDEORESIZE:
                    new_w, new_h = enforce_aspect_ratio(event.w, event.h, design_ratio)
                    current_size = (new_w, new_h)
                    screen = pygame.display.set_mode(current_size, flags)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    # Convert from current window size to design coordinates.
                    scaled_pos = (mx * design_width / current_size[0],
                                  my * design_height / current_size[1])
                    board_pos = screen_to_board(scaled_pos)
                    if board_pos:
                        idx = pos_to_index(*board_pos)
                        if selected_index is None:
                            if game.board[idx] != 0:
                                selected_index = idx
                        else:
                            legal_moves = game.get_legal_moves_for(selected_index)
                            legal_destinations = [move[1] for move in legal_moves]
                            if idx in legal_destinations:
                                move = [move for move in legal_moves if move[1] == idx][0]
                                game.make_move(move)
                                selected_index = None
                            else:
                                selected_index = None
            legal_destinations = None
            if selected_index is not None:
                legal_destinations = [move[1] for move in game.get_legal_moves_for(selected_index)]
            game.update()
            draw_board(base_surface, game, assets, bg, selected_index, legal_destinations)
            scaled = pygame.transform.smoothscale(base_surface, current_size)
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()
