import os
import pygame

# Shared constants.
CELL_SIZE = 40
BOARD_GAP = 50
BOARD_LEFT_MARGIN = 50
BOARD_TOP_MARGIN = 120
SIDE_PANEL_WIDTH = 200

# Board dimensions.
BOARD_ROWS = 8
BOARD_COLS = 12
NUM_BOARDS = 3

# Colors.
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LINE_COLOR = (0, 0, 0)

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

def screen_to_board(design_pos, board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """Convert a position in design-space (x,y) to a (layer, row, col) tuple."""
    x, y = design_pos
    board_width = board_cols * CELL_SIZE
    board_height = board_rows * CELL_SIZE
    for layer in range(num_boards):
        board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
        board_y_start = BOARD_TOP_MARGIN
        if board_x_start <= x < board_x_start + board_width and board_y_start <= y < board_y_start + board_height:
            col = int((x - board_x_start) // CELL_SIZE)
            row = int((y - board_y_start) // CELL_SIZE)
            return (layer, row, col)
    return None

def draw_board(surface, game, assets, bg, selected_index=None, legal_destinations=None,
               board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """Render the game board onto the provided surface."""
    surface.blit(bg, (0, 0))
    board_width = board_cols * CELL_SIZE
    board_height = board_rows * CELL_SIZE
    font = pygame.font.Font("assets/pixel.ttf", 20)
    frozen_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    frozen_overlay.fill((0,150,255,100))
    # Import pos_to_index dynamically.
    from game import pos_to_index  
    for layer in range(num_boards):
        board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
        board_y_start = BOARD_TOP_MARGIN
        for row in range(board_rows):
            for col in range(board_cols):
                rect = pygame.Rect(board_x_start + col * CELL_SIZE,
                                   board_y_start + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                square_color = LIGHT_SQUARE if (row+col) % 2 == 0 else DARK_SQUARE
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
        title_rect = title_text.get_rect(center=(board_x_start + board_width//2, BOARD_TOP_MARGIN//2))
        surface.blit(title_text, title_rect)
    total_width = surface.get_width()
    total_height = surface.get_height()
    pane_rect = pygame.Rect(total_width - SIDE_PANEL_WIDTH, 0, SIDE_PANEL_WIDTH, total_height)
    pygame.draw.rect(surface, (30,30,30), pane_rect)
    pygame.draw.rect(surface, LINE_COLOR, pane_rect, 3)
    log_font = pygame.font.Font("assets/pixel.ttf", 24)
    y_offset = 10
    for move_str in game.move_notations[-int((total_height-y_offset)/20):]:
        text = log_font.render(move_str, True, (200,200,200))
        surface.blit(text, (pane_rect.x+5, y_offset))
        y_offset += 20

def window_to_design(mouse_pos, design_resolution):
    """Convert window mouse coordinates to design-surface coordinates.
       Returns None if mouse is outside design area (letterbox)."""
    window_width, window_height = pygame.display.get_surface().get_size()
    design_width, design_height = design_resolution
    scale = min(window_width/design_width, window_height/design_height)
    offset_x = (window_width - design_width*scale) // 2
    offset_y = (window_height - design_height*scale) // 2
    mx, my = mouse_pos
    if mx < offset_x or mx > offset_x + design_width*scale or my < offset_y or my > offset_y + design_height*scale:
        return None
    design_x = (mx - offset_x) / scale
    design_y = (my - offset_y) / scale
    return (design_x, design_y)
