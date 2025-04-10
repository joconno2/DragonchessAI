import os
import pygame

# ===== Global Layout Variables (tweak these for debugging/layout) =====
DESIGN_WIDTH = 720                # Overall design width
DESIGN_HEIGHT = 1280              # Overall design height
DESIGN_RESOLUTION = (DESIGN_WIDTH, DESIGN_HEIGHT)

BG_SCALE_X = 1  # Scale factor for width
BG_SCALE_Y = 1   # Scale factor for height
BACKGROUND_SCALE = 1            # Scale factor for the background image (e.g., 0.8)
OFFSET_ADJUST = 100                # Fixed offset adjustment to nudge the background up/left

CELL_SIZE = 40                    # Size for each board cell (square)

BOARD_COLS = 12                   # Number of columns per board
BOARD_ROWS = 8                    # Number of rows per board
NUM_BOARDS = 3                   # Total boards (stacked vertically)

BOARD_GAP = 37                    # Vertical gap between boards
BOARD_AREA_WIDTH = BOARD_COLS * CELL_SIZE
BOARD_AREA_HEIGHT = NUM_BOARDS * BOARD_ROWS * CELL_SIZE + (NUM_BOARDS - 1) * BOARD_GAP

SIDE_PANEL_WIDTH = 185            # Width reserved for the move list panel (right side)
TEXTBOX_HEIGHT = 200              # Height reserved for the text box at the bottom

MOVE_FONT_SIZE = 34  # Increase as desired
TEXTBOX_FONT_SIZE = 24  # Increase as desired

MOVE_FONT_SIZE = 24            # Font size for the side panel move list text
SIDE_PANEL_SCALE_X = 1         # Horizontal scale factor for the side panel image
SIDE_PANEL_SCALE_Y = 1         # Vertical scale factor for the side panel image

BOTTOM_PANEL_SCALE_X = 1       # Horizontal scale factor for the bottom panel image
BOTTOM_PANEL_SCALE_Y = 1       # Vertical scale factor for the bottom panel image
TEXTBOX_FONT_SIZE = 24         # Font size for the bottom panel text

# ===== Colors =====
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LINE_COLOR = (0, 0, 0)
PANEL_BG = (40, 40, 40)
TEXTBOX_BG = (58, 31, 28)

# ===== Asset Names Mapping =====
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

# ===== Screen-to-Board Mapping =====
def screen_to_board(design_pos, board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """
    Convert a click position (x,y) in design space to a (layer, row, col) tuple.
    The boards are drawn centered relative to the background image which is scaled by BACKGROUND_SCALE
    and nudged by OFFSET_ADJUST.
    """
    x, y = design_pos
    # Compute background dimensions and placement
    bg_width = int(DESIGN_WIDTH * BACKGROUND_SCALE)
    bg_height = int(DESIGN_HEIGHT * BACKGROUND_SCALE)
    bg_x = (DESIGN_WIDTH - bg_width) // 2 - OFFSET_ADJUST
    bg_y = (DESIGN_HEIGHT - bg_height) // 2 - OFFSET_ADJUST

    # Compute the board drawing origin within the background image.
    board_origin_x = bg_x + (bg_width - BOARD_AREA_WIDTH) // 2
    board_origin_y = bg_y + (bg_height - BOARD_AREA_HEIGHT) // 2

    # If the click is within the board area, compute its cell.
    if board_origin_x <= x < board_origin_x + BOARD_AREA_WIDTH and board_origin_y <= y < board_origin_y + BOARD_AREA_HEIGHT:
        relative_x = x - board_origin_x
        relative_y = y - board_origin_y
        board_unit_height = BOARD_ROWS * CELL_SIZE + BOARD_GAP
        layer = min(relative_y // board_unit_height, num_boards - 1)
        row = (relative_y - layer * board_unit_height) // CELL_SIZE  # <--- Changed line!
        col = relative_x // CELL_SIZE
        return (int(layer), int(row), int(col))
    return None


# ===== Draw the Game Screen =====
def draw_board(surface, game, assets, bg, selected_index=None, legal_destinations=None,
               board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """
    Render the complete game screen including:
      - The repositioned background image.
      - The game boards (stacked vertically) centered relative to the background.
      - A move list panel on the right.
      - A text box panel at the bottom.
    """
    # Draw background image at the computed position
    bg_width, bg_height = bg.get_width(), bg.get_height()
    bg_x = 0
    bg_y = 0
    surface.blit(bg, (bg_x, bg_y))

    
    # Compute board origin (center boards within the background image)
    board_origin_x = bg_x + (bg_width - BOARD_AREA_WIDTH) // 2
    board_origin_y = bg_y + (bg_height - BOARD_AREA_HEIGHT) // 2

    font = pygame.font.Font("assets/pixel.ttf", 20)
    frozen_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    frozen_overlay.fill((0, 150, 255, 100))
    
    # Draw each board (vertically stacked)
    for layer in range(num_boards):
        board_y_start = board_origin_y + layer * (BOARD_ROWS * CELL_SIZE + BOARD_GAP)
        for row in range(board_rows):
            for col in range(board_cols):
                rect = pygame.Rect(board_origin_x + col * CELL_SIZE,
                                   board_y_start + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                square_color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(surface, square_color, rect)
                pygame.draw.rect(surface, LINE_COLOR, rect, 1)
                from game import pos_to_index
                idx = pos_to_index(layer, row, col)
                piece = game.board[idx]
                if piece != 0:
                    asset = assets.get(piece)
                    if asset:
                        surface.blit(asset, rect.topleft)
                    else:
                        text = font.render(game.piece_letter(piece), True, (0, 0, 0))
                        text_rect = text.get_rect(center=rect.center)
                        surface.blit(text, text_rect)
                if game.frozen[idx]:
                    surface.blit(frozen_overlay, rect.topleft)
                if selected_index is not None and idx == selected_index:
                    pygame.draw.rect(surface, (0, 255, 0), rect, 3)
                if legal_destinations and idx in legal_destinations:
                    pygame.draw.rect(surface, (0, 0, 255), rect, 3)
        board_rect = pygame.Rect(board_origin_x, board_y_start, BOARD_AREA_WIDTH, BOARD_ROWS * CELL_SIZE)
        pygame.draw.rect(surface, LINE_COLOR, board_rect, 3)
    
    # --- Draw the side panel with the side_panel.png image ---
    # Load the side panel image from assets (it is loaded fresh each time; for performance you might cache it)
    side_panel_img = pygame.image.load(os.path.join("assets", "side_panel.png")).convert_alpha()
    # Scale the side panel image using the global scaling factors;
    # the target size is the panel area: width = SIDE_PANEL_WIDTH, height = DESIGN_HEIGHT - TEXTBOX_HEIGHT
    side_panel_target_width = SIDE_PANEL_WIDTH
    side_panel_target_height = DESIGN_HEIGHT - TEXTBOX_HEIGHT
    side_panel_img = pygame.transform.scale(
        side_panel_img,
        (int(side_panel_target_width * SIDE_PANEL_SCALE_X), int(side_panel_target_height * SIDE_PANEL_SCALE_Y))
    )
    # Blit the side panel image at the correct position (right side)
    side_panel_x = DESIGN_WIDTH - side_panel_target_width
    side_panel_y = 0
    surface.blit(side_panel_img, (side_panel_x, side_panel_y))

    # --- Draw the move list text on top of the side panel ---
    move_font = pygame.font.Font("assets/pixel.ttf", MOVE_FONT_SIZE)
    move_color = (0, 0, 0)  # black font color
    text_y = 70
    # Determine how many lines fit; here we approximate using MOVE_FONT_SIZE for line height (plus a little spacing)
    max_lines = (side_panel_target_height - 20) // (MOVE_FONT_SIZE + 2)
    for move_str in game.move_notations[-max_lines:]:
        txt = move_font.render(move_str, True, move_color)
        surface.blit(txt, (side_panel_x + 10, text_y))
        text_y += MOVE_FONT_SIZE + 2


    # --- Draw the bottom panel with the bottom_panel.png image ---
    # Load the bottom panel image from assets (you could cache this if desired)
    bottom_panel_img = pygame.image.load(os.path.join("assets", "bottom_panel.png")).convert_alpha()
    # Set the target dimensions to match the bottom panel area.
    bottom_panel_target_width = DESIGN_WIDTH
    bottom_panel_target_height = TEXTBOX_HEIGHT
    # Scale the bottom panel image using the global scaling factors.
    bottom_panel_img = pygame.transform.scale(
        bottom_panel_img,
        (int(bottom_panel_target_width * BOTTOM_PANEL_SCALE_X), int(bottom_panel_target_height * BOTTOM_PANEL_SCALE_Y))
    )
    # Determine the position for the bottom panel (flush at the bottom)
    bottom_panel_x = 0
    bottom_panel_y = DESIGN_HEIGHT - bottom_panel_target_height
    # Blit the bottom panel image onto the surface.
    surface.blit(bottom_panel_img, (bottom_panel_x, bottom_panel_y))
    
    # --- Draw the text on top of the bottom panel ---
    #tb_font = pygame.font.Font("assets/pixel.ttf", TEXTBOX_FONT_SIZE)
    #placeholder = tb_font.render("Enter move:", True, (0, 0, 0))
    #surface.blit(placeholder, (10, bottom_panel_y + (bottom_panel_target_height - placeholder.get_height()) // 2))
