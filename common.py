import os
import pygame

# ===== Global Layout Variables =====
DESIGN_WIDTH = 1825                # Horizontal resolution
DESIGN_HEIGHT = 660                # Height for 16:9 aspect ratio
DESIGN_RESOLUTION = (DESIGN_WIDTH, DESIGN_HEIGHT)

BG_SCALE_X = 1                    # Background scale factors
BG_SCALE_Y = 1
BACKGROUND_SCALE = 1              # Background image scaling
OFFSET_ADJUST = 50                # Fixed offset adjustment

CELL_SIZE = 40                    # Size of each board cell
BOARD_COLS = 12                   # Columns per board
BOARD_ROWS = 8                    # Rows per board
NUM_BOARDS = 3                    # Total boards (arranged horizontally)
BOARD_GAP = 60                    # Gap between boards (horizontal gap)
BOARD_OFFSET_ADJUST = -50          # Offset adjustment for board origin

SIDE_PANEL_WIDTH = 185            # Width reserved for the move list panel (right side)
TEXTBOX_HEIGHT = 200              # Height reserved for the text box at the bottom

MOVE_FONT_SIZE = 34  
TEXTBOX_FONT_SIZE = 24  

MOVE_FONT_SIZE = 24            # Font size for the side panel move list text
SIDE_PANEL_SCALE_X = 1         # Horizontal scale factor for the side panel image
SIDE_PANEL_SCALE_Y = 1         # Vertical scale factor for the side panel image

BOTTOM_PANEL_SCALE_X = 1       # Horizontal scale factor for the bottom panel image
BOTTOM_PANEL_SCALE_Y = 1       # Vertical scale factor for the bottom panel image
TEXTBOX_FONT_SIZE = 24         # Font size for the bottom panel text


# Compute total board area for horizontal layout.
BOARD_AREA_WIDTH = NUM_BOARDS * BOARD_COLS * CELL_SIZE + (NUM_BOARDS - 1) * BOARD_GAP
BOARD_AREA_HEIGHT = BOARD_ROWS * CELL_SIZE

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

def screen_to_board(design_pos, board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """
    Converts a click position (x, y) in design space into a (layer, row, col) tuple,
    for the horizontal board layout with a right-side panel.
    
    In this version, a vertical click offset (VERTICAL_CLICK_OFFSET) is added to the y coordinate,
    so that clicks register at the expected vertical location.
    
    Returns None if the click is outside the board area or falls within a gap.
    """
    # Adjust this value until the click aligns correctly.
    VERTICAL_CLICK_OFFSET = 100
    
    x, y = design_pos
    # Apply the vertical offset.
    y += VERTICAL_CLICK_OFFSET

    # Compute the width available for boards (i.e. the game area).
    game_area_width = DESIGN_WIDTH - SIDE_PANEL_WIDTH

    # Center the board area (which is BOARD_AREA_WIDTH wide) within the game area.
    board_origin_x = (game_area_width - BOARD_AREA_WIDTH) // 2
    board_origin_y = (DESIGN_HEIGHT - BOARD_AREA_HEIGHT) // 2 - BOARD_OFFSET_ADJUST

    # Check if the click (with adjusted y) falls within the board area.
    if board_origin_x <= x < board_origin_x + BOARD_AREA_WIDTH and \
       board_origin_y <= y < board_origin_y + BOARD_AREA_HEIGHT:
        relative_x = x - board_origin_x
        relative_y = y - board_origin_y
        # Each board occupies a slot with width = (BOARD_COLS * CELL_SIZE + BOARD_GAP).
        slot_width = BOARD_COLS * CELL_SIZE + BOARD_GAP
        layer = int(relative_x // slot_width)
        within_layer_x = relative_x - layer * slot_width
        # If the click falls in the gap between boards, ignore it.
        if within_layer_x >= BOARD_COLS * CELL_SIZE:
            return None
        col = int(within_layer_x // CELL_SIZE)
        row = int(relative_y // CELL_SIZE)
        return (layer, row, col)
    return None




# ===== Draw the Game Screen for Horizontal Layout =====
def draw_board(surface, game, assets, bg, selected_index=None, legal_destinations=None,
               board_cols=BOARD_COLS, board_rows=BOARD_ROWS, num_boards=NUM_BOARDS):
    """
    Renders the complete game board onto a surface.
    The background is drawn, and then the boards are rendered side-by-side (horizontally)
    centered within the background image.
    """
    # Draw background image.
    bg_width, bg_height = bg.get_width(), bg.get_height()
    bg_x = 0
    bg_y = 0
    surface.blit(bg, (bg_x, bg_y))

    # Compute board origin to center the horizontal board area.
    board_origin_x = bg_x + (bg_width - BOARD_AREA_WIDTH) // 2
    board_origin_y = bg_y + (bg_height - BOARD_AREA_HEIGHT) // 2 - BOARD_OFFSET_ADJUST

    font = pygame.font.Font("assets/pixel.ttf", 20)
    frozen_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    frozen_overlay.fill((0, 150, 255, 100))

    # Iterate over each board (layer) and draw its cells.
    for layer in range(num_boards):
        board_x_start = board_origin_x + layer * (BOARD_COLS * CELL_SIZE + BOARD_GAP)
        board_y_start = board_origin_y
        for row in range(board_rows):
            for col in range(board_cols):
                rect = pygame.Rect(board_x_start + col * CELL_SIZE,
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
                # Highlight selected piece and legal moves.
                if selected_index == idx:
                    pygame.draw.rect(surface, (0, 255, 0), rect, 3)
                if legal_destinations and idx in legal_destinations:
                    pygame.draw.rect(surface, (255, 0, 0), rect, 3)
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
    text_y = 40
    # Determine how many lines fit; here we approximate using MOVE_FONT_SIZE for line height (plus a little spacing)
    max_lines = (side_panel_target_height - 80) // (MOVE_FONT_SIZE + 2)
    for move_str in game.move_notations[-max_lines:]:
        txt = move_font.render(move_str, True, move_color)
        surface.blit(txt, (side_panel_x + 30, text_y))
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

    # Define the board names for each board layer.
    board_names = ["Sky", "Ground", "Underworld"]

    # Load the custom font (you can adjust the font size as needed).
    board_name_font = pygame.font.Font("assets/font.ttf", 45)

    # Loop over each board (layer) and draw its name above the board.
    for layer in range(NUM_BOARDS):
        # Compute the starting x-coordinate for this board.
        board_x_start = board_origin_x + layer * (BOARD_COLS * CELL_SIZE + BOARD_GAP)
        # Render the board name text.
        name_text = board_name_font.render(board_names[layer], True, (255, 255, 255))
        # Calculate an x position to center the text above the board.
        text_x = board_x_start + (BOARD_COLS * CELL_SIZE) // 2 - name_text.get_width() // 2
        # Position the text slightly above the board (adjust the offset as needed).
        text_y = board_origin_y - name_text.get_height() - 45
        # Blit (draw) the board name text onto the surface.
        surface.blit(name_text, (text_x, text_y))
