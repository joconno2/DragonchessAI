import os
import sys
import io
import pygame
import cairosvg
from pieces import create_initial_setup

# Board/layout constants.
DEFAULT_CELL_SIZE  = 40
BOARD_COLS         = 12
BOARD_ROWS         = 8
NUM_BOARDS         = 3

# Layout and UI constants.
BOARD_LEFT_MARGIN  = 20        # Left margin for boards.
BOARD_GAP          = 20        # Gap between boards.
BOARD_TOP_MARGIN   = 60        # Top margin (for board titles).
BOARD_BOTTOM_MARGIN= 20
SIDE_PANEL_WIDTH   = 200       # Width for the move log pane.


def coord_to_algebraic(pos):
    """
    Convert board coordinate (layer, row, col) to algebraic notation.
    Board number = layer+1,
    File letter: 0 -> a, 1 -> b, …, 11 -> l,
    Rank: 8 - row (since row 0 is top and row 7 is bottom).
    """
    layer, row, col = pos
    board_num = layer + 1
    file_letter = chr(ord('a') + col)
    rank = 8 - row
    return f"{board_num}{file_letter}{rank}"

def get_piece_letter(piece):
    """
    Return the letter used for logging this piece.
    For Dragon, always use R (or r for Scarlet); for others, use the first letter.
    """
    if piece.name.lower() == "dragon":
        return "R" if piece.color == "Gold" else "r"
    else:
        letter = piece.name[0]
        return letter.upper() if piece.color == "Gold" else letter.lower()

def move_to_algebraic(piece, start, end, flag):
    """
    Convert a move into algebraic notation.
    Uses 'x' for capture/afar moves and '-' for non-capturing moves.
    """
    start_alg = coord_to_algebraic(start)
    end_alg   = coord_to_algebraic(end)
    sep = "x" if flag in ["capture", "afar"] else "-"
    return f"{get_piece_letter(piece)}{start_alg}{sep}{end_alg}"


def load_svg_image(filepath, size):
    """
    Convert the SVG file at filepath into a pygame Surface scaled to 'size',
    using cairosvg to produce PNG data in memory.
    """
    png_data = cairosvg.svg2png(url=filepath, output_width=size[0], output_height=size[1])
    image = pygame.image.load(io.BytesIO(png_data))
    return image

# --- The Game Class ---
class Game:
    def __init__(self, screen, headless=False):
        self.screen = screen
        self.headless = headless
        self.cell_size = DEFAULT_CELL_SIZE
        self.current_turn = "Gold"  # Gold moves first.
        self.selected_piece = None
        self.selected_pos = None
        self.board = {}
        self.load_board()
        self.load_assets()
        self.game_over = False
        self.winner = None
        self.game_log = []  # List of algebraic move strings.

    def load_board(self):
        # Create an empty board.
        for layer in range(NUM_BOARDS):
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    self.board[(layer, row, col)] = None
        initial_setup = create_initial_setup()
        for pos, piece in initial_setup.items():
            self.board[pos] = piece

    def load_assets(self):
        self.assets = {}
        # Determine base path: if running in a PyInstaller bundle, use sys._MEIPASS.
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        assets_dir = os.path.join(base_path, "assets")

        colors = ["gold", "scarlet"]
        piece_names = ["sylph", "griffin", "dragon", "oliphant", "unicorn", "hero",
                       "thief", "cleric", "mage", "king", "paladin", "warrior",
                       "basilisk", "elemental", "dwarf"]
        for color in colors:
            for name in piece_names:
                key = f"{color}_{name}"
                png_path = os.path.join(assets_dir, f"{key}.png")
                try:
                    image = pygame.image.load(png_path)
                    image = pygame.transform.scale(image, (self.cell_size, self.cell_size))
                    self.assets[key] = image
                except Exception as e:
                    # If image loading fails, create a placeholder.
                    surf = pygame.Surface((self.cell_size, self.cell_size))
                    surf.fill((200,200,200))
                    font = pygame.font.SysFont("Arial", 24)
                    text = font.render(key, True, (0,0,0))
                    surf.blit(text, (5,5))
                    self.assets[key] = surf


    def set_cell_size(self, new_size):
        """Update the cell size and reload assets accordingly."""
        self.cell_size = new_size
        self.load_assets()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            board_pos = self.screen_to_board(pos)
            if board_pos and not self.game_over:
                layer, row, col = board_pos
                piece = self.board.get((layer, row, col))
                if self.selected_piece:
                    legal_moves = self.get_legal_moves(self.selected_piece, self.selected_pos)
                    move_candidate = None
                    for move in legal_moves:
                        if move[0] == self.selected_pos and move[1] == board_pos:
                            move_candidate = move
                            break
                    if move_candidate:
                        self.make_move(move_candidate)
                        self.selected_piece = None
                        self.selected_pos = None
                    else:
                        if piece and piece.color == self.current_turn:
                            self.selected_piece = piece
                            self.selected_pos = board_pos
                        else:
                            self.selected_piece = None
                            self.selected_pos = None
                else:
                    if piece and piece.color == self.current_turn:
                        self.selected_piece = piece
                        self.selected_pos = board_pos

    def screen_to_board(self, pos):
        x, y = pos
        board_width = BOARD_COLS * self.cell_size
        board_height = BOARD_ROWS * self.cell_size
        for layer in range(NUM_BOARDS):
            board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
            board_x_end = board_x_start + board_width
            board_y_start = BOARD_TOP_MARGIN
            board_y_end = board_y_start + board_height
            if board_x_start <= x <= board_x_end and board_y_start <= y <= board_y_end:
                col = (x - board_x_start) // self.cell_size
                row = (y - board_y_start) // self.cell_size
                return (layer, int(row), int(col))
        return None

    def draw(self):
        if self.headless or self.screen is None:
            return
        self.screen.fill((50, 50, 50))
        board_width = BOARD_COLS * self.cell_size
        board_height = BOARD_ROWS * self.cell_size

        # Draw each board.
        for layer in range(NUM_BOARDS):
            board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
            board_y_start = BOARD_TOP_MARGIN
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    rect = pygame.Rect(board_x_start + col * self.cell_size,
                                       board_y_start + row * self.cell_size,
                                       self.cell_size, self.cell_size)
                    color = (240, 217, 181) if (row+col) % 2 == 0 else (181, 136, 99)
                    pygame.draw.rect(self.screen, color, rect)
                    if self.selected_pos == (layer, row, col):
                        pygame.draw.rect(self.screen, (0,255,0), rect, 3)
                    if self.selected_piece and self.selected_pos:
                        legal_moves = self.get_legal_moves(self.selected_piece, self.selected_pos)
                        for move in legal_moves:
                            if move[1] == (layer, row, col):
                                pygame.draw.rect(self.screen, (0,0,255), rect, 3)
            board_rect = pygame.Rect(board_x_start, board_y_start, board_width, board_height)
            pygame.draw.rect(self.screen, (0,0,0), board_rect, 3)
            # Draw board title above the board.
            titles = ["Sky", "Ground", "Underworld"]
            title_font = pygame.font.SysFont("Arial", 28)
            title_surf = title_font.render(titles[layer], True, (255,255,255))
            title_rect = title_surf.get_rect(center=(board_x_start + board_width//2, BOARD_TOP_MARGIN//2))
            self.screen.blit(title_surf, title_rect)
        # Draw pieces.
        for pos, piece in self.board.items():
            if piece:
                layer, row, col = pos
                board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
                board_y_start = BOARD_TOP_MARGIN
                rect = pygame.Rect(board_x_start + col * self.cell_size,
                                   board_y_start + row * self.cell_size,
                                   self.cell_size, self.cell_size)
                img = self.assets.get(piece.symbol)
                # Draw the piece.
                if img:
                    self.screen.blit(img, rect.topleft)
                else:
                    piece_font = pygame.font.SysFont("Arial", 24)
                    text = piece_font.render(piece.symbol, True, (0,0,0))
                    self.screen.blit(text, rect.topleft)

                # If the piece is frozen, overlay a subtle blue tint.
                if hasattr(piece, "frozen") and piece.frozen:
                    overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    overlay.fill((0, 150, 255, 100))  # RGBA: blue with 100/255 alpha
                    self.screen.blit(overlay, rect.topleft)

        # Draw move log pane.
        total_width = self.screen.get_width()
        total_height = self.screen.get_height()
        pane_rect = pygame.Rect(total_width - SIDE_PANEL_WIDTH, 0, SIDE_PANEL_WIDTH, total_height)
        pygame.draw.rect(self.screen, (30, 30, 30), pane_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), pane_rect, 3)
        log_title_font = pygame.font.SysFont("Arial", 24)
        log_font = pygame.font.SysFont("Arial", 20)
        title_surf = log_title_font.render("Game Log", True, (255,255,255))
        self.screen.blit(title_surf, (pane_rect.x + 10, 10))
        y_offset = 40
        line_spacing = 22
        for move in self.game_log[-int((total_height - y_offset) // line_spacing):]:
            move_surf = log_font.render(move, True, (200,200,200))
            self.screen.blit(move_surf, (pane_rect.x + 10, y_offset))
            y_offset += line_spacing

        # If game over, overlay a game-over message.
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            game_over_font = pygame.font.SysFont("Arial", 48)
            message = f"Game Over! Winner: {self.winner}"
            text_surf = game_over_font.render(message, True, (255,255,255))
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)

    def get_legal_moves(self, piece, pos):
        # If the piece is frozen, it cannot move.
        if hasattr(piece, "frozen") and piece.frozen:
            return []
        moves = piece.get_moves(pos, self.board)
        legal_moves = []
        for move in moves:
            if len(move) == 2:
                start, end = move
                flag = "quiet"
            else:
                start, end, flag = move
            if not self.is_within_bounds(end):
                continue
            dest = self.board.get(end)
            if flag == "quiet":
                if dest is None:
                    legal_moves.append(move)
            elif flag == "capture":
                if dest is not None and dest.color != piece.color:
                    legal_moves.append(move)
            elif flag == "afar":
                if dest is not None and dest.color != piece.color:
                    legal_moves.append(move)
            elif flag in ["ambiguous", "3d"]:
                if dest is None or (dest is not None and dest.color != piece.color):
                    legal_moves.append(move)
            else:
                legal_moves.append(move)
        return legal_moves

    def is_within_bounds(self, pos):
        layer, row, col = pos
        return 0 <= layer < NUM_BOARDS and 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS

    def make_move(self, move):
        start, end, flag = move if len(move) == 3 else (move[0], move[1], "quiet")
        piece = self.board[start]
        dest_piece = self.board[end]
        alg_move = move_to_algebraic(piece, start, end, flag)
        self.game_log.append(alg_move)
        if flag == "afar":
            if dest_piece is not None and dest_piece.color != piece.color:
                self.board[end] = None
        else:
            if dest_piece is not None and dest_piece.color != piece.color:
                self.board[end] = None
            self.board[end] = piece
            self.board[start] = None
        self.current_turn = "Scarlet" if self.current_turn == "Gold" else "Gold"

    def update(self):
        # First, unfreeze all pieces on the middle board.
        for pos, piece in self.board.items():
            if piece is not None and pos[0] == 1:
                piece.frozen = False
        # Then, for every Basilisk on the bottom board, freeze the opposing piece
        # (if any) directly above it on the middle board.
        for pos, piece in self.board.items():
            if piece is not None and piece.name.lower() == "basilisk" and pos[0] == 2:
                layer, row, col = pos
                above = (1, row, col)
                target = self.board.get(above)
                if target is not None and target.color != piece.color:
                    target.frozen = True

        # --- WIN CONDITION CHECK ---
        gold_king_found = False
        scarlet_king_found = False
        for pos, piece in self.board.items():
            if piece and piece.name.lower() == "king":
                if piece.color == "Gold":
                    gold_king_found = True
                elif piece.color == "Scarlet":
                    scarlet_king_found = True
        if not gold_king_found:
            self.game_over = True
            self.winner = "Scarlet"
        elif not scarlet_king_found:
            self.game_over = True
            self.winner = "Gold"
