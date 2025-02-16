import hashlib
import os
import pygame
import numpy as np
from pieces import create_initial_setup

# Board/layout constants.
DEFAULT_CELL_SIZE   = 40
BOARD_COLS          = 12
BOARD_ROWS          = 8
NUM_BOARDS          = 3

# Layout and UI constants.
BOARD_LEFT_MARGIN   = 20
BOARD_GAP           = 20
BOARD_TOP_MARGIN    = 60
BOARD_BOTTOM_MARGIN = 20
SIDE_PANEL_WIDTH    = 200

# --- Mapping from piece symbol to integer code ---
# (Positive for Gold; negative for Scarlet.)
piece_to_int = {
    "gold_sylph":      1,
    "scarlet_sylph":  -1,
    "gold_griffin":    2,
    "scarlet_griffin": -2,
    "gold_dragon":     3,
    "scarlet_dragon": -3,
    "gold_oliphant":   4,
    "scarlet_oliphant":-4,
    "gold_unicorn":    5,
    "scarlet_unicorn": -5,
    "gold_hero":       6,
    "scarlet_hero":    -6,
    "gold_thief":      7,
    "scarlet_thief":   -7,
    "gold_cleric":     8,
    "scarlet_cleric":  -8,
    "gold_mage":       9,
    "scarlet_mage":    -9,
    "gold_king":       10,
    "scarlet_king":    -10,
    "gold_paladin":    11,
    "scarlet_paladin": -11,
    "gold_warrior":    12,
    "scarlet_warrior": -12,
    "gold_basilisk":   13,
    "scarlet_basilisk":-13,
    "gold_elemental":  14,
    "scarlet_elemental": -14,
    "gold_dwarf":      15,
    "scarlet_dwarf":   -15
}



# --- New: Fast state hash using NumPy ---
def board_state_hash_numpy(np_board, turn):
    """
    Compute a hash for the state given a NumPy array representation of the board.
    """
    state_bytes = np_board.tobytes() + turn.encode()
    return hashlib.sha256(state_bytes).hexdigest()

# --- Helper functions for Algebraic Notation ---
def coord_to_algebraic(pos):
    layer, row, col = pos
    board_num = layer + 1
    file_letter = chr(ord('a') + col)
    rank = 8 - row
    return f"{board_num}{file_letter}{rank}"

def get_piece_letter(piece):
    if piece.name.lower() == "dragon":
        return "R" if piece.color == "Gold" else "r"
    else:
        letter = piece.name[0]
        return letter.upper() if piece.color == "Gold" else letter.lower()

def move_to_algebraic(piece, start, end, flag):
    start_alg = coord_to_algebraic(start)
    end_alg   = coord_to_algebraic(end)
    sep = "x" if flag in ["capture", "afar"] else "-"
    return f"{get_piece_letter(piece)}{start_alg}{sep}{end_alg}"

# --- The Game Class ---
class Game:
    def __init__(self, screen, headless=False):
        self.screen = screen
        self.headless = headless
        self.cell_size = DEFAULT_CELL_SIZE
        self.current_turn = "Gold"
        self.selected_piece = None
        self.selected_pos = None
        self.board = {}
        self.load_board()
        self.load_assets()
        self.game_over = False
        self.winner = None
        self.game_log = []  # List of algebraic move strings.
        self.state_history = []    
        self.no_capture_count = 0  # Count consecutive moves with no capture.

    def load_board(self):
        for layer in range(NUM_BOARDS):
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    self.board[(layer, row, col)] = None
        initial_setup = create_initial_setup()
        for pos, piece in initial_setup.items():
            self.board[pos] = piece

    def load_assets(self):
        self.assets = {}
        colors = ["gold", "scarlet"]
        piece_names = ["sylph", "griffin", "dragon", "oliphant", "unicorn", "hero",
                       "thief", "cleric", "mage", "king", "paladin", "warrior",
                       "basilisk", "elemental", "dwarf"]
        for color in colors:
            for name in piece_names:
                key = f"{color}_{name}"
                png_path = os.path.join("assets", f"{key}.png")
                try:
                    image = pygame.image.load(png_path)
                    image = pygame.transform.scale(image, (self.cell_size, self.cell_size))
                    self.assets[key] = image
                except Exception as e:
                    surf = pygame.Surface((self.cell_size, self.cell_size))
                    surf.fill((200,200,200))
                    font = pygame.font.Font("font.ttf", 24)
                    text = font.render(key, True, (0,0,0))
                    surf.blit(text, (5,5))
                    self.assets[key] = surf

    def set_cell_size(self, new_size):
        self.cell_size = new_size
        self.load_assets()

    def get_numpy_board(self):
        """
        Convert the dictionary board into a NumPy array of shape (NUM_BOARDS, BOARD_ROWS, BOARD_COLS)
        where each cell contains an integer code (0 if empty).
        """
        np_board = np.zeros((NUM_BOARDS, BOARD_ROWS, BOARD_COLS), dtype=np.int16)
        for pos, piece in self.board.items():
            layer, row, col = pos
            if piece:
                # Use our mapping. We assume piece.symbol returns keys like "gold_sylph".
                np_board[layer, row, col] = piece_to_int.get(piece.symbol, 0)
            else:
                np_board[layer, row, col] = 0
        return np_board

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
            titles = ["Sky", "Ground", "Underworld"]
            title_font = pygame.font.Font("font.ttf", 34)
            title_surf = title_font.render(titles[layer], True, (255,255,255))
            title_rect = title_surf.get_rect(center=(board_x_start + board_width//2, BOARD_TOP_MARGIN//2))
            self.screen.blit(title_surf, title_rect)
        for pos, piece in self.board.items():
            if piece:
                layer, row, col = pos
                board_x_start = BOARD_LEFT_MARGIN + layer * (board_width + BOARD_GAP)
                board_y_start = BOARD_TOP_MARGIN
                rect = pygame.Rect(board_x_start + col * self.cell_size,
                                   board_y_start + row * self.cell_size,
                                   self.cell_size, self.cell_size)
                img = self.assets.get(piece.symbol)
                if img:
                    self.screen.blit(img, rect.topleft)
                else:
                    piece_font = pygame.font.Font("font.ttf", 26)
                    text = piece_font.render(piece.symbol, True, (0,0,0))
                    self.screen.blit(text, rect.topleft)
                if hasattr(piece, "frozen") and piece.frozen:
                    overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    overlay.fill((0, 150, 255, 100))
                    self.screen.blit(overlay, rect.topleft)
        total_width = self.screen.get_width()
        total_height = self.screen.get_height()
        pane_rect = pygame.Rect(total_width - SIDE_PANEL_WIDTH, 0, SIDE_PANEL_WIDTH, total_height)
        pygame.draw.rect(self.screen, (30, 30, 30), pane_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), pane_rect, 3)
        log_title_font = pygame.font.Font("font.ttf", 30)
        log_font = pygame.font.SysFont("Arial", 20)
        title_surf = log_title_font.render("Game Log", True, (255,255,255))
        self.screen.blit(title_surf, (pane_rect.x + 10, 10))
        y_offset = 40
        line_spacing = 22
        for move in self.game_log[-int((total_height - y_offset) // line_spacing):]:
            move_surf = log_font.render(move, True, (200,200,200))
            self.screen.blit(move_surf, (pane_rect.x + 10, y_offset))
            y_offset += line_spacing

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            game_over_font = pygame.font.Font("font.ttf", 48)
            message = f"Game Over! Winner: {self.winner}"
            text_surf = game_over_font.render(message, True, (255,255,255))
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)

    def get_legal_moves(self, piece, pos):
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
        capture_occurred = False
        if flag == "afar":
            if dest_piece is not None and dest_piece.color != piece.color:
                self.board[end] = None
                capture_occurred = True
        else:
            if dest_piece is not None and dest_piece.color != piece.color:
                self.board[end] = None
                capture_occurred = True
            self.board[end] = piece
            self.board[start] = None
        if capture_occurred:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1
        # Use the new fast NumPy board hash.
        np_board = self.get_numpy_board()
        current_state_hash = board_state_hash_numpy(np_board, self.current_turn)
        self.state_history.append(current_state_hash)
        self.current_turn = "Scarlet" if self.current_turn == "Gold" else "Gold"

    def update(self):
        for pos, piece in self.board.items():
            if piece is not None and pos[0] == 1:
                piece.frozen = False
        for pos, piece in self.board.items():
            if piece is not None and piece.name.lower() == "basilisk" and pos[0] == 2:
                layer, row, col = pos
                above = (1, row, col)
                target = self.board.get(above)
                if target is not None and target.color != piece.color:
                    target.frozen = True
        if self.no_capture_count >= 50:
            self.game_over = True
            self.winner = "Draw"
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

    def get_numpy_board(self):
        np_board = np.zeros((NUM_BOARDS, BOARD_ROWS, BOARD_COLS), dtype=np.int16)
        for pos, piece in self.board.items():
            layer, row, col = pos
            if piece:
                np_board[layer, row, col] = piece_to_int.get(piece.symbol, 0)
            else:
                np_board[layer, row, col] = 0
        return np_board
