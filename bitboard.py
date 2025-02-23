import numpy as np
from numba import njit

# Board dimensions (constants)
NUM_BOARDS = 3
BOARD_ROWS = 8
BOARD_COLS = 12
TOTAL_SQUARES = NUM_BOARDS * BOARD_ROWS * BOARD_COLS

# Piece constants (positive for Gold; negative for Scarlet)
GOLD_SYLPH       = 1
GOLD_GRIFFIN     = 2
GOLD_DRAGON      = 3
GOLD_OLIPHANT    = 4
GOLD_UNICORN     = 5
GOLD_HERO        = 6
GOLD_THIEF       = 7
GOLD_CLERIC      = 8
GOLD_MAGE        = 9
GOLD_KING        = 10
GOLD_PALADIN     = 11
GOLD_WARRIOR     = 12
GOLD_BASILISK    = 13
GOLD_ELEMENTAL   = 14
GOLD_DWARF       = 15

SCARLET_SYLPH    = -1
SCARLET_GRIFFIN  = -2
SCARLET_DRAGON   = -3
SCARLET_OLIPHANT = -4
SCARLET_UNICORN  = -5
SCARLET_HERO     = -6
SCARLET_THIEF    = -7
SCARLET_CLERIC   = -8
SCARLET_MAGE     = -9
SCARLET_KING     = -10
SCARLET_PALADIN  = -11
SCARLET_WARRIOR  = -12
SCARLET_BASILISK = -13
SCARLET_ELEMENTAL= -14
SCARLET_DWARF    = -15

@njit
def pos_to_index(layer, row, col):
    """Convert (layer, row, col) to a flat index [0, TOTAL_SQUARES)."""
    return layer * (BOARD_ROWS * BOARD_COLS) + row * BOARD_COLS + col

@njit
def index_to_pos(index):
    """Convert a flat index into (layer, row, col)."""
    layer = index // (BOARD_ROWS * BOARD_COLS)
    rem = index % (BOARD_ROWS * BOARD_COLS)
    row = rem // BOARD_COLS
    col = rem % BOARD_COLS
    return layer, row, col

def create_initial_board():
    """
    Create and return a NumPy array of shape (TOTAL_SQUARES,)
    representing the starting board.
    """
    board = np.zeros(TOTAL_SQUARES, dtype=np.int16)
    # --- TOP Board (layer 0) ---
    board[pos_to_index(0, 0, 2)]  = SCARLET_GRIFFIN
    board[pos_to_index(0, 0, 6)]  = SCARLET_DRAGON
    board[pos_to_index(0, 0, 10)] = SCARLET_GRIFFIN
    for col in range(0, BOARD_COLS, 2):
        board[pos_to_index(0, 1, col)] = SCARLET_SYLPH
    for col in range(0, BOARD_COLS, 2):
        board[pos_to_index(0, 6, col)] = GOLD_SYLPH
    board[pos_to_index(0, 7, 2)]  = GOLD_GRIFFIN
    board[pos_to_index(0, 7, 6)]  = GOLD_DRAGON
    board[pos_to_index(0, 7, 10)] = GOLD_GRIFFIN

    # --- MIDDLE Board (layer 1) ---
    scarlet_middle = [SCARLET_OLIPHANT, SCARLET_UNICORN, SCARLET_HERO, SCARLET_THIEF,
                      SCARLET_CLERIC, SCARLET_MAGE, SCARLET_KING, SCARLET_PALADIN,
                      SCARLET_THIEF, SCARLET_HERO, SCARLET_UNICORN, SCARLET_OLIPHANT]
    for col, piece in enumerate(scarlet_middle):
        board[pos_to_index(1, 0, col)] = piece
    for col in range(BOARD_COLS):
        board[pos_to_index(1, 1, col)] = SCARLET_WARRIOR
    for col in range(BOARD_COLS):
        board[pos_to_index(1, 6, col)] = GOLD_WARRIOR
    gold_middle = [GOLD_OLIPHANT, GOLD_UNICORN, GOLD_HERO, GOLD_THIEF,
                   GOLD_CLERIC, GOLD_MAGE, GOLD_KING, GOLD_PALADIN,
                   GOLD_THIEF, GOLD_HERO, GOLD_UNICORN, GOLD_OLIPHANT]
    for col, piece in enumerate(gold_middle):
        board[pos_to_index(1, 7, col)] = piece

    # --- BOTTOM Board (layer 2) ---
    board[pos_to_index(2, 0, 2)]  = SCARLET_BASILISK
    board[pos_to_index(2, 0, 6)]  = SCARLET_ELEMENTAL
    board[pos_to_index(2, 0, 10)] = SCARLET_BASILISK
    for col in range(1, BOARD_COLS, 2):
        board[pos_to_index(2, 1, col)] = SCARLET_DWARF
    board[pos_to_index(2, 7, 2)]  = GOLD_BASILISK
    board[pos_to_index(2, 7, 6)]  = GOLD_ELEMENTAL
    board[pos_to_index(2, 7, 10)] = GOLD_BASILISK
    for col in range(1, BOARD_COLS, 2):
        board[pos_to_index(2, 6, col)] = GOLD_DWARF

    return board
