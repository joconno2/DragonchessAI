#!/usr/bin/env python3
"""
dragonfish.py

This module implements an evaluation function modeled on Stockfish’s pre‐NNUE evaluation – using numeric constants adapted from Stockfish’s piece values – for the Dragonchess domain.
It works with your engine’s flat NumPy board (from bitboard.py) and game state.

A position is represented as a dict with:
  • "board": a flat NumPy array (length = TOTAL_SQUARES) as created by create_initial_board().
  • "turn": "Gold" or "Scarlet".
  • "no_capture_count": an integer.

This module defines evaluation components, a 25‑parameter decoding function (14 parameters for piece value scaling and 11 for positional weights), a main evaluation function (with debug printing), search functions (alphabeta and iterative deepening), and a CustomAI class.
It also provides a helper function load_weights(filename) so that evolved parameters can be read from a file.

USAGE EXAMPLE:
    from game import Game
    from bots.dragonfish import CustomAI, load_weights
    game = Game()
    ai_bot = CustomAI(game, "Gold", weights_file="best_weights.txt")
    move = ai_bot.choose_move()
    print("Chosen move:", move)
"""

import math
import copy
import numpy as np
import time
from collections import namedtuple
from numba import njit

# --- Global Debug Flag ---
DEBUG = True

# --- Move Flag Constants ---
QUIET = 0
CAPTURE = 1
AFAR = 2
AMBIGUOUS = 3
THREED = 4

from bitboard import pos_to_index, index_to_pos, BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES

# --- Precompute index -> (layer, row, col) mapping for speed ---
INDEX_TO_POS = [index_to_pos(i) for i in range(TOTAL_SQUARES)]

# --- Stockfish-Inspired Baseline Piece Values for Dragonchess ---
# We map Dragonchess pieces to standard chess pieces as follows:
#   Sylph (1) -> Pawn (100)
#   Griffin (2) -> Knight (320)
#   Dragon (3) -> Queen (900)
#   Oliphant (4) -> Rook (500)
#   Unicorn (5) -> Bishop (330)
#   Hero (6) -> Knight (320)
#   Thief (7) -> Pawn (100)
#   Cleric (8) -> Bishop (330)
#   Mage (9) -> Bishop (330)
#   King (10) -> King (20000)
#   Paladin (11) -> Rook (500)
#   Warrior (12) -> Pawn (100)
#   Basilisk (13) -> Knight (320)
#   Elemental (14) -> Bishop (330)
#   Dwarf (15) -> Pawn (100)
stockfish_original = {
    1: 100,
    2: 320,
    3: 900,
    4: 500,
    5: 330,
    6: 320,
    7: 100,
    8: 330,
    9: 330,
    10: 20000,  # King
    11: 500,
    12: 100,
    13: 320,
    14: 330,
    15: 100
}

# --- Data Structures ---
Square3D = namedtuple("Square3D", ["layer", "x", "y"])

# --- Game-to-Position Conversion ---
def position_from_game(game):
    return {
        "board": game.board,
        "turn": game.current_turn,
        "no_capture_count": game.no_capture_count
    }

# --- Helper Functions ---
def phase(pos):
    # Use stockfish_original as phase weights (ignoring king which is fixed)
    phase_weights = {
        1: 100, 2: 320, 3: 900, 4: 500, 5: 330,
        6: 320, 7: 100, 8: 330, 9: 330, 10: 0,
        11: 500, 12: 100, 13: 320, 14: 330, 15: 100
    }
    total = 0.0
    board = pos["board"]
    for idx in range(TOTAL_SQUARES):
        piece = board[idx]
        if piece:
            total += phase_weights.get(abs(piece), 0)
    initial_phase = sum(phase_weights.values())
    phase_val = int(round(128 * (total / initial_phase)))
    return max(0, min(128, phase_val))

def rule50(pos):
    nc = pos.get("no_capture_count", 0)
    return min(100, int(nc * 100 / 250))

def scale_factor(pos, val):
    return 64

def tempo(pos):
    return 10 if pos.get("turn", "Gold") == "Gold" else -10

def colorflip(pos):
    new_board = np.copy(pos["board"])
    new_board *= -1
    new_turn = "Gold" if pos.get("turn", "Gold") == "Scarlet" else "Scarlet"
    return {"board": new_board, "turn": new_turn, "no_capture_count": pos.get("no_capture_count", 0)}

def board_at(pos, layer, row, col):
    return pos["board"][pos_to_index(layer, row, col)]

def board_sum_all(pos, func, *args, **kwargs):
    total = 0
    for idx, (layer, row, col) in enumerate(INDEX_TO_POS):
        sq = Square3D(layer, col, row)
        total += func(pos, sq, *args, **kwargs)
    return total

def board_sum_middle(pos, func, *args, **kwargs):
    total = 0
    for row in range(BOARD_ROWS):
        for col in range(min(BOARD_COLS, 8)):
            sq = Square3D(1, col, row)
            total += func(pos, sq, *args, **kwargs)
    return total

# --- New Helper Function: Board Activity Penalty ---
def board_activity_penalty(pos):
    """
    Returns a penalty for pieces not on the middle board.
    For every Gold piece (positive value) on a board other than the middle (layer 1), subtract 20.
    For every Scarlet piece (negative value) off the middle board, add 20.
    """
    penalty = 0
    board = pos["board"]
    for idx, (layer, row, col) in enumerate(INDEX_TO_POS):
        piece = board[idx]
        if piece != 0 and layer != 1:
            if piece > 0:
                penalty -= 2000
            else:
                penalty += 2000
    return penalty

# --- Mapping of Dragonchess Pieces ---
piece_letter_map = {
    1: 'P', 2: 'N', 3: 'Q', 4: 'R', 5: 'B',
    6: 'N', 7: 'P', 8: 'B', 9: 'B', 10: 'K',
    11: 'R', 12: 'P', 13: 'N', 14: 'B', 15: 'P'
}

def map_piece(piece):
    if piece == 0:
        return None
    return piece_letter_map.get(abs(piece), None)

# --- Material Evaluation ---
# Use the best-effort transfer from Stockfish’s piece values.
def piece_value_bonus(pos, sq, mg):
    p = board_at(pos, sq.layer, sq.y, sq.x)
    letter = map_piece(p)
    if letter is None or letter == 'K':
        return 0
    # Use a conversion: Pawn, Knight, Bishop, Rook, Queen based on stockfish_original.
    value_map = {
        'P': stockfish_original[1],
        'N': stockfish_original[2],
        'B': stockfish_original[5],  # Using Unicorn's value (330)
        'R': stockfish_original[4],  # Using Oliphant's value (500)
        'Q': stockfish_original[3]   # Dragon as Queen (900)
    }
    return value_map.get(letter, 0)

def piece_value_mg(pos, sq=None):
    if sq is None:
        return board_sum_all(pos, piece_value_mg)
    return piece_value_bonus(pos, sq, mg=True)

# --- Piece-Square Table Evaluation ---
def psqt_bonus(pos, sq, mg):
    if mg:
        bonus = [
            [[-175,-92,-74,-73],[-77,-41,-27,-15],[-61,-17,6,12],[-35,8,40,49],[-34,13,44,51],[-9,22,58,53],[-67,-27,4,37],[-201,-83,-56,-26]],
            [[-53,-5,-8,-23],[-15,8,19,4],[-7,21,-5,17],[-5,11,25,39],[-12,29,22,31],[-16,6,1,11],[-17,-14,5,0],[-48,1,-14,-23]],
            [[-31,-20,-14,-5],[-21,-13,-8,6],[-25,-11,-1,3],[-13,-5,-4,-6],[-27,-15,-4,3],[-22,-2,6,12],[-2,12,16,18],[-17,-19,-1,9]],
            [[3,-5,-5,4],[-3,5,8,12],[-3,6,13,7],[4,5,9,8],[0,14,12,5],[-4,10,6,8],[-5,6,10,8],[-2,-2,1,-2]],
            [[271,327,271,198],[278,303,234,179],[195,258,169,120],[164,190,138,98],[154,179,105,70],[123,145,81,31],[88,120,65,33],[59,89,45,-1]]
        ]
        pbonus = [
            [0,0,0,0,0,0,0,0],
            [3,3,10,19,16,19,7,-5],
            [-9,-15,11,15,32,22,5,-22],
            [-4,-23,6,20,40,17,4,-8],
            [13,0,-13,1,11,-2,-13,5],
            [5,-12,-7,22,-8,-5,-15,-8],
            [-7,7,-3,-13,5,-16,10,-8],
            [0,0,0,0,0,0,0,0]
        ]
    else:
        bonus = [
            [[-96,-65,-49,-21],[-67,-54,-18,8],[-40,-27,-8,29],[-35,-2,13,28],[-45,-16,9,39],[-51,-44,-16,17],[-69,-50,-51,12],[-100,-88,-56,-17]],
            [[-57,-30,-37,-12],[-37,-13,-17,1],[-16,-1,-2,10],[-20,-6,0,17],[-17,-1,-14,15],[-30,6,4,6],[-31,-20,-1,1],[-46,-42,-37,-24]],
            [[-9,-13,-10,-9],[-12,-9,-1,-2],[6,-8,-2,-6],[-6,1,-9,7],[-5,8,7,-6],[6,1,-7,10],[4,5,20,-5],[18,0,19,13]],
            [[-69,-57,-47,-26],[-55,-31,-22,-4],[-39,-18,-9,3],[-23,-3,13,24],[-29,-6,9,21],[-38,-18,-12,1],[-50,-27,-24,-8],[-75,-52,-43,-36]],
            [[1,45,85,76],[53,100,133,135],[88,130,169,175],[103,156,172,172],[96,166,199,199],[92,172,184,191],[47,121,116,131],[11,59,73,78]]
        ]
        pbonus = [
            [0,0,0,0,0,0,0,0],
            [-10,-6,10,0,14,7,-5,-19],
            [-10,-10,-10,4,4,3,-6,-4],
            [6,-2,-8,-4,-13,-12,-10,-9],
            [10,5,4,-5,-5,-5,14,9],
            [28,20,21,28,30,7,6,13],
            [0,-11,12,21,25,19,4,7],
            [0,0,0,0,0,0,0,0]
        ]
    p = board_at(pos, sq.layer, sq.y, sq.x)
    letter = map_piece(p)
    if letter is None or letter not in "PNBRQK":
        return 0
    i = "PNBRQK".index(letter)
    if i < 0:
        return 0
    if i == 0:
        return pbonus[7 - sq.y][sq.x]
    else:
        return bonus[i-1][7 - sq.y][min(sq.x, 7 - sq.x)]

def psqt_mg(pos, sq=None, mg=True):
    if sq is None:
        return board_sum_middle(pos, psqt_mg, mg=mg)
    return psqt_bonus(pos, sq, mg=mg)

# For simplicity, we use the same PSQT evaluation at all stages.
psqt_eval = psqt_mg

# --- Positional Heuristics (applied only on the middle board) ---
def pawns_mg(pos, sq=None):
    board = pos["board"]
    count = 0
    for row in range(BOARD_ROWS):
        for col in range(min(BOARD_COLS, 8)):
            p = board[pos_to_index(1, row, col)]
            if p == 12:
                count += 1
            elif p == -12:
                count -= 1
    return 100 * count

def passed_mg(pos, sq=None):
    board = pos["board"]
    count = 0
    for row in range(BOARD_ROWS):
        for col in range(min(BOARD_COLS, 8)):
            p = board[pos_to_index(1, row, col)]
            if p == 12:
                count += (BOARD_ROWS - row)
            elif p == -12:
                count -= (row + 1)
    return count

def mobility_mg(pos, sq=None):
    return mobility_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

def threats_mg(pos):
    return threats_on_board(pos["board"], BOARD_ROWS, min(BOARD_COLS, 8))

def king_mg(pos, sq=None):
    return king_safety_on_board(pos["board"], BOARD_ROWS, min(BOARD_COLS, 8))

def space(pos, sq=None):
    board = pos["board"]
    empty = np.count_nonzero(board == 0)
    return empty / TOTAL_SQUARES * 100

def imbalance_total(pos, sq=None):
    board = pos["board"]
    gold_bishops = 0
    scarlet_bishops = 0
    for row in range(BOARD_ROWS):
        for col in range(min(BOARD_COLS, 8)):
            p = board[pos_to_index(1, row, col)]
            if p > 0 and map_piece(p) == 'B':
                gold_bishops += 1
            elif p < 0 and map_piece(p) == 'B':
                scarlet_bishops += 1
    bonus = 50
    return bonus * ((gold_bishops >= 2) - (scarlet_bishops >= 2))

# --- Numba-Accelerated Helper Functions ---
@njit
def mobility_on_board(board, total_squares, board_rows, board_cols):
    mobility_gold = 0
    mobility_scarlet = 0
    for row in range(board_rows):
        for col in range(min(board_cols, 8)):
            idx = (1 * (board_rows * board_cols)) + row * board_cols + col
            p = board[idx]
            if p == 0:
                continue
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    new_row = row + dr
                    new_col = col + dc
                    if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < min(board_cols, 8):
                        new_idx = (1 * (board_rows * board_cols)) + new_row * board_cols + new_col
                        if board[new_idx] == 0:
                            count += 1
            if p > 0:
                mobility_gold += count
            else:
                mobility_scarlet += count
    return mobility_gold - mobility_scarlet

@njit
def passed_on_board(board, board_rows, board_cols):
    passed_gold = 0
    passed_scarlet = 0
    for row in range(board_rows):
        for col in range(min(board_cols, 8)):
            idx = (1 * (board_rows * board_cols)) + row * board_cols + col
            p = board[idx]
            if p == 0:
                continue
            if p == 12:
                passed_gold += (board_rows - row)
            elif p == -12:
                passed_scarlet += (row + 1)
    return passed_gold - passed_scarlet

@njit
def threats_on_board(board, board_rows, board_cols):
    threat_gold = 0
    threat_scarlet = 0
    for row in range(board_rows):
        for col in range(min(board_cols, 8)):
            idx = (1 * (board_rows * board_cols)) + row * board_cols + col
            p = board[idx]
            if p == 0:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    new_row = row + dr
                    new_col = col + dc
                    if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < min(board_cols, 8):
                        new_idx = (1 * (board_rows * board_cols)) + new_row * board_cols + new_col
                        target = board[new_idx]
                        if target != 0 and (p * target < 0):
                            if p > 0:
                                threat_gold += 1
                            else:
                                threat_scarlet += 1
    return threat_gold - threat_scarlet

@njit
def king_safety_on_board(board, board_rows, board_cols):
    king_safety_gold = 0
    king_safety_scarlet = 0
    for row in range(board_rows):
        for col in range(min(board_cols, 8)):
            idx = (1 * (board_rows * board_cols)) + row * board_cols + col
            p = board[idx]
            if p == 10 or p == -10:
                count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        new_row = row + dr
                        new_col = col + dc
                        if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < min(board_cols, 8):
                            new_idx = (1 * (board_rows * board_cols)) + new_row * board_cols + new_col
                            if p > 0 and board[new_idx] > 0:
                                count += 1
                            elif p < 0 and board[new_idx] < 0:
                                count += 1
                if p > 0:
                    king_safety_gold = count
                else:
                    king_safety_scarlet = count
    return king_safety_gold - king_safety_scarlet

# --- Python Wrappers for Numba Functions ---
def mobility_mg(pos, sq=None):
    return mobility_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

def passed_mg(pos, sq=None):
    return passed_on_board(pos["board"], BOARD_ROWS, BOARD_COLS)

def threats_mg(pos):
    return threats_on_board(pos["board"], BOARD_ROWS, min(BOARD_COLS, 8))

def king_mg(pos, sq=None):
    return king_safety_on_board(pos["board"], BOARD_ROWS, min(BOARD_COLS, 8))

def mobility_eg(pos, sq=None):
    return mobility_mg(pos, sq)

def passed_eg(pos, sq=None):
    return passed_mg(pos, sq)

def threats_eg(pos):
    return threats_mg(pos)

def king_eg(pos, sq=None):
    return king_mg(pos, sq)

def space(pos, sq=None):
    board = pos["board"]
    empty = np.count_nonzero(board == 0)
    return empty / TOTAL_SQUARES * 100

def imbalance_total(pos, sq=None):
    board = pos["board"]
    gold_bishops = 0
    scarlet_bishops = 0
    for row in range(BOARD_ROWS):
        for col in range(min(BOARD_COLS, 8)):
            p = board[pos_to_index(1, row, col)]
            if p > 0 and map_piece(p) == 'B':
                gold_bishops += 1
            elif p < 0 and map_piece(p) == 'B':
                scarlet_bishops += 1
    bonus = 50
    return bonus * ((gold_bishops >= 2) - (scarlet_bishops >= 2))

# --- Decoding a 25-Dimensional Parameter Vector ---
# The vector comprises 14 parameters for piece value scaling and 11 positional weights.
def decode_vector(param_vector):
    if len(param_vector) != 25:
        raise ValueError("Parameter vector must have 25 elements.")
    pv = np.zeros(16, dtype=np.float64)
    # For pieces 1-9 and 11-15, use stockfish_original as baseline.
    for i, gene in enumerate([1,2,3,4,5,6,7,8,9,11,12,13,14,15]):
        orig = stockfish_original[gene]
        lower = 0.5 * orig
        upper = 2 * orig
        s = 1 / (1 + math.exp(-param_vector[i]))
        pv[gene] = lower + s * (upper - lower)
    pv[0] = 0.0
    pv[10] = stockfish_original[10]
    weights_mg = param_vector[14:25]
    return pv, weights_mg

# --- Modified Evaluation Function Using Weights ---
def middle_game_evaluation(pos, nowinnable=False, weights_mg=None):
    if weights_mg is None or len(weights_mg) != 11:
        raise ValueError("weights_mg must be a sequence of 11 values.")
    # Compute each evaluation component:
    material_diff = weights_mg[0] * (piece_value_mg(pos) - piece_value_mg(colorflip(pos)))
    psqt_diff = weights_mg[1] * (psqt_mg(pos) - psqt_mg(colorflip(pos)))
    imbalance = weights_mg[2] * imbalance_total(pos)
    pawn_diff = weights_mg[3] * (pawns_mg(pos) - pawns_mg(colorflip(pos)))
    board_activity_diff = weights_mg[4] * (board_activity_penalty(pos) - board_activity_penalty(colorflip(pos)))
    mobility_diff = weights_mg[5] * (mobility_mg(pos) - mobility_mg(colorflip(pos)))
    threats_diff = weights_mg[6] * (threats_mg(pos) - threats_mg(colorflip(pos)))
    passed_diff = weights_mg[7] * (passed_mg(pos) - passed_mg(colorflip(pos)))
    space_diff = weights_mg[8] * (space(pos) - space(colorflip(pos)))
    king_safety_diff = weights_mg[9] * (king_mg(pos) - king_mg(colorflip(pos)))
    # Winnability placeholder is zero.
    mg = material_diff + psqt_diff + imbalance + pawn_diff + board_activity_diff + mobility_diff + threats_diff + passed_diff + space_diff + king_safety_diff
    if DEBUG:
        print("----- Debug Evaluation Breakdown -----")
        print("Material diff: ", material_diff)
        print("PSQT diff: ", psqt_diff)
        print("Imbalance: ", imbalance)
        print("Pawn diff: ", pawn_diff)
        print("Board activity diff: ", board_activity_diff)
        print("Mobility diff: ", mobility_diff)
        print("Threats diff: ", threats_diff)
        print("Passed diff: ", passed_diff)
        print("Space diff: ", space_diff)
        print("King safety diff: ", king_safety_diff)
        print("Total mg: ", mg)
    return mg

# --- Main Evaluation Function (used at all stages) ---
def main_evaluation(pos, weights_mg):
    mg = middle_game_evaluation(pos, nowinnable=False, weights_mg=weights_mg)
    p_val = phase(pos)
    r50 = rule50(pos)
    eg_scaled = int(mg * scale_factor(pos, mg) / 64)
    v = int((mg * p_val + (eg_scaled * (128 - p_val))) / 128)
    v = int(v / 16) * 16
    v += tempo(pos)
    v = int(v * (100 - r50) / 100)
    if DEBUG:
        print(f"DEBUG EVAL: mg={mg}, p_val={p_val}, r50={r50}, eg_scaled={eg_scaled}, final={v}")
    return v

def evaluate_game(game, param_vector):
    pos = position_from_game(game)
    _, weights_mg = decode_vector(param_vector)
    return main_evaluation(pos, weights_mg)

def load_weights(filename):
    try:
        vec = np.loadtxt(filename, delimiter=None)
        if vec.size != 25:
            raise ValueError("Expected 25 values, got {}".format(vec.size))
        return vec.tolist()
    except Exception as e:
        raise IOError("Error reading weights file {}: {}".format(filename, e))

# --- Search Functions for CustomAI ---
transposition_table = {}

def board_state_hash(state):
    board, turn_flag = state
    return hash((board.tobytes(), turn_flag))

class TimeOutException(Exception):
    pass

def get_all_moves(state, color):
    from game import move_generators
    from bitboard import index_to_pos
    board, _ = state
    moves = []
    for idx in range(board.size):
        piece = board[idx]
        if piece != 0 and (color * piece > 0):
            pos = index_to_pos(idx)
            abs_code = abs(piece)
            gen_func = move_generators.get(abs_code)
            if gen_func is not None:
                candidate_moves = gen_func(pos, board, "Gold" if color == 1 else "Scarlet")
                for move in candidate_moves:
                    from_idx, to_idx, flag = move
                    if flag == QUIET and board[to_idx] != 0:
                        continue
                    elif flag in (CAPTURE, AFAR):
                        if board[to_idx] == 0 or (color * board[to_idx] > 0):
                            continue
                    moves.append(move)
    moves.sort(key=lambda m: 0 if m[2] in (CAPTURE, AFAR) else 1)
    return moves

def simulate_move(state, move):
    board, turn_flag = state
    new_board = board.copy()
    from_idx, to_idx, flag = move
    piece = new_board[from_idx]
    if flag in (CAPTURE, AFAR):
        new_board[to_idx] = 0
    new_board[to_idx] = piece
    new_board[from_idx] = 0
    new_turn = -turn_flag
    return (new_board, new_turn)

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history,
              current_depth=0, start_time=None, time_limit=None, weights_mg=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            raise TimeOutException
    key = board_state_hash(state)
    moves = get_all_moves(state, state[1])
    if depth == 0 or not moves:
        pos = {"board": state[0], "turn": "Gold" if state[1] == 1 else "Scarlet",
               "no_capture_count": len(history)}
        from bots.dragonfish import main_evaluation
        return main_evaluation(pos, weights_mg), None
    best_move = None
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, False, my_color, history,
                                      current_depth + 1, start_time, time_limit, weights_mg)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, True, my_color, history,
                                      current_depth + 1, start_time, time_limit, weights_mg)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0, weights_mg=None):
    global transposition_table
    transposition_table = {}
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        try:
            eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history,
                                        0, start_time, time_limit, weights_mg)
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            break
    return best_eval, best_move

# --- CustomAI Class for Tournament Usage ---
class CustomAI:
    """
    A CustomAI for Dragonchess that uses the Dragonfish evaluation.
    It loads a 25-dimensional weight vector from a file (if provided) and uses
    iterative deepening minimax search to select moves.
    """
    def __init__(self, game, color, weights_file=None):
        self.game = game
        self.color = color
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 5  # Increased search depth by 3 (from default 2)
        # Default parameter vector: 14 zeros for piece scaling; then 11 positional weights (example defaults)
        default_params = [0.0] * 14 + [1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0]
        if weights_file:
            self.weights = load_weights(weights_file)
        else:
            self.weights = default_params
        _, self.weights_mg = decode_vector(self.weights)
    def choose_move(self):
        state = (self.game.board.copy(), 1 if self.game.current_turn == "Gold" else -1)
        history = self.game.state_history
        eval_val, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history,
                                                    time_limit=5.0, weights_mg=self.weights_mg)
        if DEBUG:
            print(f"CustomAI.choose_move: depth={self.max_depth}, eval={eval_val}, move={best_move}")
        return best_move

if __name__ == "__main__":
    from bitboard import create_initial_board
    board = create_initial_board()
    pos = {"board": board, "turn": "Gold", "no_capture_count": 0}
    # Dummy parameter vector of 25 zeros.
    params = [0.0] * 25
    try:
        from bots.dragonfish import main_evaluation
        score = main_evaluation(pos, weights_mg=params[14:25])
        print("Evaluation value:", score)
    except Exception as e:
        print("Evaluation failed:", e)
