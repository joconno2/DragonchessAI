#!/usr/bin/env python3
"""
dragonfish.py

This module implements a full evaluation function modeled on Stockfish’s pre‐NNUE
evaluation – using numeric constants from its JavaScript code – adapted to the
Dragonchess domain. It works with your engine’s flat NumPy board (from bitboard.py)
and game state.

A position is represented as a dict with:
  • "board": a flat NumPy array (length = TOTAL_SQUARES) from create_initial_board().
  • "turn": "Gold" or "Scarlet".
  • "no_capture_count": an integer.

This module defines evaluation components, a 35‑parameter decoding function, a main
evaluation function, search functions (alphabeta/iterative_deepening), and a CustomAI
class. It also provides load_weights(filename) so that evolved parameters can be read
from a file.

Usage Example:
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

# --- Move Flag Constants (globally defined) ---
QUIET = 0
CAPTURE = 1
AFAR = 2
AMBIGUOUS = 3
THREED = 4

from bitboard import (pos_to_index, index_to_pos, BOARD_ROWS, BOARD_COLS, 
                      NUM_BOARDS, TOTAL_SQUARES)

# --- Precompute index -> (layer, row, col) mapping for speed ---
INDEX_TO_POS = [index_to_pos(i) for i in range(TOTAL_SQUARES)]

# --- Define Original Piece Values ---
original_values = {
    1: 1, 2: 5, 3: 20, 4: 6, 5: 2.5, 6: 5, 7: 4,
    8: 9, 9: 11, 11: 15, 12: 1, 13: 3, 14: 4, 15: 2
}

# --- Data Structures ---
Square3D = namedtuple("Square3D", ["layer", "x", "y"])

# --- Converting a Game into a Position Dictionary ---
def position_from_game(game):
    return {
        "board": game.board,
        "turn": game.current_turn,
        "no_capture_count": game.no_capture_count
    }

# --- Helper Functions (Python level) ---
def phase(pos):
    phase_weights = {1:1,2:5,3:20,4:6,5:2.5,6:5,7:4,8:9,9:11,10:0,11:15,12:1,13:3,14:4,15:2}
    total = 0.0
    board = pos["board"]
    for idx in range(TOTAL_SQUARES):
        piece = board[idx]
        if piece:
            total += phase_weights.get(abs(piece), 0)
    initial_phase = 280.0
    phase_val = int(round(128 * (total / initial_phase)))
    return max(0, min(128, phase_val))

def rule50(pos):
    nc = pos.get("no_capture_count", 0)
    return min(100, int(nc * 100 / 250))

def scale_factor(pos, eg):
    return 64

def tempo(pos):
    return 10 if pos.get("turn", "Gold")=="Gold" else -10

def colorflip(pos):
    new_board = np.copy(pos["board"])
    new_board *= -1
    new_turn = "Gold" if pos.get("turn", "Gold")=="Scarlet" else "Scarlet"
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
def piece_value_bonus(pos, sq, mg):
    a = [124, 781, 825, 1276, 2538] if mg else [206, 854, 915, 1380, 2682]
    p = board_at(pos, sq.layer, sq.y, sq.x)
    letter = map_piece(p)
    if letter is None or letter not in "PNBRQ":
        return 0
    i = "PNBRQ".index(letter)
    return a[i]

def piece_value_mg(pos, sq=None):
    if sq is None:
        return board_sum_all(pos, piece_value_mg)
    return piece_value_bonus(pos, sq, mg=True)

def piece_value_eg(pos, sq=None):
    if sq is None:
        return board_sum_all(pos, piece_value_eg)
    return piece_value_bonus(pos, sq, mg=False)

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

def psqt_eg(pos, sq=None, mg=False):
    if sq is None:
        return board_sum_middle(pos, psqt_eg, mg=mg)
    return psqt_bonus(pos, sq, mg=mg)

# --- Numba-Accelerated Helper Functions ---
@njit
def mobility_on_board(board, total_squares, board_rows, board_cols):
    mobility_gold = 0
    mobility_scarlet = 0
    for idx in range(total_squares):
        piece = board[idx]
        if piece == 0:
            continue
        layer = idx // (board_rows * board_cols)
        rem = idx % (board_rows * board_cols)
        row = rem // board_cols
        col = rem % board_cols
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < board_cols:
                    new_idx = layer * (board_rows * board_cols) + new_row * board_cols + new_col
                    if board[new_idx] == 0:
                        count += 1
        if piece > 0:
            mobility_gold += count
        else:
            mobility_scarlet += count
    return mobility_gold - mobility_scarlet

@njit
def passed_on_board(board, total_squares, board_rows, board_cols):
    passed_gold = 0
    passed_scarlet = 0
    # For passed pawn evaluation, only consider pieces with absolute codes 1, 7, 12, 15 (mapped to 'P')
    for idx in range(total_squares):
        piece = board[idx]
        if piece == 0:
            continue
        abs_piece = abs(piece)
        if abs_piece in (1, 7, 12, 15):
            layer = idx // (board_rows * board_cols)
            rem = idx % (board_rows * board_cols)
            row = rem // board_cols
            if piece > 0:
                passed_gold += (board_rows - row)
            else:
                passed_scarlet += (row + 1)
    return passed_gold - passed_scarlet

@njit
def threats_on_board(board, total_squares, board_rows, board_cols):
    threat_gold = 0
    threat_scarlet = 0
    for idx in range(total_squares):
        piece = board[idx]
        if piece == 0:
            continue
        layer = idx // (board_rows * board_cols)
        rem = idx % (board_rows * board_cols)
        row = rem // board_cols
        col = rem % board_cols
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < board_cols:
                    new_idx = layer * (board_rows * board_cols) + new_row * board_cols + new_col
                    target = board[new_idx]
                    if target != 0 and (piece * target < 0):
                        if piece > 0:
                            threat_gold += 1
                        else:
                            threat_scarlet += 1
    return threat_gold - threat_scarlet

@njit
def king_safety_on_board(board, total_squares, board_rows, board_cols):
    king_safety_gold = 0
    king_safety_scarlet = 0
    for idx in range(total_squares):
        if board[idx] == 10 or board[idx] == -10:
            piece = board[idx]
            layer = idx // (board_rows * board_cols)
            rem = idx % (board_rows * board_cols)
            row = rem // board_cols
            col = rem % board_cols
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    new_row = row + dr
                    new_col = col + dc
                    if new_row >= 0 and new_row < board_rows and new_col >= 0 and new_col < board_cols:
                        new_idx = layer * (board_rows * board_cols) + new_row * board_cols + new_col
                        if piece > 0 and board[new_idx] > 0:
                            count += 1
                        elif piece < 0 and board[new_idx] < 0:
                            count += 1
            if piece > 0:
                king_safety_gold = count
            else:
                king_safety_scarlet = count
    return king_safety_gold - king_safety_scarlet

# --- Python Wrappers for Numba Functions ---
def mobility_mg(pos, sq=None):
    return mobility_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

def passed_mg(pos, sq=None):
    return passed_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

def threats_mg(pos):
    return threats_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

def king_mg(pos, sq=None):
    return king_safety_on_board(pos["board"], TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS)

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
    for p in board:
        if p > 0 and map_piece(p) == 'B':
            gold_bishops += 1
        elif p < 0 and map_piece(p) == 'B':
            scarlet_bishops += 1
    bonus = 50
    return bonus * ((gold_bishops >= 2) - (scarlet_bishops >= 2))

def pawns_mg(pos, sq=None):
    board = pos["board"]
    gold_pawns = 0
    scarlet_pawns = 0
    for p in board:
        if p != 0 and abs(p) in (1,7,12,15):  # pieces mapped as 'P'
            if p > 0:
                gold_pawns += 1
            else:
                scarlet_pawns += 1
    return 100 * (gold_pawns - scarlet_pawns)

def pawns_eg(pos, sq=None):
    return pawns_mg(pos, sq)

def pieces_mg(pos, sq=None):
    return 0

def pieces_eg(pos, sq=None):
    return 0

# --- Decoding a 35-Dimensional Parameter Vector ---
def decode_vector(param_vector):
    if len(param_vector) != 35:
        raise ValueError("Parameter vector must have 35 elements.")
    pv = np.zeros(16, dtype=np.float64)
    for i, gene in enumerate([1,2,3,4,5,6,7,8,9,11,12,13,14,15]):
        orig = original_values[gene]
        lower = 0.5 * orig
        upper = 2 * orig
        s = 1 / (1 + math.exp(-param_vector[i]))
        pv[gene] = lower + s * (upper - lower)
    pv[0] = 0.0
    pv[10] = 10000.0
    weights_mg = param_vector[14:25]
    weights_eg = param_vector[25:35]
    return pv, weights_mg, weights_eg

# --- Modified Evaluation Functions Using Weights ---
def middle_game_evaluation(pos, nowinnable=False, weights_mg=None):
    if weights_mg is None or len(weights_mg) != 11:
        raise ValueError("weights_mg must be a sequence of 11 values.")
    v = 0
    v += weights_mg[0] * (piece_value_mg(pos) - piece_value_mg(colorflip(pos)))
    v += weights_mg[1] * (psqt_mg(pos) - psqt_mg(colorflip(pos)))
    v += weights_mg[2] * imbalance_total(pos)
    v += weights_mg[3] * (pawns_mg(pos) - pawns_mg(colorflip(pos)))
    v += weights_mg[4] * (pieces_mg(pos) - pieces_mg(colorflip(pos)))
    v += weights_mg[5] * (mobility_mg(pos) - mobility_mg(colorflip(pos)))
    v += weights_mg[6] * (threats_mg(pos) - threats_mg(colorflip(pos)))
    v += weights_mg[7] * (passed_mg(pos) - passed_mg(colorflip(pos)))
    v += weights_mg[8] * (space(pos) - space(colorflip(pos)))
    v += weights_mg[9] * (king_mg(pos) - king_mg(colorflip(pos)))
    if not nowinnable:
        v += weights_mg[10] * 0  # winnable_total_mg is 0 for now.
    return v

def end_game_evaluation(pos, nowinnable=False, weights_eg=None):
    if weights_eg is None or len(weights_eg) != 10:
        raise ValueError("weights_eg must be a sequence of 10 values.")
    v = 0
    v += weights_eg[0] * (piece_value_eg(pos) - piece_value_eg(colorflip(pos)))
    v += weights_eg[1] * (psqt_eg(pos) - psqt_eg(colorflip(pos)))
    v += weights_eg[2] * imbalance_total(pos)
    v += weights_eg[3] * (pawns_eg(pos) - pawns_eg(colorflip(pos)))
    v += weights_eg[4] * (pieces_eg(pos) - pieces_eg(colorflip(pos)))
    v += weights_eg[5] * (mobility_eg(pos) - mobility_eg(colorflip(pos)))
    v += weights_eg[6] * (threats_eg(pos) - threats_eg(colorflip(pos)))
    v += weights_eg[7] * (passed_eg(pos) - passed_eg(colorflip(pos)))
    v += weights_eg[8] * (king_eg(pos) - king_eg(colorflip(pos)))
    if not nowinnable:
        v += weights_eg[9] * 0  # winnable_total_eg is 0 for now.
    return v

def main_evaluation(pos, weights_mg, weights_eg):
    mg = middle_game_evaluation(pos, nowinnable=False, weights_mg=weights_mg)
    eg = end_game_evaluation(pos, nowinnable=False, weights_eg=weights_eg)
    p_val = phase(pos)
    r50 = rule50(pos)
    eg_scaled = int(eg * scale_factor(pos, eg) / 64)
    v = int((mg * p_val + (eg_scaled * (128 - p_val))) / 128)
    v = int(v / 16) * 16
    v += tempo(pos)
    v = int(v * (100 - r50) / 100)
    return v

def evaluate_game(game, param_vector):
    pos = position_from_game(game)
    _, weights_mg, weights_eg = decode_vector(param_vector)
    return main_evaluation(pos, weights_mg, weights_eg)

def load_weights(filename):
    try:
        vec = np.loadtxt(filename, delimiter=None)
        if vec.size != 35:
            raise ValueError("Expected 35 values, got {}".format(vec.size))
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
                candidate_moves = gen_func(pos, board, "Gold" if color==1 else "Scarlet")
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
              current_depth=0, start_time=None, time_limit=None, weights_mg=None, weights_eg=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            raise TimeOutException
    key = board_state_hash(state)
    moves = get_all_moves(state, state[1])
    if depth == 0 or not moves:
        pos = {"board": state[0], "turn": "Gold" if state[1]==1 else "Scarlet", "no_capture_count": len(history)}
        from bots.dragonfish import main_evaluation
        return main_evaluation(pos, weights_mg, weights_eg), None
    best_move = None
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, False, my_color, history,
                                      current_depth + 1, start_time, time_limit, weights_mg, weights_eg)
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
                                      current_depth + 1, start_time, time_limit, weights_mg, weights_eg)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0, weights_mg=None, weights_eg=None):
    global transposition_table
    transposition_table = {}
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth+1):
        try:
            eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history,
                                        0, start_time, time_limit, weights_mg, weights_eg)
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            break
    return best_eval, best_move

# --- CustomAI Class for Tournament Usage ---
class CustomAI:
    """
    A CustomAI for Dragonchess that uses the Dragonfish evaluation.
    It loads a 35-dimensional weight vector from a file (if provided) and uses
    iterative deepening minimax search to select moves.
    """
    def __init__(self, game, color, weights_file=None):
        self.game = game
        self.color = color
        self.color_flag = 1 if color=="Gold" else -1
        # Lower default search depth for speed.
        self.max_depth = 2
        if weights_file:
            self.weights = load_weights(weights_file)
        else:
            self.weights = [0.0] * 35
        _, self.weights_mg, self.weights_eg = decode_vector(self.weights)
    def choose_move(self):
        state = (self.game.board.copy(), 1 if self.game.current_turn=="Gold" else -1)
        history = self.game.state_history
        eval_val, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history,
                                                    time_limit=5.0, weights_mg=self.weights_mg, weights_eg=self.weights_eg)
        return best_move

if __name__ == "__main__":
    from bitboard import create_initial_board
    board = create_initial_board()
    pos = {"board": board, "turn": "Gold", "no_capture_count": 0}
    params = [0.0] * 35
    try:
        from bots.dragonfish import main_evaluation
        score = main_evaluation(pos, weights_mg=params[14:25], weights_eg=params[25:35])
        print("Evaluation value:", score)
    except Exception as e:
        print("Evaluation failed:", e)
