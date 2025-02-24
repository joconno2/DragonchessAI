#!/usr/bin/env python3
"""
dragonfish.py

This module implements an evaluation function modeled on Stockfish’s pre‐NNUE evaluation – using numeric constants adapted from Stockfish’s piece values – for the Dragonchess domain.
It works with your engine’s flat NumPy board (from bitboard.py) and game state.

A position is represented as a dict with:
  • "board": a flat NumPy array (length = TOTAL_SQUARES) as created by create_initial_board().
  • "turn": "Gold" or "Scarlet".
  • "no_capture_count": an integer.
  • (For evolved evaluations, the position dictionary will also include key "pv" which holds the material scaling factors.)

This module defines evaluation components, a 25‑parameter decoding function (first 14 parameters for piece value scaling and the remaining 11 for positional weights), a main evaluation function (with debug printing), search functions (alphabeta and iterative deepening), and a CustomAI class.
It also provides a helper function load_weights(filename) so that evolved parameters can be read from a file.

USAGE EXAMPLE:
    from game import Game
    from bots.dragonfish import CustomAI, load_weights
    game = Game()
    ai_bot = CustomAI(game, "Gold", weights_file="best_weights.txt")
    move = ai_bot.choose_move()
    print("Chosen move:", move)

Additionally, if run as a standalone script, this file will run a CMA‐ES evolution routine to optimize the 25‐dimensional weight vector.
"""

import math
import copy
import numpy as np
import time
import random
from collections import namedtuple
from numba import njit

# --- Global Debug Flag ---
DEBUG = False

# --- Move Flag Constants ---
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

from bitboard import pos_to_index, index_to_pos, BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES

# --- Precompute index -> (layer, row, col) mapping for speed ---
INDEX_TO_POS = [index_to_pos(i) for i in range(TOTAL_SQUARES)]

# --- Stockfish-Inspired Baseline Piece Values for Dragonchess ---
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

# --- Evaluation Helper Functions ---
def phase(pos):
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

# --- Board Activity Penalty ---
def board_activity_penalty(pos):
    penalty = 0
    board = pos["board"]
    for idx, (layer, row, col) in enumerate(INDEX_TO_POS):
        piece = board[idx]
        if piece != 0 and layer != 1:
            if piece > 0:
                penalty -= 20
            else:
                penalty += 20
    return penalty

# --- Dragon Center Bonus ---
def dragon_center_bonus(pos):
    bonus_val = 0.0
    board = pos["board"]
    center_row = (BOARD_ROWS - 1) / 2.0
    center_col = (BOARD_COLS - 1) / 2.0
    max_dist = ((center_row)**2 + (center_col)**2)**0.5
    for idx, (layer, row, col) in enumerate(INDEX_TO_POS):
        piece = board[idx]
        if abs(piece) == 3 and layer == 0:
            dist = ((row - center_row)**2 + (col - center_col)**2)**0.5
            bonus_val += (max_dist - dist)
    return bonus_val

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

# --- Material Evaluation using Evolved Scaling (pv) ---
def piece_value_bonus(pos, sq, pv):
    p = board_at(pos, sq.layer, sq.y, sq.x)
    # Do not add material value for kings.
    if abs(p) == 10:
        return 0
    gene = abs(p)
    if gene not in pv:
        return 0
    return pv[gene]

def piece_value_mg(pos, pv, sq=None):
    if sq is None:
        return board_sum_all(pos, lambda pos, sq: piece_value_bonus(pos, sq, pv))
    return piece_value_bonus(pos, sq, pv)

# --- Piece-Square Table Evaluation (unchanged) ---
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
    bonus_val = 50
    return bonus_val * ((gold_bishops >= 2) - (scarlet_bishops >= 2))

@njit
def mobility_on_board(board, total_squares, board_rows, board_cols):
    mobility_gold = 0
    mobility_scarlet = 0
    for row in range(board_rows):
        for col in range(min(board_cols, 8)):
            idx = row * board_cols + col
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
                        new_idx = new_row * board_cols + new_col
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
            idx = row * board_cols + col
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
            idx = row * board_cols + col
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
                        new_idx = new_row * board_cols + new_col
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
            idx = row * board_cols + col
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
                            new_idx = new_row * board_cols + new_col
                            if p > 0 and board[new_idx] > 0:
                                count += 1
                            elif p < 0 and board[new_idx] < 0:
                                count += 1
                if p > 0:
                    king_safety_gold = count
                else:
                    king_safety_scarlet = count
    return king_safety_gold - king_safety_scarlet

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
    bonus_val = 50
    return bonus_val * ((gold_bishops >= 2) - (scarlet_bishops >= 2))

# --- Decoding the 25-Dimensional Parameter Vector ---
def decode_vector(param_vector):
    if len(param_vector) != 25:
        raise ValueError("Parameter vector must have 25 elements.")
    pv = np.zeros(16, dtype=np.float64)
    # For pieces 1-9 and 11-15, scale stockfish_original using the first 14 parameters.
    for i, gene in enumerate([1,2,3,4,5,6,7,8,9,11,12,13,14,15]):
        orig = stockfish_original[gene]
        lower = 0.5 * orig
        upper = 2 * orig
        s = 1 / (1 + math.exp(-param_vector[i]))
        pv[gene] = lower + s * (upper - lower)
    pv[0] = 0.0
    pv[10] = stockfish_original[10]  # King remains fixed.
    weights_mg = param_vector[14:25]
    return pv, weights_mg

# --- New Helper: Build candidate moves from a position dictionary ---
def get_all_moves_from_pos(pos, color):
    # Convert a position dictionary to a state tuple and use get_all_moves.
    state = (pos["board"], 1 if pos["turn"]=="Gold" else -1)
    return get_all_moves(state, color)

# --- New Override Heuristic: King Attack Bonus ---
def king_attack_bonus(pos, our_color):
    # our_color is 1 if we are Gold, -1 if we are Scarlet.
    enemy_king = -10 * our_color
    board = pos["board"]
    enemy_king_indices = np.where(board == enemy_king)[0]
    if enemy_king_indices.size == 0:
        return 0
    enemy_king_index = enemy_king_indices[0]
    bonus = 0
    candidate_moves = get_all_moves_from_pos(pos, our_color)
    for move in candidate_moves:
        from_idx, to_idx, flag = move
        if to_idx == enemy_king_index:
            bonus += 10000  # Huge bonus for a move that can capture the enemy king.
    return bonus

# --- Revised Material and Positional Evaluation ---
def middle_game_evaluation(pos, weights_mg=None, pv=None):
    if weights_mg is None or len(weights_mg) != 11:
        raise ValueError("weights_mg must be a sequence of 11 values.")
    if pv is None:
        raise ValueError("Material scaling vector (pv) must be provided.")
    material_diff = weights_mg[0] * (piece_value_mg(pos, pv) - piece_value_mg(colorflip(pos), pv))
    psqt_diff = weights_mg[1] * (psqt_mg(pos) - psqt_mg(colorflip(pos)))
    imbalance = weights_mg[2] * imbalance_total(pos)
    pawn_diff = weights_mg[3] * (pawns_mg(pos) - pawns_mg(colorflip(pos)))
    board_activity_diff = weights_mg[4] * (board_activity_penalty(pos) - board_activity_penalty(colorflip(pos)))
    mobility_diff = weights_mg[5] * (mobility_mg(pos) - mobility_mg(colorflip(pos)))
    threats_diff = weights_mg[6] * (threats_mg(pos) - threats_mg(colorflip(pos)))
    passed_diff = weights_mg[7] * (passed_mg(pos) - passed_mg(colorflip(pos)))
    space_diff = weights_mg[8] * (space(pos) - space(colorflip(pos)))
    king_safety_diff = weights_mg[9] * (king_mg(pos) - king_mg(colorflip(pos)))
    dragon_bonus_diff = weights_mg[10] * (dragon_center_bonus(pos) - dragon_center_bonus(colorflip(pos)))
    v = (material_diff + psqt_diff + imbalance + pawn_diff + board_activity_diff +
         mobility_diff + threats_diff + passed_diff + space_diff + king_safety_diff + dragon_bonus_diff)
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
        print("Dragon bonus diff: ", dragon_bonus_diff)
        print("Total mg: ", v)
    return v

def main_evaluation(pos, weights_mg, pv):
    mg = middle_game_evaluation(pos, weights_mg, pv)
    p_val = phase(pos)
    r50 = rule50(pos)
    eg_scaled = int(mg * scale_factor(pos, mg) / 64)
    v = int((mg * p_val + (eg_scaled * (128 - p_val))) / 128)
    v = int(v / 16) * 16
    v += tempo(pos)
    v = int(v * (100 - r50) / 100)
    # Add override: if any move can capture enemy king, boost evaluation.
    our_color = 1 if pos["turn"] == "Gold" else -1
    v += king_attack_bonus(pos, our_color)
    if DEBUG:
        print(f"DEBUG EVAL: mg={mg}, p_val={p_val}, r50={r50}, eg_scaled={eg_scaled}, final={v}")
    return v

def evaluate_game(game, param_vector):
    pos = position_from_game(game)
    pv, weights_mg = decode_vector(param_vector)
    pos["pv"] = pv  # store scaling factors in pos (if needed)
    return main_evaluation(pos, weights_mg, pv)

def load_weights(filename):
    try:
        # Read file line by line and return the first numeric vector of 25 floats.
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit() or line[0] in "-.":
                    parts = line.split()
                    vector = [float(x) for x in parts]
                    if len(vector) != 25:
                        raise ValueError(f"Expected 25 values, got {len(vector)}")
                    return vector
        raise ValueError("No numeric weight vector found in file")
    except Exception as e:
        raise IOError(f"Error reading weights file {filename}: {e}")

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
    color_str = "Gold" if color == 1 else "Scarlet"
    moves = []
    for idx in range(board.size):
        piece = board[idx]
        if piece != 0 and (color * piece > 0):
            pos_ = index_to_pos(idx)
            abs_code = abs(piece)
            gen_func = move_generators.get(abs_code)
            if gen_func:
                candidate_moves = gen_func(pos_, board, color_str)
                for move in candidate_moves:
                    from_idx, to_idx, flag = move
                    if flag == QUIET and board[to_idx] != 0:
                        continue
                    elif flag == AMBIGUOUS:
                        if board[to_idx] != 0 and ((piece > 0 and board[to_idx] > 0) or (piece < 0 and board[to_idx] < 0)):
                            continue
                    elif flag in (CAPTURE, AFAR):
                        if board[to_idx] == 0:
                            continue
                        if (piece > 0 and board[to_idx] > 0) or (piece < 0 and board[to_idx] < 0):
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

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history, current_depth=0, start_time=None, time_limit=None, weights_mg=None, pv=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            raise TimeOutException
    key = board_state_hash(state)
    moves = get_all_moves(state, state[1])
    if depth == 0 or not moves:
        pos = {"board": state[0],
               "turn": "Gold" if state[1] == 1 else "Scarlet",
               "no_capture_count": len(history),
               "pv": pv}
        val = main_evaluation(pos, weights_mg, pv)
        transposition_table[key] = (depth, val, None)
        return val, None
    best_move = None
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth-1, alpha, beta, False, my_color, history,
                                      current_depth+1, start_time, time_limit, weights_mg, pv)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth-1, alpha, beta, True, my_color, history,
                                      current_depth+1, start_time, time_limit, weights_mg, pv)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0, weights_mg=None, pv=None):
    global transposition_table
    transposition_table = {}
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth+1):
        try:
            eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history,
                                        current_depth=0, start_time=start_time, time_limit=time_limit, weights_mg=weights_mg, pv=pv)
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            break
    return best_eval, best_move

# --- CustomAI Class ---
class CustomAI:
    """
    A CustomAI for Dragonchess that uses the Dragonfish evaluation.
    It loads a 25-dimensional weight vector from a file (if provided) and uses
    iterative deepening minimax search to select moves.
    """
    def __init__(self, game, color, weights_file="best_weights.txt"):
        self.game = game
        self.color = color  # "Gold" or "Scarlet"
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 5
        default_params = [-0.6931] * 14 + [1.0, 1.0, 0.5, 0.5, 0.1, 1.0, 1.0, 0.5, 0.5, 1.0, 0.1]
        if weights_file:
            self.weights = load_weights(weights_file)
        else:
            self.weights = default_params
        self.pv, self.weights_mg = decode_vector(self.weights)
        if DEBUG:
            print(f"CustomAI initialized for {self.color}")
            print("Loaded piece scaling factors (pv):", self.pv)
            print("Loaded middle-game weights:", self.weights_mg)
    def choose_move(self):
        if self.game.current_turn != self.color:
            if DEBUG:
                print(f"CustomAI.choose_move: Not {self.color}'s turn ({self.game.current_turn} instead)")
            return None
        state = (self.game.board.copy(), 1 if self.color=="Gold" else -1)
        history = self.game.state_history
        best_eval, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history,
                                                     time_limit=5.0, weights_mg=self.weights_mg, pv=self.pv)
        if best_eval is None:
            fallback = get_all_moves(state, self.color_flag)
            if fallback:
                best_move = random.choice(fallback[:min(5, len(fallback))])
                if DEBUG:
                    print("Fallback move chosen (due to no best_eval):", best_move)
            return best_move
        best_eval = -best_eval
        candidate_moves = get_all_moves(state, self.color_flag)
        best_moves = []
        for move in candidate_moves:
            new_state = simulate_move(state, move)
            pos = {"board": new_state[0],
                   "turn": "Gold" if new_state[1] == 1 else "Scarlet",
                   "no_capture_count": len(history),
                   "pv": self.pv}
            eval_val = main_evaluation(pos, self.weights_mg, self.pv)
            if eval_val == best_eval:
                best_moves.append(move)
        if DEBUG:
            print(f"Candidate moves with score {best_eval}: {best_moves}")
        if best_moves:
            best_move = random.choice(best_moves)
        elif best_move is None:
            fallback = get_all_moves(state, self.color_flag)
            if fallback:
                best_move = random.choice(fallback[:min(5, len(fallback))])
                if DEBUG:
                    print("Fallback move chosen (random from top moves):", best_move)
        if DEBUG:
            print(f"CustomAI.choose_move: depth={self.max_depth}, best_eval={best_eval}, move={best_move}")
        return best_move

# --- Evolution Routine (using CMA-ES and multiprocessing) ---
if __name__ == "__main__":
    import sys
    import multiprocessing
    multiprocessing.freeze_support()
    # If run with argument "evolve", run the evolution routine;
    # otherwise, run a simple evaluation test.
    if len(sys.argv) > 1 and sys.argv[1] == "evolve":
        from concurrent.futures import ProcessPoolExecutor
        import cma

        # Evolution parameters
        DIM = 25
        EVAL_GAMES = 3
        BONUS_FACTOR = 300.0

        from game import Game
        from ai import RandomAI

        class CMAESBot:
            def __init__(self, game, color, param_vector):
                self.game = game
                self.color = color
                self.color_flag = 1 if color == "Gold" else -1
                self.max_depth = 3
                self.param_vector = param_vector
                self.pv, self.weights_mg = decode_vector(param_vector)
            def choose_move(self):
                state = (np.copy(self.game.board), 1 if self.game.current_turn=="Gold" else -1)
                history = self.game.state_history
                best_eval, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history,
                                                             time_limit=5.0, weights_mg=self.weights_mg, pv=self.pv)
                if best_eval is None:
                    fallback = get_all_moves(state, self.color_flag)
                    if fallback:
                        best_move = random.choice(fallback[:min(5, len(fallback))])
                        if DEBUG:
                            print("Fallback move chosen (due to no best_eval):", best_move)
                    return best_move
                best_eval = -best_eval
                candidate_moves = get_all_moves(state, self.color_flag)
                best_moves = []
                for move in candidate_moves:
                    new_state = simulate_move(state, move)
                    pos = {"board": new_state[0],
                           "turn": "Gold" if new_state[1]==1 else "Scarlet",
                           "no_capture_count": len(history),
                           "pv": self.pv}
                    eval_val = main_evaluation(pos, self.weights_mg, self.pv)
                    if eval_val == best_eval:
                        best_moves.append(move)
                print(f"Candidate moves with score {best_eval}: {best_moves}")
                if best_moves:
                    best_move = random.choice(best_moves)
                elif best_move is None:
                    fallback = get_all_moves(state, self.color_flag)
                    if fallback:
                        best_move = random.choice(fallback[:min(5, len(fallback))])
                        print("Fallback move chosen (random from top moves):", best_move)
                print(f"CustomAI.choose_move: depth={self.max_depth}, best_eval={best_eval}, move={best_move}")
                return best_move

        def simulate_game(param_vector):
            game = Game()
            bot = CMAESBot(game, "Gold", param_vector)
            opponent = RandomAI(game, "Scarlet")
            moves = 0
            while not game.game_over:
                if game.current_turn == "Gold":
                    move = bot.choose_move()
                else:
                    move = opponent.choose_move()
                if move:
                    game.make_move(move)
                    moves += 1
                game.update()
            if game.winner == "Gold":
                return 1.0 + (BONUS_FACTOR / moves)
            elif game.winner == "Draw":
                return 0.5
            else:
                return 0.0

        def objective(param_vector):
            total = 0.0
            for _ in range(EVAL_GAMES):
                total += simulate_game(param_vector)
            avg = total / EVAL_GAMES
            return -avg

        def hash_state(state):
            board, turn_flag = state
            turn = "Gold" if turn_flag == 1 else "Scarlet"
            from game import board_state_hash
            return board_state_hash(board, turn)

        log_file = open("cma_es_log.txt", "w")
        x0 = [0.0] * DIM
        sigma0 = 1.0
        opts = {
            'maxiter': 2,
            'popsize': 5,
            'verb_disp': 1,
            'verb_log': 0
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        generation = 0
        while not es.stop():
            generation += 1
            solutions = es.ask()
            with ProcessPoolExecutor() as executor:
                fitnesses = list(executor.map(objective, solutions))
            best_in_gen = min(fitnesses)
            best_solution = solutions[fitnesses.index(best_in_gen)]
            log_file.write(f"Generation {generation}: Best fitness (neg objective): {best_in_gen}\n")
            log_file.write(f"Best solution: {best_solution}\n")
            log_file.flush()
            es.tell(solutions, fitnesses)
            es.disp()
        res = es.result
        best_params = res.xbest
        best_score = -res.fbest
        log_file.write("CMA-ES optimization complete.\n")
        log_file.write(f"Best average game score achieved: {best_score}\n")
        log_file.write(f"Best parameter vector:\n{best_params}\n")
        log_file.close()
        print("CMA-ES optimization complete.")
        print("Best average game score achieved:", best_score)
        print("Best parameter vector:")
        print(best_params)
        best_piece_values, best_weights_mg = decode_vector(best_params)
        print("Decoded piece values:")
        print(best_piece_values)
        print("Decoded middle-game weights:")
        print(best_weights_mg)
        with open("best_weights.txt", "w") as f:
            f.write("Best parameter vector:\n" + " ".join(map(str, best_params)) + "\n")
            f.write("Decoded piece values:\n" + " ".join(map(str, best_piece_values)) + "\n")
            f.write("Decoded middle-game weights:\n" + " ".join(map(str, best_weights_mg)) + "\n")
    else:
        # Simple evaluation test.
        from bitboard import create_initial_board
        from game import Game
        board = create_initial_board()
        pos = {"board": board, "turn": "Gold", "no_capture_count": 0}
        params = [0.0] * 25
        score = evaluate_game(Game(), params)
        print("Evaluation value:", score)
