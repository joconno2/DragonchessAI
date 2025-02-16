import copy
import math
import hashlib
import numpy as np
from game import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, piece_to_int, board_state_hash_numpy

piece_values = {
    "King": 10000,
    "Mage": 11,
    "Paladin": 10,
    "Cleric": 9,
    "Dragon": 8,
    "Griffin": 5,
    "Oliphant": 5,
    "Hero": 4.5,
    "Thief": 4,
    "Elemental": 4,
    "Basilisk": 3,
    "Unicorn": 2.5,
    "Dwarf": 2,
    "Sylph": 1,
    "Warrior": 1
}

def in_bounds(pos):
    layer, row, col = pos
    return 0 <= layer < 3 and 0 <= row < 8 and 0 <= col < 12

def is_empty(pos, board):
    if not in_bounds(pos):
        return False
    return board.get(pos) is None

def is_enemy(pos, board, color):
    if not in_bounds(pos):
        return False
    piece = board.get(pos)
    return piece is not None and piece.color != color

# NEW: Convert dictionary board to a NumPy array.
def dict_to_numpy(board):
    np_board = np.zeros((NUM_BOARDS, BOARD_ROWS, BOARD_COLS), dtype=np.int16)
    for pos, piece in board.items():
        layer, row, col = pos
        if piece:
            np_board[layer, row, col] = piece_to_int.get(piece.symbol, 0)
        else:
            np_board[layer, row, col] = 0
    return np_board

# NEW: Use the NumPy hash for state hashing.
def board_state_hash(state):
    board, turn = state
    np_board = dict_to_numpy(board)
    return board_state_hash_numpy(np_board, turn)

def get_all_moves(state, color):
    board, turn = state
    moves = []
    for pos, piece in board.items():
        if piece is not None and piece.color == color:
            raw_moves = piece.get_moves(pos, board)
            for move in raw_moves:
                if len(move) == 2:
                    start, end = move
                    flag = "quiet"
                else:
                    start, end, flag = move
                if not in_bounds(end):
                    continue
                dest = board.get(end)
                if flag == "quiet":
                    if dest is None:
                        moves.append(move)
                elif flag == "capture":
                    if dest is not None and dest.color != piece.color:
                        moves.append(move)
                elif flag == "afar":
                    if dest is not None and dest.color != piece.color:
                        moves.append(move)
                elif flag in ["ambiguous", "3d"]:
                    if dest is None or (dest is not None and dest.color != piece.color):
                        moves.append(move)
                else:
                    moves.append(move)
    moves.sort(key=lambda m: 0 if (len(m)==3 and m[2] in ["capture","afar"]) else 1)
    return moves

def evaluate_state(state, my_color, history):
    board, turn = state
    score = 0
    for pos, piece in board.items():
        if piece is not None:
            val = piece_values.get(piece.name, 0)
            if piece.color == my_color:
                score += val
            else:
                score -= val
    # Use our fast hash.
    h = board_state_hash(state)
    repetition_count = history.count(h)
    penalty = repetition_count * 5000
    return score - penalty

def simulate_move(state, move):
    board, turn = state
    new_board = copy.deepcopy(board)  # Consider writing a custom board copy if needed.
    if len(move) == 2:
        start, end = move
        flag = "quiet"
    else:
        start, end, flag = move
    piece = new_board[start]
    dest_piece = new_board[end]
    if flag == "afar":
        if dest_piece is not None and dest_piece.color != piece.color:
            new_board[end] = None
    else:
        if dest_piece is not None and dest_piece.color != piece.color:
            new_board[end] = None
        new_board[end] = piece
        new_board[start] = None
    new_turn = "Scarlet" if turn == "Gold" else "Gold"
    return (new_board, new_turn)

transposition_table = {}

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history):
    key = board_state_hash(state)
    if key in transposition_table:
        cached_depth, cached_val, cached_move = transposition_table[key]
        if cached_depth >= depth:
            return cached_val, cached_move

    moves = get_all_moves(state, state[1])
    if depth == 0 or not moves:
        eval_val = evaluate_state(state, my_color, history)
        transposition_table[key] = (depth, eval_val, None)
        return eval_val, None

    best_move = None
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth-1, alpha, beta, False, my_color, history)
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
            eval_val, _ = alphabeta(new_state, depth-1, alpha, beta, True, my_color, history)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

class CustomAI:
    def __init__(self, game, color):
        self.game = game
        self.color = color
        self.depth = 2  # Increase as performance permits.

    def choose_move(self):
        state = (copy.deepcopy(self.game.board), self.game.current_turn)
        history = self.game.state_history
        eval_val, best_move = alphabeta(state, self.depth, -math.inf, math.inf, True, self.color, history)
        return best_move
