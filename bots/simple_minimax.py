import copy
import math
import time
import numpy as np
from bitboard import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, pos_to_index, index_to_pos

# Precomputed piece values (indices 1..15, with 0 for empty)
piece_values_arr = np.array([0, 1, 5, 8, 5, 2.5, 4.5, 4, 9, 11, 10000, 10, 1, 3, 4, 2], dtype=np.float64)

# Move flag constants (must match those in your move generators)
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

def board_state_hash(state):
    board, turn_flag = state
    # Fast hash using Python’s built‑in hash on the board’s bytes and turn flag.
    return hash((board.tobytes(), turn_flag))

# Import the Numba‑compiled move generators (they work on positions expressed as 1D indices)
from game import move_generators

def get_all_moves(state, color):
    """
    Given a state (board, turn_flag) where board is a flat NumPy array and
    color is 1 for Gold or -1 for Scarlet, return all legal moves.
    Each move is a triple (from_index, to_index, flag).
    Moves are ordered so that captures (or "afar" moves) come first.
    """
    board, _ = state
    moves = []
    for idx in range(board.size):
        piece = board[idx]
        if piece != 0 and (color * piece > 0):
            pos = index_to_pos(idx)
            abs_code = abs(piece)
            gen_func = move_generators.get(abs_code)
            if gen_func is not None:
                candidate_moves = gen_func(pos, board, color)
                for move in candidate_moves:
                    from_idx, to_idx, flag = move
                    # For QUIET moves, the destination must be empty.
                    if flag == QUIET and board[to_idx] != 0:
                        continue
                    # For CAPTURE/AFAR moves, destination must hold an enemy.
                    elif flag in (CAPTURE, AFAR):
                        if board[to_idx] == 0 or (color * board[to_idx] > 0):
                            continue
                    moves.append(move)
    # Order moves so that capture/afar moves come first (this helps alpha–beta pruning)
    moves.sort(key=lambda m: 0 if m[2] in (CAPTURE, AFAR) else 1)
    return moves

def evaluate_state(state, my_color, history):
    """
    Evaluate the state from the perspective of my_color.
    Uses a vectorized computation on the NumPy board and subtracts a penalty for repetitions.
    """
    board, _ = state
    gold_mask = board > 0
    scarlet_mask = board < 0
    score = np.sum(piece_values_arr[board[gold_mask]]) - np.sum(piece_values_arr[-board[scarlet_mask]])
    # Penalize repeated positions (draw detection)
    h = board_state_hash(state)
    penalty = history.count(h) * 5000
    # Multiply by my_color so that positive means favorable for the AI.
    return my_color * (score - penalty)

def simulate_move(state, move):
    """
    Simulate applying a move to a state.
    Returns a new state (new_board, new_turn_flag).
    The board is copied (using np.copy) to preserve the original.
    """
    board, turn_flag = state
    new_board = np.copy(board)
    from_idx, to_idx, flag = move
    piece = new_board[from_idx]
    if flag in (CAPTURE, AFAR):
        new_board[to_idx] = 0
    new_board[to_idx] = piece
    new_board[from_idx] = 0
    new_turn = -turn_flag
    return (new_board, new_turn)

# Global transposition table (caching evaluations)
transposition_table = {}

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history, current_depth=0):
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
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, False, my_color, history, current_depth + 1)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break  # Beta cutoff
        transposition_table[key] = (depth, max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, True, my_color, history, current_depth + 1)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break  # Alpha cutoff
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, time_limit):
    """
    Iteratively search deeper until a time limit (in seconds) is reached or max_depth is reached.
    Returns the best move found.
    """
    best_move = None
    best_eval = None
    start_time = time.time()
    depth = 1
    while depth <= max_depth:
        if time.time() - start_time >= time_limit:
            break
        eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, [])
        best_eval = eval_val
        best_move = move
        depth += 1
    return best_eval, best_move

# --- Custom AI using Minimax with Iterative Deepening and a Time Limit ---
class CustomAI:
    """
    A simple minimax AI with alpha–beta pruning, iterative deepening, and a 5‑second
    time limit. The board is stored as a flat NumPy array, so move indices are integers.
    """
    def __init__(self, game, color, time_limit=5.0):
        self.game = game
        self.color = color  # "Gold" or "Scarlet"
        self.time_limit = time_limit
        self.max_depth = 10  # Maximum depth to attempt

    def choose_move(self):
        turn = self.game.current_turn  # "Gold" or "Scarlet"
        turn_flag = 1 if turn == "Gold" else -1
        # Our game.board is a flat NumPy array.
        state = (self.game.board.copy(), turn_flag)
        history = self.game.state_history
        # Use 1 if "Gold", -1 if "Scarlet" for my_color.
        my_color_flag = 1 if self.color == "Gold" else -1
        _, best_move = iterative_deepening(state, self.max_depth, my_color_flag, self.time_limit)
        if best_move is not None:
            # Ensure that best_move is in the correct format (i.e. integer indices).
            # Some move generators might return positions as (layer, row, col) tuples.
            if isinstance(best_move[0], tuple):
                from_pos, to_pos, flag = best_move
                from_idx = pos_to_index(*from_pos)
                to_idx = pos_to_index(*to_pos)
                return (from_idx, to_idx, flag)
        return best_move
