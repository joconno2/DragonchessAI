import copy
import math
import time
import numpy as np
from bitboard import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, pos_to_index, index_to_pos

# Precompute piece values: index 0 = empty; indices 1..15 for piece codes.
piece_values_arr = np.array(
    [0, 1.42, 2.88, 37.76, 8.89, 1.44, 5.24, 3.01, 13.13, 18.51, 10000, 14.12, 1.09, 3.97, 6.71, 2.53],
    dtype=np.float64
)

# Move flag constants (must match moves.py)
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

def board_state_hash(state):
    board, turn_flag = state
    # Use built-in hash over the board's raw bytes and the turn flag.
    return hash((board.tobytes(), turn_flag))

# Import the Numba‐compiled move generators from the game module.
from game import move_generators

# Custom exception used to abort search if time runs out.
class TimeOutException(Exception):
    pass

def get_all_moves(state, color):
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
                    if flag == QUIET and board[to_idx] != 0:
                        continue
                    elif flag in (CAPTURE, AFAR):
                        if board[to_idx] == 0:
                            continue
                        if color * board[to_idx] > 0:
                            continue
                    moves.append(move)
    # --- Improved Move Ordering ---
    # Sort moves so that capture moves (CAPTURE or AFAR) are considered first.
    # For capture moves, we sort in descending order by the value of the piece being captured.
    def move_key(m):
        flag = m[2]
        if flag in (CAPTURE, AFAR):
            captured = board[m[1]]
            captured_value = piece_values_arr[abs(captured)] if captured != 0 else 0
            return (0, -captured_value)
        else:
            return (1, 0)
    moves.sort(key=move_key)
    return moves

def evaluate_state(state, my_color, history):
    board, _ = state
    # Vectorized evaluation: add values for Gold pieces, subtract for Scarlet.
    gold_mask = board > 0
    scarlet_mask = board < 0
    score = np.sum(piece_values_arr[board[gold_mask]]) - np.sum(piece_values_arr[-board[scarlet_mask]])
    h = board_state_hash(state)
    penalty = history.count(h) * 5000
    # Multiply by my_color so that from our AI’s perspective, favorable states are positive.
    return my_color * (score - penalty)

def simulate_move(state, move):
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

# Global transposition table.
transposition_table = {}

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history, current_depth=0, start_time=None, time_limit=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            raise TimeOutException
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
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, False, my_color, history, current_depth + 1, start_time, time_limit)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break  # Beta cutoff.
        transposition_table[key] = (depth, max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, True, my_color, history, current_depth + 1, start_time, time_limit)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break  # Alpha cutoff.
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0):
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        try:
            eval_val, move = alphabeta(
                state, depth, -math.inf, math.inf, True, my_color, history,
                current_depth=0, start_time=start_time, time_limit=time_limit
            )
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            # Time limit reached; break and return the best move from the last completed iteration.
            break
    return best_eval, best_move

class CustomAI:
    def __init__(self, game, color):
        self.game = game
        # Convert color string to numeric flag: "Gold" -> 1, "Scarlet" -> -1.
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 3  # Set your maximum depth as desired.

    def choose_move(self):
        turn = self.game.current_turn  # "Gold" or "Scarlet"
        turn_flag = 1 if turn == "Gold" else -1
        state = (np.copy(self.game.board), turn_flag)
        history = self.game.state_history
        # Use iterative deepening with a 5-second time limit.
        _, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history, time_limit=5.0)
        return best_move
