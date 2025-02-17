import math
import numpy as np
from bitboard import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, pos_to_index, index_to_pos

# Precompute a NumPy array for piece values (indices 1..15; index 0 for empty)
piece_values_arr = np.array([0, 1, 5, 8, 5, 2.5, 4.5, 4, 9, 11, 10000, 10, 1, 3, 4, 2], dtype=np.float64)

# Move flag constants (must match those in moves.py)
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

def board_state_hash(state):
    board, turn_flag = state
    # Use built-in hash (much faster than SHA256)
    return hash((board.tobytes(), turn_flag))

# Import the Numba‐compiled move generators from the game module.
from game import move_generators

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
    # Order moves so that capture/afar moves come first (improves pruning)
    moves.sort(key=lambda m: 0 if m[2] in (CAPTURE, AFAR) else 1)
    return moves

def evaluate_state(state, my_color, history):
    board, _ = state
    # Vectorized evaluation: add values for Gold pieces, subtract for Scarlet.
    gold_mask = board > 0
    scarlet_mask = board < 0
    score = np.sum(piece_values_arr[board[gold_mask]]) - np.sum(piece_values_arr[-board[scarlet_mask]])
    # Count repetitions (if any) for draw detection
    h = board_state_hash(state)
    penalty = history.count(h) * 5000
    # Multiply by my_color so that from the AI’s perspective a favorable state is positive.
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

# Global transposition table for caching evaluations.
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
                break
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
                break
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history):
    best_eval = None
    best_move = None
    # Optionally, you can clear the transposition table at each iteration:
    # transposition_table.clear()
    for depth in range(1, max_depth + 1):
        eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history)
        best_eval = eval_val
        best_move = move
    return best_eval, best_move

class CustomAI:
    def __init__(self, game, color):
        self.game = game
        # Convert color string to integer flag: "Gold" -> 1, "Scarlet" -> -1.
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 3  # You can increase this if you have more time

    def choose_move(self):
        turn = self.game.current_turn  # "Gold" or "Scarlet"
        turn_flag = 1 if turn == "Gold" else -1
        state = (np.copy(self.game.board), turn_flag)
        history = self.game.state_history
        _, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history)
        return best_move
