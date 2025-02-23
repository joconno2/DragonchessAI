# ga_bot.py
import copy
import math
import time
import numpy as np
from bitboard import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, pos_to_index, index_to_pos
from game import move_generators

# Move flag constants.
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

# The “original” piece values for indices 1–15 (index 0 is empty).
# We will evolve values for indices: 1–9 and 11–15 (leaving king at index 10 fixed).
original_values = {
    1: 1,
    2: 5,
    3: 20,
    4: 6,
    5: 2.5,
    6: 5,
    7: 4,
    8: 9,
    9: 11,
    11: 15,
    12: 1,
    13: 3,
    14: 4,
    15: 2
}
# The gene order for the evolved values.
gene_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
BITS_PER_GENE = 8

def decode_chromosome(chromosome):
    """
    Given a binary string of length 112 (14 genes × 8 bits), decode to a piece–values
    array of length 16. Index 0 is 0 and index 10 (king) is fixed at 10000.
    For each evolved gene, we scale from [0, 255] to [0.5×original, 2×original].
    """
    if len(chromosome) != BITS_PER_GENE * len(gene_indices):
        raise ValueError("Chromosome length must be {}".format(BITS_PER_GENE * len(gene_indices)))
    pv = np.zeros(16, dtype=np.float64)
    for i, gene_index in enumerate(gene_indices):
        gene_bits = chromosome[i*BITS_PER_GENE:(i+1)*BITS_PER_GENE]
        gene_int = int(gene_bits, 2)
        lower = 0.5 * original_values[gene_index]
        upper = 2 * original_values[gene_index]
        value = lower + (gene_int / 255.0) * (upper - lower)
        pv[gene_index] = value
    pv[0] = 0.0
    pv[10] = 10000.0  # King’s value fixed.
    return pv

# Global variable that our evaluation function uses.
current_piece_values = None
transposition_table = {}

def board_state_hash(state):
    board, turn_flag = state
    return hash((board.tobytes(), turn_flag))

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
    # Sort moves so that capture moves come first.
    def move_key(m):
        flag = m[2]
        if flag in (CAPTURE, AFAR):
            captured = board[m[1]]
            captured_value = current_piece_values[abs(captured)] if captured != 0 else 0
            return (0, -captured_value)
        else:
            return (1, 0)
    moves.sort(key=move_key)
    return moves

def evaluate_state(state, my_color, history):
    board, _ = state
    gold_mask = board > 0
    scarlet_mask = board < 0
    score = np.sum(current_piece_values[board[gold_mask].astype(int)]) - \
            np.sum(current_piece_values[-board[scarlet_mask].astype(int)])
    h = board_state_hash(state)
    penalty = history.count(h) * 5000
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

class TimeOutException(Exception):
    pass

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
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, False, my_color, history,
                                      current_depth + 1, start_time, time_limit)
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
            eval_val, _ = alphabeta(new_state, depth - 1, alpha, beta, True, my_color, history,
                                      current_depth + 1, start_time, time_limit)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0):
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        try:
            eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history,
                                        current_depth=0, start_time=start_time, time_limit=time_limit)
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            break
    return best_eval, best_move

class GeneticBot:
    """
    A minimax bot whose evaluation function uses a set of piece values
    decoded from a binary chromosome.
    """
    def __init__(self, game, color, chromosome):
        self.game = game
        self.color = color  # "Gold" or "Scarlet"
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 3  # Adjust depth as needed.
        self.piece_values = decode_chromosome(chromosome)
        global current_piece_values
        current_piece_values = self.piece_values
        global transposition_table
        transposition_table = {}
    
    def choose_move(self):
        turn = self.game.current_turn
        turn_flag = 1 if turn == "Gold" else -1
        state = (np.copy(self.game.board), turn_flag)
        history = self.game.state_history
        _, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history, time_limit=5.0)
        return best_move
