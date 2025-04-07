import random
import game
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

def index_to_algebraic(idx):
            layer, row, col = game.index_to_pos(idx)
            board_num = layer + 1
            return board_num

def get_sky_moves(state, color):
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
                    if index_to_algebraic(idx) == 1:
                            moves.append(move)
    return moves

def index_to_algebraic_orig(idx):
    layer, row, col = index_to_pos(idx)
    board_num = layer + 1
    file_letter = chr(ord('a') + col)
    rank = BOARD_ROWS - row
    return f"{board_num}{file_letter}{rank}"

def get_king_moves(state, current_turn, all_moves):
    board, _ = state
    pos = False

    for idx in range(game.TOTAL_SQUARES):
        piece = board[idx]
        if piece != 0:
            # Only consider moves for the player whose turn it is.
            if (current_turn == "Gold" and piece==-10) or (current_turn == "Scarlet" and piece == 10):
                pos = index_to_algebraic_orig(idx)
                #print(pos)
    
    if pos!=False:
        for move in all_moves:
            if(move[2]==pos):
                return move
    return False


        
# --- Custom AI using Heuristic ---
class CustomAI:
    """
    A simple minimax AI with alpha–beta pruning, iterative deepening, and a 5‑second
    time limit. The board is stored as a flat NumPy array, so move indices are integers.
    """
    def __init__(self, game, color):
        self.game = game
        self.color = color  # "Gold" or "Scarlet"

    def choose_move(self):
        turn = self.game.current_turn  # "Gold" or "Scarlet"
        turn_flag = 1 if turn == "Gold" else -1
        # Our game.board is a flat NumPy array.
        state = (self.game.board.copy(), turn_flag)
        history = self.game.state_history
        # Use 1 if "Gold", -1 if "Scarlet" for my_color.
        my_color_flag = 1 if self.color == "Gold" else -1
        #sky_moves = get_sky_moves(state, state[1])
        all_moves = self.game.get_all_moves()
        king_move = get_king_moves(state, turn, all_moves) 

        if king_move!=False:
             return random.choice(king_move)
        
        # if len(sky_moves)!=0: 
        #     return random.choice(sky_moves)
        else: 
            return random.choice(all_moves)
