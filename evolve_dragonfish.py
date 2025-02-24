#!/usr/bin/env python3
"""
evolve_dragonfish.py

This module uses CMA-ES to evolve a 25-dimensional parameter vector for the
Dragonfish evaluation function (implemented in bots.dragonfish). The vector consists of:
  • 14 parameters for decoding piece values.
  • 11 weights for middle-game evaluation components.
  
For a given parameter vector, a CMAESBot is constructed that uses these weights
during its minimax search. The objective function simulates several games against
RandomAI and returns the negative average game score (for minimization).

This version uses a ProcessPoolExecutor to evaluate candidate solutions concurrently.
It also writes detailed logs (to "cma_es_log.txt") and saves the best individual’s
parameters to "best_weights.txt" for use with your main AI GUI.

Requirements:
  • Ensure that bitboard.py, game.py, ai.py, and bots/dragonfish.py are in PYTHONPATH.
  • Install the cma package (pip install cma).
"""

import math
import time
import numpy as np
import cma
import copy
import random
from concurrent.futures import ProcessPoolExecutor

from bitboard import pos_to_index, index_to_pos, BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES, create_initial_board
from game import Game, move_generators, board_state_hash
from ai import RandomAI
from bots.dragonfish import evaluate_game, decode_vector, main_evaluation

DEBUG = False

# Our new parameter vector dimension is 25
DIM = 25

def hash_state(state):
    board, turn_flag = state
    turn = "Gold" if turn_flag == 1 else "Scarlet"
    return board_state_hash(board, turn)

# Custom AI for CMA-ES evaluation.
class CMAESBot:
    def __init__(self, game, color, param_vector):
        self.game = game
        self.color = color
        self.color_flag = 1 if color == "Gold" else -1
        self.max_depth = 3
        self.param_vector = param_vector
        # decode_vector now returns (piece_values, weights_mg)
        self.piece_values, self.weights_mg = decode_vector(param_vector)
    
    def choose_move(self):
        state = (np.copy(self.game.board), 1 if self.game.current_turn=="Gold" else -1)
        history = self.game.state_history
        best_eval, best_move = iterative_deepening(state, self.max_depth, self.color_flag, history,
                                                     time_limit=5.0, weights_mg=self.weights_mg)
        # If no evaluation was returned, fall back immediately.
        if best_eval is None:
            fallback = get_all_moves(state, self.color_flag)
            if fallback:
                best_move = random.choice(fallback[:min(5, len(fallback))])
                if DEBUG:
                    print("Fallback move chosen (due to no best_eval):", best_move)
            return best_move
        # Invert best_eval to compare from our perspective.
        best_eval = -best_eval
        # Now, even if we got a move, check if there are other candidate moves with the same evaluation.
        candidate_moves = get_all_moves(state, self.color_flag)
        best_moves = []
        for move in candidate_moves:
            new_state = simulate_move(state, move)
            pos = {"board": new_state[0],
                   "turn": "Gold" if new_state[1] == 1 else "Scarlet",
                   "no_capture_count": len(history)}
            eval_val = main_evaluation(pos, self.weights_mg)
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

# Move flag constants (must match moves.py)
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

transposition_table = {}

def get_all_moves(state, color):
    from game import move_generators
    from bitboard import index_to_pos
    board, _ = state
    # Convert numeric color flag to string.
    color_str = "Gold" if color == 1 else "Scarlet"
    moves = []
    for idx in range(board.size):
        piece = board[idx]
        if piece != 0 and (color * piece > 0):
            pos_ = index_to_pos(idx)
            abs_code = abs(piece)
            gen_func = move_generators.get(abs_code)
            if gen_func is not None:
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

class TimeOutException(Exception):
    pass

def alphabeta(state, depth, alpha, beta, maximizingPlayer, my_color, history, current_depth=0, start_time=None, time_limit=None, weights_mg=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            raise TimeOutException
    key = hash_state(state)
    if key in transposition_table:
        cached_depth, cached_val, cached_move = transposition_table[key]
        if cached_depth >= depth:
            return cached_val, cached_move
    moves = get_all_moves(state, state[1])
    if depth == 0 or not moves:
        pos = {"board": state[0],
               "turn": "Gold" if state[1] == 1 else "Scarlet",
               "no_capture_count": len(history)}
        val = main_evaluation(pos, weights_mg)
        transposition_table[key] = (depth, val, None)
        return val, None
    best_move = None
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            new_state = simulate_move(state, move)
            eval_val, _ = alphabeta(new_state, depth-1, alpha, beta, False, my_color, history,
                                      current_depth+1, start_time, time_limit, weights_mg)
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
                                      current_depth+1, start_time, time_limit, weights_mg)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        transposition_table[key] = (depth, min_eval, best_move)
        return min_eval, best_move

def iterative_deepening(state, max_depth, my_color, history, time_limit=5.0, weights_mg=None):
    global transposition_table
    transposition_table = {}
    best_eval = None
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth+1):
        try:
            eval_val, move = alphabeta(state, depth, -math.inf, math.inf, True, my_color, history,
                                        current_depth=0, start_time=start_time, time_limit=time_limit, weights_mg=weights_mg)
            best_eval = eval_val
            best_move = move
        except TimeOutException:
            break
    return best_eval, best_move

EVAL_GAMES = 3
BONUS_FACTOR = 300.0

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

def main():
    log_file = open("cma_es_log.txt", "w")
    x0 = [0.0] * DIM
    sigma0 = 1.0
    opts = {
        'maxiter': 10,
        'popsize': 20,
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
        log_file.write("Generation {}: Best fitness (neg objective): {}\n".format(generation, best_in_gen))
        log_file.write("Best solution: {}\n".format(best_solution))
        log_file.flush()
        es.tell(solutions, fitnesses)
        es.disp()
    res = es.result
    best_params = res.xbest
    best_score = -res.fbest
    log_file.write("CMA-ES optimization complete.\n")
    log_file.write("Best average game score achieved: {}\n".format(best_score))
    log_file.write("Best parameter vector:\n{}\n".format(best_params))
    log_file.close()
    print("CMA-ES optimization complete.")
    print("Best average game score achieved:", best_score)
    print("Best parameter vector:")
    print(best_params)
    from bots.dragonfish import decode_vector
    best_piece_values, best_weights_mg = decode_vector(best_params)
    print("Decoded piece values:")
    print(best_piece_values)
    print("Decoded middle-game weights:")
    print(best_weights_mg)
    with open("best_weights.txt", "w") as f:
        f.write("Best parameter vector:\n")
        f.write(" ".join(map(str, best_params)) + "\n")
        f.write("Decoded piece values:\n")
        f.write(" ".join(map(str, best_piece_values)) + "\n")
        f.write("Decoded middle-game weights:\n")
        f.write(" ".join(map(str, best_weights_mg)) + "\n")

if __name__ == "__main__":
    main()
