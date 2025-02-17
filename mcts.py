import copy
import math
import random
import hashlib
import numpy as np
from game import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, piece_to_int, board_state_hash_numpy

def in_bounds(pos):
    layer, row, col = pos
    return 0 <= layer < 3 and 0 <= row < 8 and 0 <= col < 12

def dict_to_numpy(board):
    np_board = np.zeros((NUM_BOARDS, BOARD_ROWS, BOARD_COLS), dtype=np.int16)
    for pos, piece in board.items():
        layer, row, col = pos
        if piece:
            np_board[layer, row, col] = piece_to_int.get(piece.symbol, 0)
        else:
            np_board[layer, row, col] = 0
    return np_board

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
                if len(move)==2:
                    start, end = move
                    flag = "quiet"
                else:
                    start, end, flag = move
                if not in_bounds(end):
                    continue
                dest = board.get(end)
                if flag == "quiet" and dest is None:
                    moves.append(move)
                elif flag == "capture" and dest is not None and dest.color != piece.color:
                    moves.append(move)
                elif flag == "afar" and dest is not None and dest.color != piece.color:
                    moves.append(move)
                elif flag in ["ambiguous", "3d"] and (dest is None or (dest is not None and dest.color != piece.color)):
                    moves.append(move)
    return moves

def simulate_move(state, move):
    board, turn = state
    new_board = copy.deepcopy(board)
    if len(move)==2:
        start, end = move
        flag = "quiet"
    else:
        start, end, flag = move
    piece = new_board[start]
    dest_piece = new_board[end]
    if flag=="afar":
        if dest_piece is not None and dest_piece.color != piece.color:
            new_board[end] = None
    else:
        if dest_piece is not None and dest_piece.color != piece.color:
            new_board[end] = None
        new_board[end] = piece
        new_board[start] = None
    new_turn = "Scarlet" if turn=="Gold" else "Gold"
    return (new_board, new_turn)

def is_terminal(state):
    board, turn = state
    gold_king = False
    scarlet_king = False
    for pos, piece in board.items():
        if piece and piece.name.lower() == "king":
            if piece.color=="Gold":
                gold_king = True
            elif piece.color=="Scarlet":
                scarlet_king = True
    return not (gold_king and scarlet_king)

def rollout(state, max_moves=50):
    current_state = copy.deepcopy(state)
    for _ in range(max_moves):
        if is_terminal(current_state):
            break
        moves = get_all_moves(current_state, current_state[1])
        if not moves:
            break
        move = random.choice(moves)
        current_state = simulate_move(current_state, move)
    return current_state

def result(state, root_player):
    board, turn = state
    gold_king = False
    scarlet_king = False
    for pos, piece in board.items():
        if piece and piece.name.lower() == "king":
            if piece.color=="Gold":
                gold_king = True
            elif piece.color=="Scarlet":
                scarlet_king = True
    if gold_king and not scarlet_king:
        return 1 if root_player=="Gold" else -1
    elif scarlet_king and not gold_king:
        return 1 if root_player=="Scarlet" else -1
    return 0

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = get_all_moves(state, state[1])
        self.visits = 0
        self.wins = 0

    def uct_select_child(self):
        return max(self.children, key=lambda c: c.wins/c.visits + math.sqrt(2 * math.log(self.visits)/c.visits))

    def add_child(self, move, state):
        child = MCTSNode(state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, outcome):
        self.visits += 1
        self.wins += outcome

class CustomAI:
    """
    A Monte Carlo Tree Search AI for Dragonchess.
    """
    def __init__(self, game, color, iterations=500):
        self.game = game
        self.color = color
        self.iterations = iterations

    def choose_move(self):
        state = (copy.deepcopy(self.game.board), self.game.current_turn)
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = root
            state_copy = copy.deepcopy(state)
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                state_copy = simulate_move(state_copy, node.move)
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state_copy = simulate_move(state_copy, move)
                node = node.add_child(move, state_copy)
            final_state = rollout(state_copy)
            outcome = result(final_state, self.color)
            while node is not None:
                node.update(outcome)
                node = node.parent
        if not root.children:
            return None
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move
