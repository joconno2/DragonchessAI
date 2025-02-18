import math
import random
import numpy as np
from bitboard import NUM_BOARDS, BOARD_ROWS, BOARD_COLS, pos_to_index, index_to_pos

# Our state is represented as (board, turn_flag)
# where board is a 1D NumPy array of length NUM_BOARDS * BOARD_ROWS * BOARD_COLS,
# and turn_flag is 1 for Gold and -1 for Scarlet.

# Move flag constants (must match those used by your move generators)
QUIET     = 0
CAPTURE   = 1
AFAR      = 2
AMBIGUOUS = 3
THREED    = 4

def board_state_hash(state):
    board, turn_flag = state
    # Fast hash based on board bytes and turn flag
    return hash((board.tobytes(), turn_flag))

# Import the Numba‐compiled move generators from the game module.
from game import move_generators

def get_all_moves(state, color):
    board, turn_flag = state
    moves = []
    # Board is a flattened NumPy array
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
                    # For QUIET moves, ensure destination is empty.
                    if flag == QUIET and board[to_idx] != 0:
                        continue
                    # For CAPTURE/AFAR moves, destination must hold an enemy.
                    elif flag in (CAPTURE, AFAR):
                        if board[to_idx] == 0 or (color * board[to_idx] > 0):
                            continue
                    moves.append(move)
    # Order moves so that capture/afar moves come first (improves pruning)
    moves.sort(key=lambda m: 0 if m[2] in (CAPTURE, AFAR) else 1)
    return moves

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

def is_terminal(state):
    board, _ = state
    # Gold king is represented by 10; Scarlet king by -10.
    gold_king = np.any(board == 10)
    scarlet_king = np.any(board == -10)
    return not (gold_king and scarlet_king)

def rollout(state, max_moves=250):
    board, turn_flag = state
    current_state = (np.copy(board), turn_flag)
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
    board, _ = state
    gold_king = np.any(board == 10)
    scarlet_king = np.any(board == -10)
    if gold_king and not scarlet_king:
        return 1 if root_player == "Gold" else -1
    elif scarlet_king and not gold_king:
        return 1 if root_player == "Scarlet" else -1
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
        # UCT formula: average win rate plus exploration term.
        return max(self.children, key=lambda c: (c.wins / c.visits) + math.sqrt(2 * math.log(self.visits) / c.visits))

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
    A Monte Carlo Tree Search (MCTS) AI for Dragonchess using the new state representation.
    """
    def __init__(self, game, color, iterations=10):
        self.game = game
        self.color = color  # "Gold" or "Scarlet"
        self.iterations = iterations

    def choose_move(self):
        turn = self.game.current_turn  # "Gold" or "Scarlet"
        turn_flag = 1 if turn == "Gold" else -1
        state = (np.copy(self.game.board), turn_flag)
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = root
            state_copy = (np.copy(state[0]), state[1])
            # SELECTION: traverse the tree to a leaf node.
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                state_copy = simulate_move(state_copy, node.move)
            # EXPANSION: if untried moves remain, expand one.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state_copy = simulate_move(state_copy, move)
                node = node.add_child(move, state_copy)
            # SIMULATION: run a rollout from this new node.
            final_state = rollout(state_copy)
            outcome = result(final_state, self.color)
            # BACKPROPAGATION: update nodes with the outcome.
            while node is not None:
                node.update(outcome)
                node = node.parent
        if not root.children:
            return None
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move
