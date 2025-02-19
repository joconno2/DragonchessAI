import copy
from pieces import in_bounds,is_empty,is_enemy 

# Given a move (start, end, flag) and a board, return whether it is legal.
def legal_move(move, board, color):
    start, end, flag = move
    if not in_bounds(end):
        return False
    dest = board.get(end)
    if flag == "quiet":
        return dest is None
    elif flag in ["capture", "afar"]:
        return dest is not None and dest.color != color
    elif flag in ["ambiguous", "3d"]:
        return dest is None or (dest is not None and dest.color != color)
    return False

# Generate all legal moves for pieces of the given color from the state.
# The state is a tuple: (board, turn)
def get_all_moves(state, color):
    board, turn = state
    moves = []
    for pos, piece in board.items():
        if piece is not None and piece.color == color:
            for move in piece.get_moves(pos, board):
                if legal_move(move, board, color):
                    moves.append(move)
    return moves

# Piece values for evaluation function.
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

# Evaluate a board state from the perspective of my_color.
def evaluate_state(state, my_color):
    board, turn = state
    my_score = 0
    enemy_score = 0
    for pos, piece in board.items():
        if piece is not None:
            value = piece_values.get(piece.name, 0)
            if piece.color == my_color:
                my_score += value
            else:
                enemy_score += value
    return my_score - enemy_score

# Simulate applying a move to the state.
def simulate_move(state, move):
    board, turn = state
    new_board = copy.deepcopy(board)
    start, end, flag = move
    piece = new_board[start]
    if flag == "afar":
        # Dragon's "capture from afar" – do not move the Dragon.
        new_board[end] = None
    else:
        # If there's an enemy, remove it.
        if new_board[end] is not None and new_board[end].color != piece.color:
            new_board[end] = None
        # Move the piece.
        new_board[end] = piece
        new_board[start] = None
    new_turn = "Scarlet" if turn == "Gold" else "Gold"
    return (new_board, new_turn)

def minimax(state, depth, maximizingPlayer, my_color):
    board, turn = state
    moves = get_all_moves(state, turn)
    # Terminal condition: depth limit reached or no legal moves.
    if depth == 0 or not moves:
        return evaluate_state(state, my_color), None
    best_move = None
    if maximizingPlayer:
        max_eval = -float('inf')
        for move in moves:
            new_state = simulate_move(state, move)
            eval_score, _ = minimax(new_state, depth - 1, False, my_color)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            new_state = simulate_move(state, move)
            eval_score, _ = minimax(new_state, depth - 1, True, my_color)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move


class CustomAI:
    def __init__(self, game, color):
        self.game = game
        self.color = color
        self.depth = 2  

    def choose_move(self):
        state = (copy.deepcopy(self.game.board), self.game.current_turn)
        _, best_move = minimax(state, self.depth, True, self.color)
        return best_move
