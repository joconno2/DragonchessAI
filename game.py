import numpy as np
import hashlib
from bitboard import create_initial_board, pos_to_index, index_to_pos, TOTAL_SQUARES, BOARD_ROWS, BOARD_COLS
import moves
from moves import QUIET, CAPTURE, AFAR, AMBIGUOUS, THREED

# Map absolute piece code to its move generator function.
move_generators = {
    1: moves.generate_sylph_moves,
    2: moves.generate_griffin_moves,
    3: moves.generate_dragon_moves,
    4: moves.generate_oliphant_moves,
    5: moves.generate_unicorn_moves,
    6: moves.generate_hero_moves,
    7: moves.generate_thief_moves,
    8: moves.generate_cleric_moves,
    9: moves.generate_mage_moves,
    10: moves.generate_king_moves,
    11: moves.generate_paladin_moves,
    12: moves.generate_warrior_moves,
    13: moves.generate_basilisk_moves,
    14: moves.generate_elemental_moves,
    15: moves.generate_dwarf_moves
}

def board_state_hash(board, turn):
    state_bytes = board.tobytes() + turn.encode()
    return hashlib.sha256(state_bytes).hexdigest()

class Game:
    def __init__(self):
        self.board = create_initial_board()  # NumPy array (flat length TOTAL_SQUARES)
        self.current_turn = "Gold"
        self.state_history = []
        self.game_log = []           # list of move tuples: (from_idx, to_idx, flag)
        self.move_notations = []     # list of algebraic notation strings computed at move time
        self.no_capture_count = 0
        self.game_over = False
        self.winner = None
        # Frozen pieces array for the middle board.
        self.frozen = np.zeros(TOTAL_SQUARES, dtype=np.bool_)
    
    def get_all_moves(self):
        moves_list = []
        for idx in range(TOTAL_SQUARES):
            piece = self.board[idx]
            if piece != 0:
                if (self.current_turn == "Gold" and piece > 0) or (self.current_turn == "Scarlet" and piece < 0):
                    pos = index_to_pos(idx)
                    abs_code = abs(piece)
                    gen_func = move_generators.get(abs_code)
                    if gen_func:
                        candidate_moves = gen_func(pos, self.board, self.current_turn)
                        for move in candidate_moves:
                            from_idx, to_idx, flag = move
                            if flag == QUIET and self.board[to_idx] != 0:
                                continue
                            elif flag in (CAPTURE, AFAR):
                                if self.board[to_idx] == 0:
                                    continue
                                if (self.current_turn == "Gold" and self.board[to_idx] > 0) or \
                                   (self.current_turn == "Scarlet" and self.board[to_idx] < 0):
                                    continue
                            moves_list.append(move)
        return moves_list

    def get_legal_moves_for(self, from_index):
        all_moves = self.get_all_moves()
        return [move for move in all_moves if move[0] == from_index]

    def _move_to_algebraic(self, move, moving_piece):
        from_idx, to_idx, flag = move
        piece_letter = self.piece_letter(moving_piece)
        separator = "x" if flag == CAPTURE or flag == AFAR else "-"
        return f"{piece_letter}{self.index_to_algebraic(from_idx)}{separator}{self.index_to_algebraic(to_idx)}"

    def make_move(self, move):
        from_idx, to_idx, flag = move
        moving_piece = self.board[from_idx]  # retrieve before updating board
        move_alg = self._move_to_algebraic(move, moving_piece)
        self.move_notations.append(move_alg)
        self.game_log.append(move)
        capture_occurred = False
        if flag in (CAPTURE, AFAR):
            if self.board[to_idx] != 0:
                capture_occurred = True
            self.board[to_idx] = 0
        self.board[to_idx] = moving_piece
        self.board[from_idx] = 0
        if capture_occurred:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1
        self.state_history.append(board_state_hash(self.board, self.current_turn))
        self.current_turn = "Scarlet" if self.current_turn == "Gold" else "Gold"

    def update(self):
        self.frozen[:] = False
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                idx_bottom = pos_to_index(2, row, col)
                piece = self.board[idx_bottom]
                if piece == 13 or piece == -13:
                    idx_middle = pos_to_index(1, row, col)
                    target = self.board[idx_middle]
                    if target != 0 and (target * piece < 0):
                        self.frozen[idx_middle] = True
        if self.no_capture_count >= 250:
            self.game_over = True
            self.winner = "Draw"
        gold_king_exists = False
        scarlet_king_exists = False
        for idx in range(TOTAL_SQUARES):
            piece = self.board[idx]
            if piece == 10:
                gold_king_exists = True
            elif piece == -10:
                scarlet_king_exists = True
        if not gold_king_exists:
            self.game_over = True
            self.winner = "Scarlet"
        elif not scarlet_king_exists:
            self.game_over = True
            self.winner = "Gold"

    def piece_letter(self, piece):
        mapping = {
            1: "S", 2:"G", 3:"R", 4:"O", 5:"U", 6:"H", 7:"T", 8:"C",
            9: "M", 10:"K", 11:"P", 12:"W", 13:"B", 14:"E", 15:"D"
        }
        if piece > 0:
            return mapping.get(piece, "?")
        else:
            return mapping.get(-piece, "?").lower()

    def index_to_algebraic(self, idx):
        layer, row, col = index_to_pos(idx)
        board_num = layer + 1
        file_letter = chr(ord('a') + col)
        rank = BOARD_ROWS - row
        return f"{board_num}{file_letter}{rank}"