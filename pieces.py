# -------------
# Base Piece
# -------------
class Piece:
    def __init__(self, name, color):
        self.name = name       # e.g. "Sylph", "Griffin", etc.
        self.color = color     # "Gold" or "Scarlet"
        self.symbol = self.get_symbol()
        self.frozen = False

    def get_symbol(self):
        # Returns keys like "gold_sylph" (all lowercase).
        return f"{self.color.lower()}_{self.name.lower()}"

    def get_moves(self, pos, board):
        # To be overridden in subclasses.
        return []


# ------------------
# Helper Functions
# ------------------
def in_bounds(pos):
    layer, row, col = pos
    return 0 <= layer < 3 and 0 <= row < 8 and 0 <= col < 12

def is_empty(pos, board):
    if not in_bounds(pos):
        return False
    return board.get(pos) is None

def is_enemy(pos, board, color):
    if not in_bounds(pos):
        return False
    piece = board.get(pos)
    return piece is not None and piece.color != color

def get_sylph_home_cells(color):
    # Home cells (on the TOP board) for Sylphs.
    cells = []
    if color == "Gold":
        # For Gold, assume home cells on row 7 (rank 1) at columns 0,2,4,6,8,10.
        row = 7
        for col in [0, 2, 4, 6, 8, 10]:
            cells.append((0, row, col))
    else:
        # For Scarlet, assume home cells on row 0 (rank 8) at columns 1,3,5,7,9,11.
        row = 0
        for col in [1, 3, 5, 7, 9, 11]:
            cells.append((0, row, col))
    return cells


# ------------------
# Pieces on the Top Board
# ------------------

class Sylph(Piece):
    def __init__(self, color):
        super().__init__("Sylph", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        direction = -1 if self.color == "Gold" else 1
        if layer == 0:  # TOP board
            # Non-capturing: diagonal move (only if target is empty)
            for dc in [-1, 1]:
                new_pos = (layer, row + direction, col + dc)
                if in_bounds(new_pos) and is_empty(new_pos, board):
                    moves.append((pos, new_pos, "quiet"))
            # Capturing: directly forward.
            new_pos = (layer, row + direction, col)
            if in_bounds(new_pos) and is_enemy(new_pos, board, self.color):
                moves.append((pos, new_pos, "capture"))
            # Capturing: straight down to the middle board.
            new_pos = (1, row, col)
            if in_bounds(new_pos) and is_enemy(new_pos, board, self.color):
                moves.append((pos, new_pos, "capture"))
        elif layer == 1:  # MIDDLE board
            new_pos = (0, row, col)
            if in_bounds(new_pos) and is_empty(new_pos, board):
                moves.append((pos, new_pos, "quiet"))
            for cell in get_sylph_home_cells(self.color):
                if in_bounds(cell) and is_empty(cell, board):
                    moves.append((pos, cell, "quiet"))
        return moves


class Griffin(Piece):
    def __init__(self, color):
        super().__init__("Griffin", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 0:
            offsets = [(3,2), (3,-2), (-3,2), (-3,-2), (2,3), (2,-3), (-2,3), (-2,-3)]
            for dr, dc in offsets:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    new_pos = (1, row+dr, col+dc)
                    if in_bounds(new_pos):
                        moves.append((pos, new_pos, "ambiguous"))
        elif layer == 1:
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    new_pos = (layer, row+dr, col+dc)
                    if in_bounds(new_pos):
                        moves.append((pos, new_pos, "ambiguous"))
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    new_pos = (0, row+dr, col+dc)
                    if in_bounds(new_pos):
                        moves.append((pos, new_pos, "ambiguous"))
        return moves


class Dragon(Piece):
    def __init__(self, color):
        super().__init__("Dragon", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 0:
            return moves
        # King–like moves: now ambiguous so they allow capturing.
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            new_pos = (layer, row+dr, col+dc)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        # Bishop–like sliding moves.
        diag_dirs = [(-1,-1), (-1,1), (1,-1), (1,1)]
        for dr, dc in diag_dirs:
            r, c = row, col
            while True:
                r += dr; c += dc
                new_pos = (layer, r, c)
                if not in_bounds(new_pos):
                    break
                moves.append((pos, new_pos, "ambiguous"))
                if board.get(new_pos) is not None:
                    break
        # "Capture from afar" moves.
        target_cells = []
        direct = (1, row, col)
        if in_bounds(direct):
            target_cells.append(direct)
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_pos = (1, row+dr, col+dc)
            if in_bounds(new_pos):
                target_cells.append(new_pos)
        for target in target_cells:
            moves.append((pos, target, "afar"))
        return moves


# ------------------
# Pieces on the Middle Board
# ------------------

class Oliphant(Piece):
    def __init__(self, color):
        super().__init__("Oliphant", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 1:
            return moves
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        for dr, dc in directions:
            r, c = row, col
            while True:
                r += dr; c += dc
                new_pos = (layer, r, c)
                if not in_bounds(new_pos):
                    break
                moves.append((pos, new_pos, "ambiguous"))
                if board.get(new_pos) is not None:
                    break
        return moves


class Unicorn(Piece):
    def __init__(self, color):
        super().__init__("Unicorn", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 1:
            return moves
        offsets = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        for dr, dc in offsets:
            new_pos = (layer, row+dr, col+dc)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        return moves


class Hero(Piece):
    def __init__(self, color):
        super().__init__("Hero", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 1:
            # On MIDDLE, one or two cells diagonally.
            for dr in [-1, -2, 1, 2]:
                for dc in [-1, -2, 1, 2]:
                    if abs(dr) == abs(dc):
                        new_pos = (layer, row+dr, col+dc)
                        if in_bounds(new_pos):
                            moves.append((pos, new_pos, "ambiguous"))
            # Also, move to TOP or BOTTOM.
            for target_layer in [0, 2]:
                for dr in [-1, 1]:
                    for dc in [-1, 1]:
                        new_pos = (target_layer, row+dr, col+dc)
                        if in_bounds(new_pos):
                            moves.append((pos, new_pos, "ambiguous"))
        else:
            target_layer = 1
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    new_pos = (target_layer, row+dr, col+dc)
                    if in_bounds(new_pos):
                        moves.append((pos, new_pos, "ambiguous"))
        return moves


class Thief(Piece):
    def __init__(self, color):
        super().__init__("Thief", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 1:
            return moves
        directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
        for dr, dc in directions:
            r, c = row, col
            while True:
                r += dr; c += dc
                new_pos = (layer, r, c)
                if not in_bounds(new_pos):
                    break
                moves.append((pos, new_pos, "ambiguous"))
                if board.get(new_pos) is not None:
                    break
        return moves


class Cleric(Piece):
    def __init__(self, color):
        super().__init__("Cleric", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            new_pos = (layer, row+dr, col+dc)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        if layer == 0:
            new_pos = (1, row, col)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        elif layer == 1:
            for target in [0, 2]:
                new_pos = (target, row, col)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        elif layer == 2:
            new_pos = (1, row, col)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        return moves


class Mage(Piece):
    def __init__(self, color):
        super().__init__("Mage", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 1:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dr, dc in directions:
                r, c = row, col
                while True:
                    r += dr; c += dc
                    new_pos = (layer, r, c)
                    if not in_bounds(new_pos):
                        break
                    moves.append((pos, new_pos, "ambiguous"))
                    if board.get(new_pos) is not None:
                        break
            for target in [0, 2]:
                new_pos = (target, row, col)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        else:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
            for dr in [-2, -1, 1, 2]:
                new_pos = (layer, row+dr, col)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        return moves


class King(Piece):
    def __init__(self, color):
        super().__init__("King", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 1:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dr, dc in directions:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
            for target in [0, 2]:
                new_pos = (target, row, col)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        else:
            new_pos = (1, row, col)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        return moves


class Paladin(Piece):
    def __init__(self, color):
        super().__init__("Paladin", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 1:
            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
            for dr, dc in [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        else:
            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos):
                    moves.append((pos, new_pos, "ambiguous"))
        # Add 3D Knight moves (flag "3d" remains unchanged).
        for d_layer in [-2, -1, 1, 2]:
            for d_row in [-2, -1, 0, 1, 2]:
                for d_col in [-2, -1, 0, 1, 2]:
                    if d_layer == 0:
                        continue
                    diffs = sorted([abs(d_layer), abs(d_row), abs(d_col)])
                    if diffs == [0, 1, 2]:
                        new_pos = (layer+d_layer, row+d_row, col+d_col)
                        if in_bounds(new_pos):
                            moves.append((pos, new_pos, "3d"))
        return moves


class Warrior(Piece):
    def __init__(self, color):
        super().__init__("Warrior", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 1:
            return moves
        direction = -1 if self.color=="Gold" else 1
        new_pos = (layer, row+direction, col)
        if in_bounds(new_pos) and is_empty(new_pos, board):
            moves.append((pos, new_pos, "quiet"))
        for dc in [-1, 1]:
            new_pos = (layer, row+direction, col+dc)
            if in_bounds(new_pos) and is_enemy(new_pos, board, self.color):
                moves.append((pos, new_pos, "capture"))
        return moves


# ------------------
# Pieces on the Bottom Board
# ------------------

class Basilisk(Piece):
    def __init__(self, color):
        super().__init__("Basilisk", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer != 2:
            return moves
        direction = -1 if self.color=="Gold" else 1
        # Forward moves: allow moving or capturing.
        for dc in [0, -1, 1]:
            new_pos = (layer, row+direction, col+dc)
            if in_bounds(new_pos):
                moves.append((pos, new_pos, "ambiguous"))
        # Backward move (non-capturing only).
        new_pos = (layer, row - direction, col)
        if in_bounds(new_pos) and is_empty(new_pos, board):
            moves.append((pos, new_pos, "quiet"))
        return moves


class Elemental(Piece):
    def __init__(self, color):
        super().__init__("Elemental", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer == 2:
            # Orthogonal moves (one or two cells) with the intermediate cell check.
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                for dist in [1,2]:
                    new_pos = (layer, row+dr*dist, col+dc*dist)
                    if not in_bounds(new_pos):
                        break
                    if dist == 1:
                        if is_empty(new_pos, board) or is_enemy(new_pos, board, self.color):
                            moves.append((pos, new_pos, "ambiguous"))
                        else:
                            break
                    else:
                        inter_pos = (layer, row+dr, col+dc)
                        if not is_empty(inter_pos, board):
                            break
                        if is_empty(new_pos, board) or is_enemy(new_pos, board, self.color):
                            moves.append((pos, new_pos, "ambiguous"))
                        else:
                            break
            # One-cell diagonal move (non-capturing).
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                new_pos = (layer, row+dr, col+dc)
                if in_bounds(new_pos) and is_empty(new_pos, board):
                    moves.append((pos, new_pos, "quiet"))
            # Capturing move to the MIDDLE board.
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                inter_pos = (layer, row+dr, col+dc)
                target = (1, row+dr, col+dc)
                if in_bounds(inter_pos) and is_empty(inter_pos, board) and in_bounds(target) and is_enemy(target, board, self.color):
                    moves.append((pos, target, "capture"))
        elif layer == 1:
            # From MIDDLE, Elemental may only return to BOTTOM.
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                inter_pos = (layer, row+dr, col+dc)
                target = (2, row+dr, col+dc)
                if in_bounds(inter_pos) and is_empty(inter_pos, board) and in_bounds(target):
                    if is_enemy(target, board, self.color):
                        moves.append((pos, target, "capture"))
                    elif is_empty(target, board):
                        moves.append((pos, target, "quiet"))
        return moves


class Dwarf(Piece):
    def __init__(self, color):
        super().__init__("Dwarf", color)
    def get_moves(self, pos, board):
        moves = []
        layer, row, col = pos
        if layer not in [1,2]:
            return moves
        direction = -1 if self.color=="Gold" else 1
        new_pos = (layer, row+direction, col)
        if in_bounds(new_pos) and is_empty(new_pos, board):
            moves.append((pos, new_pos, "quiet"))
        for dc in [-1, 1]:
            new_pos = (layer, row, col+dc)
            if in_bounds(new_pos) and is_empty(new_pos, board):
                moves.append((pos, new_pos, "quiet"))
        for dc in [-1, 1]:
            new_pos = (layer, row+direction, col+dc)
            if in_bounds(new_pos) and is_enemy(new_pos, board, self.color):
                moves.append((pos, new_pos, "capture"))
        if layer == 2:
            target = (1, row, col)
            if in_bounds(target) and is_enemy(target, board, self.color):
                moves.append((pos, target, "capture"))
        if layer == 1:
            target = (2, row, col)
            if in_bounds(target) and is_empty(target, board):
                moves.append((pos, target, "quiet"))
        # Sean Shubin's rule: allow reverse direction if at an end.
        if not in_bounds((layer, row+direction, col)):
            reverse = -direction
            new_pos = (layer, row+reverse, col)
            if in_bounds(new_pos) and is_empty(new_pos, board):
                moves.append((pos, new_pos, "quiet"))
        return moves


# ------------------
# Helper Functions
# ------------------
def in_bounds(pos):
    layer, row, col = pos
    return 0 <= layer < 3 and 0 <= row < 8 and 0 <= col < 12

def is_empty(pos, board):
    if not in_bounds(pos):
        return False
    return board.get(pos) is None

def is_enemy(pos, board, color):
    if not in_bounds(pos):
        return False
    piece = board.get(pos)
    return piece is not None and piece.color != color

def get_sylph_home_cells(color):
    # Return a list of home–cell positions for Sylphs (on the TOP board).
    cells = []
    if color == "Gold":
        # For Gold, assume home cells on row 7 (rank 1) at columns 0,2,4,6,8,10.
        row = 7
        for col in [0,2,4,6,8,10]:
            cells.append((0, row, col))
    else:
        # For Scarlet, assume home cells on row 0 (rank 8) at columns 1,3,5,7,9,11.
        row = 0
        for col in [1,3,5,7,9,11]:
            cells.append((0, row, col))
    return cells

def create_initial_setup():
    """
    Returns a dictionary mapping positions (layer, row, col) to Piece objects,
    according to the Dragonchess starting layout.
    """
    board = {}
    for layer in range(3):
        for row in range(8):
            for col in range(12):
                board[(layer, row, col)] = None
    # --- TOP BOARD (layer 0) ---
    # Rows: row 0 corresponds to rank 8, row 7 to rank 1.
    # Rank 8 (row 0): “- - g - - - r - - - g -”
    board[(0,0,2)]  = Griffin("Scarlet")
    board[(0,0,6)]  = Dragon("Scarlet")
    board[(0,0,10)] = Griffin("Scarlet")
    # Rank 7 (row 1): “s - s - s - s - s - s -”
    row = 1
    for col in range(0, 12, 2):
        board[(0,1,col)] = Sylph("Scarlet")
    # Rank 2 (row 6): “S - S - S - S - S - S -”
    row = 6
    for col in range(0, 12, 2):
        board[(0,6,col)] = Sylph("Gold")
    # Rank 1 (row 7): “- - G - - - R - - - G -”
    board[(0,7,2)]  = Griffin("Gold")
    board[(0,7,6)]  = Dragon("Gold")
    board[(0,7,10)] = Griffin("Gold")
    
    # --- MIDDLE BOARD (layer 1) ---
    # Rank 8 (row 0): “o u h t c m k p t h u o”
    row = 0
    middle_row0 = ["o","u","h","t","c","m","k","p","t","h","u","o"]
    for col, sym in enumerate(middle_row0):
        piece = create_middle_piece(sym, "Scarlet")
        board[(1, row, col)] = piece
    # Rank 7 (row 1): “w w w w w w w w w w w w”
    row = 1
    for col in range(12):
        board[(1, row, col)] = Warrior("Scarlet")
    # Rank 2 (row 6): “W W W W W W W W W W W W”
    row = 6
    for col in range(12):
        board[(1, row, col)] = Warrior("Gold")
    # Rank 1 (row 7): “O U H T C M K P T H U O”
    row = 7
    middle_row7 = ["O","U","H","T","C","M","K","P","T","H","U","O"]
    for col, sym in enumerate(middle_row7):
        piece = create_middle_piece(sym, "Gold")
        board[(1, row, col)] = piece

    # --- BOTTOM BOARD (layer 2) ---
    # Rank 8 (row 0): “- - b - - - e - - - b -”
    row = 0
    board[(2,0,2)]  = Basilisk("Scarlet")
    board[(2,0,6)]  = Elemental("Scarlet")
    board[(2,0,10)] = Basilisk("Scarlet")
    # Rank 7 (row 1): “- d - d - d - d - d - d”
    row = 1
    for col in range(1, 12, 2):
        board[(2,1,col)] = Dwarf("Scarlet")
    # Rank 1 (row 7): “- - B - - - E - - - B -”
    row = 7
    board[(2,7,2)]  = Basilisk("Gold")
    board[(2,7,6)]  = Elemental("Gold")
    board[(2,7,10)] = Basilisk("Gold")
    # Also place the Gold Dwarves on the appropriate row.
    row = 5
    for col in range(1, 12, 2):
        board[(2,5,col)] = Dwarf("Gold")
    
    return board

def create_middle_piece(symbol, color):
    """Creates a middle–board piece based on its symbol."""
    mapping = {
        'o': Oliphant,
        'u': Unicorn,
        'h': Hero,
        't': Thief,
        'c': Cleric,
        'm': Mage,
        'k': King,
        'p': Paladin,
        'w': Warrior
    }
    piece_class = mapping.get(symbol.lower())
    return piece_class(color) if piece_class else None
