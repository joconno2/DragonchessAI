from numba import njit
from bitboard import BOARD_ROWS, BOARD_COLS, NUM_BOARDS, pos_to_index

# Move flag constants (use integers instead of strings)
QUIET      = 0
CAPTURE    = 1
AFAR       = 2
AMBIGUOUS  = 3
THREED     = 4

@njit
def in_bounds(layer, row, col):
    return (0 <= layer < NUM_BOARDS) and (0 <= row < BOARD_ROWS) and (0 <= col < BOARD_COLS)

@njit
def generate_sylph_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    # For Gold (color==1), we assume movement is upward (row decreases); for Scarlet (color==-1), downward.
    direction = -1 if color == 1 else 1
    if layer == 0:
        # Non-capturing: diagonal moves
        for dc in (-1, 1):
            new_row = row + direction
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                to_idx = pos_to_index(layer, new_row, new_col)
                if board[to_idx] == 0:
                    moves.append((from_idx, to_idx, QUIET))
        # Capturing: straight forward
        new_row = row + direction
        new_col = col
        if in_bounds(layer, new_row, new_col):
            moves.append((from_idx, pos_to_index(layer, new_row, new_col), CAPTURE))
        # Capturing: move down to middle board
        new_layer = 1
        if in_bounds(new_layer, row, col):
            moves.append((from_idx, pos_to_index(new_layer, row, col), CAPTURE))
    elif layer == 1:
        # On middle board, allow quiet move back to top board
        new_layer = 0
        if in_bounds(new_layer, row, col) and board[pos_to_index(new_layer, row, col)] == 0:
            moves.append((from_idx, pos_to_index(new_layer, row, col), QUIET))
        # Also allow moves to designated home cells:
        target_layer = 0
        if color == 1:
            home_row = BOARD_ROWS - 1  # row 7
            for c in range(0, BOARD_COLS, 2):
                if in_bounds(target_layer, home_row, c) and board[pos_to_index(target_layer, home_row, c)] == 0:
                    moves.append((from_idx, pos_to_index(target_layer, home_row, c), QUIET))
        else:
            home_row = 0
            for c in range(1, BOARD_COLS, 2):
                if in_bounds(target_layer, home_row, c) and board[pos_to_index(target_layer, home_row, c)] == 0:
                    moves.append((from_idx, pos_to_index(target_layer, home_row, c), QUIET))
    return moves

@njit
def generate_griffin_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 0:
        offsets = ((3,2), (3,-2), (-3,2), (-3,-2),
                   (2,3), (2,-3), (-2,3), (-2,-3))
        for dr, dc in offsets:
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        new_layer = 1
        for dr in (-1, 1):
            for dc in (-1, 1):
                new_row = row + dr
                new_col = col + dc
                if in_bounds(new_layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(new_layer, new_row, new_col), AMBIGUOUS))
    elif layer == 1:
        for dr in (-1, 1):
            for dc in (-1, 1):
                new_row = row + dr
                new_col = col + dc
                if in_bounds(layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        new_layer = 0
        for dr in (-1, 1):
            for dc in (-1, 1):
                new_row = row + dr
                new_col = col + dc
                if in_bounds(new_layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(new_layer, new_row, new_col), AMBIGUOUS))
    return moves

@njit
def generate_dragon_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 0:
        return moves
    # King–like moves (excluding the null move)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    # Bishop–like sliding moves (diagonals)
    for dr, dc in ((-1,-1), (-1,1), (1,-1), (1,1)):
        r = row
        c = col
        while True:
            r += dr
            c += dc
            if not in_bounds(layer, r, c):
                break
            moves.append((from_idx, pos_to_index(layer, r, c), AMBIGUOUS))
            if board[pos_to_index(layer, r, c)] != 0:
                break
    # "Capture from afar" moves (to middle board)
    target_layer = 1
    if in_bounds(target_layer, row, col):
        moves.append((from_idx, pos_to_index(target_layer, row, col), AFAR))
    for dr, dc in ((0,1), (0,-1), (1,0), (-1,0)):
        new_row = row + dr
        new_col = col + dc
        if in_bounds(target_layer, new_row, new_col):
            moves.append((from_idx, pos_to_index(target_layer, new_row, new_col), AFAR))
    return moves

@njit
def generate_oliphant_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 1:
        return moves
    for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
        r = row
        c = col
        while True:
            r += dr
            c += dc
            if not in_bounds(layer, r, c):
                break
            moves.append((from_idx, pos_to_index(layer, r, c), AMBIGUOUS))
            if board[pos_to_index(layer, r, c)] != 0:
                break
    return moves

@njit
def generate_unicorn_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 1:
        return moves
    offsets = ((2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2))
    for dr, dc in offsets:
        new_row = row + dr
        new_col = col + dc
        if in_bounds(layer, new_row, new_col):
            moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    return moves

@njit
def generate_hero_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 1:
        # Move 1 or 2 cells diagonally.
        for dr in (-2, -1, 1, 2):
            for dc in (-2, -1, 1, 2):
                if abs(dr) == abs(dc):
                    new_row = row + dr
                    new_col = col + dc
                    if in_bounds(layer, new_row, new_col):
                        moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        # Move to top or bottom board via diagonal.
        for target_layer in (0, 2):
            for dr in (-1, 1):
                for dc in (-1, 1):
                    new_row = row + dr
                    new_col = col + dc
                    if in_bounds(target_layer, new_row, new_col):
                        moves.append((from_idx, pos_to_index(target_layer, new_row, new_col), AMBIGUOUS))
    else:
        target_layer = 1
        for dr in (-1, 1):
            for dc in (-1, 1):
                new_row = row + dr
                new_col = col + dc
                if in_bounds(target_layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(target_layer, new_row, new_col), AMBIGUOUS))
    return moves

@njit
def generate_thief_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 1:
        return moves
    for dr, dc in ((-1,-1), (-1,1), (1,-1), (1,1)):
        r = row
        c = col
        while True:
            r += dr
            c += dc
            if not in_bounds(layer, r, c):
                break
            moves.append((from_idx, pos_to_index(layer, r, c), AMBIGUOUS))
            if board[pos_to_index(layer, r, c)] != 0:
                break
    return moves

@njit
def generate_cleric_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    # Exclude the null move (dr=0, dc=0)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    if layer == 0:
        new_layer = 1
        if in_bounds(new_layer, row, col):
            moves.append((from_idx, pos_to_index(new_layer, row, col), AMBIGUOUS))
    elif layer == 1:
        for target_layer in (0, 2):
            if in_bounds(target_layer, row, col):
                moves.append((from_idx, pos_to_index(target_layer, row, col), AMBIGUOUS))
    elif layer == 2:
        new_layer = 1
        if in_bounds(new_layer, row, col):
            moves.append((from_idx, pos_to_index(new_layer, row, col), AMBIGUOUS))
    return moves

@njit
def generate_mage_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 1:
        for dr, dc in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)):
            r = row
            c = col
            while True:
                r += dr
                c += dc
                if not in_bounds(layer, r, c):
                    break
                moves.append((from_idx, pos_to_index(layer, r, c), AMBIGUOUS))
                if board[pos_to_index(layer, r, c)] != 0:
                    break
        for d_layer in (-1, 1):
            new_layer = layer + d_layer
            if in_bounds(new_layer, row, col):
                moves.append((from_idx, pos_to_index(new_layer, row, col), AMBIGUOUS))
    else:
        for dr, dc in ((0,1), (0,-1), (1,0), (-1,0)):
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        for d in (-2, -1, 1, 2):
            new_row = row + d
            if in_bounds(layer, new_row, col):
                moves.append((from_idx, pos_to_index(layer, new_row, col), AMBIGUOUS))
    return moves

@njit
def generate_king_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 1:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if in_bounds(layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        for d_layer in (-1, 1):
            new_layer = layer + d_layer
            if in_bounds(new_layer, row, col):
                moves.append((from_idx, pos_to_index(new_layer, row, col), AMBIGUOUS))
    else:
        new_layer = 1
        if in_bounds(new_layer, row, col):
            moves.append((from_idx, pos_to_index(new_layer, row, col), AMBIGUOUS))
    return moves

@njit
def generate_paladin_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 1:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if in_bounds(layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
        offsets = ((2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2))
        for dr, dc in offsets:
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col):
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    else:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if in_bounds(layer, new_row, new_col):
                    moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    # 3D knight moves (unblockable)
    for d_layer in (-2, -1, 1, 2):
        for d_row in (-2, -1, 0, 1, 2):
            for d_col in (-2, -1, 0, 1, 2):
                if d_layer == 0:
                    continue
                diffs = [abs(d_layer), abs(d_row), abs(d_col)]
                diffs.sort()
                if diffs[0] == 0 and diffs[1] == 1 and diffs[2] == 2:
                    new_layer = layer + d_layer
                    new_row = row + d_row
                    new_col = col + d_col
                    if in_bounds(new_layer, new_row, new_col):
                        moves.append((from_idx, pos_to_index(new_layer, new_row, new_col), THREED))
    return moves

@njit
def generate_warrior_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 1:
        return moves
    direction = -1 if color == 1 else 1
    new_row = row + direction
    if in_bounds(layer, new_row, col) and board[pos_to_index(layer, new_row, col)] == 0:
        moves.append((from_idx, pos_to_index(layer, new_row, col), QUIET))
    for dc in (-1, 1):
        new_col = col + dc
        if in_bounds(layer, new_row, new_col):
            moves.append((from_idx, pos_to_index(layer, new_row, new_col), CAPTURE))
    return moves

@njit
def generate_basilisk_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer != 2:
        return moves
    direction = -1 if color == 1 else 1
    for dc in (0, -1, 1):
        new_row = row + direction
        new_col = col + dc
        if in_bounds(layer, new_row, new_col):
            moves.append((from_idx, pos_to_index(layer, new_row, new_col), AMBIGUOUS))
    new_row = row - direction
    if in_bounds(layer, new_row, col) and board[pos_to_index(layer, new_row, col)] == 0:
        moves.append((from_idx, pos_to_index(layer, new_row, col), QUIET))
    return moves

@njit
def generate_elemental_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer == 2:
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            for dist in (1, 2):
                new_row = row + dr * dist
                new_col = col + dc * dist
                if not in_bounds(layer, new_row, new_col):
                    break
                to_idx = pos_to_index(layer, new_row, new_col)
                if dist == 1:
                    if board[to_idx] == 0 or (board[to_idx] != 0 and board[to_idx] * (1 if color == 1 else -1) < 0):
                        moves.append((from_idx, to_idx, AMBIGUOUS))
                    else:
                        break
                else:
                    inter_idx = pos_to_index(layer, row + dr, col + dc)
                    if board[inter_idx] != 0:
                        break
                    if board[to_idx] == 0 or (board[to_idx] != 0 and board[to_idx] * (1 if color == 1 else -1) < 0):
                        moves.append((from_idx, to_idx, AMBIGUOUS))
                    else:
                        break
        for dr, dc in ((-1,-1), (-1,1), (1,-1), (1,1)):
            new_row = row + dr
            new_col = col + dc
            if in_bounds(layer, new_row, new_col) and board[pos_to_index(layer, new_row, new_col)] == 0:
                moves.append((from_idx, pos_to_index(layer, new_row, new_col), QUIET))
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            inter_row = row + dr
            inter_col = col + dc
            target_layer = 1
            if in_bounds(layer, inter_row, inter_col) and board[pos_to_index(layer, inter_row, inter_col)] == 0:
                moves.append((from_idx, pos_to_index(target_layer, row+dr, col+dc), CAPTURE))
    elif layer == 1:
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            inter_row = row + dr
            inter_col = col + dc
            target_layer = 2
            if in_bounds(layer, inter_row, inter_col) and board[pos_to_index(layer, inter_row, inter_col)] == 0:
                to_idx = pos_to_index(target_layer, row+dr, col+dc)
                if board[to_idx] == 0:
                    moves.append((from_idx, to_idx, QUIET))
                elif board[to_idx] != 0 and board[to_idx] * (1 if color == 1 else -1) < 0:
                    moves.append((from_idx, to_idx, CAPTURE))
    return moves

@njit
def generate_dwarf_moves(pos, board, color):
    moves = []
    layer, row, col = pos
    from_idx = pos_to_index(layer, row, col)
    if layer not in (1, 2):
        return moves
    direction = -1 if color == 1 else 1
    new_row = row + direction
    if in_bounds(layer, new_row, col) and board[pos_to_index(layer, new_row, col)] == 0:
        moves.append((from_idx, pos_to_index(layer, new_row, col), QUIET))
    for dc in (-1, 1):
        if in_bounds(layer, row, col+dc) and board[pos_to_index(layer, row, col+dc)] == 0:
            moves.append((from_idx, pos_to_index(layer, row, col+dc), QUIET))
    for dc in (-1, 1):
        new_row = row + direction
        if in_bounds(layer, new_row, col+dc):
            moves.append((from_idx, pos_to_index(layer, new_row, col+dc), CAPTURE))
    if layer == 2:
        target_layer = 1
        if in_bounds(target_layer, row, col):
            moves.append((from_idx, pos_to_index(target_layer, row, col), CAPTURE))
    if layer == 1:
        target_layer = 2
        if in_bounds(target_layer, row, col) and board[pos_to_index(target_layer, row, col)] == 0:
            moves.append((from_idx, pos_to_index(target_layer, row, col), QUIET))
    if not in_bounds(layer, row + direction, col):
        reverse = -direction
        new_row = row + reverse
        if in_bounds(layer, new_row, col) and board[pos_to_index(layer, new_row, col)] == 0:
            moves.append((from_idx, pos_to_index(layer, new_row, col), QUIET))
    return moves
