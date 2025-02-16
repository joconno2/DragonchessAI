import random

class BaseAI:
    def __init__(self, game, color):
        self.game = game
        self.color = color

    def choose_move(self):
        moves = []
        board = self.game.board
        for pos, piece in board.items():
            if piece and piece.color == self.color:
                legal = self.game.get_legal_moves(piece, pos)
                moves.extend(legal)
        if moves:
            return random.choice(moves)
        return None

class RandomAI(BaseAI):
    def __init__(self, game, color):
        super().__init__(game, color)
