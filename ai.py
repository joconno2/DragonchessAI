import random

class BaseAI:
    def __init__(self, game, color):
        self.game = game
        self.color = color

    def choose_move(self):
        moves = self.game.get_all_moves()
        if moves:
            return random.choice(moves)
        return None

class RandomAI(BaseAI):
    def __init__(self, game, color):
        super().__init__(game, color)
