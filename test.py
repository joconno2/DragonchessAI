import unittest
import numpy as np

# Import key functions and constants from the project.
from bitboard import (
    BOARD_ROWS, BOARD_COLS, NUM_BOARDS, TOTAL_SQUARES,
    pos_to_index, index_to_pos, create_initial_board,
    GOLD_SYLPH, SCARLET_GRIFFIN, GOLD_GRIFFIN, SCARLET_BASILISK,
    SCARLET_WARRIOR, GOLD_WARRIOR, GOLD_DWARF, GOLD_KING, SCARLET_KING
)
import moves
from moves import (
    generate_sylph_moves, generate_warrior_moves, generate_dwarf_moves
)
from game import Game

#########################
# Basic Bitboard Tests  #
#########################

class TestBitboard(unittest.TestCase):
    def test_pos_index_conversion(self):
        """Test conversion between (layer, row, col) and flat index."""
        print("\n[Bitboard] Testing position-index conversion")
        for layer in range(NUM_BOARDS):
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    idx = pos_to_index(layer, row, col)
                    self.assertEqual((layer, row, col), index_to_pos(idx),
                                     f"Failed at layer={layer}, row={row}, col={col}")

    def test_create_initial_board(self):
        """Test that the initial board is set up correctly."""
        print("\n[Bitboard] Testing creation of the initial board")
        board = create_initial_board()
        self.assertEqual(len(board), TOTAL_SQUARES)
        # Top board (layer 0): Check Scarlet Griffin at (0,0,2)
        self.assertEqual(board[pos_to_index(0, 0, 2)], SCARLET_GRIFFIN,
                         "Scarlet Griffin not at expected position (0,0,2)")
        # Top board (layer 0): Check Gold Griffin at (0,7,2)
        self.assertEqual(board[pos_to_index(0, 7, 2)], GOLD_GRIFFIN,
                         "Gold Griffin not at expected position (0,7,2)")
        # Middle board (layer 1): Check Scarlet Warrior at (1,1,0)
        self.assertEqual(board[pos_to_index(1, 1, 0)], SCARLET_WARRIOR,
                         "Scarlet Warrior not at expected position (1,1,0)")
        # Bottom board (layer 2): Check Scarlet Basilisk at (2,0,2)
        self.assertEqual(board[pos_to_index(2, 0, 2)], SCARLET_BASILISK,
                         "Scarlet Basilisk not at expected position (2,0,2)")

#########################
# Basic Moves Tests     #
#########################

class TestMoves(unittest.TestCase):
    def setUp(self):
        # Create a fresh empty board for each test.
        self.board = np.zeros(TOTAL_SQUARES, dtype=np.int16)

    def test_sylph_moves_gold(self):
        """
        For a Gold Sylph at (0,3,3) with direction -1, expect:
          - Quiet diagonal moves to (0,2,2) and (0,2,4)
          - A forward capture move to (0,2,3)
          - A capture move to the middle board at (1,3,3)
        """
        print("\n[Moves] Testing Gold Sylph moves at (0,3,3)")
        layer, row, col = 0, 3, 3
        from_idx = pos_to_index(layer, row, col)
        self.board[from_idx] = GOLD_SYLPH  # Gold Sylph
        moves_list = generate_sylph_moves((layer, row, col), self.board, "Gold")
        expected_destinations = [
            (0, 2, 2),  # Diagonal left quiet move
            (0, 2, 4),  # Diagonal right quiet move
            (0, 2, 3),  # Forward capture move
            (1, 3, 3)   # Capture move to middle board
        ]
        expected_indices = {pos_to_index(*dest) for dest in expected_destinations}
        result_indices = {move[1] for move in moves_list}
        for idx in expected_indices:
            self.assertIn(idx, result_indices,
                          f"Expected destination index {idx} not found in Gold Sylph moves.")

    def test_warrior_moves_gold(self):
        """
        For a Gold Warrior at (1,4,5) with direction -1, expect:
          - Quiet move forward to (1,3,5)
          - Capture moves diagonally to (1,3,4) and (1,3,6)
        """
        print("\n[Moves] Testing Gold Warrior moves at (1,4,5)")
        layer, row, col = 1, 4, 5
        from_idx = pos_to_index(layer, row, col)
        self.board[from_idx] = GOLD_WARRIOR  # Gold Warrior
        moves_list = generate_warrior_moves((layer, row, col), self.board, "Gold")
        expected_quiet = pos_to_index(layer, 3, 5)
        quiet_moves = [move for move in moves_list if move[2] == moves.QUIET]
        quiet_destinations = {move[1] for move in quiet_moves}
        self.assertIn(expected_quiet, quiet_destinations,
                      "Quiet move for Gold Warrior not generated correctly.")
        expected_capture_left = pos_to_index(layer, 3, 4)
        expected_capture_right = pos_to_index(layer, 3, 6)
        capture_moves = [move for move in moves_list if move[2] == moves.CAPTURE]
        capture_destinations = {move[1] for move in capture_moves}
        self.assertIn(expected_capture_left, capture_destinations,
                      "Left diagonal capture for Gold Warrior not generated correctly.")
        self.assertIn(expected_capture_right, capture_destinations,
                      "Right diagonal capture for Gold Warrior not generated correctly.")

    def test_dwarf_moves_gold(self):
        """
        For a Gold Dwarf at (1,4,5) with direction -1, expect a quiet forward move to (1,3,5).
        """
        print("\n[Moves] Testing Gold Dwarf moves at (1,4,5)")
        layer, row, col = 1, 4, 5
        from_idx = pos_to_index(layer, row, col)
        self.board[from_idx] = GOLD_DWARF  # Gold Dwarf
        moves_list = generate_dwarf_moves((layer, row, col), self.board, "Gold")
        expected_quiet = pos_to_index(layer, 3, 5)
        quiet_moves = [move for move in moves_list if move[2] == moves.QUIET]
        quiet_destinations = {move[1] for move in quiet_moves}
        self.assertIn(expected_quiet, quiet_destinations,
                      "Quiet forward move for Gold Dwarf not generated correctly.")

#########################
# Extended Moves Tests  #
#########################

class TestExtendedMoves(unittest.TestCase):
    def setUp(self):
        self.board = np.zeros(TOTAL_SQUARES, dtype=np.int16)

    def test_dragon_moves(self):
        print("\n[Extended Moves] Testing Gold Dragon moves at (0,4,4)")
        pos = (0, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 3  # Gold Dragon
        moves_list = moves.generate_dragon_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Dragon moves should not be empty.")
        for move in moves_list:
            l, r, c = index_to_pos(move[1])
            self.assertTrue(0 <= l < NUM_BOARDS, "Dragon move layer out of bounds.")

    def test_oliphant_moves(self):
        print("\n[Extended Moves] Testing Gold Oliphant moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 4  # Gold Oliphant
        moves_list = moves.generate_oliphant_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Oliphant moves should not be empty.")
        for move in moves_list:
            l, r, c = index_to_pos(move[1])
            self.assertEqual(l, 1, "Oliphant must remain on middle board.")

    def test_unicorn_moves(self):
        print("\n[Extended Moves] Testing Gold Unicorn moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 5  # Gold Unicorn
        moves_list = moves.generate_unicorn_moves(pos, self.board, "Gold")
        self.assertEqual(len(moves_list), 8, "Unicorn should have 8 moves from center.")

    def test_hero_moves(self):
        print("\n[Extended Moves] Testing Gold Hero moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 6  # Gold Hero
        moves_list = moves.generate_hero_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Hero moves should not be empty.")

    def test_thief_moves(self):
        print("\n[Extended Moves] Testing Gold Thief moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 7  # Gold Thief
        moves_list = moves.generate_thief_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Thief moves should not be empty.")

    def test_cleric_moves(self):
        print("\n[Extended Moves] Testing Gold Cleric moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 8  # Gold Cleric
        moves_list = moves.generate_cleric_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Cleric moves should not be empty.")

    def test_mage_moves(self):
        print("\n[Extended Moves] Testing Gold Mage moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 9  # Gold Mage
        moves_list = moves.generate_mage_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Mage moves should not be empty.")

    def test_king_moves(self):
        print("\n[Extended Moves] Testing Gold King moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 10  # Gold King
        moves_list = moves.generate_king_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "King moves should not be empty.")

    def test_paladin_moves(self):
        print("\n[Extended Moves] Testing Gold Paladin moves at (1,4,4)")
        pos = (1, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 11  # Gold Paladin
        moves_list = moves.generate_paladin_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Paladin moves should not be empty.")

    def test_elemental_moves(self):
        print("\n[Extended Moves] Testing Gold Elemental moves at (2,4,4)")
        pos = (2, 4, 4)
        from_idx = pos_to_index(*pos)
        self.board[from_idx] = 14  # Gold Elemental
        moves_list = moves.generate_elemental_moves(pos, self.board, "Gold")
        self.assertGreater(len(moves_list), 0, "Elemental moves should not be empty.")

#########################
# Game & Utility Tests  #
#########################

class TestGame(unittest.TestCase):
    def test_make_move_and_turn_switch(self):
        """Test that making a move updates the board and switches the turn."""
        print("\n[Game] Testing make_move and turn switch")
        game = Game()
        initial_turn = game.current_turn
        moves_list = game.get_all_moves()
        self.assertTrue(len(moves_list) > 0, "No moves available at game start.")
        move = moves_list[0]
        game.make_move(move)
        expected_turn = "Scarlet" if initial_turn == "Gold" else "Gold"
        self.assertEqual(game.current_turn, expected_turn,
                         "Turn did not switch correctly after move.")
        self.assertEqual(game.board[move[0]], 0,
                         "Source square not emptied after move.")

    def test_game_over_when_king_missing(self):
        """Test that the game ends when one side's king is missing."""
        print("\n[Game] Testing game over condition when a king is missing")
        game = Game()
        # Remove Gold King.
        for idx in range(len(game.board)):
            if game.board[idx] == GOLD_KING:
                game.board[idx] = 0
        game.update()
        self.assertTrue(game.game_over, "Game should be over when Gold King is missing.")
        self.assertEqual(game.winner, "Scarlet",
                         "Winner should be Scarlet when Gold King is missing.")
        # Test the reverse condition.
        game = Game()
        for idx in range(len(game.board)):
            if game.board[idx] == SCARLET_KING:
                game.board[idx] = 0
        game.update()
        self.assertTrue(game.game_over, "Game should be over when Scarlet King is missing.")
        self.assertEqual(game.winner, "Gold",
                         "Winner should be Gold when Scarlet King is missing.")

class TestFrozenPieces(unittest.TestCase):
    def test_frozen_piece_update(self):
        """Test that a Basilisk on the bottom board freezes an opposing piece above."""
        print("\n[Frozen Pieces] Testing frozen piece update")
        game = Game()
        basilisk_idx = pos_to_index(2, 3, 3)
        enemy_idx = pos_to_index(1, 3, 3)
        from bitboard import SCARLET_BASILISK
        # Place a Scarlet Basilisk below a Gold Warrior.
        game.board[basilisk_idx] = SCARLET_BASILISK
        game.board[enemy_idx] = GOLD_WARRIOR
        game.update()
        self.assertTrue(game.frozen[enemy_idx],
                        "The enemy piece should be frozen by the Basilisk.")

class TestPieceLetter(unittest.TestCase):
    def test_piece_letter(self):
        """Test that the piece_letter method returns the correct symbol."""
        print("\n[Utility] Testing piece_letter conversion")
        game = Game()
        self.assertEqual(game.piece_letter(1), "S", "Gold Sylph letter should be 'S'")
        self.assertEqual(game.piece_letter(-1), "s", "Scarlet Sylph letter should be 's'")

if __name__ == '__main__':
    unittest.main(verbosity=2)
