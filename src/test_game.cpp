#include "bitboard.h"
#include "moves.h"
#include "game.h"
#include "ai.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace dragonchess;

// Test counters
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) \
    std::cout << "Testing " << #name << "... "; \
    if (test_##name()) { \
        std::cout << "✓ PASSED" << std::endl; \
        tests_passed++; \
    } else { \
        std::cout << "✗ FAILED" << std::endl; \
        tests_failed++; \
    }

// Test helper
#define ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "  ASSERTION FAILED: " << message << std::endl; \
        return false; \
    }

// Test: Initial board setup
bool test_initial_board() {
    Board board = create_initial_board();
    
    // Check total pieces
    int piece_count = 0;
    for (int i = 0; i < TOTAL_SQUARES; ++i) {
        if (board[i] != EMPTY) piece_count++;
    }
    ASSERT(piece_count == 84, "Should have 84 pieces (42 per side)");
    
    // Check specific pieces
    ASSERT(board[pos_to_index(0, 0, 6)] == SCARLET_DRAGON, "Scarlet dragon at top board");
    ASSERT(board[pos_to_index(0, 7, 6)] == GOLD_DRAGON, "Gold dragon at top board");
    ASSERT(board[pos_to_index(1, 0, 6)] == SCARLET_KING, "Scarlet king at middle board");
    ASSERT(board[pos_to_index(1, 7, 6)] == GOLD_KING, "Gold king at middle board");
    
    // Check symmetry
    ASSERT(board[pos_to_index(1, 0, 0)] == SCARLET_OLIPHANT, "Scarlet oliphant corner");
    ASSERT(board[pos_to_index(1, 7, 0)] == GOLD_OLIPHANT, "Gold oliphant corner");
    
    return true;
}

// Test: Position conversion
bool test_position_conversion() {
    // Test pos_to_index and index_to_pos
    for (int layer = 0; layer < NUM_BOARDS; ++layer) {
        for (int row = 0; row < BOARD_ROWS; ++row) {
            for (int col = 0; col < BOARD_COLS; ++col) {
                int idx = pos_to_index(layer, row, col);
                auto [l, r, c] = index_to_pos(idx);
                
                ASSERT(l == layer && r == row && c == col, 
                       "Position conversion round-trip failed");
            }
        }
    }
    
    // Test bounds
    ASSERT(pos_to_index(0, 0, 0) == 0, "First square should be index 0");
    ASSERT(pos_to_index(2, 7, 11) == TOTAL_SQUARES - 1, "Last square should be index 287");
    
    return true;
}

// Test: Warrior moves
bool test_warrior_moves() {
    Board board{};
    board.fill(EMPTY);
    
    // Place Gold warrior at (1, 6, 5)
    board[pos_to_index(1, 6, 5)] = GOLD_WARRIOR;
    
    std::vector<Move> moves = generate_warrior_moves({1, 6, 5}, board, Color::GOLD);
    
    // Gold warrior should move forward (decreasing row)
    ASSERT(moves.size() == 1, "Warrior should have 1 forward move");
    
    auto [from, to, flag] = moves[0];
    ASSERT(to == pos_to_index(1, 5, 5), "Warrior should move forward");
    ASSERT(flag == QUIET, "Forward move should be QUIET");
    
    // Add enemy for capture
    board[pos_to_index(1, 5, 4)] = SCARLET_WARRIOR;
    board[pos_to_index(1, 5, 6)] = SCARLET_WARRIOR;
    
    moves = generate_warrior_moves({1, 6, 5}, board, Color::GOLD);
    ASSERT(moves.size() == 3, "Warrior should have 1 forward + 2 diagonal captures");
    
    return true;
}

// Test: King moves
bool test_king_moves() {
    Board board{};
    board.fill(EMPTY);
    
    // Place Gold king at (1, 4, 6)
    board[pos_to_index(1, 4, 6)] = GOLD_KING;
    
    std::vector<Move> moves = generate_king_moves({1, 4, 6}, board, Color::GOLD);
    
    // King on middle board: 8 adjacent + 2 vertical (up/down boards)
    ASSERT(moves.size() == 10, "King on middle board should have 10 moves");
    
    // Test king on top board
    board.fill(EMPTY);
    board[pos_to_index(0, 4, 6)] = GOLD_KING;
    moves = generate_king_moves({0, 4, 6}, board, Color::GOLD);
    ASSERT(moves.size() == 1, "King on top board can only move to middle");
    
    return true;
}

// Test: Dragon moves
bool test_dragon_moves() {
    Board board{};
    board.fill(EMPTY);
    
    // Place Gold dragon at (0, 4, 6)
    board[pos_to_index(0, 4, 6)] = GOLD_DRAGON;
    
    std::vector<Move> moves = generate_dragon_moves({0, 4, 6}, board, Color::GOLD);
    
    // Dragon has king-like moves + bishop slides
    ASSERT(moves.size() > 10, "Dragon should have many moves on empty board");
    
    // Test capture from afar
    board[pos_to_index(1, 4, 6)] = SCARLET_WARRIOR;
    moves = generate_dragon_moves({0, 4, 6}, board, Color::GOLD);
    
    bool has_afar = false;
    for (const auto& move : moves) {
        if (std::get<2>(move) == AFAR) {
            has_afar = true;
            ASSERT(std::get<1>(move) == pos_to_index(1, 4, 6), 
                   "AFAR capture should target middle board piece");
        }
    }
    ASSERT(has_afar, "Dragon should have AFAR capture move");
    
    return true;
}

// Test: Paladin 3D knight moves
bool test_paladin_3d_moves() {
    Board board{};
    board.fill(EMPTY);
    
    // Place Gold paladin at top board corner
    board[pos_to_index(0, 0, 0)] = GOLD_PALADIN;
    
    std::vector<Move> moves = generate_paladin_moves({0, 0, 0}, board, Color::GOLD);
    
    // Should have 3D knight moves (unblockable)
    bool has_3d = false;
    for (const auto& move : moves) {
        if (std::get<2>(move) == THREED) {
            has_3d = true;
            break;
        }
    }
    ASSERT(has_3d, "Paladin on edge should have 3D knight moves");
    
    return true;
}

// Test: Game initialization
bool test_game_initialization() {
    Game game;
    
    ASSERT(game.current_turn == Color::GOLD, "Gold should move first");
    ASSERT(game.game_over == false, "Game should not be over");
    ASSERT(game.no_capture_count == 0, "No captures yet");
    ASSERT(game.move_notations.empty(), "No moves yet");
    
    // Check initial move count
    std::vector<Move> moves = game.get_all_moves();
    ASSERT(moves.size() > 0, "Should have legal moves at start");
    
    // Gold should have warrior moves
    bool has_warrior_move = false;
    for (const auto& move : moves) {
        auto [from, to, flag] = move;
        if (game.board[from] == GOLD_WARRIOR) {
            has_warrior_move = true;
            break;
        }
    }
    ASSERT(has_warrior_move, "Gold should have warrior moves");
    
    return true;
}

// Test: Make a move
bool test_make_move() {
    Game game;
    
    // Get a legal move
    std::vector<Move> moves = game.get_all_moves();
    ASSERT(moves.size() > 0, "Should have moves");
    
    Move first_move = moves[0];
    auto [from, to, flag] = first_move;
    int16_t piece = game.board[from];
    
    game.make_move(first_move);
    
    ASSERT(game.board[from] == EMPTY || (flag == AFAR && std::abs(piece) == 3), 
           "Source should be empty (unless AFAR dragon)");
    ASSERT(game.board[to] == piece || flag == AFAR, "Piece should move to destination");
    ASSERT(game.current_turn == Color::SCARLET, "Turn should switch to Scarlet");
    ASSERT(game.move_notations.size() == 1, "Should have one move notation");
    
    return true;
}

// Test: Capture detection
bool test_capture_detection() {
    Game game;
    game.board.fill(EMPTY);
    
    // Set up a capture scenario
    game.board[pos_to_index(1, 3, 5)] = GOLD_WARRIOR;
    game.board[pos_to_index(1, 2, 4)] = SCARLET_WARRIOR;
    game.current_turn = Color::GOLD;
    
    std::vector<Move> moves = game.get_all_moves();
    
    bool has_capture = false;
    for (const auto& move : moves) {
        auto [from, to, flag] = move;
        if (flag == CAPTURE && to == pos_to_index(1, 2, 4)) {
            has_capture = true;
            break;
        }
    }
    ASSERT(has_capture, "Should detect capture move");
    
    return true;
}

// Test: Frozen pieces (Basilisk)
bool test_frozen_pieces() {
    Game game;
    game.board.fill(EMPTY);
    
    // Place Scarlet basilisk on bottom board
    game.board[pos_to_index(2, 3, 5)] = SCARLET_BASILISK;
    
    // Place Gold warrior directly above on middle board
    game.board[pos_to_index(1, 3, 5)] = GOLD_WARRIOR;
    
    game.current_turn = Color::GOLD;
    game.update();
    
    ASSERT(game.frozen[pos_to_index(1, 3, 5)] == true, 
           "Warrior should be frozen by basilisk");
    
    // Frozen piece should not generate moves
    std::vector<Move> moves = game.get_all_moves();
    for (const auto& move : moves) {
        auto [from, to, flag] = move;
        ASSERT(from != pos_to_index(1, 3, 5), "Frozen piece should not move");
    }
    
    return true;
}

// Test: Win condition
bool test_win_condition() {
    Game game;
    game.board.fill(EMPTY);
    
    // Only kings remaining
    game.board[pos_to_index(1, 4, 6)] = GOLD_KING;
    game.board[pos_to_index(1, 3, 6)] = SCARLET_KING;
    
    game.update();
    ASSERT(game.game_over == false, "Game should not be over with both kings");
    
    // Remove Scarlet king
    game.board[pos_to_index(1, 3, 6)] = EMPTY;
    game.update();
    
    ASSERT(game.game_over == true, "Game should be over");
    ASSERT(game.winner == "Gold", "Gold should win");
    
    return true;
}

// Test: Draw by 250 moves
bool test_draw_condition() {
    Game game;
    game.no_capture_count = 249;
    game.update();
    ASSERT(game.game_over == false, "Should not be draw at 249 moves");
    
    game.no_capture_count = 250;
    game.update();
    ASSERT(game.game_over == true, "Should be draw at 250 moves");
    ASSERT(game.winner == "Draw", "Winner should be Draw");
    
    return true;
}

// Test: AI can choose move
bool test_ai_move() {
    Game game;
    RandomAI ai(game, Color::GOLD);
    
    auto move = ai.choose_move();
    ASSERT(move.has_value(), "AI should return a move");
    
    // Verify move is legal
    std::vector<Move> legal_moves = game.get_all_moves();
    bool is_legal = false;
    for (const auto& m : legal_moves) {
        if (std::get<0>(m) == std::get<0>(move.value()) &&
            std::get<1>(m) == std::get<1>(move.value()) &&
            std::get<2>(m) == std::get<2>(move.value())) {
            is_legal = true;
            break;
        }
    }
    ASSERT(is_legal, "AI move should be legal");
    
    return true;
}

// Test: Full game simulation
bool test_full_game() {
    Game game;
    RandomAI gold_ai(game, Color::GOLD);
    RandomAI scarlet_ai(game, Color::SCARLET);
    
    int max_moves = 500;
    int move_count = 0;
    
    while (!game.game_over && move_count < max_moves) {
        auto move = (game.current_turn == Color::GOLD) 
                    ? gold_ai.choose_move() 
                    : scarlet_ai.choose_move();
        
        if (!move.has_value()) {
            // No moves available - stalemate (not implemented, treat as draw)
            std::cout << "  (No moves available at move " << move_count << ")";
            break;
        }
        
        game.make_move(move.value());
        game.update();
        move_count++;
    }
    
    ASSERT(move_count > 0, "Game should have at least some moves");
    
    std::cout << "  (Simulated " << move_count << " moves";
    if (game.game_over) {
        std::cout << ", winner: " << game.winner;
    } else {
        std::cout << ", stopped at move limit";
    }
    std::cout << ")";
    
    return true;
}

// Test: Algebraic notation
bool test_algebraic_notation() {
    Game game;
    
    // Get a warrior move
    std::vector<Move> moves = game.get_all_moves();
    Move warrior_move;
    bool found = false;
    
    for (const auto& move : moves) {
        auto [from, to, flag] = move;
        if (game.board[from] == GOLD_WARRIOR) {
            warrior_move = move;
            found = true;
            break;
        }
    }
    
    ASSERT(found, "Should find a warrior move");
    
    auto [from, to, flag] = warrior_move;
    std::string notation = game.move_to_algebraic(warrior_move, GOLD_WARRIOR);
    
    // Should start with 'W' for warrior
    ASSERT(notation[0] == 'W', "Warrior notation should start with W");
    ASSERT(notation.length() > 4, "Notation should have reasonable length");
    ASSERT(notation.find('-') != std::string::npos || notation.find('x') != std::string::npos,
           "Notation should contain - or x");
    
    return true;
}

// Test: All piece types generate moves
bool test_all_pieces_generate_moves() {
    std::vector<int16_t> piece_types = {
        GOLD_SYLPH, GOLD_GRIFFIN, GOLD_DRAGON, GOLD_OLIPHANT,
        GOLD_UNICORN, GOLD_HERO, GOLD_THIEF, GOLD_CLERIC,
        GOLD_MAGE, GOLD_KING, GOLD_PALADIN, GOLD_WARRIOR,
        GOLD_BASILISK, GOLD_ELEMENTAL, GOLD_DWARF
    };
    
    for (int16_t piece : piece_types) {
        Board board{};
        board.fill(EMPTY);
        
        // Place piece at appropriate layer
        int layer = 1; // Default to middle
        if (piece == GOLD_SYLPH || piece == GOLD_GRIFFIN || piece == GOLD_DRAGON) {
            layer = 0; // Top board
        } else if (piece == GOLD_BASILISK || piece == GOLD_ELEMENTAL || piece == GOLD_DWARF) {
            layer = 2; // Bottom board (dwarf can be on both)
        }
        
        int row = 4, col = 6;
        board[pos_to_index(layer, row, col)] = piece;
        
        Game game;
        game.board = board;
        game.current_turn = Color::GOLD;
        
        std::vector<Move> moves = game.get_all_moves();
        
        // Most pieces should have at least one move in the center of an empty board
        ASSERT(moves.size() > 0, "Piece should generate at least one move");
    }
    
    return true;
}

int main() {
    std::cout << "=== Dragonchess C++ Test Suite ===" << std::endl << std::endl;
    
    // Run all tests
    TEST(initial_board);
    TEST(position_conversion);
    TEST(warrior_moves);
    TEST(king_moves);
    TEST(dragon_moves);
    TEST(paladin_3d_moves);
    TEST(game_initialization);
    TEST(make_move);
    TEST(capture_detection);
    TEST(frozen_pieces);
    TEST(win_condition);
    TEST(draw_condition);
    TEST(ai_move);
    TEST(algebraic_notation);
    TEST(all_pieces_generate_moves);
    TEST(full_game);
    
    // Print summary
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Tests Passed: " << tests_passed << std::endl;
    std::cout << "Tests Failed: " << tests_failed << std::endl;
    std::cout << "Total Tests:  " << (tests_passed + tests_failed) << std::endl;
    
    if (tests_failed == 0) {
        std::cout << std::endl << "✓ ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "✗ SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
