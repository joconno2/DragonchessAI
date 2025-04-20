# === main.py ===
import sys
import os
import csv
import time
import pygame
import importlib.util
import numpy as np
from menu import run_menu, run_ai_vs_ai_menu, run_ai_vs_player_menu, run_tournament_menu
from game import Game, pos_to_index, move_generators
from bitboard import TOTAL_SQUARES
from ai import RandomAI
from simulation import simulate_ai_vs_ai_game
from common import (BG_SCALE_X, BG_SCALE_Y, CELL_SIZE, load_assets,
                    draw_board, screen_to_board, DESIGN_WIDTH, DESIGN_HEIGHT)


def warmup_moves():
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode((DESIGN_WIDTH, DESIGN_HEIGHT))
    font = pygame.font.Font("assets/pixel.ttf", 36)
    text = font.render("Loading...", True, (255, 255, 255))
    screen.fill((0, 0, 0))
    screen.blit(text, ((screen.get_width() - text.get_width()) // 2,
                        (screen.get_height() - text.get_height()) // 2))
    pygame.display.update()
    dummy_board = np.zeros(TOTAL_SQUARES, dtype=np.int16)
    dummy_pos = (0, 0, 0)
    for func in move_generators.values():
        try:
            func(dummy_pos, dummy_board, "Gold")
        except Exception:
            pass
    time.sleep(1)


def load_custom_ai(filepath, game, color):
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)


def show_post_game_screen(screen, game, assets, bg, clock, auto_continue=False):
    # Helper to draw outlined text
    def draw_outlined_text(txt, font, x, y):
        outline = font.render(txt, True, (0, 0, 0))
        # draw outline offsets
        for ox, oy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            screen.blit(outline, (x + ox, y + oy))
        # draw main text
        main_surf = font.render(txt, True, (255, 255, 255))
        screen.blit(main_surf, (x, y))

    font_large = pygame.font.Font("assets/pixel.ttf", 48)
    font_small = pygame.font.Font("assets/pixel.ttf", 32)
    win_w, win_h = screen.get_size()
    # positions
    wx = (win_w - font_large.size(f"Winner: {game.winner}")[0]) // 2
    wy = int(win_h * 0.3)
    mx = (win_w - font_small.size(f"Moves: {len(game.move_notations)}")[0]) // 2
    my = int(win_h * 0.4)
    btn_w, btn_h = 200, 50
    btn_x = (win_w - btn_w) // 2
    btn_y = int(win_h * 0.75)
    button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
    showing = True
    start_time = pygame.time.get_ticks()

    while showing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and not auto_continue:
                if button_rect.collidepoint(event.pos):
                    showing = False
        if auto_continue and pygame.time.get_ticks() - start_time > 2000:
            showing = False

        screen.fill((0, 0, 0))
        # background board
        base = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        draw_board(base, game, assets, bg)
        scaled = pygame.transform.scale(base, (win_w, win_h))
        screen.blit(scaled, (0, 0))
        # outlined texts
        draw_outlined_text(f"Winner: {game.winner}", font_large, wx, wy)
        draw_outlined_text(f"Moves: {len(game.move_notations)}", font_small, mx, my)
        # button
        if not auto_continue:
            pygame.draw.rect(screen, (255, 255, 255), button_rect)
            # draw button text
            btxt = font_small.render("Back to Menu", True, (255, 255, 255))
            # outline button text
            draw_outlined_text("Back to Menu", font_small,
                               btn_x + (btn_w - font_small.size("Back to Menu")[0]) // 2,
                               btn_y + (btn_h - font_small.size("Back to Menu")[1]) // 2)
        pygame.display.flip()
        clock.tick(30)


def main():
    pygame.init()
    flags = pygame.RESIZABLE
    pygame.display.set_icon(pygame.image.load(os.path.join("assets", "small_icon.png")))
    pygame.display.set_caption("Dragonchess")

    assets = None
    bg = None
    clock = pygame.time.Clock()
    app_running = True

    while app_running:
        mode, _ = run_menu()
        warmup_moves()

        if mode == "Campaign":
            import campaign
            campaign.run_campaign_menu()
            continue

        if mode == "AI vs AI":
            options = run_ai_vs_ai_menu()
        elif mode == "AI vs Player":
            options = run_ai_vs_player_menu()
        elif mode == "Tournament":
            options = run_tournament_menu()
        else:
            options = {}

        if mode == "Tournament":
            import tournament
            tournament.run_tournament(options)
            continue

        screen = pygame.display.set_mode((DESIGN_WIDTH, DESIGN_HEIGHT), flags)
        pygame.display.set_caption("Dragonchess")
        if assets is None:
            assets = load_assets(CELL_SIZE)
        if bg is None:
            bg_img = pygame.image.load(os.path.join("assets", "bg.png")).convert()
            bg = pygame.transform.scale(bg_img, (int(bg_img.get_width() * BG_SCALE_X),
                                                int(bg_img.get_height() * BG_SCALE_Y)))

        if mode in ["2 Player", "AI vs Player"]:
            game = Game()
            if mode == "AI vs Player":
                ai_side = options.get("ai_side", "Gold")
                human_side = "Scarlet" if ai_side == "Gold" else "Gold"
                ai = load_custom_ai(options.get("ai_file"), game, ai_side) if options.get("ai_file") else RandomAI(game, ai_side)
            else:
                human_side = None
            selected = None
            legal_dest = None
            running = True
            while running:
                dt = clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        app_running = False
                    if event.type == pygame.MOUSEBUTTONUP:
                        if mode == "AI vs Player" and game.current_turn != human_side:
                            continue
                        mx, my = pygame.mouse.get_pos()
                        win_w, win_h = screen.get_size()
                        sx, sy = DESIGN_WIDTH / win_w, DESIGN_HEIGHT / win_h
                        dm = (mx * sx, my * sy)
                        bp = screen_to_board(dm)
                        if bp:
                            idx = pos_to_index(*bp)
                            valid = True if human_side is None else (game.board[idx] > 0 if human_side == "Gold" else game.board[idx] < 0)
                            if selected is None:
                                if valid:
                                    selected = idx
                                    legal_dest = {m[1] for m in game.get_legal_moves_for(selected)}
                            else:
                                if idx in legal_dest:
                                    for mv in game.get_legal_moves_for(selected):
                                        if mv[1] == idx:
                                            game.make_move(mv)
                                            break
                                    selected, legal_dest = None, None
                                else:
                                    if valid:
                                        selected = idx
                                        legal_dest = {m[1] for m in game.get_legal_moves_for(selected)}
                                    else:
                                        selected, legal_dest = None, None
                if mode == "AI vs Player" and game.current_turn != human_side:
                    mv = ai.choose_move()
                    if mv:
                        game.make_move(mv)
                game.update()
                if game.game_over:
                    running = False
                base = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
                draw_board(base, game, assets, bg, selected, legal_dest)
                screen.blit(pygame.transform.scale(base, screen.get_size()), (0, 0))
                pygame.display.flip()
            if app_running:
                show_post_game_screen(screen, game, assets, bg, clock, auto_continue=False)
            continue

        if mode == "AI vs AI":
            headless = options.get("headless", True)
            num_games = options.get("num_games", 10)
            log_file = options.get("log_filename", "logs/ai_vs_ai_log.csv")
            if headless:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Game", "Winner", "Moves"])
                    for n in range(1, num_games + 1):
                        gn, mts, wn = simulate_ai_vs_ai_game(n, options)
                        writer.writerow([gn, wn, len(mts)])
                continue
            else:
                scarlet_path = options.get("scarlet_ai")
                gold_path = options.get("gold_ai")
                for _ in range(num_games):
                    game = Game()
                    gold_ai = load_custom_ai(gold_path, game, "Gold") if gold_path else RandomAI(game, "Gold")
                    scarlet_ai = load_custom_ai(scarlet_path, game, "Scarlet") if scarlet_path else RandomAI(game, "Scarlet")
                    running_aa = True
                    while running_aa:
                        clock.tick(60)
                        mv = gold_ai.choose_move() if game.current_turn == "Gold" else scarlet_ai.choose_move()
                        if mv:
                            game.make_move(mv)
                        game.update()
                        base = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
                        draw_board(base, game, assets, bg)
                        screen.blit(pygame.transform.scale(base, screen.get_size()), (0, 0))
                        pygame.display.flip()
                        if game.game_over:
                            running_aa = False
                    show_post_game_screen(screen, game, assets, bg, clock, auto_continue=True)
                continue

    pygame.quit()


if __name__ == "__main__":
    main()

# campaign.py remains unchanged
