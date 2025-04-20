import sys
import os
import json
import pygame
import pygame_gui
import importlib.util
from game import Game, pos_to_index
from common import (
    DESIGN_WIDTH, DESIGN_HEIGHT,
    BG_SCALE_X, BG_SCALE_Y,
    CELL_SIZE, load_assets,
    screen_to_board, draw_board
)

# --- Helpers ---
def add_gold_border(image, border_thickness=2):
    gold = (255, 215, 0)
    w, h = image.get_size()
    surf = pygame.Surface((w + 2*border_thickness, h + 2*border_thickness), pygame.SRCALPHA)
    surf.fill(gold)
    surf.blit(image, (border_thickness, border_thickness))
    return surf

# --- Profile Persistence ---
def load_profile(slot):
    path = os.path.join("saves", f"slot{slot}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_profile(slot, data):
    os.makedirs("saves", exist_ok=True)
    path = os.path.join("saves", f"slot{slot}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def load_custom_ai(filepath, game, color):
    spec = importlib.util.spec_from_file_location("custom_ai", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CustomAI(game, color)

# --- Campaign Menu ---
def run_campaign_menu():
    pygame.init()
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("Dragonchess Campaign - Save Selection")
    manager = pygame_gui.UIManager((360, 640))
    clock = pygame.time.Clock()

    slot_buttons = []
    for i in range(1, 4):
        profile = load_profile(i)
        text = profile["player_name"] if profile else f"Empty Slot {i}"
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((80, 100 + (i-1)*80), (200, 50)),
            text=text,
            manager=manager
        )
        slot_buttons.append((i, btn))
        if profile:
            avatar_idx = profile.get("portrait", 0)
            path = os.path.join("assets", f"player{avatar_idx+1}.PNG")
            try:
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.scale(img, (40, 40))
                img = add_gold_border(img, 2)
                pygame_gui.elements.UIImage(
                    relative_rect=pygame.Rect((20, 100 + (i-1)*80 + 5), (40, 40)),
                    image_surface=img,
                    manager=manager
                )
            except Exception:
                pass

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                for slot, btn in slot_buttons:
                    if event.ui_element == btn:
                        profile = load_profile(slot)
                        if profile:
                            run_opponent_selection(profile, slot)
                        else:
                            run_character_creation(slot)
                        running = False
                        break
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50, 50, 50))
        manager.draw_ui(screen)
        pygame.display.update()

# --- Opponent Selection with World Map ---
def run_opponent_selection(profile, slot):
    pygame.init()
    screen = pygame.display.set_mode((891, 899))
    pygame.display.set_caption("Dragonchess Campaign - World Map")
    manager = pygame_gui.UIManager((891, 899))
    clock = pygame.time.Clock()

    background = pygame.image.load(os.path.join("assets", "world_map.png")).convert()

    unlocked = profile.get("unlocked_opponents", [0])
    opponent_files = [
        os.path.join("bots", "novice.py"),
        os.path.join("bots", "apprentice.py"),
        os.path.join("bots", "veteran.py"),
        os.path.join("bots", "champion.py"),
        os.path.join("bots", "warlord.py")
    ]

    positions = [(753, 728), (685, 593), (413, 427), (200, 444), (397, 190)]
    stage_buttons = []
    for idx, (x, y) in enumerate(positions):
        active = os.path.join("assets", f"stage_{idx+1}_active.png")
        inactive = os.path.join("assets", f"stage_{idx+1}_inactive.png")
        img = pygame.image.load(active if idx in unlocked else inactive).convert_alpha()
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x, y), (107, 103)),
            text="",
            manager=manager,
            tool_tip_text=f"Stage {idx+1}" if idx in unlocked else "Locked"
        )
        btn.normal_image = img
        btn.hover_image = img
        btn.pressed_image = img
        btn.disabled_image = img
        btn.rebuild()
        if idx not in unlocked:
            btn.disable()
        stage_buttons.append((idx, btn))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                for idx, btn in stage_buttons:
                    if event.ui_element == btn:
                        run_campaign_game(profile, {
                            "name": f"Stage {idx+1}",
                            "ai_file": opponent_files[idx]
                        })
                        running = False
                        break
            manager.process_events(event)
        manager.update(dt)
        screen.blit(background, (0, 0))
        manager.draw_ui(screen)
        pygame.display.update()

# --- Character Creation ---
def run_character_creation(slot):
    pygame.init()
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("Dragonchess Campaign - Character Creation")
    manager = pygame_gui.UIManager((360, 640))
    clock = pygame.time.Clock()

    name_entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((80, 20), (200, 40)),
        manager=manager
    )
    name_entry.set_text("")

    avatars = []
    for idx in range(8):
        path = os.path.join("assets", f"player{idx+1}.PNG")
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (90, 90))
        avatars.append(img)

    avatar_buttons = []
    gap = 10
    grid_x = (360 - (2*90 + gap)) // 2
    grid_y = 100
    for idx, img in enumerate(avatars):
        col = idx % 2
        row = idx // 2
        x = grid_x + col*(90+gap)
        y = grid_y + row*(90+gap)
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x, y), (90, 90)),
            text=" ",
            manager=manager
        )
        btn.normal_image = img.copy()
        btn.hover_image = img.copy()
        btn.pressed_image = img.copy()
        btn.disabled_image = img.copy()
        btn.rebuild()
        avatar_buttons.append((idx, btn))

    confirm_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((80, 560), (200, 40)),
        text="Confirm",
        manager=manager
    )

    selected_portrait = None
    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                for idx, btn in avatar_buttons:
                    if event.ui_element == btn:
                        selected_portrait = idx
                        for j, b in avatar_buttons:
                            if j == idx:
                                bordered = add_gold_border(avatars[j], 3)
                                b.normal_image = bordered
                            else:
                                b.normal_image = avatars[j].copy()
                            b.hover_image = b.normal_image
                            b.pressed_image = b.normal_image
                            b.disabled_image = b.normal_image
                            b.rebuild()
                if event.ui_element == confirm_btn:
                    name = name_entry.get_text().strip()
                    if name and selected_portrait is not None:
                        profile = {
                            "player_name": name,
                            "portrait": selected_portrait,
                            "wins": 0,
                            "unlocked_opponents": [0],
                            "slot": slot
                        }
                        save_profile(slot, profile)
                        run_opponent_selection(profile, slot)
                        running = False
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50, 50, 50))
        manager.draw_ui(screen)
        pygame.display.update()

# --- Game Loop ---
def run_campaign_game(profile, opponent):
    pygame.init()
    screen = pygame.display.set_mode((DESIGN_WIDTH, DESIGN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(f"{profile['player_name']} vs {opponent['name']}")
    clock = pygame.time.Clock()

    base_surface = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
    assets = load_assets(CELL_SIZE)
    bg_img = pygame.image.load(os.path.join("assets", "bg.png")).convert()
    bg = pygame.transform.scale(bg_img, (int(bg_img.get_width()*BG_SCALE_X), int(bg_img.get_height()*BG_SCALE_Y)))

    game = Game()
    enemy_ai = load_custom_ai(opponent['ai_file'], game, 'Scarlet')
    selected = None
    legal_dest = None
    running = True

    while running and not game.game_over:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and game.current_turn == 'Gold':
                mx, my = pygame.mouse.get_pos()
                win_w, win_h = screen.get_size()
                sx, sy = DESIGN_WIDTH/win_w, DESIGN_HEIGHT/win_h
                dm = (mx*sx, my*sy)
                bp = screen_to_board(dm)
                if bp:
                    idx = pos_to_index(*bp)
                    if selected is None and game.board[idx] > 0:
                        selected = idx
                        legal_dest = {m[1] for m in game.get_legal_moves_for(selected)}
                    elif selected is not None:
                        if idx in legal_dest:
                            for mv in game.get_legal_moves_for(selected):
                                if mv[1]==idx:
                                    game.make_move(mv)
                                    break
                            selected, legal_dest = None, None
                        elif game.board[idx]>0:
                            selected = idx
                            legal_dest = {m[1] for m in game.get_legal_moves_for(selected)}
        if game.current_turn == 'Scarlet' and not game.game_over:
            mv = enemy_ai.choose_move()
            if mv:
                game.make_move(mv)
        game.update()

        base_surface.fill((0,0,0))
        draw_board(base_surface, game, assets, bg, selected, legal_dest)
        win_w, win_h = screen.get_size()
        screen.blit(pygame.transform.scale(base_surface,(win_w,win_h)),(0,0))
        pygame.display.flip()

    font = pygame.font.Font("assets/pixel.ttf",48)
    small = pygame.font.Font("assets/pixel.ttf",32)
    win_w, win_h = screen.get_size()
    wtxt = small.render(f"Winner: {game.winner}",True,(255,255,255))
    mtxt = small.render(f"Moves: {len(game.move_notations)}",True,(255,255,255))
    btn = pygame.Rect((win_w-200)//2,int(win_h*0.75),200,50)
    showing = True
    while showing:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: sys.exit()
            if event.type==pygame.MOUSEBUTTONUP and btn.collidepoint(event.pos):
                showing=False
        base_surface.fill((0,0,0))
        draw_board(base_surface, game, assets, bg)
        screen.blit(pygame.transform.scale(base_surface,(win_w,win_h)),(0,0))
        screen.blit(wtxt,((win_w-wtxt.get_width())//2,int(win_h*0.3)))
        screen.blit(mtxt,((win_w-mtxt.get_width())//2,int(win_h*0.4)))
        pygame.draw.rect(screen,(255,255,255),btn)
        text=small.render("Back to Selection",True,(0,0,0))
        screen.blit(text,(btn.x+(200-text.get_width())//2,btn.y+(50-text.get_height())//2))
        pygame.display.flip()

    if game.winner=='Gold':
        profile['wins']=profile.get('wins',0)+1
        unlocked=profile.get('unlocked_opponents',[])
        next_idx=len(unlocked)
        if next_idx<5 and next_idx not in unlocked:
            unlocked.append(next_idx)
            profile['unlocked_opponents']=unlocked
    save_profile(profile.get('slot',1),profile)
