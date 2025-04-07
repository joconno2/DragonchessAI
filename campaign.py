import os
import json
import pygame
import pygame_gui
import importlib.util
from game import Game, pos_to_index
from common import (CELL_SIZE, BOARD_GAP, BOARD_LEFT_MARGIN, BOARD_TOP_MARGIN, SIDE_PANEL_WIDTH,
                    load_assets, screen_to_board, draw_board, window_to_design,
                    BOARD_ROWS, BOARD_COLS, NUM_BOARDS)

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

def run_campaign_menu():
    pygame.init()
    screen = pygame.display.set_mode((360,640))
    pygame.display.set_caption("Dragonchess Campaign - Save Selection")
    manager = pygame_gui.UIManager((360,640))
    
    slot_buttons = []
    for i in range(1,4):
        profile = load_profile(i)
        text = profile["player_name"] if profile else f"Empty Slot {i}"
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((40,100+(i-1)*80), (280,50)),
            text=text,
            manager=manager
        )
        slot_buttons.append((i,btn))
    
    running = True
    clock = pygame.time.Clock()
    while running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    for slot, btn in slot_buttons:
                        if event.ui_element == btn:
                            profile = load_profile(slot)
                            if profile:
                                run_opponent_selection(profile, slot)
                            else:
                                run_character_creation(slot)
                            running = False
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50,50,50))
        manager.draw_ui(screen)
        pygame.display.update()

def run_character_creation(slot):
    pygame.init()
    screen = pygame.display.set_mode((360,640))
    pygame.display.set_caption("Dragonchess Campaign - Character Creation")
    manager = pygame_gui.UIManager((360,640))
    
    name_entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((80,80), (200,40)),
        manager=manager
    )
    name_entry.set_text("Enter Name")
    
    portrait_colors = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,0),
        (255,0,255),(0,255,255),(128,128,128),(255,128,0)
    ]
    portrait_buttons = []
    for idx, color in enumerate(portrait_colors):
        x = 20+(idx % 4)*80
        y = 150+(idx//4)*80
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x,y), (70,70)),
            text="",
            manager=manager
        )
        portrait_buttons.append((idx,btn))
        
    confirm_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((80,550), (200,40)),
        text="Confirm",
        manager=manager
    )
    
    selected_portrait = None
    running = True
    clock = pygame.time.Clock()
    while running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    for idx, btn in portrait_buttons:
                        if event.ui_element == btn:
                            selected_portrait = idx
                    if event.ui_element == confirm_button:
                        player_name = name_entry.get_text()
                        if not player_name or selected_portrait is None:
                            print("Please enter a name and select a portrait.")
                        else:
                            profile = {
                                "player_name": player_name,
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
        screen.fill((50,50,50))
        manager.draw_ui(screen)
        pygame.display.update()

def run_opponent_selection(profile, slot):
    pygame.init()
    screen = pygame.display.set_mode((360,640))
    pygame.display.set_caption("Dragonchess Campaign - Opponent Selection")
    manager = pygame_gui.UIManager((360,640))
    
    opponents = [
        {"name": "Novice", "ai_file": os.path.join("bots","novice.py")},
        {"name": "Apprentice", "ai_file": os.path.join("bots","apprentice.py")},
        {"name": "Veteran", "ai_file": os.path.join("bots","veteran.py")},
        {"name": "Champion", "ai_file": os.path.join("bots","champion.py")}
    ]
    unlocked = profile.get("unlocked_opponents", [0])
    opponent_buttons = []
    for idx, opp in enumerate(opponents):
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((40,100+idx*80), (280,50)),
            text=opp["name"],
            manager=manager
        )
        if idx not in unlocked:
            btn.disable()
        opponent_buttons.append((idx,btn))
    
    running = True
    clock = pygame.time.Clock()
    while running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    for idx, btn in opponent_buttons:
                        if event.ui_element == btn and idx in unlocked:
                            print(f"Starting campaign game against {opponents[idx]['name']}")
                            run_campaign_game(profile, opponents[idx])
                            running = False
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50,50,50))
        manager.draw_ui(screen)
        pygame.display.update()

def run_campaign_game(profile, opponent):
    board_width = CELL_SIZE * 12
    total_board_width = BOARD_LEFT_MARGIN * 2 + 3 * board_width + 2 * BOARD_GAP
    design_width = total_board_width + SIDE_PANEL_WIDTH
    design_height = BOARD_TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 20
    design_resolution = (design_width, design_height)
    flags = pygame.RESIZABLE
    screen = pygame.display.set_mode((1280,720), flags)
    pygame.display.set_caption(f"Campaign Game: {profile['player_name']} vs {opponent['name']}")
    
    assets = load_assets(CELL_SIZE)
    bg = pygame.image.load(os.path.join("assets","bg.png")).convert()
    bg = pygame.transform.scale(bg, (design_width - SIDE_PANEL_WIDTH, design_height))
    
    game = Game()
    enemy_ai = load_custom_ai(opponent["ai_file"], game, "Scarlet")
    
    selected_index = None
    legal_destinations = None
    clock = pygame.time.Clock()
    base_surface = pygame.Surface(design_resolution)
    running_game = True
    while not game.game_over and running_game:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_game = False
                break
            if event.type == pygame.MOUSEBUTTONUP:
                if game.current_turn == "Gold":
                    design_mouse = window_to_design(pygame.mouse.get_pos(), design_resolution)
                    if design_mouse:
                        board_pos = screen_to_board(design_mouse)
                        if board_pos:
                            idx = pos_to_index(*board_pos)
                            if selected_index is None:
                                if game.board[idx] > 0:
                                    selected_index = idx
                                    moves_list = game.get_legal_moves_for(selected_index)
                                    legal_destinations = {move[1] for move in moves_list}
                            else:
                                if idx in legal_destinations:
                                    moves_list = game.get_legal_moves_for(selected_index)
                                    for move in moves_list:
                                        if move[1] == idx:
                                            game.make_move(move)
                                            break
                                    selected_index = None
                                    legal_destinations = None
                                else:
                                    if game.board[idx] > 0:
                                        selected_index = idx
                                        moves_list = game.get_legal_moves_for(selected_index)
                                        legal_destinations = {move[1] for move in moves_list}
                                    else:
                                        selected_index = None
                                        legal_destinations = None
        if game.current_turn == "Scarlet":
            move = enemy_ai.choose_move()
            if move:
                game.make_move(move)
        game.update()
        draw_board(base_surface, game, assets, bg, selected_index, legal_destinations)
        window_width, window_height = pygame.display.get_surface().get_size()
        scale = min(window_width/design_width, window_height/design_height)
        scaled_width = int(design_width * scale)
        scaled_height = int(design_height * scale)
        scaled_surface = pygame.transform.scale(base_surface, (scaled_width, scaled_height))
        final_surface = pygame.Surface((window_width, window_height))
        final_surface.fill((0,0,0))
        final_surface.blit(scaled_surface, ((window_width-scaled_width)//2, (window_height-scaled_height)//2))
        screen.blit(final_surface, (0,0))
        pygame.display.update()
    
    if game.winner == "Gold":
        print("You win!")
        profile["wins"] = profile.get("wins", 0) + 1
        unlocked = profile.get("unlocked_opponents", [0])
        if len(unlocked) < 4:
            next_enemy = len(unlocked)
            if next_enemy not in unlocked:
                unlocked.append(next_enemy)
                profile["unlocked_opponents"] = unlocked
    else:
        print("You lose or draw. No new enemy unlocked.")
    slot = profile.get("slot", 1)
    save_profile(slot, profile)
    return
