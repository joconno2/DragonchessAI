import pygame
import pygame_gui
import sys
import os

def run_menu():
    pygame.init()
    # 9:16 resolution for menus (360x640)
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("Dragonchess Menu")
    manager = pygame_gui.UIManager((360, 640))

    # Load and scale the main menu background image.
    background = pygame.image.load(os.path.join("assets", "main_menu_1.png")).convert()
    background = pygame.transform.scale(background, (360, 640))

    # Title configuration.
    title_font = pygame.font.Font("assets/pixel.ttf", 36)
    title_text = title_font.render("Dragonchess", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(180, 80))

    # Button configuration.
    button_width = 140
    button_height = 40
    button_x = 110
    button_y_start = 140
    button_gap = 60

    button_2_player = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((button_x, button_y_start), (button_width, button_height)),
        text='2 Player',
        manager=manager
    )
    button_ai_vs_player = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((button_x, button_y_start+button_gap), (button_width, button_height)),
        text='AI vs Player',
        manager=manager
    )
    button_ai_vs_ai = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((button_x, button_y_start+2*button_gap), (button_width, button_height)),
        text='AI vs AI',
        manager=manager
    )
    button_tournament = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((button_x, button_y_start+3*button_gap), (button_width, button_height)),
        text='Tournament',
        manager=manager
    )
    button_campaign = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((button_x, button_y_start+4*button_gap), (button_width, button_height)),
        text='Campaign',
        manager=manager
    )

    mode = None
    custom_ai = {"scarlet": None, "gold": None}
    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == button_2_player:
                        mode = "2 Player"
                        running = False
                    elif event.ui_element == button_ai_vs_player:
                        mode = "AI vs Player"
                        running = False
                    elif event.ui_element == button_ai_vs_ai:
                        mode = "AI vs AI"
                        running = False
                    elif event.ui_element == button_tournament:
                        mode = "Tournament"
                        running = False
                    elif event.ui_element == button_campaign:
                        mode = "Campaign"
                        running = False
            manager.process_events(event)
        manager.update(time_delta)
        # Blit the main menu background image first...
        screen.blit(background, (0, 0))
        # ...then draw the title and the UI elements on top.
        screen.blit(title_text, title_rect)
        manager.draw_ui(screen)
        pygame.display.update()

    return mode, custom_ai


def run_ai_vs_ai_menu():
    pygame.init()
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("AI vs AI Options")
    manager = pygame_gui.UIManager((360, 640))

    label_num_games = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 50), (120, 30)),
        text="Number of Games:",
        manager=manager
    )
    input_num_games = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((150, 50), (180, 30)),
        manager=manager
    )
    input_num_games.set_text("10")

    label_log_filename = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 100), (120, 30)),
        text="Log Filename:",
        manager=manager
    )
    input_log_filename = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((150, 100), (180, 30)),
        manager=manager
    )
    input_log_filename.set_text("logs/ai_vs_ai_log.csv")

    label_headless = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 150), (120, 30)),
        text="Headless Mode:",
        manager=manager
    )
    button_headless_toggle = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((150, 150), (180, 30)),
        text="Yes",
        manager=manager
    )

    button_browse_scarlet = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((20, 200), (140, 30)),
        text="Browse Scarlet AI",
        manager=manager
    )
    button_browse_gold = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((180, 200), (150, 30)),
        text="Browse Gold AI",
        manager=manager
    )

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((80, 550), (200, 40)),
        text="Start",
        manager=manager
    )

    options = {
        "num_games": 10,
        "log_filename": "logs/ai_vs_ai_log.csv",
        "headless": True,
        "scarlet_ai": None,
        "gold_ai": None
    }

    clock = pygame.time.Clock()
    running = True
    active_file_dialog = None

    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == button_headless_toggle:
                        options["headless"] = not options["headless"]
                        event.ui_element.set_text("Yes" if options["headless"] else "No")
                    elif event.ui_element == button_browse_scarlet:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((50, 150), (260, 300)),
                            manager=manager,
                            window_title="Select Scarlet AI File",
                            initial_file_path=os.getcwd()
                        )
                        active_file_dialog.custom_title = "Select Scarlet AI File"
                    elif event.ui_element == button_browse_gold:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((50, 150), (260, 300)),
                            manager=manager,
                            window_title="Select Gold AI File",
                            initial_file_path=os.getcwd()
                        )
                        active_file_dialog.custom_title = "Select Gold AI File"
                    elif event.ui_element == button_start:
                        try:
                            options["num_games"] = int(input_num_games.get_text().strip() or "10")
                        except ValueError:
                            options["num_games"] = 10
                        options["log_filename"] = input_log_filename.get_text() or "logs/ai_vs_ai_log.csv"
                        running = False
                if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                    if hasattr(event.ui_element, "custom_title"):
                        if event.ui_element.custom_title == "Select Scarlet AI File":
                            options["scarlet_ai"] = event.text
                        elif event.ui_element.custom_title == "Select Gold AI File":
                            options["gold_ai"] = event.text
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50, 50, 50))
        manager.draw_ui(screen)
        pygame.display.update()
    return options

def run_ai_vs_player_menu():
    pygame.init()
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("AI vs Player Options")
    manager = pygame_gui.UIManager((360, 640))

    label_ai_side = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 50), (120, 30)),
        text="AI Side:",
        manager=manager
    )
    button_toggle_ai_side = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((150, 50), (180, 30)),
        text="Gold",
        manager=manager
    )

    button_browse_ai = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((20, 100), (140, 30)),
        text="Browse AI File",
        manager=manager
    )
    label_ai_file = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((170, 100), (160, 30)),
        text="None",
        manager=manager
    )

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((80, 550), (200, 40)),
        text="Start",
        manager=manager
    )

    options = {
        "ai_side": "Gold",
        "ai_file": None
    }

    clock = pygame.time.Clock()
    running = True
    active_file_dialog = None

    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == button_toggle_ai_side:
                        if options["ai_side"] == "Gold":
                            options["ai_side"] = "Scarlet"
                            button_toggle_ai_side.set_text("Scarlet")
                        else:
                            options["ai_side"] = "Gold"
                            button_toggle_ai_side.set_text("Gold")
                    elif event.ui_element == button_browse_ai:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((50, 150), (260, 300)),
                            manager=manager,
                            window_title="Select AI File",
                            initial_file_path=os.getcwd()
                        )
                        active_file_dialog.custom_title = "Select AI File"
                    elif event.ui_element == button_start:
                        running = False
                if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                    if hasattr(event.ui_element, "custom_title") and event.ui_element.custom_title == "Select AI File":
                        options["ai_file"] = event.text
                        label_ai_file.set_text(event.text)
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50, 50, 50))
        manager.draw_ui(screen)
        pygame.display.update()
    return options

def run_tournament_menu():
    pygame.init()
    screen = pygame.display.set_mode((360, 640))
    pygame.display.set_caption("Tournament Options")
    manager = pygame_gui.UIManager((360, 640))

    label_rounds = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 50), (120, 30)),
        text="Rounds:",
        manager=manager
    )
    input_rounds = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((150, 50), (180, 30)),
        manager=manager
    )
    input_rounds.set_text("5")

    label_csv = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((20, 100), (120, 30)),
        text="Output CSV:",
        manager=manager
    )
    input_csv = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((150, 100), (180, 30)),
        manager=manager
    )
    input_csv.set_text("logs/tournament_results.csv")

    bot_buttons = []
    bot_labels = []
    default_paths = [None for _ in range(8)]
    for i in range(8):
        y_pos = 150 + i * 35
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, y_pos), (140, 30)),
            text=f"Browse Bot {i+1}",
            manager=manager
        )
        lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((170, y_pos), (160, 30)),
            text=str(default_paths[i]),
            manager=manager
        )
        bot_buttons.append(btn)
        bot_labels.append(lbl)

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((80, 580), (200, 40)),
        text="Start Tournament",
        manager=manager
    )

    options = {
        "tournament_rounds": 5,
        "bot_file_paths": default_paths.copy(),
        "output_csv": "logs/tournament_results.csv",
        "headless": True
    }

    clock = pygame.time.Clock()
    running = True
    active_file_dialog = None

    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    for i, btn in enumerate(bot_buttons):
                        if event.ui_element == btn:
                            active_file_dialog = pygame_gui.windows.UIFileDialog(
                                rect=pygame.Rect((50, 150), (260, 300)),
                                manager=manager,
                                window_title=f"Select Bot {i+1} File",
                                initial_file_path=os.getcwd()
                            )
                            active_file_dialog.custom_title = f"Select Bot {i+1} File"
                    if event.ui_element == button_start:
                        try:
                            rounds_str = input_rounds.get_text().strip()
                            options["tournament_rounds"] = int(rounds_str) if rounds_str != "" else 5
                        except ValueError:
                            options["tournament_rounds"] = 5
                        options["output_csv"] = input_csv.get_text() or "logs/tournament_results.csv"
                        for i, lbl in enumerate(bot_labels):
                            options["bot_file_paths"][i] = lbl.text
                        running = False
                if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                    if hasattr(event.ui_element, "custom_title"):
                        title = event.ui_element.custom_title
                        if title.startswith("Select Bot"):
                            try:
                                bot_index = int(title.split()[2]) - 1
                                options["bot_file_paths"][bot_index] = event.text
                                bot_labels[bot_index].set_text(event.text)
                            except Exception:
                                pass
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((50, 50, 50))
        manager.draw_ui(screen)
        pygame.display.update()
    return options
