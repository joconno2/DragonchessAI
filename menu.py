import pygame
import pygame_gui
import sys
import os

def run_menu():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Dragonchess Menu")
    manager = pygame_gui.UIManager((600, 400))

    button_2_player = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 100), (200, 50)),
        text='2 Player',
        manager=manager
    )
    button_ai_vs_player = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 170), (200, 50)),
        text='AI vs Player',
        manager=manager
    )
    button_ai_vs_ai = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 240), (200, 50)),
        text='AI vs AI',
        manager=manager
    )
    button_tournament = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 310), (200, 50)),
        text='Tournament',
        manager=manager
    )

    mode = None
    custom_ai = {"scarlet": None, "gold": None}
    clock = pygame.time.Clock()
    running = True

    title_font = pygame.font.Font("pixel.ttf", 48)
    title_text = title_font.render("Dragonchess", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(300, 50))

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
            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((30, 30, 30))
        screen.blit(title_text, title_rect)
        manager.draw_ui(screen)
        pygame.display.update()

    return mode, custom_ai

def run_ai_vs_ai_menu():
    pygame.init()
    screen = pygame.display.set_mode((600, 500))
    pygame.display.set_caption("AI vs AI Options")
    manager = pygame_gui.UIManager((600, 500))

    label_num_games = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 50), (200, 40)),
        text="Number of Games:",
        manager=manager
    )
    input_num_games = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((300, 50), (200, 40)),
        manager=manager
    )
    input_num_games.set_text("10")

    label_log_filename = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 110), (200, 40)),
        text="Log Filename:",
        manager=manager
    )
    input_log_filename = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((300, 110), (200, 40)),
        manager=manager
    )
    input_log_filename.set_text("logs/ai_vs_ai_log.csv")

    label_headless = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 170), (200, 40)),
        text="Headless Mode:",
        manager=manager
    )
    button_headless_toggle = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((300, 170), (200, 40)),
        text="Yes",
        manager=manager
    )

    button_browse_scarlet = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((50, 230), (200, 40)),
        text="Browse Scarlet AI",
        manager=manager
    )
    button_browse_gold = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((350, 230), (200, 40)),
        text="Browse Gold AI",
        manager=manager
    )

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 350), (200, 50)),
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
                            rect=pygame.Rect((100, 50), (400, 300)),
                            manager=manager,
                            window_title="Select Scarlet AI File",
                            initial_file_path=os.getcwd()
                        )
                        active_file_dialog.custom_title = "Select Scarlet AI File"
                    elif event.ui_element == button_browse_gold:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((100, 50), (400, 300)),
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
    screen = pygame.display.set_mode((600, 500))
    pygame.display.set_caption("AI vs Player Options")
    manager = pygame_gui.UIManager((600, 500))

    label_ai_side = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 50), (200, 40)),
        text="AI Side:",
        manager=manager
    )
    button_toggle_ai_side = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((300, 50), (200, 40)),
        text="Gold",
        manager=manager
    )

    button_browse_ai = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((50, 110), (200, 40)),
        text="Browse AI File",
        manager=manager
    )
    label_ai_file = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((300, 110), (200, 40)),
        text="None",
        manager=manager
    )

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 350), (200, 50)),
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
                            rect=pygame.Rect((100, 50), (400, 300)),
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
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Tournament Options")
    manager = pygame_gui.UIManager((600, 600))

    label_rounds = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 50), (200, 40)),
        text="Rounds:",
        manager=manager
    )
    input_rounds = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((300, 50), (200, 40)),
        manager=manager
    )
    input_rounds.set_text("5")

    label_csv = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((50, 110), (200, 40)),
        text="Output CSV:",
        manager=manager
    )
    input_csv = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((300, 110), (200, 40)),
        manager=manager
    )
    input_csv.set_text("tournament_results.csv")

    # Create 8 file selection buttons and labels for 8 bots.
    bot_buttons = []
    bot_labels = []
    # Default file paths are now set to None (meaning use RandomAI).
    default_paths = [None for _ in range(8)]
    for i in range(8):
        y_pos = 180 + i * 40
        btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((50, y_pos), (200, 35)),
            text=f"Browse Bot {i+1}",
            manager=manager
        )
        lbl = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((300, y_pos), (250, 35)),
            text=str(default_paths[i]),
            manager=manager
        )
        bot_buttons.append(btn)
        bot_labels.append(lbl)

    button_start = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 520), (200, 50)),
        text="Start Tournament",
        manager=manager
    )

    options = {
        "tournament_rounds": 5,
        "bot_file_paths": default_paths.copy(),
        "output_csv": "tournament_results.csv",
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
                                rect=pygame.Rect((100, 50), (400, 300)),
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
                        options["output_csv"] = input_csv.get_text() or "tournament_results.csv"
                        # Update the bot file paths from the labels.
                        for i, lbl in enumerate(bot_labels):
                            options["bot_file_paths"][i] = lbl.text  # using the .text property
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
