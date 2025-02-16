import pygame
import pygame_gui
import sys
import os

def run_menu():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Dragonchess Menu")
    manager = pygame_gui.UIManager((600, 400))

    # Create main menu buttons.
    button_2_player = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 50), (200, 50)),
        text='2 Player',
        manager=manager
    )
    button_ai_scarlet = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 120), (200, 50)),
        text='AI as Scarlet',
        manager=manager
    )
    button_ai_gold = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 190), (200, 50)),
        text='AI as Gold',
        manager=manager
    )
    button_ai_vs_ai = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 260), (200, 50)),
        text='AI vs AI',
        manager=manager
    )

    # Create browse buttons for custom AI files.
    button_browse_scarlet = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((200, 330), (90, 40)),
        text='Browse Scarlet',
        manager=manager
    )
    button_browse_gold = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((310, 330), (90, 40)),
        text='Browse Gold',
        manager=manager
    )

    mode = None
    custom_ai = {"scarlet": None, "gold": None}
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
                    if event.ui_element == button_2_player:
                        mode = "2 Player"
                        running = False
                    elif event.ui_element == button_ai_scarlet:
                        mode = "AI as Scarlet"
                        running = False
                    elif event.ui_element == button_ai_gold:
                        mode = "AI as Gold"
                        running = False
                    elif event.ui_element == button_ai_vs_ai:
                        mode = "AI vs AI"
                        running = False
                    elif event.ui_element == button_browse_scarlet:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((100, 50), (400, 300)),
                            manager=manager,
                            window_title="Select Scarlet AI File",
                            initial_file_path=os.getcwd()
                        )
                        # Set our custom attribute:
                        active_file_dialog.custom_title = "Select Scarlet AI File"
                    elif event.ui_element == button_browse_gold:
                        active_file_dialog = pygame_gui.windows.UIFileDialog(
                            rect=pygame.Rect((100, 50), (400, 300)),
                            manager=manager,
                            window_title="Select Gold AI File",
                            initial_file_path=os.getcwd()
                        )
                        active_file_dialog.custom_title = "Select Gold AI File"
                if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                    # Check our custom_title attribute instead.
                    if hasattr(event.ui_element, "custom_title"):
                        if event.ui_element.custom_title == "Select Scarlet AI File":
                            custom_ai["scarlet"] = event.text
                        elif event.ui_element.custom_title == "Select Gold AI File":
                            custom_ai["gold"] = event.text

            manager.process_events(event)
        manager.update(time_delta)
        screen.fill((30, 30, 30))
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
    input_log_filename.set_text("ai_vs_ai_log.csv")

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
        "log_filename": "ai_vs_ai_log.csv",
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
                            options["num_games"] = int(input_num_games.get_text())
                        except ValueError:
                            options["num_games"] = 10
                        options["log_filename"] = input_log_filename.get_text() or "ai_vs_ai_log.csv"
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
