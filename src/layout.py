import PySimpleGUI as sg
import constants as ct


def popup(message):
    sg.popup_quick_message(message, keep_on_top=True,
                           line_width=200, auto_close_duration=3)


def field_label(text, key, offset):
    label = text + ':' if text is not None and len(text.strip()) > 0 else text
    return sg.Text(label,
                   justification='r', size=(offset, 1), pad=(20, 0), key=key)


def st_label(text, key=None):
    return field_label(text, key, 20)


def text_label(text, key=None):
    return field_label(text, key, 10)


def text_field(text, key=None):
    return sg.Text(text, justification='l', size=(10 + ct.OFFSET, 1), key=key)


def text_output(key, length, padding=None):
    bg_col = ct.OUTPUT_BACKGROUND_COLOR
    return sg.Input(size=(length, 1), readonly=True, key=key, pad=padding,
                    disabled_readonly_background_color=bg_col,
                    text_color=ct.OUTPUT_TEXT_COLOR)


def pad(s):
    return sg.Text("", size=(s, 1))


def row_separator(text, line=False):
    if line:
        return (sg.Text(text, font=ct.SEPARATOR_FONT),
                sg.HSeparator(pad=((5, 18), (4, 2))))
    else:
        return sg.Text(text, font=ct.SEPARATOR_FONT), sg.Text(" ")


def get_color_square(player):
    # create a small Graph (peg) to display the player color
    size = 16
    color_square = sg.Graph(canvas_size=(size, size),
                            graph_bottom_left=(0, 0),
                            graph_top_right=(size, size),
                            background_color=sg.theme_background_color(),
                            key=ct.K_COLOR[player],
                            pad=((6, 72 + ct.OFFSET * 8), (0, 0)),
                            enable_events=False)
    return color_square


# row_ functions make up the layout of the control bar


def row_colors():
    return [text_label(ct.K_COLOR[0]),
            get_color_square(1), get_color_square(2)]


def row_turn_indicators():
    bg_col = ct.OUTPUT_BACKGROUND_COLOR
    return [text_label(ct.K_TURN_INDICATOR[0]),
            sg.Input(size=(14, 1),
                     disabled_readonly_background_color=bg_col, readonly=True,
                     key=ct.K_TURN_INDICATOR[1], text_color=ct.TURN_COLOR),
            pad(1),
            sg.Input(size=(14, 1), disabled_readonly_background_color=bg_col,
                     readonly=True, key=ct.K_TURN_INDICATOR[2],
                     text_color=ct.TURN_COLOR)
            ]


def row_names():
    return [text_label(ct.K_NAME[0]),
            text_output(ct.K_NAME[1], 14), pad(1),
            text_output(ct.K_NAME[2], 14)]


def row_auto_moves():
    return [text_label(ct.K_AUTO_MOVE[0]),
            sg.Checkbox(text="", enable_events=True, default=False,
                        key=ct.K_AUTO_MOVE[1], size=(7 + ct.OFFSET, 1)),
            sg.Checkbox(text="", enable_events=True, default=False,
                        key=ct.K_AUTO_MOVE[2], size=(7 + ct.OFFSET, 1))]


def row_trials():
    return [text_label(ct.K_TRIALS[0]),
            sg.Slider(range=(0, ct.TRIALS_MAX), default_value=0, size=(11, 20),
                      resolution=ct.TRIALS_RESOLUTION, enable_events=True,
                      orientation='horizontal', key=ct.K_TRIALS[1]),
            pad(2),
            sg.Slider(range=(0, ct.TRIALS_MAX), default_value=0, size=(11, 20),
                      resolution=ct.TRIALS_RESOLUTION, enable_events=True,
                      orientation='horizontal', key=ct.K_TRIALS[2])]


def get_progress_bar():
    return sg.ProgressBar(1000, orientation='h', size=(19, 20),
                          key=ct.K_PROGRESS_BAR[1], relief='RELIEF_RIDGE',
                          bar_color=(ct.PROGRESS_BAR_COLOR,
                                     ct.OUTPUT_BACKGROUND_COLOR))


def row_progress_bar():
    return [text_label(ct.K_PROGRESS_BAR[0]),
            get_progress_bar()]


def row_progress_nums():
    return [text_label(ct.K_PROGRESS_NUM[0]),
            text_output(ct.K_PROGRESS_NUM[1], 29, padding=(4, 6)),
            sg.Image(ct.SPINNER_IMAGE, key=ct.K_SPINNER[1], visible=False)]


def row_visits():
    return [text_label(ct.K_VISITS[0]),
            sg.Canvas(background_color=ct.OUTPUT_BACKGROUND_COLOR,
                      size=(240, 80), key=ct.K_VISITS[1])]


def row_visualize_mcts():
    return [text_label(ct.K_VISUALIZE_MCTS[0]),
            sg.Checkbox(text="", enable_events=True,
                        default=ct.K_VISUALIZE_MCTS[3],
                        key=ct.K_VISUALIZE_MCTS[1])]


def row_moves():
    return [text_label(ct.K_MOVES[0]),
            sg.Multiline(default_text='', font=ct.MOVES_FONT,
                         background_color=ct.OUTPUT_BACKGROUND_COLOR,
                         text_color=ct.OUTPUT_TEXT_COLOR, autoscroll=True,
                         key=ct.K_MOVES[1], disabled=True, size=(28, 5))]


def row_eval_show_num():
    return [text_label(""),

            sg.Checkbox(text=ct.K_SHOW_EVALUATION[0], enable_events=True,
                        default=ct.K_SHOW_EVALUATION[3],
                        key=ct.K_SHOW_EVALUATION[1]),
            text_output(ct.K_EVAL_NUM[1], 7)
            ]


def row_heatmap():
    return [text_label(text=ct.K_HEATMAP[0]),
            sg.Checkbox("", enable_events=True, default=False,
                        key=ct.K_HEATMAP[1], size=(7 + ct.OFFSET, 1))
            ]


def row_eval_moves():
    bg_col = ct.OUTPUT_BACKGROUND_COLOR
    return [text_label(ct.K_EVAL_MOVES[0]),
            sg.Canvas(size=(240, 80), background_color=bg_col,
                      key=ct.K_EVAL_MOVES[1])]


def row_eval_hist():
    bg_col = ct.OUTPUT_BACKGROUND_COLOR
    return [text_label(ct.K_EVAL_HIST[0]),
            sg.Canvas(size=(240, 80), background_color=bg_col,
                      key=ct.K_EVAL_HIST[1])]


class MainWindowLayout:

    def __init__(self, board, stgs):
        self.board = board
        self.stgs = stgs
        self.layout = self.build_layout()

    def col_eval_bar(self):
        colors = (self.stgs.get(ct.K_COLOR[1]),
                  self.stgs.get(ct.K_COLOR[2]))

        # create dummy window to get line height in pixels
        layout = [[sg.Text(text='test', key='-TEXT-', font=('Helvetica', 10))]]
        window = sg.Window('test', layout, finalize=True)
        size = window['-TEXT-'].get_size()
        window.close()
        vsize = int(self.stgs.get(ct.K_BOARD_SIZE[1]) / (size[1] * 0.65))
        return [sg.ProgressBar(2000, orientation='v', size=(vsize, 8),
                               key=ct.K_EVAL_BAR[1], bar_color=colors,
                               pad=(0, 0)),
                sg.ProgressBar(1, orientation='v', size=(0, 5),
                               key="TESTEVAL", bar_color=("white", "white"),
                               pad=(0, 0))
                ]

    def build_layout(self):
        menu_def = [[ct.ITEM_FILE, [ct.ITEM_OPEN_FILE, ct.ITEM_SAVE_FILE,
                                    ct.ITEM_SETTINGS, ct.ITEM_EXIT]],
                    [ct.ITEM_HELP, [ct.ITEM_ABOUT]]]

        button_count = 7
        bw = int(self.stgs.get(ct.K_BOARD_SIZE[1]) / (button_count * 10))
        button_row = [
            sg.Button(ct.B_BOT_MOVE, size=(bw, 1), focus=True),
            sg.Button(ct.B_ACCEPT, size=(bw, 1)),
            sg.Button(ct.B_CANCEL, size=(bw, 1)),
            sg.Button(ct.B_UNDO, size=(bw, 1)),
            sg.Button(ct.B_REDO, size=(bw, 1)),
            sg.Button(ct.B_RESIGN, size=(bw, 1)),
            sg.Button(ct.B_RESET, size=(bw, 1))
        ]

        eval_bar_col = sg.Column([self.col_eval_bar()])

        control_col = sg.Column([row_colors(),
                                 row_names(),
                                 row_turn_indicators(),
                                 row_auto_moves(),
                                 row_moves(),
                                 row_separator("evaluation", True),
                                 row_eval_show_num(),
                                 row_eval_hist(),
                                 row_eval_moves(),
                                 row_heatmap(),
                                 row_separator("MCTS", True),
                                 row_trials(),
                                 row_visits(),
                                 row_visualize_mcts(),
                                 row_progress_bar(),
                                 row_progress_nums()
                                 ],
                                vertical_alignment='top')

        board_col = sg.Column([[self.board.graph]])

        layout = [
            [sg.Menu(menu_def, tearoff=False)],
            [
                board_col,
                eval_bar_col,
                control_col
            ],
            button_row
        ]

        return layout

    def get_layout(self):
        return self.layout


# SettingsDialog


def st_row_allow_swap():
    return [st_label(ct.K_ALLOW_SWAP[0]),
            sg.Checkbox(text="", default=ct.K_ALLOW_SWAP[3],
                        key=ct.K_ALLOW_SWAP[1])]


def st_row_allow_scl():
    return [st_label(ct.K_ALLOW_SCL[0]),
            sg.Checkbox(text="", default=ct.K_ALLOW_SCL[3],
                        key=ct.K_ALLOW_SCL[1])]


def st_row_smart_accept():
    return [st_label(ct.K_SMART_ACCEPT[0]),
            sg.Checkbox(text="", default=ct.K_SMART_ACCEPT[3],
                        key=ct.K_SMART_ACCEPT[1])]


def st_row_resign_threshold():
    return [st_label(ct.K_RESIGN_THRESHOLD[0]),
            sg.Spin(values=[float((x + 70.0) / 100.0) for x in range(31)],
                    initial_value=ct.K_RESIGN_THRESHOLD[3],
                    key=ct.K_RESIGN_THRESHOLD[1], size=(5, 0), readonly=True)]


def st_row_color(player):
    return [st_label(ct.K_COLOR[0]),
            sg.Combo(ct.COLOR_LIST, ct.K_COLOR[player + 2],
                     size=(15, 1), key=ct.K_COLOR[player], readonly=True)]


def st_row_name(player):
    return [st_label(ct.K_NAME[0]),
            sg.Input(ct.K_NAME[player + 2], size=(15, 1),
                     key=ct.K_NAME[player])]


def st_row_auto_move(player):
    return [st_label(ct.K_AUTO_MOVE[0]),
            sg.Checkbox(text="", default=ct.K_AUTO_MOVE[player + 2],
                        key=ct.K_AUTO_MOVE[player])]


def st_row_model_folder(player):
    return [st_label(ct.K_MODEL_FOLDER[0]),
            sg.Input(key=ct.K_MODEL_FOLDER[player], size=(30, 1)),
            sg.FolderBrowse(
                target=ct.K_MODEL_FOLDER[player],
                initial_folder=ct.K_MODEL_FOLDER[player + 2]),
            sg.Text(ct.MSG_REQUIRES_RESTART)]


def st_row_trials(player):
    return [st_label(ct.K_TRIALS[0]),
            sg.Slider(range=(0, ct.TRIALS_MAX), default_value=0,
                      resolution=ct.TRIALS_RESOLUTION,
                      orientation='horizontal', enable_events=False,
                      size=(11, 20), key=ct.K_TRIALS[player])]


def st_row_temperature(player):
    return [st_label(ct.K_TEMPERATURE[0]),
            sg.Combo(ct.TEMPERATURE_LIST, ct.K_TEMPERATURE[player + 2],
                     size=(5, 1), key=ct.K_TEMPERATURE[player], readonly=True)]


def st_row_add_noise(player):
    return [st_label(ct.K_ADD_NOISE[0]),
            sg.Spin(values=[float(x / 100.0) for x in range(101)],
                    initial_value=ct.K_ADD_NOISE[player + 2],
                    key=ct.K_ADD_NOISE[player], size=(5, 0), readonly=True)]


def st_row_cpuct(player):
    return [st_label(ct.K_CPUCT[0]),
            sg.Spin(values=[float(x / 100.0) for x in range(101)],
                    initial_value=ct.K_CPUCT[player + 2],
                    key=ct.K_CPUCT[player], size=(5, 0), readonly=True)]


def st_row_rotation(player):
    return [st_label(ct.K_ROTATION[0]),
            sg.Combo(ct.ROTATION_LIST, ct.K_ROTATION[player + 2], size=(15, 1),
                     key=ct.K_ROTATION[player], readonly=True)]


def st_row_level(player):
    return [st_label(ct.K_LEVEL[0]),
            sg.Spin(values=[float(x / 100.0) for x in range(101)],
                    initial_value=ct.K_LEVEL[player + 2],
                    key=ct.K_LEVEL[player], size=(5, 0), readonly=True)]

def st_row_smart_root(player):
    return [st_label(ct.K_SMART_ROOT[0]),
            sg.Checkbox(text="", default=ct.K_SMART_ROOT[player + 2],
                        key=ct.K_SMART_ROOT[player])]


def st_tab_player(player):
    return [[sg.Text("")],
            st_row_color(player),
            st_row_name(player),
            st_row_auto_move(player),
            row_separator("   evaluation"),
            st_row_level(player),
            st_row_model_folder(player),
            st_row_rotation(player),
            row_separator("   MCTS"),
            st_row_trials(player),
            st_row_smart_root(player),
            st_row_temperature(player),
            st_row_add_noise(player),
            st_row_cpuct(player),
            [sg.Text("")]
            ]


class SettingsDialogLayout:

    def __init__(self):
        self.layout = self.build_layout()

    @staticmethod
    def build_layout():

        st_tab_general = [[sg.Text("")],
                          st_row_allow_swap(),
                          st_row_allow_scl(),
                          row_separator(""),
                          [st_label(ct.K_BOARD_SIZE[0]),
                           sg.Combo(ct.BOARD_SIZE_LIST, ct.K_BOARD_SIZE[3],
                                    size=(15, 1), key=ct.K_BOARD_SIZE[1],
                                    readonly=True),
                           sg.Text(ct.MSG_REQUIRES_RESTART,
                                   pad=((0, 20), (0, 0)))],
                          [st_label(ct.K_SHOW_LABELS[0]),
                           sg.Checkbox(text="", default=ct.K_SHOW_LABELS[3],
                                       key=ct.K_SHOW_LABELS[1])],
                          [st_label(ct.K_SHOW_GUIDELINES[0]),
                           sg.Checkbox(text="",
                                       default=ct.K_SHOW_GUIDELINES[3],
                                       key=ct.K_SHOW_GUIDELINES[1])],
                          [st_label(ct.K_SHOW_CURSOR_LABEL[0]),
                           sg.Checkbox(text="",
                                       default=ct.K_SHOW_CURSOR_LABEL[3],
                                       key=ct.K_SHOW_CURSOR_LABEL[1])],
                          [st_label(ct.K_HIGHLIGHT_LAST_MOVE[0]),
                           sg.Checkbox(text="",
                                       default=ct.K_HIGHLIGHT_LAST_MOVE[3],
                                       key=ct.K_HIGHLIGHT_LAST_MOVE[1])],
                          row_separator(""),
                          st_row_smart_accept(),
                          st_row_resign_threshold(),
                          row_separator(""),
                          [st_label(ct.K_LOG_LEVEL[0]),
                           sg.Combo(ct.LOG_LEVEL_LIST, ct.K_LOG_LEVEL[3],
                                    size=(15, 1), key=ct.K_LOG_LEVEL[1],
                                    readonly=True)],
                          row_separator("")
                          ]

        st_tab_player1 = st_tab_player(1)
        st_tab_player2 = st_tab_player(2)

        layout = [
            [sg.TabGroup(
                [
                    [
                        sg.Tab(ct.TAB_LABEL_GENERAL, st_tab_general),
                        sg.Tab(ct.TAB_LABEL_PLAYER1,
                               st_tab_player1, pad=(20, 0)),
                        sg.Tab(ct.TAB_LABEL_PLAYER2,
                               st_tab_player2, pad=(10, 0))
                    ]
                ]
            )],
            [sg.Button(ct.B_APPLY_SAVE, size=(12, 1)),
             sg.Button(ct.B_RESET_DEFAULT, size=(15, 1)),
             sg.Button(ct.B_CANCEL, size=(10, 1), focus=True)]
        ]

        return layout

    def get_layout(self):
        return self.layout


class AboutDialogLayout:

    def __init__(self):
        self.layout = self.build_layout()

    def build_layout(self) -> object:
        s = (40, 1)
        return [
            [sg.Text("twixtbot engine and network by Jordan Lampe", size=s)],
            [sg.Text("https://github.com/BonyJordan/twixtbot", size=s)],
            [sg.Text("", size=s)],
            [sg.Text(
                "twixtbot-ui frontend by stevens68 and contributors", size=s)],
            [sg.Text("https://github.com/stevens68/twixtbot-ui", size=s)],
            [sg.Text("", size=s)],
            [sg.Button(ct.B_OK, size=(10, 1), focus=True)]
        ]

    def get_layout(self):
        return self.layout


class SplashScreenLayout:
    def __init__(self):
        width = 35
        self.layout = [
            [sg.ProgressBar(100, orientation='h', size=(30, 20),
                            # 'default', 'winnative', 'clam', 'alt', 'classic',
                            # 'vista', 'xpnative'
                            # relief='RELIEF_RIDGE',
                            key=ct.K_SPLASH_PROGRESS_BAR[0],
                            style='clam',
                            bar_color=(ct.PROGRESS_BAR_COLOR,
                                       'lightslategrey'))],
            [sg.Text("", size=(width, 1), key=ct.K_SPLASH_STATUS_TEXT[0],
                     background_color=sg.theme_background_color(),
                     text_color=ct.OUTPUT_TEXT_COLOR)],
        ]

    def get_layout(self):
        return self.layout
