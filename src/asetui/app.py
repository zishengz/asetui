from __future__ import annotations

import curses
from dataclasses import dataclass, field

import numpy as np

from ase.atoms import Atoms
from ase.data.colors import jmol_colors

from asetui.render import (
    BOND_COLOR,
    RENDER_BALLSTICK,
    RENDER_CPK,
    RENDER_WIRE,
    PreparedAtoms,
    RenderOptions,
    build_frame,
    prepare_atoms,
)

MODE_SEQUENCE = (RENDER_WIRE, RENDER_BALLSTICK, RENDER_CPK)
BASE_ROTATION_STEP = 0.2618  # 15 degrees
BASE_TRANSLATION_STEP = 0.12
BASE_ZOOM_DELTA = 0.15
STEP_INCREMENT = 0.25
MIN_STEP_MULTIPLIER = 0.25


@dataclass
class AppState:
    label_mode: str = "symbol"
    zoom: float = 1.5
    offset_x: float = 0.0
    offset_y: float = 0.0
    step_multiplier: float = 1.0
    mode: str = "rotate"
    render_mode: str = RENDER_WIRE
    show_help: bool = True
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))


@dataclass
class ScreenCache:
    width: int = -1
    height: int = -1
    rows: dict[int, tuple[str, tuple[int, ...]]] = field(default_factory=dict)


def _rotation_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def _rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _apply_view_rotation(state: AppState, rotation: np.ndarray) -> None:
    state.orientation = rotation @ state.orientation


def _next_render_mode(current: str) -> str:
    try:
        index = MODE_SEQUENCE.index(current)
    except ValueError:
        return MODE_SEQUENCE[0]
    return MODE_SEQUENCE[(index + 1) % len(MODE_SEQUENCE)]


def _adjust_step_multiplier(current: float, direction: int) -> float:
    updated = current + direction * STEP_INCREMENT
    return max(MIN_STEP_MULTIPLIER, updated)


def _rotation_step(state: AppState) -> float:
    return BASE_ROTATION_STEP * state.step_multiplier


def _translation_step(state: AppState) -> float:
    return BASE_TRANSLATION_STEP * state.step_multiplier


def _zoom_factor(state: AppState) -> float:
    return 1.0 + BASE_ZOOM_DELTA * state.step_multiplier


def _preset_orientation(key: str) -> np.ndarray:
    if key == "1":
        # Front: look along +y, up = +z
        x3 = np.array([0.0, 1.0, 0.0])
        x2 = np.array([0.0, 0.0, 1.0])
    elif key == "2":
        # Side: look along +x, up = +y (CCW 90° from z-up)
        x3 = np.array([1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0])
    elif key == "3":
        # Top: look along +z, up = +y
        x3 = np.array([0.0, 0.0, 1.0])
        x2 = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"Unsupported preset key: {key}")
    x1 = np.cross(x2, x3)
    return np.array([x1, x2, x3], dtype=float).T


def _rgb_to_xterm_index(rgb: tuple[float, float, float]) -> int:
    steps = [0, 95, 135, 175, 215, 255]
    channels = [max(0, min(255, round(value * 255))) for value in rgb]
    cube = [min(range(6), key=lambda idx: abs(steps[idx] - channel)) for channel in channels]
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]


def _nearest_basic_color(rgb: tuple[float, float, float]) -> int:
    basic = {
        curses.COLOR_BLACK: (0, 0, 0),
        curses.COLOR_RED: (205, 49, 49),
        curses.COLOR_GREEN: (13, 188, 121),
        curses.COLOR_YELLOW: (229, 229, 16),
        curses.COLOR_BLUE: (36, 114, 200),
        curses.COLOR_MAGENTA: (188, 63, 188),
        curses.COLOR_CYAN: (17, 168, 205),
        curses.COLOR_WHITE: (229, 229, 229),
    }
    channels = tuple(max(0, min(255, round(value * 255))) for value in rgb)
    return min(
        basic,
        key=lambda color_id: sum((basic[color_id][i] - channels[i]) ** 2 for i in range(3)),
    )


class ColorManager:
    def __init__(self) -> None:
        self.enabled = False
        self.background = -1
        self.pairs: dict[tuple[int, int], int] = {}

    def setup(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            self.background = curses.COLOR_BLACK
        self.enabled = True

    def attr_for(self, color_code: int, depth: float, is_label: bool = False) -> int:
        emphasis = curses.A_BOLD if depth > 0.33 else curses.A_DIM if depth < -0.33 else curses.A_NORMAL
        if color_code == BOND_COLOR:
            return emphasis
        if color_code <= 0 or not self.enabled:
            return emphasis

        rgb = tuple(float(value) for value in jmol_colors[color_code])
        if is_label:
            scale = 0.55 if depth < -0.33 else 1.2 if depth > 0.33 else 1.0
            bg_rgb = tuple(min(1.0, c * scale) for c in rgb)
            pair_bg = _rgb_to_xterm_index(bg_rgb) if curses.COLORS >= 256 else _nearest_basic_color(bg_rgb)
            lum = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
            pair_fg = curses.COLOR_BLACK if lum > 0.35 else curses.COLOR_WHITE
            emphasis = curses.A_NORMAL
        else:
            element_color = _rgb_to_xterm_index(rgb) if curses.COLORS >= 256 else _nearest_basic_color(rgb)
            pair_fg = element_color
            pair_bg = self.background
        pair_key = (pair_fg, pair_bg)
        pair_id = self.pairs.get(pair_key)
        if pair_id is None:
            pair_id = len(self.pairs) + 1
            if pair_id >= curses.COLOR_PAIRS:
                return emphasis
            try:
                curses.init_pair(pair_id, pair_fg, pair_bg)
            except curses.error:
                return emphasis
            self.pairs[pair_key] = pair_id
        return curses.color_pair(pair_id) | emphasis


def _render_runs(
    line: str,
    color_row: list[int],
    depth_row: list[float],
    label_row: list[bool],
    width_limit: int,
    colors: ColorManager,
) -> tuple[str, tuple[int, ...]]:
    if width_limit <= 0:
        return "", ()
    text = line[:width_limit]
    attrs = tuple(
        colors.attr_for(color_row[index], depth_row[index], label_row[index])
        for index in range(len(text))
    )
    return text, attrs


def _draw_row_runs(stdscr: curses.window, row: int, text: str, attrs: tuple[int, ...]) -> None:
    if not text:
        return
    start = 0
    while start < len(text):
        attr = attrs[start]
        end = start + 1
        while end < len(text) and attrs[end] == attr:
            end += 1
        stdscr.addstr(row, start, text[start:end], attr)
        start = end


def _draw_screen(
    stdscr: curses.window,
    atoms: PreparedAtoms,
    state: AppState,
    colors: ColorManager,
    cache: ScreenCache,
) -> None:
    height, width = stdscr.getmaxyx()
    frame = build_frame(
        atoms,
        RenderOptions(
            width=width,
            height=height,
            label_mode=state.label_mode,
            zoom=state.zoom,
            offset_x=state.offset_x,
            offset_y=state.offset_y,
            orientation=state.orientation,
            render_mode=state.render_mode,
        ),
    )

    if height <= 0 or width <= 0:
        return
    if cache.width != width or cache.height != height:
        stdscr.erase()
        cache.width = width
        cache.height = height
        cache.rows.clear()

    try:
        title_text = frame.title[: max(width - 1, 0)]
        title_state = (title_text, ())
        if cache.rows.get(0) != title_state:
            stdscr.move(0, 0)
            stdscr.clrtoeol()
            stdscr.addstr(0, 0, title_text)
            cache.rows[0] = title_state
        canvas_top = 1
        for row_index, line in enumerate(frame.canvas, start=canvas_top):
            source_row = row_index - canvas_top
            if row_index >= height - 3:
                break
            row_state = _render_runs(
                line,
                frame.colors[source_row],
                frame.depths[source_row],
                frame.label_mask[source_row],
                max(width - 1, 0),
                colors,
            )
            if cache.rows.get(row_index) != row_state:
                stdscr.move(row_index, 0)
                stdscr.clrtoeol()
                _draw_row_runs(stdscr, row_index, row_state[0], row_state[1])
                cache.rows[row_index] = row_state
        if height >= 3:
            status1_text = frame.status[: max(width - 1, 0)]
            status1_state = (status1_text, ())
            if cache.rows.get(height - 3) != status1_state:
                stdscr.move(height - 3, 0)
                stdscr.clrtoeol()
                stdscr.addstr(height - 3, 0, status1_text)
                cache.rows[height - 3] = status1_state
        if height >= 2:
            status2 = f"mode={state.mode} step={state.step_multiplier:.2f}x"
            status2_text = status2[: max(width - 1, 0)]
            status2_state = (status2_text, ())
            if cache.rows.get(height - 2) != status2_state:
                stdscr.move(height - 2, 0)
                stdscr.clrtoeol()
                stdscr.addstr(height - 2, 0, status2_text)
                cache.rows[height - 2] = status2_state
        if height >= 1:
            help_text = (frame.help_text if state.show_help else "h show keys")[: max(width - 1, 0)]
            help_state = (help_text, ())
            if cache.rows.get(height - 1) != help_state:
                stdscr.move(height - 1, 0)
                stdscr.clrtoeol()
                stdscr.addstr(height - 1, 0, help_text)
                cache.rows[height - 1] = help_state
        max_used_row = min(height - 1, len(frame.canvas) + 2)
        stale_rows = [row for row in cache.rows if row > max_used_row or row >= height]
        for row in stale_rows:
            stdscr.move(row, 0)
            stdscr.clrtoeol()
            del cache.rows[row]
    except curses.error:
        pass

    stdscr.refresh()


def run_app(atoms: Atoms, initial_state: AppState | None = None) -> int:
    def _main(stdscr: curses.window) -> int:
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(100)
        colors = ColorManager()
        colors.setup()
        prepared = prepare_atoms(atoms)
        cache = ScreenCache()

        state = initial_state or AppState()
        _draw_screen(stdscr, prepared, state, colors, cache)

        while True:
            key = stdscr.getch()
            if key == -1 or key == curses.KEY_RESIZE:
                _draw_screen(stdscr, prepared, state, colors, cache)
                continue
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("t"), ord("T")):
                state.mode = "translate"
            elif key in (ord("r"), ord("R")):
                state.mode = "rotate"
            elif key == curses.KEY_LEFT:
                if state.mode == "translate":
                    state.offset_x -= _translation_step(state) / max(state.zoom, 1e-6)
                else:
                    _apply_view_rotation(state, _rotation_y(-_rotation_step(state)))
            elif key == curses.KEY_RIGHT:
                if state.mode == "translate":
                    state.offset_x += _translation_step(state) / max(state.zoom, 1e-6)
                else:
                    _apply_view_rotation(state, _rotation_y(_rotation_step(state)))
            elif key == curses.KEY_UP:
                if state.mode == "translate":
                    state.offset_y += _translation_step(state) / max(state.zoom, 1e-6)
                else:
                    _apply_view_rotation(state, _rotation_x(-_rotation_step(state)))
            elif key == curses.KEY_DOWN:
                if state.mode == "translate":
                    state.offset_y -= _translation_step(state) / max(state.zoom, 1e-6)
                else:
                    _apply_view_rotation(state, _rotation_x(_rotation_step(state)))
            elif key == ord("="):
                state.zoom *= _zoom_factor(state)
            elif key == ord("-"):
                state.zoom = max(0.2, state.zoom / _zoom_factor(state))
            elif key == ord(">"):
                state.step_multiplier = _adjust_step_multiplier(state.step_multiplier, 1)
            elif key == ord("<"):
                state.step_multiplier = _adjust_step_multiplier(state.step_multiplier, -1)
            elif key in (ord("l"), ord("L")):
                state.label_mode = {"symbol": "index", "index": "off", "off": "symbol"}[state.label_mode]
            elif key in (ord("h"), ord("H")):
                state.show_help = not state.show_help
            elif key in (ord("1"), ord("2"), ord("3")):
                state.orientation = _preset_orientation(chr(key))
            elif key == ord("0"):
                state.render_mode = _next_render_mode(state.render_mode)
            elif key in (ord("c"), ord("C")):
                state = AppState()
            _draw_screen(stdscr, prepared, state, colors, cache)

    return curses.wrapper(_main)
