from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs

if TYPE_CHECKING:
    from numpy.typing import NDArray

BOND_COLOR = -1
RENDER_WIRE = "wire"
RENDER_BALLSTICK = "ballstick"
RENDER_CPK = "cpk"
CELL_ASPECT_Y = 0.5


@dataclass(slots=True)
class RenderOptions:
    width: int
    height: int
    show_labels: bool = False
    yaw: float = 0.0
    pitch: float = 0.0
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    orientation: NDArray[np.float64] | None = None
    render_mode: str = RENDER_WIRE


@dataclass(slots=True)
class Frame:
    title: str
    canvas: list[str]
    colors: list[list[int]]
    depths: list[list[float]]
    label_mask: list[list[bool]]
    status: str
    help_text: str


@dataclass(slots=True)
class PreparedAtoms:
    formula: str
    centered: np.ndarray
    base_radius: float
    numbers: np.ndarray
    radii: np.ndarray
    symbols: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class Scene:
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    numbers: np.ndarray
    radii: np.ndarray
    symbols: list[str]
    plot_width: int
    plot_height: int
    project_x: object
    project_y: object
    min_z: float
    max_z: float


def _euler_rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)

    book_flip_matrix = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ]
    )
    return book_flip_matrix @ pitch_matrix


def _rotate_positions(
    positions: np.ndarray,
    yaw: float,
    pitch: float,
    orientation: NDArray[np.float64] | None,
) -> np.ndarray:
    rotation = orientation if orientation is not None else _euler_rotation(yaw, pitch)
    return positions @ rotation.T


def _projector(span: float, plot_width: int, plot_height: int):
    def project_x(value: float) -> int:
        normalized = ((value / span) + 0.5) * (plot_width - 1)
        return max(0, min(plot_width - 1, round(normalized)))

    def project_y(value: float) -> int:
        normalized = (0.5 - ((value * CELL_ASPECT_Y) / span)) * (plot_height - 1)
        return max(0, min(plot_height - 1, round(normalized)))

    return project_x, project_y


def _bond_pairs(atoms: Atoms) -> list[tuple[int, int]]:
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)

    pairs: set[tuple[int, int]] = set()
    for index in range(len(atoms)):
        neighbors, _ = neighbor_list.get_neighbors(index)
        for neighbor in neighbors:
            pair = tuple(sorted((int(index), int(neighbor))))
            if pair[0] != pair[1]:
                pairs.add(pair)
    return sorted(pairs)


def prepare_atoms(atoms: Atoms) -> PreparedAtoms:
    positions = atoms.get_positions()
    centered = positions - positions.mean(axis=0, keepdims=True)
    numbers = atoms.get_atomic_numbers()
    radii = np.array([float(covalent_radii[number]) for number in numbers], dtype=float)
    extents = np.linalg.norm(centered, axis=1) + radii
    base_radius = max(float(extents.max()), 1e-9)
    return PreparedAtoms(
        formula=atoms.get_chemical_formula(),
        centered=centered,
        base_radius=base_radius,
        numbers=numbers,
        radii=radii,
        symbols=tuple(atoms.get_chemical_symbols()),
        bonds=tuple(_bond_pairs(atoms)),
    )


def _normalize_depth(value: float, minimum: float, maximum: float) -> float:
    span = max(maximum - minimum, 1e-9)
    return ((value - minimum) / span) * 2.0 - 1.0


def _build_scene(prepared: PreparedAtoms, options: RenderOptions) -> Scene:
    rotated = _rotate_positions(prepared.centered, options.yaw, options.pitch, options.orientation)
    rotated[:, 0] += options.offset_x
    rotated[:, 1] += options.offset_y

    plot_width = max(4, options.width - 2)
    plot_height = max(3, options.height - 4)
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    zs = rotated[:, 2]
    radius = prepared.base_radius / max(options.zoom, 1e-6)
    span = max(radius * 2.4, 1e-9)
    project_x, project_y = _projector(span, plot_width, plot_height)

    return Scene(
        xs=xs,
        ys=ys,
        zs=zs,
        numbers=prepared.numbers,
        radii=prepared.radii,
        symbols=list(prepared.symbols),
        plot_width=plot_width,
        plot_height=plot_height,
        project_x=project_x,
        project_y=project_y,
        min_z=float(zs.min()),
        max_z=float(zs.max()),
    )


def _empty_frame(message: str) -> Frame:
    return Frame(
        title="asetui",
        canvas=[],
        colors=[],
        depths=[],
        label_mask=[],
        status=message,
        help_text="Resize the terminal to at least 12x6." if "small" in message.lower() else "q quit",
    )


def _base_status(options: RenderOptions) -> str:
    return (
        f"mode={options.render_mode} view=relative zoom={options.zoom:.2f} "
        f"pan=({options.offset_x:+.2f},{options.offset_y:+.2f}) "
        f"depth=on aspect={CELL_ASPECT_Y:.2f} labels={'on' if options.show_labels else 'off'}"
    )


def _help_text() -> str:
    return "t translate  r rotate  arrows move  1/2/3 views  </> step  =/- zoom  l labels  0 mode-cycle  c reset  q quit"


def _draw_wire_line(
    canvas: list[list[str]],
    colors: list[list[int]],
    depths: list[list[float]],
    start: tuple[int, int],
    stop: tuple[int, int],
    depth: float,
) -> None:
    x0, y0 = start
    x1, y1 = stop
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    if dx == 0 and dy == 0:
        return

    slope = dy / max(dx, 1)
    if depth > 0.33:
        horizontal, vertical, forward, backward = ("-", ":", "'", "`")
    elif depth < -0.33:
        horizontal, vertical, forward, backward = (".", ".", ",", ".")
    else:
        horizontal, vertical, forward, backward = (".", ":", "\\", "/")

    if dx == 0:
        glyph = vertical
    elif dy == 0:
        glyph = horizontal
    elif slope < 0.35:
        glyph = forward if (x1 - x0) * (y1 - y0) > 0 else backward
    elif slope < 0.85:
        glyph = "\\" if (x1 - x0) * (y1 - y0) > 0 else "/"
    elif slope < 1.6:
        glyph = "/" if (x1 - x0) * (y1 - y0) > 0 else "\\"
    else:
        glyph = vertical

    while True:
        if 0 <= y0 < len(canvas) and 0 <= x0 < len(canvas[0]) and canvas[y0][x0] == " ":
            canvas[y0][x0] = glyph
            colors[y0][x0] = BOND_COLOR
            depths[y0][x0] = depth
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_ballstick_line(
    canvas: list[list[str]],
    colors: list[list[int]],
    depths: list[list[float]],
    start: tuple[int, int],
    stop: tuple[int, int],
    depth: float,
    color_code: int,
) -> None:
    x0, y0 = start
    x1, y1 = stop
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    glyph = "▓" if depth > 0.33 else "▒" if depth > -0.33 else "░"

    while True:
        if 0 <= y0 < len(canvas) and 0 <= x0 < len(canvas[0]) and depth >= depths[y0][x0]:
            canvas[y0][x0] = glyph
            colors[y0][x0] = color_code
            depths[y0][x0] = depth
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _wire_atom_token(symbol: str, show_labels: bool, depth: float) -> str:
    core = symbol if show_labels else " " * len(symbol)
    if depth > 0.33:
        left, right = "[", "]"
    elif depth < -0.33:
        left, right = ".", "."
    else:
        left, right = "(", ")"
    return f"{left}{core}{right}"


def _overlay_label(
    canvas: list[list[str]],
    colors: list[list[int]],
    depths: list[list[float]],
    label_mask: list[list[bool]],
    row: int,
    col: int,
    label: str,
    color_code: int,
    depth: float,
) -> None:
    if row < 0 or row >= len(canvas):
        return
    start_col = col - len(label) // 2
    end_col = start_col + len(label)
    if start_col < 0 or end_col > len(canvas[0]):
        return
    for offset in range(len(label)):
        target_col = start_col + offset
        if depth < depths[row][target_col]:
            return
    for offset, char in enumerate(label):
        target_col = start_col + offset
        canvas[row][target_col] = char
        colors[row][target_col] = color_code
        depths[row][target_col] = depth
        label_mask[row][target_col] = True


def _build_wire_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas = [[" " for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    colors = [[0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    depths = [[-2.0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    label_mask = [[False for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]

    for left, right in prepared.bonds:
        start = (scene.project_x(float(scene.xs[left])), scene.project_y(float(scene.ys[left])))
        stop = (scene.project_x(float(scene.xs[right])), scene.project_y(float(scene.ys[right])))
        bond_depth = _normalize_depth(float((scene.zs[left] + scene.zs[right]) * 0.5), scene.min_z, scene.max_z)
        _draw_wire_line(canvas, colors, depths, start, stop, bond_depth)

    ordering = sorted(
        range(len(prepared.numbers)),
        key=lambda index: (float(scene.zs[index]), int(scene.numbers[index])),
    )
    for index in ordering:
        x = scene.project_x(float(scene.xs[index]))
        y = scene.project_y(float(scene.ys[index]))
        atom_depth = _normalize_depth(float(scene.zs[index]), scene.min_z, scene.max_z)
        token = _wire_atom_token(scene.symbols[index], options.show_labels, atom_depth)
        start_col = x - len(token) // 2
        for offset, char in enumerate(token):
            col = start_col + offset
            if col < 0:
                continue
            if col >= scene.plot_width:
                break
            canvas[y][col] = char
            colors[y][col] = int(scene.numbers[index])
            depths[y][col] = atom_depth

    return Frame(
        title=f"asetui  {prepared.formula}  atoms={len(prepared.numbers)}",
        canvas=["".join(row) for row in canvas],
        colors=colors,
        depths=depths,
        label_mask=label_mask,
        status=_base_status(options),
        help_text=_help_text(),
    )


def _build_ballstick_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas = [[" " for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    colors = [[0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    depths = [[-2.0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    label_mask = [[False for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]

    for left, right in prepared.bonds:
        start = (scene.project_x(float(scene.xs[left])), scene.project_y(float(scene.ys[left])))
        stop = (scene.project_x(float(scene.xs[right])), scene.project_y(float(scene.ys[right])))
        mid_x = (float(scene.xs[left]) + float(scene.xs[right])) * 0.5
        mid_y = (float(scene.ys[left]) + float(scene.ys[right])) * 0.5
        mid_z = (float(scene.zs[left]) + float(scene.zs[right])) * 0.5
        midpoint = (scene.project_x(mid_x), scene.project_y(mid_y))
        left_depth = _normalize_depth((float(scene.zs[left]) + mid_z) * 0.5, scene.min_z, scene.max_z)
        right_depth = _normalize_depth((float(scene.zs[right]) + mid_z) * 0.5, scene.min_z, scene.max_z)
        _draw_ballstick_line(canvas, colors, depths, start, midpoint, left_depth, int(scene.numbers[left]))
        _draw_ballstick_line(canvas, colors, depths, midpoint, stop, right_depth, int(scene.numbers[right]))

    ordering = sorted(range(len(prepared.numbers)), key=lambda index: float(scene.zs[index]))
    for index in ordering:
        cx = scene.project_x(float(scene.xs[index]))
        cy = scene.project_y(float(scene.ys[index]))
        atom_depth = _normalize_depth(float(scene.zs[index]), scene.min_z, scene.max_z)
        cov_radius = float(scene.radii[index])
        radius = 1.0 + cov_radius * 1.8 + 0.45 * ((atom_depth + 1.0) * 0.5)
        fill = "█" if atom_depth > 0.33 else "▓" if atom_depth > -0.33 else "▒"
        y_radius = max(1.0, radius * CELL_ASPECT_Y)
        for row in range(max(0, int(cy - y_radius - 1)), min(scene.plot_height, int(cy + y_radius + 2))):
            for col in range(max(0, int(cx - radius - 1)), min(scene.plot_width, int(cx + radius + 2))):
                dx = (col - cx) / max(radius, 1e-9)
                dy = (row - cy) / max(y_radius, 1e-9)
                distance = dx * dx + dy * dy
                if distance > 1.0:
                    continue
                if atom_depth < depths[row][col]:
                    continue
                canvas[row][col] = fill if distance < 0.72 else "▓" if atom_depth > -0.33 else "░"
                colors[row][col] = int(scene.numbers[index])
                depths[row][col] = atom_depth

        if options.show_labels:
            _overlay_label(
                canvas,
                colors,
                depths,
                label_mask,
                cy,
                cx,
                scene.symbols[index],
                int(scene.numbers[index]),
                atom_depth,
            )

    return Frame(
        title=f"asetui  {prepared.formula}  atoms={len(prepared.numbers)}",
        canvas=["".join(row) for row in canvas],
        colors=colors,
        depths=depths,
        label_mask=label_mask,
        status=_base_status(options),
        help_text=_help_text(),
    )


def _build_cpk_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas = [[" " for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    colors = [[0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    depths = [[-2.0 for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]
    label_mask = [[False for _ in range(scene.plot_width)] for _ in range(scene.plot_height)]

    ordering = sorted(range(len(prepared.numbers)), key=lambda index: float(scene.zs[index]))
    for index in ordering:
        cx = scene.project_x(float(scene.xs[index]))
        cy = scene.project_y(float(scene.ys[index]))
        atom_depth = _normalize_depth(float(scene.zs[index]), scene.min_z, scene.max_z)
        cov_radius = float(scene.radii[index])
        radius = (1.8 + cov_radius * 3.4 + 0.65 * ((atom_depth + 1.0) * 0.5)) * options.zoom
        y_radius = max(1.8, radius * CELL_ASPECT_Y)
        core = "█" if atom_depth > 0.33 else "▓" if atom_depth > -0.33 else "▒"
        rim = "▓" if atom_depth > -0.2 else "▒"
        for row in range(max(0, int(cy - y_radius - 1)), min(scene.plot_height, int(cy + y_radius + 2))):
            for col in range(max(0, int(cx - radius - 1)), min(scene.plot_width, int(cx + radius + 2))):
                dx = (col - cx) / max(radius, 1e-9)
                dy = (row - cy) / max(y_radius, 1e-9)
                distance = dx * dx + dy * dy
                if distance > 1.0 or atom_depth < depths[row][col]:
                    continue
                canvas[row][col] = core if distance < 0.72 else rim
                colors[row][col] = int(scene.numbers[index])
                depths[row][col] = atom_depth

        if options.show_labels:
            _overlay_label(
                canvas,
                colors,
                depths,
                label_mask,
                cy,
                cx,
                scene.symbols[index],
                int(scene.numbers[index]),
                atom_depth,
            )

    return Frame(
        title=f"asetui  {prepared.formula}  atoms={len(prepared.numbers)}",
        canvas=["".join(row) for row in canvas],
        colors=colors,
        depths=depths,
        label_mask=label_mask,
        status=_base_status(options),
        help_text=_help_text(),
    )


def build_frame(atoms: Atoms | PreparedAtoms, options: RenderOptions) -> Frame:
    if options.width < 12 or options.height < 6:
        return _empty_frame("Terminal too small")
    prepared = atoms if isinstance(atoms, PreparedAtoms) else prepare_atoms(atoms)
    if len(prepared.numbers) == 0:
        return _empty_frame("Empty structure")

    scene = _build_scene(prepared, options)
    if options.render_mode == RENDER_BALLSTICK:
        return _build_ballstick_frame(prepared, options, scene)
    if options.render_mode == RENDER_CPK:
        return _build_cpk_frame(prepared, options, scene)
    return _build_wire_frame(prepared, options, scene)


def render_atoms(atoms: Atoms | PreparedAtoms, options: RenderOptions) -> str:
    frame = build_frame(atoms, options)
    if not frame.canvas:
        return "\n".join([frame.title, frame.status, frame.help_text])

    border = "+" + "-" * len(frame.canvas[0]) + "+"
    lines = [frame.title, border]
    lines.extend("|" + row + "|" for row in frame.canvas)
    lines.append(border)
    lines.append(frame.status)
    lines.append(frame.help_text)
    return "\n".join(lines)
