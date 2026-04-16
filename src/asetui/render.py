from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, sin
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
LABEL_DEPTH_TOLERANCE = 0.15  # 0 = only front-most atom; 1 = show all
CPK_RADIUS_SCALE = 0.8


@dataclass
class RenderOptions:
    width: int
    height: int
    label_mode: str = "symbol"
    yaw: float = 0.0
    pitch: float = 0.0
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    orientation: NDArray[np.float64] | None = None
    render_mode: str = RENDER_WIRE


@dataclass
class Frame:
    title: str
    canvas: list[str]
    colors: NDArray[np.int16] | list[list[int]]
    depths: NDArray[np.float32] | list[list[float]]
    label_mask: NDArray[np.bool_] | list[list[bool]]
    status: str
    help_text: str


@dataclass
class PreparedAtoms:
    formula: str
    centered: NDArray[np.float32]
    base_radius: float
    numbers: NDArray[np.int16]
    radii: NDArray[np.float32]
    symbols: tuple[str, ...]
    index_labels: tuple[str, ...]
    blank_labels: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_left: NDArray[np.intp]
    bond_right: NDArray[np.intp]
    energy: float | None
    fmax: float | None


@dataclass
class Scene:
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    zs: NDArray[np.float64]
    px: NDArray[np.intp]
    py: NDArray[np.intp]
    depths: NDArray[np.float32]
    plot_width: int
    plot_height: int
    scale: float
    x_center: float
    y_center: float
    y_scale: float
    min_z: float
    depth_scale: float
    z_order: NDArray[np.intp]
    wire_order: NDArray[np.intp]


def _euler_rotation(yaw: float, pitch: float) -> NDArray[np.float64]:
    cy, sy = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)

    book_flip_matrix = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=float,
    )
    return book_flip_matrix @ pitch_matrix


def _rotate_positions(
    positions: NDArray[np.float32],
    yaw: float,
    pitch: float,
    orientation: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    rotation = orientation if orientation is not None else _euler_rotation(yaw, pitch)
    return np.asarray(positions @ rotation, dtype=np.float64)


def _project_values(values: NDArray[np.float64], scale: float, center: float, limit: int) -> NDArray[np.intp]:
    projected = np.rint(values * scale + center)
    np.clip(projected, 0, limit, out=projected)
    return projected.astype(np.intp, copy=False)


def _bond_pairs(atoms: Atoms) -> list[tuple[int, int]]:
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)

    pairs: list[tuple[int, int]] = []
    for index in range(len(atoms)):
        neighbors, _ = neighbor_list.get_neighbors(index)
        for neighbor in neighbors:
            neighbor_index = int(neighbor)
            if neighbor_index > index:
                pairs.append((index, neighbor_index))
    return pairs


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_observables(atoms: Atoms) -> tuple[float | None, float | None]:
    energy = None
    forces = None

    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None)
    if isinstance(results, dict):
        energy = results.get("energy", results.get("free_energy"))
        forces = results.get("forces")

    if energy is None:
        info = getattr(atoms, "info", {})
        if isinstance(info, dict):
            energy = info.get("energy", info.get("Epot"))

    if forces is None:
        arrays = getattr(atoms, "arrays", {})
        if isinstance(arrays, dict):
            forces = arrays.get("forces")

    fmax = None
    if forces is not None:
        force_array = np.asarray(forces, dtype=np.float64)
        if force_array.ndim == 2 and force_array.size:
            fmax = float(np.linalg.norm(force_array, axis=1).max())

    return _float_or_none(energy), fmax


def prepare_atoms(atoms: Atoms) -> PreparedAtoms:
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    centered = positions - positions.mean(axis=0, keepdims=True, dtype=np.float32)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int16)
    radii = np.asarray(np.take(covalent_radii, numbers), dtype=np.float32)
    extents = np.linalg.norm(centered.astype(np.float64), axis=1) + radii
    base_radius = max(float(extents.max()) if extents.size else 0.0, 1e-9)
    bonds = tuple(_bond_pairs(atoms))
    if bonds:
        bond_array = np.asarray(bonds, dtype=np.intp)
        bond_left = bond_array[:, 0]
        bond_right = bond_array[:, 1]
    else:
        bond_left = np.empty(0, dtype=np.intp)
        bond_right = np.empty(0, dtype=np.intp)
    symbols = tuple(atoms.get_chemical_symbols())
    energy, fmax = _extract_observables(atoms)
    return PreparedAtoms(
        formula=atoms.get_chemical_formula(),
        centered=centered,
        base_radius=base_radius,
        numbers=numbers,
        radii=radii,
        symbols=symbols,
        index_labels=tuple(str(index) for index in range(len(numbers))),
        blank_labels=tuple(" " * len(symbol) for symbol in symbols),
        bonds=bonds,
        bond_left=bond_left,
        bond_right=bond_right,
        energy=energy,
        fmax=fmax,
    )


def _build_scene(prepared: PreparedAtoms, options: RenderOptions) -> Scene:
    rotated = _rotate_positions(prepared.centered, options.yaw, options.pitch, options.orientation)
    rotated[:, 0] += options.offset_x
    rotated[:, 1] += options.offset_y

    plot_width = max(4, options.width - 2)
    plot_height = max(3, options.height - 5)
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    zs = rotated[:, 2]
    radius = prepared.base_radius / max(options.zoom, 1e-6)
    span = max(radius * 2.4, 1e-9)

    physical_w = plot_width - 1
    physical_h = (plot_height - 1) / CELL_ASPECT_Y
    scale = min(physical_w, physical_h) / span
    x_center = (plot_width - 1) / 2.0
    y_center = (plot_height - 1) / 2.0
    y_scale = scale * CELL_ASPECT_Y

    px = _project_values(xs, scale, x_center, plot_width - 1)
    py = _project_values(-ys, y_scale, y_center, plot_height - 1)

    min_z = float(zs.min()) if zs.size else 0.0
    max_z = float(zs.max()) if zs.size else 0.0
    z_span = max(max_z - min_z, 1e-9)
    depth_scale = 2.0 / z_span
    depths = np.asarray((zs - min_z) * depth_scale - 1.0, dtype=np.float32)

    return Scene(
        xs=xs,
        ys=ys,
        zs=zs,
        px=px,
        py=py,
        depths=depths,
        plot_width=plot_width,
        plot_height=plot_height,
        scale=scale,
        x_center=x_center,
        y_center=y_center,
        y_scale=y_scale,
        min_z=min_z,
        depth_scale=depth_scale,
        z_order=np.argsort(zs, kind="stable"),
        wire_order=np.lexsort((prepared.numbers, zs)),
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
    if options.orientation is not None:
        d = options.orientation[:, 2]
        az = round(degrees(atan2(float(d[1]), float(d[0]))))
        el = round(degrees(asin(max(-1.0, min(1.0, float(d[2]))))))
        view_str = f"az={az:+d} el={el:+d}"
    else:
        view_str = f"yaw={options.yaw:.2f} pitch={options.pitch:.2f}"
    return (
        f"render={options.render_mode} {view_str} zoom={options.zoom:.2f} "
        f"pan=({options.offset_x:+.2f},{options.offset_y:+.2f}) labels={options.label_mode}"
    )


def _help_text() -> str:
    return "arrows move  t/r translate/rotate  1/2/3 views  =/- zoom  </> step  [/] frames  l labels  0 mode  c reset  h help  q quit"


def _make_buffers(
    plot_height: int,
    plot_width: int,
) -> tuple[list[list[str]], NDArray[np.int16], NDArray[np.float32], NDArray[np.bool_]]:
    canvas = [[" "] * plot_width for _ in range(plot_height)]
    colors = np.zeros((plot_height, plot_width), dtype=np.int16)
    depths = np.full((plot_height, plot_width), -2.0, dtype=np.float32)
    label_mask = np.zeros((plot_height, plot_width), dtype=bool)
    return canvas, colors, depths, label_mask


def _draw_wire_line(
    canvas: list[list[str]],
    colors: NDArray[np.int16],
    depths: NDArray[np.float32],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    depth: float,
) -> None:
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

    height = len(canvas)
    width = len(canvas[0]) if height else 0
    while True:
        if 0 <= y0 < height and 0 <= x0 < width:
            row_canvas = canvas[y0]
            if row_canvas[x0] == " ":
                row_canvas[x0] = glyph
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
    colors: NDArray[np.int16],
    depths: NDArray[np.float32],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    depth: float,
    color_code: int,
) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    glyph = "▓" if depth > 0.33 else "▒" if depth > -0.33 else "░"

    height = len(canvas)
    width = len(canvas[0]) if height else 0
    while True:
        if 0 <= y0 < height and 0 <= x0 < width:
            row_depths = depths[y0]
            if depth >= row_depths[x0]:
                canvas[y0][x0] = glyph
                colors[y0][x0] = color_code
                row_depths[x0] = depth
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _wire_atom_token(label: str, depth: float) -> str:
    if depth > 0.33:
        left, right = "[", "]"
    elif depth < -0.33:
        left, right = ".", "."
    else:
        left, right = "(", ")"
    return f"{left}{label}{right}"


def _label_texts(prepared: PreparedAtoms, label_mode: str) -> tuple[str, ...] | None:
    if label_mode == "off":
        return None
    if label_mode == "index":
        return prepared.index_labels
    return prepared.symbols


def _select_label_indices(
    scene: Scene,
    labels: tuple[str, ...],
    ordering: NDArray[np.intp],
    depths: NDArray[np.float32],
) -> NDArray[np.bool_]:
    claimed = np.zeros((scene.plot_height, scene.plot_width), dtype=bool)
    visible = np.zeros(len(labels), dtype=bool)
    for index in ordering[::-1]:
        label = labels[int(index)]
        cx = int(scene.px[index])
        cy = int(scene.py[index])
        atom_depth = float(scene.depths[index])
        if atom_depth < float(depths[cy, cx]) - LABEL_DEPTH_TOLERANCE:
            continue
        start_col = cx - len(label) // 2
        end_col = start_col + len(label)
        if start_col < 0 or end_col > scene.plot_width:
            continue
        if claimed[cy, start_col:end_col].any():
            continue
        claimed[cy, start_col:end_col] = True
        visible[int(index)] = True
    return visible


def _overlay_label(
    canvas: list[list[str]],
    colors: NDArray[np.int16],
    depths: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
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
    row_canvas = canvas[row]
    row_colors = colors[row]
    row_depths = depths[row]
    row_label_mask = label_mask[row]
    for offset, char in enumerate(label):
        target_col = start_col + offset
        row_canvas[target_col] = char
        row_colors[target_col] = color_code
        row_depths[target_col] = depth
        row_label_mask[target_col] = True


def _draw_disc(
    canvas: list[list[str]],
    colors: NDArray[np.int16],
    depths: NDArray[np.float32],
    cx: int,
    cy: int,
    radius: float,
    y_radius: float,
    depth: float,
    color_code: int,
    core: str,
    rim: str,
) -> None:
    row_start = max(0, int(cy - y_radius - 1))
    row_stop = min(len(canvas), int(cy + y_radius + 2))
    col_start = max(0, int(cx - radius - 1))
    col_stop = min(len(canvas[0]), int(cx + radius + 2))
    inv_radius = 1.0 / max(radius, 1e-9)
    inv_y_radius = 1.0 / max(y_radius, 1e-9)

    for row in range(row_start, row_stop):
        dy = ((row - cy) * inv_y_radius) ** 2
        if dy > 1.0:
            continue
        row_canvas = canvas[row]
        row_colors = colors[row]
        row_depths = depths[row]
        for col in range(col_start, col_stop):
            dx = (col - cx) * inv_radius
            distance = dx * dx + dy
            if distance > 1.0 or depth < row_depths[col]:
                continue
            row_canvas[col] = core if distance < 0.72 else rim
            row_colors[col] = color_code
            row_depths[col] = depth


def _frame_from_buffers(
    prepared: PreparedAtoms,
    options: RenderOptions,
    canvas: list[list[str]],
    colors: NDArray[np.int16],
    depths: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
) -> Frame:
    title = f"asetui  {prepared.formula}  atoms={len(prepared.numbers)}"
    if prepared.energy is not None:
        title += f"  Epot = {prepared.energy:.3f} eV"
    if prepared.fmax is not None:
        title += f"  Fmax = {prepared.fmax:.3f} eV/A"
    return Frame(
        title=title,
        canvas=["".join(row) for row in canvas],
        colors=colors,
        depths=depths,
        label_mask=label_mask,
        status=_base_status(options),
        help_text=_help_text(),
    )


def _build_wire_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas, colors, depths, label_mask = _make_buffers(scene.plot_height, scene.plot_width)
    for left, right in zip(prepared.bond_left, prepared.bond_right):
        bond_depth = float(((scene.zs[left] + scene.zs[right]) * 0.5 - scene.min_z) * scene.depth_scale - 1.0)
        _draw_wire_line(
            canvas,
            colors,
            depths,
            int(scene.px[left]),
            int(scene.py[left]),
            int(scene.px[right]),
            int(scene.py[right]),
            bond_depth,
        )

    if options.label_mode == "index":
        labels = prepared.index_labels
    elif options.label_mode == "off":
        labels = prepared.blank_labels
    else:
        labels = prepared.symbols

    for index in scene.wire_order:
        atom_index = int(index)
        x = int(scene.px[index])
        y = int(scene.py[index])
        token = _wire_atom_token(labels[atom_index], float(scene.depths[index]))
        start_col = x - len(token) // 2
        row_canvas = canvas[y]
        row_colors = colors[y]
        row_depths = depths[y]
        number = int(prepared.numbers[index])
        for offset, char in enumerate(token):
            col = start_col + offset
            if col < 0:
                continue
            if col >= scene.plot_width:
                break
            row_canvas[col] = char
            row_colors[col] = number
            row_depths[col] = scene.depths[index]

    return _frame_from_buffers(prepared, options, canvas, colors, depths, label_mask)


def _build_ballstick_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas, colors, depths, label_mask = _make_buffers(scene.plot_height, scene.plot_width)

    for left, right in zip(prepared.bond_left, prepared.bond_right):
        left_index = int(left)
        right_index = int(right)
        start_x = int(scene.px[left_index])
        start_y = int(scene.py[left_index])
        stop_x = int(scene.px[right_index])
        stop_y = int(scene.py[right_index])
        if max(abs(stop_x - start_x), abs(stop_y - start_y)) <= 1:
            continue
        midpoint_x = int(round((start_x + stop_x) * 0.5))
        midpoint_y = int(round((start_y + stop_y) * 0.5))
        left_depth = float(((scene.zs[left_index] * 0.75 + scene.zs[right_index] * 0.25) - scene.min_z) * scene.depth_scale - 1.0)
        right_depth = float(((scene.zs[left_index] * 0.25 + scene.zs[right_index] * 0.75) - scene.min_z) * scene.depth_scale - 1.0)
        if start_x != midpoint_x or start_y != midpoint_y:
            _draw_ballstick_line(
                canvas,
                colors,
                depths,
                start_x,
                start_y,
                midpoint_x,
                midpoint_y,
                left_depth,
                int(prepared.numbers[left_index]),
            )
        if midpoint_x != stop_x or midpoint_y != stop_y:
            _draw_ballstick_line(
                canvas,
                colors,
                depths,
                midpoint_x,
                midpoint_y,
                stop_x,
                stop_y,
                right_depth,
                int(prepared.numbers[right_index]),
            )

    for index in scene.z_order:
        atom_index = int(index)
        atom_depth = float(scene.depths[index])
        radius = 1.0 + float(prepared.radii[index]) * 1.8 + 0.45 * ((atom_depth + 1.0) * 0.5)
        fill = "█" if atom_depth > 0.33 else "▓" if atom_depth > -0.33 else "▒"
        rim = "▓" if atom_depth > -0.33 else "░"
        _draw_disc(
            canvas,
            colors,
            depths,
            int(scene.px[index]),
            int(scene.py[index]),
            radius,
            max(1.0, radius * CELL_ASPECT_Y),
            atom_depth,
            int(prepared.numbers[atom_index]),
            fill,
            rim,
        )

    label_texts = _label_texts(prepared, options.label_mode)
    if label_texts is not None:
        visible_labels = _select_label_indices(scene, label_texts, scene.z_order, depths)
        for index in scene.z_order[::-1]:
            atom_index = int(index)
            if not visible_labels[atom_index]:
                continue
            _overlay_label(
                canvas,
                colors,
                depths,
                label_mask,
                int(scene.py[index]),
                int(scene.px[index]),
                label_texts[atom_index],
                int(prepared.numbers[atom_index]),
                float(scene.depths[index]),
            )

    return _frame_from_buffers(prepared, options, canvas, colors, depths, label_mask)


def _build_cpk_frame(prepared: PreparedAtoms, options: RenderOptions, scene: Scene) -> Frame:
    canvas, colors, depths, label_mask = _make_buffers(scene.plot_height, scene.plot_width)

    for index in scene.z_order:
        atom_index = int(index)
        atom_depth = float(scene.depths[index])
        radius = max(1.0, CPK_RADIUS_SCALE * float(prepared.radii[index]) * scene.scale)
        core = "█" if atom_depth > 0.33 else "▓" if atom_depth > -0.33 else "▒"
        rim = "▓" if atom_depth > -0.2 else "▒"
        _draw_disc(
            canvas,
            colors,
            depths,
            int(scene.px[index]),
            int(scene.py[index]),
            radius,
            max(0.5, radius * CELL_ASPECT_Y),
            atom_depth,
            int(prepared.numbers[atom_index]),
            core,
            rim,
        )

    label_texts = _label_texts(prepared, options.label_mode)
    if label_texts is not None:
        visible_labels = _select_label_indices(scene, label_texts, scene.z_order, depths)
        for index in scene.z_order[::-1]:
            atom_index = int(index)
            if not visible_labels[atom_index]:
                continue
            _overlay_label(
                canvas,
                colors,
                depths,
                label_mask,
                int(scene.py[index]),
                int(scene.px[index]),
                label_texts[atom_index],
                int(prepared.numbers[atom_index]),
                float(scene.depths[index]),
            )

    return _frame_from_buffers(prepared, options, canvas, colors, depths, label_mask)


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
