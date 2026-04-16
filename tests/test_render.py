import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from asetui.render import (
    RENDER_BALLSTICK,
    RENDER_CPK,
    RenderOptions,
    _overlay_label,
    build_frame,
    prepare_atoms,
    render_atoms,
)


def test_prepare_atoms_uses_compact_arrays_and_cached_observables() -> None:
    atoms = Atoms("H2", positions=[(-0.37, 0.0, 0.0), (0.37, 0.0, 0.0)])
    forces = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-12.3456, forces=forces)

    prepared = prepare_atoms(atoms)

    assert prepared.centered.dtype == np.float32
    assert prepared.numbers.dtype == np.int16
    assert prepared.radii.dtype == np.float32
    assert prepared.index_labels == ("0", "1")
    assert prepared.blank_labels == (" ", " ")
    assert np.array_equal(prepared.bond_left, np.array([0]))
    assert np.array_equal(prepared.bond_right, np.array([1]))
    assert prepared.energy == -12.3456
    assert prepared.fmax == 5.0


def test_render_atoms_includes_border_formula_and_observables_when_available() -> None:
    atoms = Atoms("H2O", positions=[(0.0, 0.0, 0.0), (0.8, 0.1, 0.0), (-0.8, 0.1, 0.0)])
    atoms.calc = SinglePointCalculator(atoms, energy=-7.0, forces=np.zeros((3, 3)))

    rendered = render_atoms(atoms, RenderOptions(width=28, height=10))

    assert rendered.startswith("asetui  H2O")
    assert "+--------------------------+" in rendered
    assert "Epot = -7.000 eV" in rendered
    assert "Fmax = 0.000 eV/A" in rendered


def test_build_frame_wire_mode_returns_numpy_backing_arrays() -> None:
    atoms = Atoms("NaCl", positions=[(0.0, 0.0, 0.0), (2.5, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=28, height=10))
    flattened = "\n".join(frame.canvas)

    assert frame.colors.shape == (5, 26)
    assert frame.depths.shape == (5, 26)
    assert frame.label_mask.shape == (5, 26)
    assert frame.colors.dtype == np.int16
    assert frame.depths.dtype == np.float32
    assert frame.label_mask.dtype == np.bool_
    assert "Na" in flattened or "Cl" in flattened
    assert any(char in flattened for char in ".:/\\'`,-")


def test_build_frame_accepts_prepared_atoms_and_orientation_matrix() -> None:
    atoms = Atoms("H2", positions=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    prepared = prepare_atoms(atoms)
    orientation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    frame = build_frame(prepared, RenderOptions(width=28, height=10, orientation=orientation))

    assert frame.canvas
    assert "H2" in frame.title
    assert "az=" in frame.status


def test_wire_and_spacefill_modes_respect_label_modes() -> None:
    atoms = Atoms("NaCl", positions=[(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)])

    wire_index = build_frame(atoms, RenderOptions(width=32, height=12, label_mode="index"))
    ball_off = build_frame(atoms, RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK, label_mode="off"))
    cpk_symbol = build_frame(atoms, RenderOptions(width=36, height=14, render_mode=RENDER_CPK, label_mode="symbol"))

    assert "0" in "\n".join(wire_index.canvas) or "1" in "\n".join(wire_index.canvas)
    assert "Na" not in "\n".join(ball_off.canvas)
    assert "Cl" not in "\n".join(ball_off.canvas)
    assert "Na" in "\n".join(cpk_symbol.canvas) or "Cl" in "\n".join(cpk_symbol.canvas)


def test_ballstick_mode_uses_block_glyphs_and_endpoint_colors() -> None:
    atoms = Atoms("HO", positions=[(-0.7, 0.0, 0.0), (0.7, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK))
    flattened = "\n".join(frame.canvas)
    bond_cells = [color for row in frame.colors for color in row if color in (1, 8)]

    assert "render=ballstick" in frame.status
    assert any(char in flattened for char in "█▓▒░")
    assert 1 in bond_cells
    assert 8 in bond_cells


def test_cpk_mode_uses_large_atoms_without_wire_bonds() -> None:
    atoms = Atoms("H2O", positions=[(-0.8, 0.0, -1.0), (0.0, 0.0, 0.0), (0.8, 0.0, 1.0)])
    frame = build_frame(atoms, RenderOptions(width=36, height=14, render_mode=RENDER_CPK, label_mode="symbol"))
    flattened = "\n".join(frame.canvas)

    assert "render=cpk" in frame.status
    assert any(char in flattened for char in "█▓▒")
    assert not any(char in flattened for char in ".:/\\'`,-")


def test_overlay_label_is_all_or_nothing_when_clipped() -> None:
    canvas = [[" " for _ in range(6)]]
    colors = [[0 for _ in range(6)]]
    depths = [[-2.0 for _ in range(6)]]
    label_mask = [[False for _ in range(6)]]

    _overlay_label(canvas, colors, depths, label_mask, 0, 0, "Na", 11, 0.0)

    assert "".join(canvas[0]).strip() == ""
    assert all(color == 0 for color in colors[0])
    assert all(depth == -2.0 for depth in depths[0])
    assert not any(label_mask[0])


def test_overlay_label_writes_full_label_when_it_fits() -> None:
    canvas = [[" " for _ in range(6)]]
    colors = [[0 for _ in range(6)]]
    depths = [[-2.0 for _ in range(6)]]
    label_mask = [[False for _ in range(6)]]

    _overlay_label(canvas, colors, depths, label_mask, 0, 3, "Cl", 17, 0.0)

    assert "".join(canvas[0]) == "  Cl  "
    assert colors[0][2] == 17 and colors[0][3] == 17
    assert label_mask[0][2] and label_mask[0][3]


def test_help_text_mentions_transient_help_overlay() -> None:
    atoms = Atoms("He", positions=[(0.0, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=24, height=10))

    assert "h help" in frame.help_text
