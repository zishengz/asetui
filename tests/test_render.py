import numpy as np
from ase import Atoms

from asetui.render import (
    RENDER_BALLSTICK,
    RENDER_CPK,
    RenderOptions,
    _overlay_label,
    build_frame,
    prepare_atoms,
    render_atoms,
)


def test_render_atoms_includes_formula_and_border() -> None:
    atoms = Atoms("H2O", positions=[(0.0, 0.0, 0.0), (0.8, 0.1, 0.0), (-0.8, 0.1, 0.0)])
    rendered = render_atoms(atoms, RenderOptions(width=20, height=8, show_labels=True))

    assert rendered.startswith("asetui")
    assert "H2O" in rendered
    assert "[H]" in rendered or "(H)" in rendered or ".H." in rendered


def test_render_atoms_supports_labels() -> None:
    atoms = Atoms("NaCl", positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
    rendered = render_atoms(atoms, RenderOptions(width=16, height=6, show_labels=True))

    assert any(token in rendered for token in ("(Na)", "[Na]", ".Na.", "(Cl)", "[Cl]", ".Cl."))


def test_build_frame_respects_terminal_size_budget() -> None:
    atoms = Atoms("CO", positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=18, height=9))

    assert len(frame.canvas) == 5
    assert all(len(row) == 16 for row in frame.canvas)


def test_build_frame_reports_pan_and_updated_controls() -> None:
    atoms = Atoms("He", positions=[(0.0, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=18, height=9, offset_x=0.5, offset_y=-0.25))

    assert "pan=(+0.50,-0.25)" in frame.status
    assert "mode=wire" in frame.status
    assert "view=relative" in frame.status
    assert "depth=on" in frame.status
    assert "aspect=0.50" in frame.status
    assert "t translate" in frame.help_text
    assert "1/2/3 views" in frame.help_text
    assert "</> step" in frame.help_text
    assert "0 mode-cycle" in frame.help_text
    assert "c reset" in frame.help_text


def test_build_frame_marks_atom_colors() -> None:
    atoms = Atoms("NaCl", positions=[(0.0, 0.0, 0.0), (2.5, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=24, height=10, show_labels=True))

    assert any(11 in row for row in frame.colors)
    assert any(17 in row for row in frame.colors)
    assert any(
        token in row
        for row in frame.canvas
        for token in ("(Na)", "[Na]", ".Na.", "(Cl)", "[Cl]", ".Cl.")
    )


def test_build_frame_wire_mode_hides_labels_when_disabled() -> None:
    atoms = Atoms("NaCl", positions=[(0.0, 0.0, 0.0), (2.5, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=24, height=10, show_labels=False))
    flattened = "\n".join(frame.canvas)

    assert "Na" not in flattened
    assert "Cl" not in flattened


def test_build_frame_draws_bonds() -> None:
    atoms = Atoms("H2", positions=[(-0.37, 0.0, 0.0), (0.37, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=28, height=10))
    flattened = "\n".join(frame.canvas)

    assert any(char in flattened for char in ".:/\\'`")


def test_build_frame_accepts_orientation_matrix() -> None:
    atoms = Atoms("H2", positions=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    orientation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    frame = build_frame(atoms, RenderOptions(width=28, height=10, orientation=orientation))

    assert frame.canvas


def test_build_frame_accepts_prepared_atoms() -> None:
    atoms = Atoms("H2O", positions=[(0.0, 0.0, 0.0), (0.8, 0.1, 0.0), (-0.8, 0.1, 0.0)])
    prepared = prepare_atoms(atoms)
    frame = build_frame(prepared, RenderOptions(width=28, height=10, show_labels=True))

    assert frame.canvas
    assert "H2O" in frame.title


def test_build_frame_depth_wrappers_change_with_z() -> None:
    atoms = Atoms("H2", positions=[(-0.5, 0.0, -1.0), (0.5, 0.0, 1.0)])
    frame = build_frame(atoms, RenderOptions(width=32, height=10, show_labels=True))
    flattened = "\n".join(frame.canvas)

    assert ".H." in flattened
    assert "[H]" in flattened


def test_build_frame_ballstick_mode_uses_block_glyphs() -> None:
    atoms = Atoms("H2O", positions=[(-0.8, 0.0, -1.0), (0.0, 0.0, 0.0), (0.8, 0.0, 1.0)])
    frame = build_frame(atoms, RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK, show_labels=True))
    flattened = "\n".join(frame.canvas)

    assert "mode=ballstick" in frame.status
    assert any(char in flattened for char in "█▓▒░")
    assert "H" in flattened or "O" in flattened


def test_build_frame_ballstick_mode_shows_labels_when_enabled() -> None:
    atoms = Atoms("NaCl", positions=[(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)])
    frame = build_frame(
        atoms,
        RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK, show_labels=True),
    )
    flattened = "\n".join(frame.canvas)

    assert "Na" in flattened or "Cl" in flattened


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


def test_build_frame_ballstick_mode_uses_endpoint_colors_for_bond_halves() -> None:
    atoms = Atoms("HO", positions=[(-0.7, 0.0, 0.0), (0.7, 0.0, 0.0)])
    frame = build_frame(atoms, RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK))

    bond_cells = [color for row in frame.colors for color in row if color in (1, 8)]
    assert 1 in bond_cells
    assert 8 in bond_cells


def test_build_frame_ballstick_mode_hides_labels_when_disabled() -> None:
    atoms = Atoms("NaCl", positions=[(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)])
    frame = build_frame(
        atoms,
        RenderOptions(width=32, height=12, render_mode=RENDER_BALLSTICK, show_labels=False),
    )
    flattened = "\n".join(frame.canvas)

    assert "Na" not in flattened
    assert "Cl" not in flattened


def test_build_frame_cpk_mode_uses_large_atoms_without_bonds() -> None:
    atoms = Atoms("H2O", positions=[(-0.8, 0.0, -1.0), (0.0, 0.0, 0.0), (0.8, 0.0, 1.0)])
    frame = build_frame(atoms, RenderOptions(width=36, height=14, render_mode=RENDER_CPK, show_labels=True))
    flattened = "\n".join(frame.canvas)

    assert "mode=cpk" in frame.status
    assert any(char in flattened for char in "█▓▒")
    assert not any(char in flattened for char in ".:/\\'`,-")
    assert "H" in flattened or "O" in flattened


def test_build_frame_cpk_mode_shows_labels_when_enabled() -> None:
    atoms = Atoms("NaCl", positions=[(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)])
    frame = build_frame(
        atoms,
        RenderOptions(width=36, height=14, render_mode=RENDER_CPK, show_labels=True),
    )
    flattened = "\n".join(frame.canvas)

    assert "Na" in flattened or "Cl" in flattened


def test_overlay_label_writes_full_label_when_it_fits() -> None:
    canvas = [[" " for _ in range(6)]]
    colors = [[0 for _ in range(6)]]
    depths = [[-2.0 for _ in range(6)]]
    label_mask = [[False for _ in range(6)]]

    _overlay_label(canvas, colors, depths, label_mask, 0, 3, "Cl", 17, 0.0)

    assert "".join(canvas[0]) == "  Cl  "
    assert colors[0][2] == 17 and colors[0][3] == 17
    assert label_mask[0][2] and label_mask[0][3]


def test_build_frame_cpk_mode_hides_labels_when_disabled() -> None:
    atoms = Atoms("NaCl", positions=[(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)])
    frame = build_frame(
        atoms,
        RenderOptions(width=36, height=14, render_mode=RENDER_CPK, show_labels=False),
    )
    flattened = "\n".join(frame.canvas)

    assert "Na" not in flattened
    assert "Cl" not in flattened


def test_prepare_atoms_uses_ase_covalent_radii() -> None:
    atoms = Atoms("HO", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    prepared = prepare_atoms(atoms)

    assert prepared.radii[0] < prepared.radii[1]
