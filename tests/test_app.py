import numpy as np
from ase import Atoms

from asetui.app import _adjust_step_multiplier, _preset_orientation, _rotation_step, _zoom_factor, AppState


def test_preset_orientation_matches_ase_gui_axis_choice() -> None:
    atoms = Atoms("H", positions=[(0.0, 0.0, 0.0)], cell=[2.0, 3.0, 4.0], pbc=True)

    view1 = _preset_orientation(atoms, "1")
    view2 = _preset_orientation(atoms, "2")
    view3 = _preset_orientation(atoms, "3")

    assert np.allclose(view1[:, 0], [0.0, 1.0, 0.0])
    assert np.allclose(view1[:, 1], [0.0, 0.0, 1.0])
    assert np.allclose(view2[:, 0], [0.0, 0.0, 1.0])
    assert np.allclose(view2[:, 1], [1.0, 0.0, 0.0])
    assert np.allclose(view3[:, 0], [1.0, 0.0, 0.0])
    assert np.allclose(view3[:, 1], [0.0, 1.0, 0.0])


def test_step_multiplier_changes_in_quarter_base_increments() -> None:
    assert _adjust_step_multiplier(1.0, 1) == 1.25
    assert _adjust_step_multiplier(1.0, -1) == 0.75
    assert _adjust_step_multiplier(0.25, -1) == 0.25


def test_step_multiplier_scales_rotation_and_zoom() -> None:
    state = AppState(step_multiplier=1.5)

    assert np.isclose(_rotation_step(state), 0.18)
    assert np.isclose(_zoom_factor(state), 1.225)
