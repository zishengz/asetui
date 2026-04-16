import numpy as np

from asetui.app import (
    AppState,
    RENDER_BALLSTICK,
    RENDER_CPK,
    RENDER_WIRE,
    _prepared_frame,
    _adjust_step_multiplier,
    _help_overlay_lines,
    _next_render_mode,
    _preset_orientation,
    _rotation_step,
    _zoom_factor,
)


def test_preset_orientation_returns_right_handed_orthonormal_views() -> None:
    expected_views = {
        "1": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        "2": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "3": np.eye(3),
    }

    for key, expected in expected_views.items():
        view = _preset_orientation(key)
        assert np.allclose(view, expected)
        assert np.allclose(view.T @ view, np.eye(3))
        assert np.isclose(np.linalg.det(view), 1.0)


def test_step_multiplier_scales_rotation_and_zoom() -> None:
    state = AppState(step_multiplier=1.5)

    assert _adjust_step_multiplier(1.0, 1) == 1.25
    assert _adjust_step_multiplier(1.0, -1) == 0.75
    assert _adjust_step_multiplier(0.25, -1) == 0.25
    assert np.isclose(_rotation_step(state), 0.3927, atol=1e-4)
    assert np.isclose(_zoom_factor(state), 1.225)


def test_render_mode_cycle_uses_all_supported_modes() -> None:
    assert _next_render_mode(RENDER_WIRE) == RENDER_BALLSTICK
    assert _next_render_mode(RENDER_BALLSTICK) == RENDER_CPK
    assert _next_render_mode(RENDER_CPK) == RENDER_WIRE


def test_help_overlay_lines_add_frame_controls_only_for_trajectories() -> None:
    single = _help_overlay_lines(1)
    multi = _help_overlay_lines(3)

    assert "[/]" not in " ".join(single)
    assert any("[/]" in line for line in multi)
    assert any("q: quit" in line for line in multi)


def test_prepared_frame_is_lazy_and_cached(monkeypatch) -> None:
    frames = [
        object(),
        object(),
        object(),
    ]
    cache = [None, None, None]
    calls: list[int] = []

    def fake_prepare(frame: object) -> str:
        calls.append(frames.index(frame))
        return f"prepared-{frames.index(frame)}"

    monkeypatch.setattr("asetui.app.prepare_atoms", fake_prepare)

    assert _prepared_frame(frames, cache, 1) == "prepared-1"
    assert _prepared_frame(frames, cache, 1) == "prepared-1"
    assert _prepared_frame(frames, cache, 2) == "prepared-2"
    assert calls == [1, 2]
