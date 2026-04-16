# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .          # install in editable mode
pytest                    # run all tests
pytest tests/test_render.py::test_name  # run a single test
atui examples/MeOH.xyz    # run the app
```

## Architecture

The pipeline has three stages:

**1. Preparation** (`render.py: prepare_atoms`)  
Reads ASE `Atoms`, centers positions, computes covalent radii and `base_radius` (max atom extent from center). Returns `PreparedAtoms`, which is cached across frames.

**2. Scene building** (`render.py: _build_scene`)  
Applies rotation (via `AppState.orientation`, a 3×3 matrix) and pan offsets, then constructs a `Scene` with projected coordinates. `_projector` computes a uniform physical-space scale (`min(physical_w, physical_h) / span`) so both axes share the same col/Å scale — this is critical for correct aspect ratio. `Scene.scale` (cols/Å) is used by CPK to size atoms relative to structure density.

**3. Frame rendering** (`render.py: _build_wire_frame / _build_ballstick_frame / _build_cpk_frame`)  
Each builder fills a character canvas plus parallel `colors` (atomic number), `depths` (−1..+1), and `label_mask` arrays. Rendering is back-to-front (ascending z). Labels use a two-pass approach: blobs are drawn first to populate the depth buffer, then `_select_label_indices` filters labels by depth visibility (`LABEL_DEPTH_TOLERANCE`) and greedy front-to-back cell claiming to prevent overlap.

**Display** (`app.py: _draw_screen → _render_runs`)  
`ColorManager.attr_for` maps atomic number + depth to curses color pairs. `is_label=True` cells get depth-scaled element color as background with black/white foreground for contrast. Uses 256-color xterm palette when available.

## Key constants (render.py)

| Constant | Purpose |
|---|---|
| `CELL_ASPECT_Y = 0.5` | Character cell height/width ratio for projection |
| `LABEL_DEPTH_TOLERANCE` | 0=front only, 1=show all; controls which back atoms get labels |

## State model

`AppState` (app.py) holds all interactive state: `orientation` (3×3 rotation matrix, updated incrementally via `_apply_view_rotation`), `zoom`, `offset_x/y`, `label_mode` (`"symbol"/"index"/"off"`), `render_mode`, `mode` (`"rotate"/"translate"`).

`RenderOptions` is a flat snapshot passed to the renderer each frame.

## Known issue

Tests in `tests/` reference the old `show_labels=True` API on `RenderOptions`; this was replaced by `label_mode="symbol"`. Tests need updating.
