# asetui

`asetui` is a lightweight terminal viewer for ASE `Atoms` objects — no GUI, no X11, no display server required.

Inspect crystal structures, molecules, and slabs directly over SSH on remote HPC clusters, where launching VESTA or ASE's GUI is slow or unavailable.

## Features

- Reads all ASE-supported formats (`xyz`, `cif`, `vasp`, and more)
- Runs entirely in the terminal over any SSH connection
- Jmol-style element coloring
- Interactive rotation, translation, and zoom
- Multiple render modes: `wire`, `ballstick`, `cpk`
- Preset views: front (`1`), side (`2`), top (`3`)
- Atom labels cycling through element symbol, index, or off
- Multi-frame trajectory support with `[` / `]` navigation

## Install

```bash
pip install --user git+https://github.com/zishengz/asetui
```

Or from a local clone:

```bash
pip install -e .
```

## Usage

```bash
atui structure.xyz
```

## Controls

| Key | Action |
|-----|--------|
| `r` / `t` | rotate / translate mode |
| Arrow keys | rotate or pan |
| `1` / `2` / `3` | preset views: front / side / top |
| `=` / `-` | zoom in / out |
| `<` / `>` | change step size |
| `[` / `]` | previous / next frame (trajectories) |
| `l` | cycle labels: symbol → index → off |
| `0` | cycle render modes |
| `h` | show / hide key reference |
| `c` | reset view |
| `q` | quit |

## Render Modes

- `wire`: character-based view with bonds and depth cues
- `ballstick`: filled atom blobs with split-color bonds
- `cpk`: space-filling view, atom size proportional to covalent radius

## Why

Visualizing structures on remote machines typically means tunneling X11, waiting for a slow GUI to load over a high-latency connection, or copying files locally first. `asetui` runs in any terminal, responds instantly, and requires nothing beyond Python and ASE.
