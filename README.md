# asetui

`asetui` is a lightweight in-terminal viewer for ASE `Atoms` objects — no GUI, no X11, no display server required.

Inspect crystal structures, molecules, and slabs directly from your terminal, either locally or over SSH on remote HPC clusters where launching any GUI application is slow or unavailable.

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

### Making `atui` available in your shell

`pip install --user` places the `atui` script in `~/.local/bin` (Linux) or
`~/Library/Python/<version>/bin` (macOS). If that directory is not already on
your `PATH`, add it to your shell config (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

On HPC clusters where you install into a custom prefix or a conda/venv
environment, the script lands in `<prefix>/bin`. Make sure that directory is on
your `PATH` or call `atui` with the full path.

## Usage

```bash
atui examples/H2O.xyz                      # single structure
atui examples/Cu111_CO.vasp                # VASP format
atui examples/Cu4_opt_traj.xyz.gz          # gzipped trajectory
atui examples/Cu4_opt_traj.xyz.gz@:5       # first 5 frames
atui examples/Cu4_opt_traj.xyz.gz@-1       # last frame only
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

## To-dos

- [ ] Manipulation of structures
- [ ] File IO
- [ ] TBD