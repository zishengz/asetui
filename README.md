# asetui

`asetui` is a lightweight terminal viewer for ASE `Atoms` objects.

It reads structure files through ASE and renders them directly in the terminal
with interactive rotation, translation, zoom, labels, and multiple display
modes.

## Features

- Read common ASE-supported formats such as `xyz`, `cif`, and more
- Interactive terminal controls
- Jmol-style element coloring
- Multiple render modes:
  - `wire`
  - `ballstick`
  - `cpk`
- ASE-style preset views on `1`, `2`, and `3`

## Install

```bash
pip install -e .
```

## Usage

```bash
atui structure.xyz
```

Example:

```bash
atui examples/H2O.xyz
```

## Controls

- `r`: rotate mode
- `t`: translate mode
- Arrow keys: move or rotate
- `1` / `2` / `3`: ASE-style preset views
- `=` / `-`: zoom in and out
- `<` / `>`: change step size
- `l`: toggle element labels
- `0`: cycle render modes
- `c`: reset view
- `q`: quit

## Render Modes

- `wire`: character-based structural view with bonds
- `ballstick`: raster-like terminal view with split-color bonds
- `cpk`: larger space-filling atom view

## Why

`asetui` is meant to be a small, direct way to inspect atomistic structures
from the terminal without opening a full desktop GUI.
