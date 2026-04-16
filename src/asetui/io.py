from __future__ import annotations

from pathlib import Path

from ase.atoms import Atoms
from ase.io import read


def read_atoms(path: str | Path) -> Atoms:
    """Read an ASE structure file into a single Atoms object."""
    atoms = read(Path(path))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError("No frames found in input file.")
        return atoms[0]
    return atoms
