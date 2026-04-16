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


def read_all_frames(path: str | Path) -> list:
    """Read all frames from an ASE-supported file. Returns a list of Atoms."""
    result = read(Path(path), index=":")
    if isinstance(result, Atoms):
        return [result]
    if not result:
        raise ValueError("No frames found in input file.")
    return list(result)
